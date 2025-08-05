from PIL import Image
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor, CLIPProcessor, CLIPModel
from decord import VideoReader, cpu
import numpy as np
import os
import easyocr
import ast
import socket
import pickle
from tools.scene_graph import generate_scene_graph_description
import ffmpeg, torchaudio

max_frames_num = 32
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float16,
                                       device_map="auto")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large",
    torch_dtype=torch.float16,
    device_map="auto"
)
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")


def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time


def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run()


def chunk_audio(audio_path, chunk_length_s=30):
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0)
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
    num_samples_per_chunk = chunk_length_s * 16000
    chunks = []
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:i + num_samples_per_chunk])
    return chunks


def transcribe_chunk(chunk):
    inputs = whisper_processor(chunk, return_tensors="pt")
    inputs["input_features"] = inputs["input_features"].to(whisper_model.device, torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            inputs["input_features"],
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def get_asr_docs(video_path, audio_path):
    full_transcription = []

    try:
        extract_audio(video_path, audio_path)
    except:
        return full_transcription
    audio_chunks = chunk_audio(audio_path, chunk_length_s=30)

    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        full_transcription.append(transcription)

    return full_transcription


def get_ocr_docs(frames):
    reader = easyocr.Reader(['en'])
    text_set = []
    ocr_docs = []
    for img in frames:
        ocr_results = reader.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 and text not in text_set:
                det_info += f"{text}; "
                text_set.append(text)
        if len(det_info) > 0:
            ocr_docs.append(det_info)

    return ocr_docs


def save_frames(frames):
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = f'restore/frame_{i}.png'
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths


def get_det_docs(frames, prompt):
    prompt = ",".join(prompt)
    frames_path = save_frames(frames)
    res = []
    if len(frames) > 0:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('0.0.0.0', 9999))
        data = (frames_path, prompt)
        client_socket.send(pickle.dumps(data))
        result_data = client_socket.recv(4096)
        try:
            res = pickle.loads(result_data)
        except:
            res = []
    return res


def det_preprocess(det_docs, location, relation, number):
    scene_descriptions = []

    for det_doc_per_frame in det_docs:
        objects = []
        scene_description = ""
        if len(det_doc_per_frame) > 0:
            for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
                obj_name = objs.split(":")[0].strip()
                obj_bbox = objs.split(":")[1].strip()
                obj_bbox = ast.literal_eval(obj_bbox)
                objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})

            scene_description = generate_scene_graph_description(objects, location, relation, number)
        scene_descriptions.append(scene_description)

    return scene_descriptions