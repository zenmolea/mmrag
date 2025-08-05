import numpy as np
import json
from tqdm import tqdm
import os
import torchaudio, ffmpeg
from pipline.asr_pipe import ASRPipeline
from pipline.ocr_pipe import OCRPipeline
from pipline.video_process import VideoProcessPipeline
from pipline.inference import InferencePipeline

max_frames_num = 64
asr_pipeline = ASRPipeline(model_name="whisper-large", chunk_length_s=30)
ocr_pipeline = OCRPipeline(languages=['en'], confidence_threshold=0.5)
video_pipeline = VideoProcessPipeline(max_frames_num=max_frames_num, fps=1, output_dir="restore")
inference_pipeline = InferencePipeline(
    model_name="LLaVA-Video-7B-Qwen2",
    clip_model_name="clip-vit-large-patch14-336",
    device="cuda",
    conv_template="qwen_1_5",
    rag_threshold=0.3,
    clip_threshold=0.3,
    beta=3.0
)





rep_list = []
USE_OCR = True
USE_ASR = True
USE_DET = True
print(f"---------------OCR: {USE_OCR}-----------------")
print(f"---------------ASR: {USE_ASR}-----------------")
print(f"---------------DET: {USE_DET}-----------------")
print(f"---------------Frames: {max_frames_num}-----------------")

file_name = f"generate_videomme"
file_path = os.path.join("restore", file_name)
if not os.path.exists(file_path):
    os.mkdir(file_path)
data_path = "/path/to/Video-MME/data"
json_file = f"results/{file_name}.json"

with open("videomme_json_file.json", 'r', encoding='utf-8') as file:
    mme_data = json.load(file)

if os.path.exists(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        rep_list = json.load(file)

index = len(rep_list)
for item in tqdm(mme_data[index:], desc="Processing items"):

    video_path = os.path.join(data_path, item['url'] + ".mp4")
    content = item.copy()
    frames, frame_time, video_time = video_pipeline.process_video(video_path, max_frames_num, 1, force_sample=True)

    video = inference_pipeline.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]

    asr_docs_total = []
    if USE_ASR: 
        asr_text_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt")
        if os.path.exists(asr_text_path): 
            with open(asr_text_path, 'r', encoding='utf-8') as f: 
                asr_docs_total = f.readlines() 
        else: 
            asr_docs_total = asr_pipeline.process_video_asr(video_path) 

    ocr_docs_total = []
    if USE_OCR:
        ocr_docs_total = ocr_pipeline.get_ocr_docs(frames)

    processed_questions = inference_pipeline.batch_process_questions(
        questions_data=content['questions'],
        video=video,
        video_time=video_time,
        frame_time=frame_time,
        frames=frames,
        asr_docs_total=asr_docs_total,
        ocr_docs_total=ocr_docs_total,
        video_pipeline=video_pipeline,
        file_name=file_name,
        max_frames_num=max_frames_num
    )
    
    content['questions'] = processed_questions

    rep_list.append(content)

    with open(json_file, "w", encoding='utf-8') as file:
        json.dump(rep_list, file, ensure_ascii=False, indent=4)
