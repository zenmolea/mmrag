import os
import torch
import torchaudio
import ffmpeg
from transformers import WhisperForConditionalGeneration, WhisperProcessor

class ASRPipeline:
    def __init__(self, model_name="whisper-large", chunk_length_s=30):
        self.model_name = model_name
        self.chunk_length_s = chunk_length_s
        
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
    
    def extract_audio(self, video_path, audio_path):
        try:
            if not os.path.exists(audio_path):
                ffmpeg.input(video_path).output(
                    audio_path, 
                    acodec='pcm_s16le', 
                    ac=1, 
                    ar='16k'
                ).run()
            return True
        except Exception as e:
            print(f"音频提取失败: {e}")
            return False
    
    def chunk_audio(self, audio_path, chunk_length_s=None):
        if chunk_length_s is None:
            chunk_length_s = self.chunk_length_s
            
        try:
            speech, sr = torchaudio.load(audio_path)
            speech = speech.mean(dim=0)
            speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
            
            num_samples_per_chunk = chunk_length_s * 16000
            chunks = []
            
            for i in range(0, len(speech), num_samples_per_chunk):
                chunks.append(speech[i:i + num_samples_per_chunk])
            
            return chunks
        except Exception as e:
            print(f"音频分块失败: {e}")
            return []
    
    def transcribe_chunk(self, chunk):
        try:
            inputs = self.whisper_processor(chunk, return_tensors="pt")
            inputs["input_features"] = inputs["input_features"].to(
                self.whisper_model.device, 
                torch.float16
            )
            
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    inputs["input_features"],
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            transcription = self.whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription
        except Exception as e:
            print(f"音频块转录失败: {e}")
            return ""
    
    def get_asr_docs(self, video_path, audio_path, chunk_length_s=None):
        full_transcription = []
        
        try:
            if not self.extract_audio(video_path, audio_path):
                return full_transcription
            
            audio_chunks = self.chunk_audio(audio_path, chunk_length_s)
            
            for chunk in audio_chunks:
                transcription = self.transcribe_chunk(chunk)
                if transcription.strip():
                    full_transcription.append(transcription)
            
            return full_transcription
            
        except Exception as e:
            print(f"ASR处理失败: {e}")
            return full_transcription
    
    def process_video_asr(self, video_path, output_audio_path=None, output_text_path=None):
        if output_audio_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_audio_path = os.path.join("restore/audio", f"{base_name}.wav")
        
        if output_text_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_text_path = os.path.join("restore/audio", f"{base_name}.txt")
        
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
        
        asr_docs = self.get_asr_docs(video_path, output_audio_path)
        
        if output_text_path and asr_docs:
            try:
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    for doc in asr_docs:
                        f.write(doc + '\n')
            except Exception as e:
                print(f"保存ASR文本文件失败: {e}")
        
        return asr_docs
