import os
import torch
import torchaudio
import ffmpeg
import json
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
    
    def get_video_info(self, video_path):
        """获取视频信息，包括时长、帧率等"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            fps = eval(video_info['r_frame_rate'])  # 例如 "30/1" -> 30.0
            total_frames = int(duration * fps)
            
            return {
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames
            }
        except Exception as e:
            print(f"获取视频信息失败: {e}")
            return {'duration': 0, 'fps': 30, 'total_frames': 0}
    
    def chunk_audio(self, audio_path, chunk_length_s=None):
        if chunk_length_s is None:
            chunk_length_s = self.chunk_length_s
            
        try:
            speech, sr = torchaudio.load(audio_path)
            speech = speech.mean(dim=0)
            speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
            
            num_samples_per_chunk = chunk_length_s * 16000
            chunks = []
            chunk_times = []
            
            for i in range(0, len(speech), num_samples_per_chunk):
                chunks.append(speech[i:i + num_samples_per_chunk])
                # 计算每个音频块的时间范围
                start_time = i / 16000
                end_time = min((i + num_samples_per_chunk) / 16000, len(speech) / 16000)
                chunk_times.append((start_time, end_time))
            
            return chunks, chunk_times
        except Exception as e:
            print(f"音频分块失败: {e}")
            return [], []
    
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
    
    def get_asr_docs_with_frame_info(self, video_path, audio_path=None, max_frames_num=64, chunk_length_s=None):
        """获取ASR文档，包含帧信息"""
        asr_docs_with_frames = []
        
        try:
            # 如果没有提供audio_path，自动生成
            if audio_path is None:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                audio_path = os.path.join("restore/audio", f"{base_name}.wav")
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            if not self.extract_audio(video_path, audio_path):
                return asr_docs_with_frames
            
            # 获取视频信息
            video_info = self.get_video_info(video_path)
            duration = video_info['duration']
            fps = video_info['fps']
            
            # 获取音频块和时间信息
            audio_chunks, chunk_times = self.chunk_audio(audio_path, chunk_length_s)
            
            for i, (chunk, (start_time, end_time)) in enumerate(zip(audio_chunks, chunk_times)):
                transcription = self.transcribe_chunk(chunk)
                if transcription.strip():
                    # 计算对应的帧索引
                    frame_indices = self.calculate_frame_indices(start_time, end_time, duration, max_frames_num)
                    
                    asr_doc = {
                        'text': transcription,
                        'start_time': start_time,
                        'end_time': end_time,
                        'frame_indices': frame_indices,
                        'chunk_index': i
                    }
                    asr_docs_with_frames.append(asr_doc)
            
            return asr_docs_with_frames
            
        except Exception as e:
            print(f"ASR处理失败: {e}")
            return asr_docs_with_frames
    
    def calculate_frame_indices(self, start_time, end_time, duration, max_frames_num):
        """计算音频时间段对应的帧索引"""
        try:
            if duration <= 0 or max_frames_num <= 0:
                return []
            
            # 计算帧的时间间隔
            frame_interval = duration / max_frames_num
            
            # 计算开始和结束时间对应的帧索引
            start_frame = int(start_time / frame_interval)
            end_frame = int(end_time / frame_interval)
            
            # 确保索引在有效范围内
            start_frame = max(0, min(start_frame, max_frames_num - 1))
            end_frame = max(0, min(end_frame, max_frames_num - 1))
            
            # 返回时间范围内的所有帧索引
            frame_indices = list(range(start_frame, end_frame + 1))
            return frame_indices
            
        except Exception as e:
            print(f"计算帧索引失败: {e}")
            return []
    
    def get_asr_docs(self, video_path, audio_path, chunk_length_s=None):
        """保持向后兼容的原始方法"""
        full_transcription = []
        
        try:
            if not self.extract_audio(video_path, audio_path):
                return full_transcription
            
            audio_chunks = self.chunk_audio(audio_path, chunk_length_s)
            
            for chunk in audio_chunks[0]:  # 只取chunks，不取时间信息
                transcription = self.transcribe_chunk(chunk)
                if transcription.strip():
                    full_transcription.append(transcription)
            
            return full_transcription
            
        except Exception as e:
            print(f"ASR处理失败: {e}")
            return full_transcription
    
    def process_video_asr(self, video_path, output_audio_path=None, output_text_path=None, max_frames_num=64):
        if output_audio_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_audio_path = os.path.join("restore/audio", f"{base_name}.wav")
        
        if output_text_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_text_path = os.path.join("restore/audio", f"{base_name}.txt")
        
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
        
        # 获取带帧信息的ASR文档
        asr_docs_with_frames = self.get_asr_docs_with_frame_info(video_path, output_audio_path, max_frames_num)
        
        # 保存带帧信息的ASR结果
        if output_text_path and asr_docs_with_frames:
            try:
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    for doc in asr_docs_with_frames:
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"保存ASR文本文件失败: {e}")
        
        # 返回纯文本列表以保持向后兼容
        return [doc['text'] for doc in asr_docs_with_frames]
