import torch
import torchaudio
import os
import ffmpeg
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tools.rag_retriever_dynamic import retrieve_documents_with_dynamic

class AudioPipeline:
    def __init__(self, chunk_length_s=30):
        """
        音频处理pipeline初始化
        
        Args:
            chunk_length_s: 音频分块长度（秒）
        """
        self.chunk_length_s = chunk_length_s
        
        # 初始化Whisper模型
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        
        # 处理结果
        self.audio_path = None
        self.asr_docs = []
        self.full_transcription = []
    
    def extract_audio(self, video_path, audio_path):
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 输出音频文件路径
        """
        if not os.path.exists(audio_path):
            try:
                ffmpeg.input(video_path).output(
                    audio_path, 
                    acodec='pcm_s16le', 
                    ac=1, 
                    ar='16k'
                ).run()
                self.audio_path = audio_path
            except Exception as e:
                print(f"音频提取失败: {e}")
                return False
        else:
            self.audio_path = audio_path
        return True
    
    def chunk_audio(self, audio_path):
        """
        将音频分块处理
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            list: 音频块列表
        """
        try:
            speech, sr = torchaudio.load(audio_path)
            speech = speech.mean(dim=0)  # 转换为单声道
            speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
            
            num_samples_per_chunk = self.chunk_length_s * 16000
            chunks = []
            
            for i in range(0, len(speech), num_samples_per_chunk):
                chunks.append(speech[i:i + num_samples_per_chunk])
            
            return chunks
        except Exception as e:
            print(f"音频分块失败: {e}")
            return []
    
    def transcribe_chunk(self, chunk):
        """
        转录音频块
        
        Args:
            chunk: 音频块
            
        Returns:
            str: 转录文本
        """
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
            print(f"音频转录失败: {e}")
            return ""
    
    def get_asr_docs(self, video_path, audio_path):
        """
        获取ASR文档
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            
        Returns:
            list: ASR文档列表
        """
        self.full_transcription = []
        
        # 提取音频
        if not self.extract_audio(video_path, audio_path):
            return self.full_transcription
        
        # 分块音频
        audio_chunks = self.chunk_audio(audio_path)
        
        # 转录每个块
        for chunk in audio_chunks:
            transcription = self.transcribe_chunk(chunk)
            if transcription.strip():
                self.full_transcription.append(transcription)
        
        return self.full_transcription
    
    def load_cached_asr(self, video_path):
        """
        加载缓存的ASR结果
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            list: ASR文档列表
        """
        audio_txt_path = os.path.join(
            "restore/audio", 
            os.path.basename(video_path).split(".")[0] + ".txt"
        )
        
        if os.path.exists(audio_txt_path):
            try:
                with open(audio_txt_path, 'r', encoding='utf-8') as f:
                    self.asr_docs = [line.strip() for line in f.readlines() if line.strip()]
                return self.asr_docs
            except Exception as e:
                print(f"加载缓存ASR失败: {e}")
        
        return []
    
    def save_asr_cache(self, video_path):
        """
        保存ASR结果到缓存
        
        Args:
            video_path: 视频文件路径
        """
        audio_txt_path = os.path.join(
            "restore/audio", 
            os.path.basename(video_path).split(".")[0] + ".txt"
        )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(audio_txt_path), exist_ok=True)
        
        try:
            with open(audio_txt_path, 'w', encoding='utf-8') as f:
                for doc in self.full_transcription:
                    f.write(doc + '\n')
        except Exception as e:
            print(f"保存ASR缓存失败: {e}")
    
    def retrieve_asr_docs(self, query, threshold=0.3):
        """
        基于查询检索相关ASR文档
        
        Args:
            query: 查询列表
            threshold: 相似度阈值
            
        Returns:
            tuple: (检索到的文档, 相似度分数)
        """
        if not self.asr_docs:
            return [], []
        
        return retrieve_documents_with_dynamic(
            self.asr_docs, 
            query, 
            threshold=threshold
        )
    
    def process_audio(self, video_path, query=None, threshold=0.3):
        """
        处理音频的完整流程
        
        Args:
            video_path: 视频文件路径
            query: 查询列表
            threshold: 相似度阈值
            
        Returns:
            dict: 包含音频处理结果的字典
        """
        # 尝试加载缓存
        cached_docs = self.load_cached_asr(video_path)
        
        if cached_docs:
            self.asr_docs = cached_docs
        else:
            # 提取音频并转录
            audio_path = os.path.join(
                "restore/audio", 
                os.path.basename(video_path).split(".")[0] + ".wav"
            )
            self.full_transcription = self.get_asr_docs(video_path, audio_path)
            self.asr_docs = self.full_transcription
            
            # 保存缓存
            self.save_asr_cache(video_path)
        
        # 检索相关文档
        if query:
            retrieved_docs, scores = self.retrieve_asr_docs(query, threshold)
        else:
            retrieved_docs, scores = [], []
        
        return {
            'full_transcription': self.full_transcription,
            'asr_docs': self.asr_docs,
            'retrieved_docs': retrieved_docs,
            'scores': scores,
            'audio_path': self.audio_path
        } 