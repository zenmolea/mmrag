from PIL import Image
import torch
import numpy as np
import os
import easyocr
import ast
import socket
import pickle
from decord import VideoReader, cpu
from transformers import CLIPProcessor, CLIPModel
from tools.scene_graph import generate_scene_graph_description

class VideoPipeline:
    def __init__(self, max_frames=32, clip_threshold=0.3, beta=3.0):
        """
        视频处理pipeline初始化
        
        Args:
            max_frames: 最大帧数
            clip_threshold: CLIP相似度阈值
            beta: CLIP权重参数
        """
        self.max_frames = max_frames
        self.clip_threshold = clip_threshold
        self.beta = beta
        
        # 初始化模型
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.ocr_reader = easyocr.Reader(['en'])
        
        # 处理结果
        self.frames = None
        self.frame_time = None
        self.video_time = None
        self.raw_video = None
        self.video_tensor = None
        self.ocr_docs = []
        self.det_docs = []
        self.det_top_idx = []
    
    def process_video(self, video_path, fps=1, force_sample=False):
        """
        处理视频，提取帧
        
        Args:
            video_path: 视频路径
            fps: 采样帧率
            force_sample: 是否强制采样
            
        Returns:
            frames, frame_time, video_time
        """
        if self.max_frames == 0:
            return np.zeros((1, 336, 336, 3)), "", 0
            
        vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
        total_frame_num = len(vr)
        self.video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        
        if len(frame_idx) > self.max_frames or force_sample:
            sample_fps = self.max_frames
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            
        self.frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        self.frames = vr.get_batch(frame_idx).asnumpy()
        self.raw_video = [f for f in self.frames]
        
        return self.frames, self.frame_time, self.video_time
    
    def prepare_video_tensor(self):
        """准备CLIP模型所需的视频张量"""
        if self.raw_video is None:
            raise ValueError("请先调用process_video方法")
            
        self.video_tensor = []
        for frame in self.raw_video:
            processed = self.clip_processor(
                images=frame, 
                return_tensors="pt"
            )["pixel_values"].to(self.clip_model.device, dtype=torch.float16)
            self.video_tensor.append(processed.squeeze(0))
        self.video_tensor = torch.stack(self.video_tensor, dim=0)
    
    def extract_ocr(self):
        """提取OCR文本"""
        if self.frames is None:
            raise ValueError("请先调用process_video方法")
            
        text_set = []
        self.ocr_docs = []
        
        for img in self.frames:
            ocr_results = self.ocr_reader.readtext(img)
            det_info = ""
            for result in ocr_results:
                text = result[1]
                confidence = result[2]
                if confidence > 0.5 and text not in text_set:
                    det_info += f"{text}; "
                    text_set.append(text)
            if len(det_info) > 0:
                self.ocr_docs.append(det_info)
        
        return self.ocr_docs
    
    def save_frames(self, frames):
        """保存帧到文件"""
        file_paths = []
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            file_path = f'restore/frame_{i}.png'
            img.save(file_path)
            file_paths.append(file_path)
        return file_paths
    
    def get_detection_docs(self, frames, prompt):
        """获取检测文档（通过socket连接APE服务）"""
        prompt = ",".join(prompt)
        frames_path = self.save_frames(frames)
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
    
    def preprocess_detection(self, det_docs, location, relation, number):
        """预处理检测结果"""
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
                
                scene_description = generate_scene_graph_description(
                    objects, location, relation, number
                )
            scene_descriptions.append(scene_description)
        
        return scene_descriptions
    
    def retrieve_detection(self, request_det, request_type):
        """
        基于CLIP相似度检索相关帧并进行对象检测
        
        Args:
            request_det: 请求检测的对象列表
            request_type: 请求的信息类型 ['location', 'relation', 'number']
        """
        if self.video_tensor is None:
            self.prepare_video_tensor()
        
        # 准备CLIP文本输入
        if request_det and len(request_det) > 0:
            clip_text = ["A picture of " + txt for txt in request_det]
        else:
            clip_text = ["A picture of object"]
        
        # 计算相似度
        clip_inputs = self.clip_processor(
            text=clip_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.clip_model.device)
        
        clip_img_feats = self.clip_model.get_image_features(self.video_tensor)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**clip_inputs)
            similarities = (clip_img_feats @ text_features.T).squeeze(0).mean(1).cpu()
            similarities = np.array(similarities, dtype=np.float64)
            alpha = self.beta * (len(similarities) / 16)
            similarities = similarities * alpha / np.sum(similarities)
        
        # 清理GPU内存
        del clip_inputs, clip_img_feats, text_features
        torch.cuda.empty_cache()
        
        # 获取高相似度帧索引
        self.det_top_idx = [idx for idx in range(self.max_frames) if similarities[idx] > self.clip_threshold]
        
        # 获取检测结果
        if request_det and len(request_det) > 0:
            det_raw = self.get_detection_docs(
                [self.frames[i] for i in self.det_top_idx], 
                request_det
            )
            
            # 解析请求类型
            L = 'location' in request_type if request_type else False
            R = 'relation' in request_type if request_type else False
            N = 'number' in request_type if request_type else False
            
            self.det_docs = self.preprocess_detection(det_raw, location=L, relation=R, number=N)
        else:
            self.det_docs = []
        
        return self.det_docs, self.det_top_idx
    
    def get_all_video_info(self, video_path, request_det=None, request_type=None):
        """
        获取视频的所有信息
        
        Args:
            video_path: 视频路径
            request_det: 请求检测的对象
            request_type: 请求的信息类型
            
        Returns:
            dict: 包含所有视频信息的字典
        """
        # 处理视频
        self.process_video(video_path)
        
        # 提取OCR
        ocr_docs = self.extract_ocr()
        
        # 提取检测信息
        det_docs, det_top_idx = self.retrieve_detection(request_det, request_type)
        
        return {
            'frames': self.frames,
            'frame_time': self.frame_time,
            'video_time': self.video_time,
            'ocr_docs': ocr_docs,
            'det_docs': det_docs,
            'det_top_idx': det_top_idx
        } 