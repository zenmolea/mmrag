import os
import json
import copy
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from transformers import CLIPProcessor, CLIPModel
from tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
from tools.filter_keywords import filter_keywords
from .prompts import get_prompt, get_all_prompts

class InferencePipeline:
    def __init__(self, 
                                   # 多模态大模型选择:
                  # - "LLaVA-Video-7B-Qwen2": LLaVA Video模型，基于Qwen2
                  # - "LongVA-7B": LongVA模型，支持长视频理解
                  # - "LLaVA-1.5-7B": 标准LLaVA模型，支持图像理解
                  # - "Qwen2-VL-7B": Qwen2视觉语言模型
                  # - "Qwen2.5-VL-7B": Qwen2.5视觉语言模型，更强的视觉理解能力
                  model_name="LLaVA-Video-7B-Qwen2",
                 
                 # CLIP模型选择:
                 # - "clip-vit-large-patch14-336": 标准CLIP模型
                 # - "clip-vit-base-patch32": 基础CLIP模型
                 # - "clip-vit-large-patch14": 大尺寸CLIP模型
                 clip_model_name="clip-vit-large-patch14-336",
                 
                 device="cuda",
                 
                 # 对话模板选择:
                 # - "qwen_1_5": Qwen2对话模板
                 # - "llava_v1": LLaVA v1对话模板
                 # - "llava_v1_5": LLaVA v1.5对话模板
                 conv_template="qwen_1_5",
                 
                 rag_threshold=0.3,
                 clip_threshold=0.3,
                 beta=3.0):
        self.device = device
        self.rag_threshold = rag_threshold
        self.clip_threshold = clip_threshold
        self.beta = beta
        self.conv_template = conv_template
        self.model_name = model_name  # 保存模型名称用于后续判断
        
        # 加载多模态大模型
         # 根据不同的模型类型选择不同的模型架构:
         # - LLaVA-Video系列: 使用 "llava_qwen" 架构
         # - LongVA系列: 使用 "longva" 架构  
         # - 标准LLaVA系列: 使用 "llava" 架构
         # - Qwen2-VL系列: 使用 "qwen2_vl" 架构
         # - Qwen2.5-VL系列: 使用 "qwen2_5_vl" 架构
        model_arch = "llava_qwen"  # 默认架构，可根据model_name动态调整
        if "LongVA" in model_name:
             model_arch = "longva"
        elif "LLaVA-1.5" in model_name or "LLaVA-1.6" in model_name:
             model_arch = "llava"
        elif "Qwen2.5-VL" in model_name:
             model_arch = "qwen2_5_vl"
        elif "Qwen2-VL" in model_name:
             model_arch = "qwen2_vl"
        
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_name, 
            None, 
            model_arch,  # 根据模型类型选择架构
            torch_dtype="bfloat16", 
            device_map="auto"
        )
        self.model.eval()
        
        # 加载CLIP模型用于视觉特征提取和相似度计算
        self.clip_model = CLIPModel.from_pretrained(
            clip_model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        self.prompts = get_all_prompts()
    

    
    def llava_inference(self, qs: str, video: Optional[torch.Tensor] = None, 
                       video_time: float = 0, frame_time: str = "", 
                       max_new_tokens: int = 4096) -> str:
        """
        多模态推理方法，支持不同模型的推理
        
        支持的模型类型:
        - LLaVA-Video: 支持视频输入，使用modalities=["video"]
        - LongVA: 支持长视频输入，使用modalities=["video"] 
        - LLaVA-1.5: 支持图像输入，使用modalities=["image"]
        - Qwen2-VL: 支持图像输入，使用modalities=["image"]
        """
        try:
            if video is not None:
                time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n" + qs
            else:
                question = qs
            
            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            
                     # 根据模型类型选择输入模态
         # LLaVA-Video和LongVA支持视频模态
         # LLaVA-1.5、Qwen2-VL和Qwen2.5-VL支持图像模态
            if "Video" in self.model_name or "LongVA" in self.model_name:
                modality = ["video"]
            else:
                modality = ["image"]
            
            if video is not None:
                cont = self.model.generate(
                    input_ids,
                    images=video,
                    modalities=modality,  # 动态选择模态
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                    top_p=1.0,
                    num_beams=1
                )
            else:
                cont = self.model.generate(
                    input_ids,
                    images=video,
                    modalities=modality,  # 动态选择模态
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                )
            
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            return text_outputs
            
        except Exception as e:
            print(f"LLaVA推理失败: {e}")
            return ""
    
    def get_retrieve_request(self, question: str, options: List[str]) -> Dict[str, Any]:
        try:
            retrieve_prompt = get_prompt("retrieve_request",
                question=question,
                options=" ".join(options)
            )
            
            json_request = self.llava_inference(retrieve_prompt, max_new_tokens=512)
            
            try:
                request_dict = json.loads(json_request)
                return request_dict
            except json.JSONDecodeError:
                print(f"JSON解析失败: {json_request}")
                return {"ASR": None, "DET": None, "TYPE": None}
                
        except Exception as e:
            print(f"获取检索请求失败: {e}")
            return {"ASR": None, "DET": None, "TYPE": None}
    
    def retrieve_asr_docs(self, asr_docs_total: List[str], query: List[str], 
                         request_asr: Optional[str] = None) -> List[str]:
        try:
            if not asr_docs_total:
                return []
            
            asr_query = query.copy()
            if request_asr:
                asr_query.append(request_asr)
            
            asr_docs, _ = retrieve_documents_with_dynamic(
                asr_docs_total, asr_query, threshold=self.rag_threshold
            )
            
            return asr_docs
            
        except Exception as e:
            print(f"ASR文档检索失败: {e}")
            return []
    
    def retrieve_ocr_docs(self, ocr_docs_total: List[str], query: List[str], 
                         request_det: Optional[List[str]] = None) -> List[str]:
        try:
            if not ocr_docs_total:
                return []
            
            ocr_query = query.copy()
            if request_det:
                ocr_query.extend(request_det)
            
            ocr_docs, _ = retrieve_documents_with_dynamic(
                ocr_docs_total, ocr_query, threshold=self.rag_threshold
            )
            
            return ocr_docs
            
        except Exception as e:
            print(f"OCR文档检索失败: {e}")
            return []
    
    def get_clip_similarities(self, video_tensor: torch.Tensor, 
                            request_det: Optional[List[str]] = None) -> Tuple[np.ndarray, List[int]]:
        try:
            if request_det is None or not request_det:
                clip_text = ["A picture of object"]
            else:
                request_det = filter_keywords(request_det)
                clip_text = ["A picture of " + txt for txt in request_det]
                if not clip_text:
                    clip_text = ["A picture of object"]
            
            clip_inputs = self.clip_processor(
                text=clip_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.clip_model.device)
            
            clip_img_feats = self.clip_model.get_image_features(video_tensor)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**clip_inputs)
                similarities = (clip_img_feats @ text_features.T).squeeze(0).mean(1).cpu()
                similarities = np.array(similarities, dtype=np.float64)
                alpha = self.beta * (len(similarities) / 16)
                similarities = similarities * alpha / np.sum(similarities)
            
            del clip_inputs, clip_img_feats, text_features
            torch.cuda.empty_cache()
            
            det_top_idx = [idx for idx in range(len(similarities)) if similarities[idx] > self.clip_threshold]
            
            return similarities, det_top_idx
            
        except Exception as e:
            print(f"CLIP相似度计算失败: {e}")
            return np.array([]), []
    
    def build_final_question(self, question: str, options: List[str], 
                           det_docs: List[str], det_top_idx: List[int],
                           asr_docs: List[str], ocr_docs: List[str],
                           max_frames_num: int, 
                           asr_docs_total: List[Dict] = None, 
                           ocr_docs_total: List[Dict] = None) -> str:
        qs = ""
        
        # 收集所有相关帧索引
        all_frame_indices = set()
        if det_top_idx:
            all_frame_indices.update(det_top_idx)
        
        # 从检索到的OCR内容中收集帧索引
        if ocr_docs_total and ocr_docs:
            for doc in ocr_docs_total:
                if doc.get('text') in ocr_docs:
                    frame_idx = doc.get('frame_index')
                    if frame_idx is not None:
                        all_frame_indices.add(frame_idx)
        
        # 从检索到的ASR内容中收集帧索引
        if asr_docs_total and asr_docs:
            for doc in asr_docs_total:
                if doc.get('text') in asr_docs:
                    frame_indices = doc.get('frame_indices', [])
                    all_frame_indices.update(frame_indices)
        
        # 对帧索引排序
        sorted_frame_indices = sorted(list(all_frame_indices))
        
        if sorted_frame_indices:
            qs += f"\nVideo have {str(max_frames_num)} frames in total. Synchronized multimodal information by time segments:\n"
            
            # 按帧索引组织多模态信息
            for frame_idx in sorted_frame_indices:
                frame_info = f"Frame {frame_idx + 1}: "
                modalities = []
                
                # 添加DET信息
                if frame_idx in det_top_idx:
                    det_idx = det_top_idx.index(frame_idx)
                    if det_idx < len(det_docs) and det_docs[det_idx]:
                        modalities.append(f"DET: {det_docs[det_idx]}")
                
                # 添加OCR信息
                if ocr_docs_total:
                    for doc in ocr_docs_total:
                        if doc.get('frame_index') == frame_idx and doc.get('text') in ocr_docs:
                            modalities.append(f"OCR: {doc['text']}")
                
                # 添加ASR信息
                if asr_docs_total:
                    for doc in asr_docs_total:
                        frame_indices = doc.get('frame_indices', [])
                        if frame_idx in frame_indices and doc.get('text') in asr_docs:
                            modalities.append(f"ASR: {doc['text']}")
                
                if modalities:
                    frame_info += " | ".join(modalities)
                    qs += frame_info + "\n"
        else:
            # 如果没有找到相关帧，使用原来的格式
            if det_docs:
                for i, info in enumerate(det_docs):
                    if info:
                        qs += f"Frame {str(det_top_idx[i]+1)}: {info}\n"
                if qs:
                    qs = f"\nVideo have {str(max_frames_num)} frames in total, the detected objects' information in specific frames: " + qs
            
            if asr_docs:
                qs += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join(asr_docs)
            
            if ocr_docs:
                qs += "\nVideo OCR information (given in chronological order of the video): " + "; ".join(ocr_docs)
        
        final_prompt = get_prompt("final_answer",
            question=question,
            options=" ".join(options)
        )
        qs += final_prompt
        
        return qs
    
    def process_question(self, question: str, options: List[str], 
                        video: torch.Tensor, video_time: float, frame_time: str,
                        frames: np.ndarray, asr_docs_total: List[Dict], 
                        ocr_docs_total: List[Dict], video_pipeline,
                        file_name: str, max_frames_num: int) -> str:
        try:
            retrieve_request = self.get_retrieve_request(question, options)
            
            query = [question] + options
            
            det_docs = []
            det_top_idx = []
            if retrieve_request.get("DET"):
                request_det = retrieve_request["DET"]
                
                video_tensor = []
                for frame in frames:
                    processed = self.clip_processor(
                        images=frame, 
                        return_tensors="pt"
                    )["pixel_values"].to(self.clip_model.device, dtype=torch.float16)
                    video_tensor.append(processed.squeeze(0))
                video_tensor = torch.stack(video_tensor, dim=0)
                
                similarities, det_top_idx = self.get_clip_similarities(video_tensor, request_det)
                
                if det_top_idx and request_det:
                    det_docs = video_pipeline.get_det_docs(frames[det_top_idx], request_det, file_name)
                    
                    L, R, N = False, False, False
                    det_retrieve_info = retrieve_request.get("TYPE")
                    if det_retrieve_info:
                        if "location" in det_retrieve_info:
                            L = True
                        if "relation" in det_retrieve_info:
                            R = True
                        if "number" in det_retrieve_info:
                            N = True
                    
                    det_docs = video_pipeline.det_preprocess(det_docs, location=L, relation=R, number=N)
            
            ocr_docs = []
            if ocr_docs_total:
                # 将带帧信息的OCR文档转换为纯文本列表进行检索
                ocr_texts = [doc['text'] for doc in ocr_docs_total if doc.get('text')]
                ocr_docs = self.retrieve_ocr_docs(ocr_texts, query, retrieve_request.get("DET"))
            
            asr_docs = []
            if asr_docs_total:
                # 将带帧信息的ASR文档转换为纯文本列表进行检索
                asr_texts = [doc['text'] for doc in asr_docs_total if doc.get('text')]
                asr_docs = self.retrieve_asr_docs(asr_texts, query, retrieve_request.get("ASR"))
            
            # 收集所有模态找到的内容对应的帧索引
            all_frame_indices = set()
            
            # 添加DET模态的帧索引
            if det_top_idx:
                all_frame_indices.update(det_top_idx)
            
            # 添加OCR模态的帧索引
            if ocr_docs_total:
                for doc in ocr_docs_total:
                    if doc.get('text') in ocr_docs:  # 只考虑检索到的OCR内容
                        frame_idx = doc.get('frame_index')
                        if frame_idx is not None:
                            all_frame_indices.add(frame_idx)
            
            # 添加ASR模态的帧索引
            if asr_docs_total:
                for doc in asr_docs_total:
                    if doc.get('text') in asr_docs:  # 只考虑检索到的ASR内容
                        frame_indices = doc.get('frame_indices', [])
                        all_frame_indices.update(frame_indices)
            
            # 对帧索引去重并排序
            unique_frame_indices = sorted(list(all_frame_indices))
            
            # 根据去重后的帧索引重新收集对应的内容
            if unique_frame_indices:
                # 重新收集OCR内容
                synchronized_ocr = []
                for doc in ocr_docs_total:
                    if doc.get('frame_index') in unique_frame_indices and doc.get('text'):
                        synchronized_ocr.append(doc['text'])
                if synchronized_ocr:
                    ocr_docs = synchronized_ocr
                
                # 重新收集ASR内容
                synchronized_asr = []
                for doc in asr_docs_total:
                    frame_indices = doc.get('frame_indices', [])
                    if any(frame_idx in unique_frame_indices for frame_idx in frame_indices) and doc.get('text'):
                        synchronized_asr.append(doc['text'])
                if synchronized_asr:
                    asr_docs = synchronized_asr
            

            final_question = self.build_final_question(
                question, options, det_docs, det_top_idx, 
                asr_docs, ocr_docs, max_frames_num,
                asr_docs_total, ocr_docs_total
            )
            
            answer = self.llava_inference(
                final_question, video, video_time, frame_time
            )
            
            return answer
            
        except Exception as e:
            print(f"问题处理失败: {e}")
            return ""
    
    def batch_process_questions(self, questions_data: List[Dict], 
                              video: torch.Tensor, video_time: float, frame_time: str,
                              frames: np.ndarray, asr_docs_total: List[Dict], 
                              ocr_docs_total: List[Dict], video_pipeline,
                              file_name: str, max_frames_num: int) -> List[Dict]:
        results = []
        
        for question_data in questions_data:
            question = question_data['question']
            options = question_data['options']
            
            answer = self.process_question(
                question, options, video, video_time, frame_time,
                frames, asr_docs_total, ocr_docs_total, video_pipeline,
                file_name, max_frames_num
            )
            
            question_data['response'] = answer
            results.append(question_data)
        
        return results


