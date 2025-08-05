import torch
import copy
import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
from tools.filter_keywords import filter_keywords
from video_pipeline import VideoPipeline
from audio_pipeline import AudioPipeline

class MultimodalRAGPipeline:
    def __init__(self, 
                 model_name="LLaVA-Video-7B-Qwen2.5",
                 conv_template="qwen_1_5",
                 rag_threshold=0.3,
                 clip_threshold=0.3,
                 beta=3.0,
                 max_frames=32):
        """
        多模态RAG pipeline初始化
        
        Args:
            model_name: LLaVA模型名称
            conv_template: 对话模板
            rag_threshold: RAG检索阈值
            clip_threshold: CLIP相似度阈值
            beta: CLIP权重参数
            max_frames: 最大帧数
        """
        self.rag_threshold = rag_threshold
        self.clip_threshold = clip_threshold
        self.beta = beta
        self.max_frames = max_frames
        
        # 初始化LLaVA模型
        self.device = "cuda"
        overwrite_config = {}
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_name,
            None, 
            "llava_qwen", 
            torch_dtype="bfloat16", 
            device_map="auto", 
            overwrite_config=overwrite_config
        )
        self.model.eval()
        self.conv_template = conv_template
        
        # 初始化子pipeline
        self.video_pipeline = VideoPipeline(
            max_frames=max_frames,
            clip_threshold=clip_threshold,
            beta=beta
        )
        self.audio_pipeline = AudioPipeline()
        
        # 处理结果
        self.video_info = {}
        self.audio_info = {}
        self.retrieval_request = {}
    
    def llava_inference(self, qs, video):
        """
        LLaVA模型推理
        
        Args:
            qs: 问题文本
            video: 视频张量
            
        Returns:
            str: 模型回答
        """
        if video is not None:
            question = DEFAULT_IMAGE_TOKEN + qs
        else:
            question = qs
            
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt_question, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        cont = self.model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    
    def generate_retrieval_request(self, question):
        """
        生成检索请求
        
        Args:
            question: 问题文本
            
        Returns:
            dict: 检索请求字典
        """
        retrieve_prompt = "Question: " + question
        retrieve_prompt += "\nTo answer the question step by step, you can provide your retrieve request to assist you by the following json format:"
        retrieve_prompt += '''{
    "ASR": Optional[str]. The subtitles of the video that may relavent to the question you want to retrieve, in two sentences. If you no need for this information, please return null.
    "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities) All the physical entities and their location related to the question you want to retrieve, not abstract concepts. If you no need for this information, please return null.
    "TYPE": Optional[list]. (The output must be specified as null or a list containing only one or more of the following strings: 'location', 'number', 'relation'. No other values are valid for this field) The information you want to obtain about the detected objects. If you need the object location in the video frame, output "location"; if you need the number of specific object, output "number"; if you need the positional relationship between objects, output "relation". 
}
## Example 1: 
Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
Your retrieve can be:
{
    "ASR": "The location and the color of balloons, the number of the blue balloons.",
    "DET": ["blue ballons", "long table"],
    "TYPE": ["relation", "number"]
}
## Example 2: 
Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
Your retrieve can be:
{
    "ASR": null,
    "DET": ["the man in black", "woman"],
    "TYPE": ["location", "relation"]
}
## Example 3: 
Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
Your retrieve can be:
{
    "ASR": "The country recognized worldwide for its comedy.",
    "DET": null,
    "TYPE": null
}
Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format.'''
        
        json_request, _ = self.llava_inference(retrieve_prompt, None)
        
        try:
            self.retrieval_request = json.loads(json_request)
        except:
            self.retrieval_request = {
                "ASR": None,
                "DET": None,
                "TYPE": None
            }
        
        return self.retrieval_request
    
    def process_video_information(self, video_path, question):
        """
        处理视频信息
        
        Args:
            video_path: 视频路径
            question: 问题文本
        """
        # 获取检索请求
        retrieval_request = self.generate_retrieval_request(question)
        
        # 处理视频
        request_det = retrieval_request.get("DET", None)
        request_type = retrieval_request.get("TYPE", None)
        
        if request_det:
            request_det = filter_keywords(request_det)
        
        self.video_info = self.video_pipeline.get_all_video_info(
            video_path, 
            request_det, 
            request_type
        )
        
        # 检索OCR文档
        if self.video_info['ocr_docs']:
            ocr_query = [question]
            if request_det:
                ocr_query.extend(request_det)
            
            ocr_docs, _ = retrieve_documents_with_dynamic(
                self.video_info['ocr_docs'], 
                ocr_query, 
                threshold=self.rag_threshold
            )
            self.video_info['retrieved_ocr_docs'] = ocr_docs
        else:
            self.video_info['retrieved_ocr_docs'] = []
    
    def process_audio_information(self, video_path, question):
        """
        处理音频信息
        
        Args:
            video_path: 视频路径
            question: 问题文本
        """
        # 获取ASR检索请求
        request_asr = self.retrieval_request.get("ASR", None)
        
        # 处理音频
        query = [question]
        if request_asr:
            query.append(request_asr)
        
        self.audio_info = self.audio_pipeline.process_audio(
            video_path, 
            query, 
            self.rag_threshold
        )
    
    def build_question_prompt(self, question):
        """
        构建问题提示
        
        Args:
            question: 原始问题
            
        Returns:
            str: 构建的提示文本
        """
        qs = ""
        
        # 添加检测信息
        if self.video_info.get('det_docs') and len(self.video_info['det_docs']) > 0:
            for i, info in enumerate(self.video_info['det_docs']):
                if len(info) > 0:
                    frame_idx = self.video_info['det_top_idx'][i] + 1
                    qs += f"Frame {str(frame_idx)}: " + info + "\n"
            
            if len(qs) > 0:
                qs = f"\nVideo have {str(self.max_frames)} frames in total, the detected objects' information in specific frames: " + qs
        
        # 添加ASR信息
        if self.audio_info.get('retrieved_docs') and len(self.audio_info['retrieved_docs']) > 0:
            qs += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join(self.audio_info['retrieved_docs'])
        
        # 添加OCR信息
        if self.video_info.get('retrieved_ocr_docs') and len(self.video_info['retrieved_ocr_docs']) > 0:
            qs += "\nVideo OCR information (given in chronological order of the video): " + "; ".join(self.video_info['retrieved_ocr_docs'])
        
        # 添加问题
        qs += "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option. Question: " + question
        
        return qs
    
    def answer_question(self, video_path, question):
        """
        回答问题的主函数
        
        Args:
            video_path: 视频路径
            question: 问题文本
            
        Returns:
            str: 答案
        """
        # 处理视频信息
        self.process_video_information(video_path, question)
        
        # 处理音频信息
        self.process_audio_information(video_path, question)
        
        # 准备视频张量
        video = self.image_processor.preprocess(
            self.video_info['frames'], 
            return_tensors="pt"
        )["pixel_values"].cuda().bfloat16()
        video = [video]
        
        # 构建问题提示
        prompt = self.build_question_prompt(question)
        
        # 推理
        answer = self.llava_inference(prompt, video)
        
        return answer
    
    def get_processing_info(self):
        """
        获取处理信息
        
        Returns:
            dict: 处理信息字典
        """
        return {
            'video_info': self.video_info,
            'audio_info': self.audio_info,
            'retrieval_request': self.retrieval_request
        } 