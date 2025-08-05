#!/usr/bin/env python3
"""
简化的Video-RAG Pipeline
使用模块化的pipeline组件
"""

import numpy as np
import json
import os
from pipline.asr_pipe import ASRPipeline
from pipline.ocr_pipe import OCRPipeline
from pipline.video_process import VideoProcessPipeline
from pipline.inference import InferencePipeline

# 配置参数
max_frames_num = 32
rag_threshold = 0.3
clip_threshold = 0.3
beta = 3.0

# 功能开关
USE_OCR = True
USE_ASR = True
USE_DET = True

print(f"---------------OCR{rag_threshold}: {USE_OCR}-----------------")
print(f"---------------ASR{rag_threshold}: {USE_ASR}-----------------")
print(f"---------------DET{beta}-{clip_threshold}: {USE_DET}-----------------")
print(f"---------------Frames: {max_frames_num}-----------------")

# 初始化所有管道
print("正在初始化管道...")

# ASR管道
asr_pipeline = ASRPipeline(model_name="whisper-large", chunk_length_s=30)

# OCR管道
ocr_pipeline = OCRPipeline(languages=['en'], confidence_threshold=0.5)

# 视频处理管道
video_pipeline = VideoProcessPipeline(max_frames_num=max_frames_num, fps=1, output_dir="restore")

# 推理管道
inference_pipeline = InferencePipeline(
    model_name="LLaVA-Video-7B-Qwen2",
    clip_model_name="clip-vit-large-patch14-336",
    device="cuda",
    conv_template="qwen_1_5",
    rag_threshold=rag_threshold,
    clip_threshold=clip_threshold,
    beta=beta
)

print("所有管道初始化完成！")

def process_single_video(video_path, question):
    print(f"\n开始处理视频: {video_path}")
    print(f"问题: {question}")
    
    try:
        # 步骤1: 处理视频帧
        print("1. 提取视频帧...")
        frames, frame_time, video_time = video_pipeline.process_video(
            video_path, max_frames_num, 1, force_sample=True
        )
        print(f"   提取了 {len(frames)} 帧，视频时长: {video_time:.2f}秒")
        
        # 步骤2: 准备视频张量
        print("2. 准备视频张量...")
        video = inference_pipeline.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        
        # 步骤3: 处理ASR
        asr_docs_total = []
        if USE_ASR:
            print("3. 处理ASR...")
            # 检查是否已有ASR文本文件
            asr_text_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt")
            if os.path.exists(asr_text_path):
                with open(asr_text_path, 'r', encoding='utf-8') as f:
                    asr_docs_total = f.readlines()
                print(f"   从缓存文件加载ASR结果: {len(asr_docs_total)} 个片段")
            else:
                # 使用ASR管道处理视频
                asr_docs_total = asr_pipeline.process_video_asr(video_path)
                print(f"   ASR处理完成: {len(asr_docs_total)} 个片段")
        
        # 步骤4: 处理OCR
        ocr_docs_total = []
        if USE_OCR:
            print("4. 处理OCR...")
            ocr_docs_total = ocr_pipeline.get_ocr_docs(frames)
            print(f"   OCR处理完成: {len(ocr_docs_total)} 个文本片段")
        
        # 步骤5: 构建问题数据
        print("5. 构建问题数据...")
        question_data = {
            'question': question,
            'options': ['A. 1', 'B. 2', 'C. 3', 'D. 4']  # 示例选项，实际使用时需要从问题中提取
        }
        
        # 步骤6: 使用推理管道处理问题
        print("6. 推理处理...")
        processed_questions = inference_pipeline.batch_process_questions(
            questions_data=[question_data],
            video=video,
            video_time=video_time,
            frame_time=frame_time,
            frames=frames,
            asr_docs_total=asr_docs_total,
            ocr_docs_total=ocr_docs_total,
            video_pipeline=video_pipeline,
            file_name="vidrag_test",
            max_frames_num=max_frames_num
        )
        
        # 步骤7: 获取答案
        answer = processed_questions[0]['response']
        print(f"7. 推理完成！")
        
        return answer
        
    except Exception as e:
        print(f"处理失败: {e}")
        return f"处理失败: {e}"

def main():
    # 示例使用
    video_path = "/path/to/your/video.mp4"  # 替换为实际视频路径
    question = "How many people appear in the video? A. 1. B. 2. C. 3. D. 4."
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        print("请修改 video_path 变量为实际的视频文件路径")
        return
    
    # 处理视频和问题
    answer = process_single_video(video_path, question)
    
    print(f"\n{'='*60}")
    print(f"最终答案: {answer}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 