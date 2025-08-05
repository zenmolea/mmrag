#!/usr/bin/env python3
"""
Video-RAG Pipeline 使用示例

这个文件展示了如何使用三个独立的pipeline：
1. VideoPipeline - 视频处理pipeline
2. AudioPipeline - 音频处理pipeline  
3. MultimodalRAGPipeline - 多模态RAG pipeline
"""

import os
from video_pipeline import VideoPipeline
from audio_pipeline import AudioPipeline
from multimodal_rag_pipeline import MultimodalRAGPipeline

def example_1_individual_pipelines():
    """
    示例1: 分别使用各个pipeline
    """
    print("=== 示例1: 分别使用各个pipeline ===")
    
    video_path = "/path/to/your/video.mp4"
    question = "How many people appear in the video? A. 1. B. 2. C. 3. D. 4."
    
    # 1. 视频处理pipeline
    print("\n1. 使用视频处理pipeline...")
    video_pipeline = VideoPipeline(max_frames=32, clip_threshold=0.3, beta=3.0)
    
    # 处理视频并获取所有信息
    video_info = video_pipeline.get_all_video_info(
        video_path=video_path,
        request_det=["people", "person"],
        request_type=["number", "location"]
    )
    
    print(f"视频时长: {video_info['video_time']:.2f}秒")
    print(f"提取帧数: {len(video_info['frames'])}")
    print(f"OCR文档数: {len(video_info['ocr_docs'])}")
    print(f"检测文档数: {len(video_info['det_docs'])}")
    
    # 2. 音频处理pipeline
    print("\n2. 使用音频处理pipeline...")
    audio_pipeline = AudioPipeline(chunk_length_s=30)
    
    audio_info = audio_pipeline.process_audio(
        video_path=video_path,
        query=[question, "people count"],
        threshold=0.3
    )
    
    print(f"音频路径: {audio_info['audio_path']}")
    print(f"完整转录数: {len(audio_info['full_transcription'])}")
    print(f"检索到的ASR文档数: {len(audio_info['retrieved_docs'])}")
    
    # 3. 多模态RAG pipeline
    print("\n3. 使用多模态RAG pipeline...")
    rag_pipeline = MultimodalRAGPipeline(
        model_name="LLaVA-Video-7B-Qwen2.5",
        rag_threshold=0.3,
        clip_threshold=0.3,
        beta=3.0,
        max_frames=32
    )
    
    answer = rag_pipeline.answer_question(video_path, question)
    print(f"答案: {answer}")
    
    # 获取处理信息
    processing_info = rag_pipeline.get_processing_info()
    print(f"检索请求: {processing_info['retrieval_request']}")

def example_2_customized_pipeline():
    """
    示例2: 自定义pipeline配置
    """
    print("\n=== 示例2: 自定义pipeline配置 ===")
    
    video_path = "/path/to/your/video.mp4"
    question = "What color is the car in the video? A. Red. B. Blue. C. Green. D. Yellow."
    
    # 自定义配置
    custom_config = {
        'max_frames': 16,  # 减少帧数以节省内存
        'rag_threshold': 0.5,  # 提高检索阈值
        'clip_threshold': 0.4,  # 提高CLIP阈值
        'beta': 2.0,  # 调整CLIP权重
        'chunk_length_s': 15  # 减少音频块长度
    }
    
    # 创建pipeline
    rag_pipeline = MultimodalRAGPipeline(
        max_frames=custom_config['max_frames'],
        rag_threshold=custom_config['rag_threshold'],
        clip_threshold=custom_config['clip_threshold'],
        beta=custom_config['beta']
    )
    
    # 回答问题
    answer = rag_pipeline.answer_question(video_path, question)
    print(f"问题: {question}")
    print(f"答案: {answer}")

def example_3_batch_processing():
    """
    示例3: 批量处理多个视频
    """
    print("\n=== 示例3: 批量处理多个视频 ===")
    
    video_paths = [
        "/path/to/video1.mp4",
        "/path/to/video2.mp4",
        "/path/to/video3.mp4"
    ]
    
    questions = [
        "How many people are in the video? A. 1. B. 2. C. 3. D. 4.",
        "What is the main object in the video? A. Car. B. Tree. C. Building. D. Animal.",
        "What color is the background? A. Red. B. Blue. C. Green. D. Yellow."
    ]
    
    # 创建pipeline（只初始化一次模型）
    rag_pipeline = MultimodalRAGPipeline()
    
    results = []
    for i, (video_path, question) in enumerate(zip(video_paths, questions)):
        print(f"\n处理视频 {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
        
        try:
            answer = rag_pipeline.answer_question(video_path, question)
            results.append({
                'video_path': video_path,
                'question': question,
                'answer': answer,
                'status': 'success'
            })
            print(f"答案: {answer}")
        except Exception as e:
            print(f"处理失败: {e}")
            results.append({
                'video_path': video_path,
                'question': question,
                'answer': None,
                'status': 'failed',
                'error': str(e)
            })
    
    # 输出结果统计
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n批量处理完成: {success_count}/{len(results)} 成功")

def example_4_error_handling():
    """
    示例4: 错误处理
    """
    print("\n=== 示例4: 错误处理 ===")
    
    # 测试不存在的视频文件
    non_existent_video = "/path/to/non_existent_video.mp4"
    question = "What is in the video?"
    
    try:
        rag_pipeline = MultimodalRAGPipeline()
        answer = rag_pipeline.answer_question(non_existent_video, question)
        print(f"答案: {answer}")
    except FileNotFoundError:
        print("错误: 视频文件不存在")
    except Exception as e:
        print(f"其他错误: {e}")
    
    # 测试空问题
    video_path = "/path/to/your/video.mp4"
    empty_question = ""
    
    try:
        rag_pipeline = MultimodalRAGPipeline()
        answer = rag_pipeline.answer_question(video_path, empty_question)
        print(f"答案: {answer}")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")

def main():
    """
    主函数
    """
    print("Video-RAG Pipeline 使用示例")
    print("=" * 50)
    
    # 检查必要的目录
    os.makedirs("restore", exist_ok=True)
    os.makedirs("restore/audio", exist_ok=True)
    
    # 运行示例
    example_1_individual_pipelines()
    example_2_customized_pipeline()
    example_3_batch_processing()
    example_4_error_handling()
    
    print("\n所有示例完成!")

if __name__ == "__main__":
    main() 