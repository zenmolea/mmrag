# Video-RAG Pipeline

这是一个模块化的视频问答系统，将原始的单一pipeline分解为三个独立的pipeline，便于维护和扩展。

## 架构概述

系统包含三个主要pipeline：

1. **VideoPipeline** (`video_pipeline.py`) - 视频处理pipeline
2. **AudioPipeline** (`audio_pipeline.py`) - 音频处理pipeline  
3. **MultimodalRAGPipeline** (`multimodal_rag_pipeline.py`) - 多模态RAG pipeline

## 功能特性

### VideoPipeline
- 视频帧提取和预处理
- OCR文本识别
- 基于CLIP的对象检测和检索
- 场景图生成

### AudioPipeline
- 音频提取和分块
- 语音识别（ASR）
- 音频缓存管理
- 基于相似度的文档检索

### MultimodalRAGPipeline
- 多模态信息整合
- 智能检索请求生成
- LLaVA模型推理
- 问答生成

## 安装依赖

```bash
pip install torch torchaudio transformers decord easyocr ffmpeg-python
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
from multimodal_rag_pipeline import MultimodalRAGPipeline

# 创建pipeline
pipeline = MultimodalRAGPipeline()

# 回答问题
video_path = "/path/to/your/video.mp4"
question = "How many people appear in the video? A. 1. B. 2. C. 3. D. 4."

answer = pipeline.answer_question(video_path, question)
print(f"答案: {answer}")
```

### 分别使用各个pipeline

```python
from video_pipeline import VideoPipeline
from audio_pipeline import AudioPipeline

# 视频处理
video_pipeline = VideoPipeline(max_frames=32)
video_info = video_pipeline.get_all_video_info(
    video_path="/path/to/video.mp4",
    request_det=["people", "car"],
    request_type=["number", "location"]
)

# 音频处理
audio_pipeline = AudioPipeline()
audio_info = audio_pipeline.process_audio(
    video_path="/path/to/video.mp4",
    query=["people count"],
    threshold=0.3
)
```

## 配置参数

### VideoPipeline 参数
- `max_frames`: 最大帧数 (默认: 32)
- `clip_threshold`: CLIP相似度阈值 (默认: 0.3)
- `beta`: CLIP权重参数 (默认: 3.0)

### AudioPipeline 参数
- `chunk_length_s`: 音频分块长度（秒）(默认: 30)

### MultimodalRAGPipeline 参数
- `model_name`: LLaVA模型名称 (默认: "LLaVA-Video-7B-Qwen2.5")
- `conv_template`: 对话模板 (默认: "qwen_1_5")
- `rag_threshold`: RAG检索阈值 (默认: 0.3)
- `clip_threshold`: CLIP相似度阈值 (默认: 0.3)
- `beta`: CLIP权重参数 (默认: 3.0)
- `max_frames`: 最大帧数 (默认: 32)

## 高级用法

### 自定义配置

```python
# 自定义配置
custom_config = {
    'max_frames': 16,
    'rag_threshold': 0.5,
    'clip_threshold': 0.4,
    'beta': 2.0
}

pipeline = MultimodalRAGPipeline(**custom_config)
```

### 批量处理

```python
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
questions = ["Question 1", "Question 2", "Question 3"]

pipeline = MultimodalRAGPipeline()
results = []

for video_path, question in zip(video_paths, questions):
    try:
        answer = pipeline.answer_question(video_path, question)
        results.append({'video': video_path, 'answer': answer})
    except Exception as e:
        print(f"处理失败: {e}")
```

### 获取处理信息

```python
pipeline = MultimodalRAGPipeline()
answer = pipeline.answer_question(video_path, question)

# 获取详细的处理信息
info = pipeline.get_processing_info()
print(f"视频信息: {info['video_info']}")
print(f"音频信息: {info['audio_info']}")
print(f"检索请求: {info['retrieval_request']}")
```

## 文件结构

```
vidrag_pipeline/
├── video_pipeline.py          # 视频处理pipeline
├── audio_pipeline.py          # 音频处理pipeline
├── multimodal_rag_pipeline.py # 多模态RAG pipeline
├── example_usage.py           # 使用示例
├── process_tool.py            # 工具函数
├── vidrag_pipeline.py         # 原始pipeline（参考）
├── vidrag_pipeline_original.py # 原始pipeline备份
└── README.md                  # 说明文档
```

## 依赖工具

系统依赖以下工具和模型：

### 模型
- LLaVA-Video-7B-Qwen2.5 (视频理解)
- CLIP-ViT-Large-Patch14-336 (图像-文本匹配)
- Whisper-Large (语音识别)

### 工具
- EasyOCR (OCR文本识别)
- APE (对象检测，通过socket连接)
- RAG检索器 (动态文档检索)

## 注意事项

1. **内存使用**: 处理长视频时注意内存使用，可以调整`max_frames`参数
2. **GPU要求**: 需要CUDA支持的GPU来运行LLaVA和CLIP模型
3. **APE服务**: 确保APE对象检测服务在端口9999上运行
4. **缓存目录**: 系统会在`restore/`目录下缓存处理结果

## 错误处理

系统包含完善的错误处理机制：

```python
try:
    answer = pipeline.answer_question(video_path, question)
except FileNotFoundError:
    print("视频文件不存在")
except ValueError as e:
    print(f"参数错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
```

## 性能优化

1. **缓存机制**: ASR结果会自动缓存到文件
2. **GPU内存管理**: 自动清理CLIP模型中间结果
3. **批量处理**: 支持批量处理多个视频
4. **参数调优**: 可根据硬件配置调整参数

## 扩展开发

### 添加新的处理模块

```python
class CustomPipeline:
    def __init__(self, **kwargs):
        # 初始化代码
        pass
    
    def process(self, video_path):
        # 处理逻辑
        return result
```

### 集成到主pipeline

```python
class MultimodalRAGPipeline:
    def __init__(self, **kwargs):
        # 添加自定义pipeline
        self.custom_pipeline = CustomPipeline(**kwargs)
    
    def process_custom_info(self, video_path):
        return self.custom_pipeline.process(video_path)
```

## 许可证

本项目遵循原始项目的许可证。 