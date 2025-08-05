# Video-RAG Pipeline: 模块化视频理解系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 项目简介

本项目是基于 [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093) 论文的改进实现。我们参考了原始代码库的设计思路，并进行了以下改进：

- **模块化设计**：将原始的单体代码重构为独立的管道模块
- **代码优化**：简化了代码结构，提高了可维护性
- **易于扩展**：支持轻松添加新的功能模块

### 🎯 主要特性

- **ASR管道**：自动语音识别，支持音频转录
- **OCR管道**：光学字符识别，提取视频中的文本信息
- **视频处理管道**：帧提取、目标检测、场景描述
- **推理管道**：LLaVA模型推理和RAG检索
- **Prompt管理**：结构化的提示模板系统

## 🏗️ 项目结构

```
mmrag/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── setup_environment.py         # 环境配置脚本
├── test_pipeline.ipynb          # 测试notebook
├── evals/
│   ├── generate_videomme.py     # 主控制文件
│   ├── videomme_json_file.json  # 数据文件
│   └── pipline/                 # 模块化管道
│       ├── asr_pipe.py          # ASR管道
│       ├── ocr_pipe.py          # OCR管道
│       ├── video_process.py     # 视频处理管道
│       ├── inference.py         # 推理管道
│       └── prompts.py           # Prompt模板
├── vidrag_pipeline_simplified.py # 简化版本
├── vidrag_pipeline.py           # 原始版本
└── ape_tools/                   # APE工具
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (用于GPU加速)
- 至少16GB GPU内存

### 自动安装

运行环境配置脚本：

```bash
python setup_environment.py
```

### 手动安装步骤

1. **克隆项目**
```bash
git clone <your-repo-url>
cd mmrag
```

2. **创建虚拟环境**
```bash
conda create -n mmrag python=3.10 -y
conda activate mmrag
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **安装LLaVA-NeXT**
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]"
cd ..
```

5. **安装APE (用于目标检测)**
```bash
git clone https://github.com/shenyunhang/APE
cd APE
pip install -r requirements.txt
python -m pip install -e .
cd ..
```

### 运行示例

1. **启动APE服务**
```bash
cd APE/demo
python ape_service.py
```

2. **运行主程序**
```bash
cd evals
python generate_videomme.py
```

3. **运行测试notebook**
```bash
jupyter notebook test_pipeline.ipynb
```

## 📚 使用指南

### 基本用法

```python
from pipline.asr_pipe import ASRPipeline
from pipline.ocr_pipe import OCRPipeline
from pipline.video_process import VideoProcessPipeline
from pipline.inference import InferencePipeline

# 初始化管道
asr_pipeline = ASRPipeline(model_name="whisper-large")
ocr_pipeline = OCRPipeline(languages=['en'])
video_pipeline = VideoProcessPipeline(max_frames_num=64)
inference_pipeline = InferencePipeline()

# 处理视频
video_path = "path/to/your/video.mp4"
frames, frame_time, video_time = video_pipeline.process_video(video_path)

# ASR处理
asr_docs = asr_pipeline.process_video_asr(video_path)

# OCR处理
ocr_docs = ocr_pipeline.get_ocr_docs(frames)

# 推理
question = "What is happening in the video?"
answer = inference_pipeline.process_question(question, frames, asr_docs, ocr_docs)
```

### 配置参数

主要配置参数位于 `evals/generate_videomme.py`：

```python
max_frames_num = 64          # 最大帧数
USE_OCR = True              # 启用OCR
USE_ASR = True              # 启用ASR
USE_DET = True              # 启用目标检测
```

## 🔧 模块说明

### ASR管道 (`asr_pipe.py`)
- 音频提取和分块
- Whisper模型转录
- 支持多种音频格式

### OCR管道 (`ocr_pipe.py`)
- 文本检测和识别
- 多语言支持
- 置信度过滤

### 视频处理管道 (`video_process.py`)
- 帧提取和采样
- 目标检测集成
- 场景描述生成

### 推理管道 (`inference.py`)
- LLaVA模型推理
- RAG检索
- CLIP相似度计算

### Prompt管理 (`prompts.py`)
- 结构化提示模板
- 动态提示生成
- 多模态融合

## 📊 性能优化

- **内存优化**：支持GPU内存清理
- **批处理**：支持批量处理多个视频
- **缓存机制**：ASR结果缓存
- **并行处理**：多进程支持

## 🧪 测试和验证

### 运行测试notebook

```bash
jupyter notebook test_pipeline.ipynb
```

测试notebook包含：
- 环境检查和配置
- 依赖包验证
- 模块导入测试
- 基本功能测试
- 性能基准测试
- 故障排除指南

### 环境验证

```bash
python setup_environment.py
```

自动检查：
- Python版本
- CUDA环境
- 依赖包安装
- 目录结构创建

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目基于MIT许可证开源。

## 🙏 致谢

本项目基于以下工作：

- **Video-RAG论文**: [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093)
- **原始代码库**: [Video-RAG GitHub Repository](https://github.com/LLaVA-VL/LLaVA-NeXT)
- **LLaVA-NeXT**: 开源大语言视觉模型
- **APE**: 目标检测工具

## 📞 联系方式

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]

---

**注意**: 本项目仅供学术研究使用，请遵守相关法律法规和伦理准则。 