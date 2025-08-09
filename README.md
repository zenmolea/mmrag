## 📖 项目简介

本项目是基于 [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093) 论文的改进实现。参考了原始代码库的设计思路，并进行了改进：


## 🏗️ 项目结构

```
mmrag/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── setup_environment.py         # 环境配置脚本
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
git clone https://github.com/zenmolea/mmrag.git
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
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e ".[train]"
cd ..
```

5. **安装APE (用于目标检测)**
```bash
git clone https://github.com/shenyunhang/APE.git
cd APE
pip install -r requirements.txt
python -m pip install -e .
cd ..
```

6. **复制项目文件到相应目录**
```bash
# 将vidrag_pipeline下的所有文件复制到LLaVA-NeXT根目录
cp -r vidrag_pipeline/* LLaVA-NeXT/

# 将ape_tools下的所有文件复制到APE的demo目录
cp -r ape_tools/* APE/demo/
```

### 运行示例

1. **启动APE服务**
```bash
cd APE/demo
# 运行APE服务（ape_tools目录下的文件已复制到此处）
python ape_service.py
```

2. **运行主程序**
```bash
cd evals
python generate_videomme.py
```


本项目基于以下工作：
- **Video-RAG论文**: [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093)
- **原始代码库**: [Video-RAG GitHub Repository](https://github.com/LLaVA-VL/LLaVA-NeXT)
- **LLaVA-NeXT**: 开源大语言视觉模型
- **APE**: 目标检测工具
