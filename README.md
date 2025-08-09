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

- **Anaconda/Miniconda** (用于环境管理)
- Python 3.8+ (通过conda安装)
- CUDA 11.8+ (用于GPU加速)
- 至少16GB GPU内存

### 自动安装

运行环境配置脚本（推荐）：

```bash
# 配置所有环境（LLaVA-NeXT + APE）
python setup_environment.py

# 或者分别配置
python setup_environment.py --env llava    # 仅配置LLaVA-NeXT环境
python setup_environment.py --env ape      # 仅配置APE环境
```

### 手动安装步骤

> **重要提示**: 由于APE和LLaVA-NeXT可能存在依赖冲突，建议分别创建独立的conda环境。

1. **克隆项目**
```bash
git clone https://github.com/zenmolea/mmrag.git
cd mmrag
```

#### 环境1: LLaVA-NeXT环境

2. **创建LLaVA-NeXT环境**
```bash
conda create -n llava-next python=3.10 -y
conda activate llava-next
```

3. **安装基础依赖**
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

5. **复制vidrag_pipeline文件**
```bash
# 将vidrag_pipeline下的所有文件复制到LLaVA-NeXT根目录
cp -r vidrag_pipeline/* LLaVA-NeXT/
```

#### 环境2: APE环境

6. **创建APE环境**
```bash
conda create -n ape python=3.8 -y
conda activate ape
```

7. **安装APE (用于目标检测)**
```bash
git clone https://github.com/shenyunhang/APE.git
cd APE
pip install -r requirements.txt
python -m pip install -e .
cd ..
```

8. **复制ape_tools文件**
```bash
# 将ape_tools下的所有文件复制到APE的demo目录
cp -r ape_tools/* APE/demo/
```

### 运行示例

> **注意**: 需要在两个不同的终端窗口中分别激活对应的环境

1. **启动APE服务** (终端1)
```bash
conda activate ape
cd APE/demo
# 运行APE服务（ape_tools目录下的文件已复制到此处）
python ape_service.py
```

2. **运行主程序** (终端2)
```bash
conda activate llava-next
cd evals
python generate_videomme.py
```


本项目基于以下工作：
- **Video-RAG论文**: [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093)
- **原始代码库**: [Video-RAG GitHub Repository](https://github.com/LLaVA-VL/LLaVA-NeXT)
- **LLaVA-NeXT**: 开源大语言视觉模型
- **APE**: 目标检测工具
