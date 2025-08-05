# MME-Video 环境配置指南

## 系统要求

- Python 3.8+
- CUDA 11.8+ (如果使用GPU)
- 至少 16GB RAM
- 至少 50GB 可用磁盘空间

## 1. 基础环境配置

### 创建虚拟环境
```bash
# 使用 conda
conda create -n vidrag python=3.9
conda activate vidrag

# 或使用 venv
python -m venv vidrag_env
# Windows
vidrag_env\Scripts\activate
# Linux/Mac
source vidrag_env/bin/activate
```

### 安装基础依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install decord opencv-python pillow
pip install easyocr
pip install ffmpeg-python torchaudio
pip install tqdm requests pandas
```

## 2. LLaVA 模型配置

### 安装 LLaVA
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
```

### 下载模型
```bash
# 下载 LLaVA-Video-7B-Qwen2.5 模型
# 注意：这是一个大模型，需要足够的存储空间
# 模型下载地址：https://huggingface.co/LLaVA-Video-7B-Qwen2.5
```

## 3. 其他依赖

### 安装 EasyOCR
```bash
pip install easyocr
```

### 安装 FFmpeg
```bash
# Windows (使用 chocolatey)
choco install ffmpeg

# Linux
sudo apt update
sudo apt install ffmpeg

# Mac
brew install ffmpeg
```

### 安装 APE 服务 (可选)
```bash
# 如果需要对象检测功能，需要配置 APE 服务
# 这需要额外的配置，详见 APE 文档
```

## 4. 验证安装

运行验证脚本：
```bash
cd Dataset
python verify_environment.py
```

## 5. 常见问题

### CUDA 相关问题
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 内存不足
- 减少 `max_frames` 参数
- 使用 CPU 模式
- 增加虚拟内存

### 模型下载失败
- 检查网络连接
- 使用镜像源
- 手动下载模型文件

## 6. 环境变量设置

```bash
# 设置模型路径
export LLAVA_MODEL_PATH="/path/to/llava/model"

# 设置缓存目录
export HF_HOME="/path/to/huggingface/cache"

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0
```

## 7. 测试环境

配置完成后，运行以下命令测试：

```bash
# 1. 基础测试
python simple_test.py

# 2. 下载数据集
python download_mme_video.py

# 3. 完整测试
python test_mme_video.py
```

## 8. 性能优化

### GPU 优化
```bash
# 使用混合精度
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 设置内存分配策略
export CUDA_LAUNCH_BLOCKING=1
```

### 内存优化
- 减少批处理大小
- 使用梯度检查点
- 启用内存高效注意力

## 9. 故障排除

### 导入错误
```bash
# 检查 Python 路径
python -c "import sys; print(sys.path)"

# 重新安装依赖
pip install --force-reinstall package_name
```

### 模型加载错误
```bash
# 检查模型文件
ls -la /path/to/model

# 验证模型完整性
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('/path/to/model')"
```

### 权限问题
```bash
# 修复权限
chmod +x *.py
chmod -R 755 Dataset/
```

## 10. 下一步

环境配置完成后，您可以：

1. 运行 `python download_mme_video.py` 下载数据集
2. 运行 `python test_mme_video.py` 进行完整测试
3. 查看 `README.md` 了解详细使用方法
4. 根据需要调整配置参数 