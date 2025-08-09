#!/usr/bin/env python3
"""
Video-RAG Pipeline 环境配置脚本
自动检查和安装所需的依赖包
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, check=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("需要Python 3.8或更高版本")
        return False
    else:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True

def check_pip():
    """检查pip是否可用"""
    print("检查pip...")
    success, stdout, stderr = run_command("pip --version")
    if success:
        print("✅ pip可用")
        return True
    else:
        print("❌ pip不可用")
        return False

def install_package(package, install_name=None):
    """安装包"""
    if install_name is None:
        install_name = package
    
    print(f"安装 {package}...")
    success, stdout, stderr = run_command(f"pip install {install_name}")
    if success:
        print(f"✅ {package} 安装成功")
        return True
    else:
        print(f"❌ {package} 安装失败: {stderr}")
        return False

def check_cuda():
    """检查CUDA是否可用"""
    print("检查CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
            return False
    except ImportError:
        print("❌ PyTorch未安装，无法检查CUDA")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Video-RAG Pipeline 环境配置脚本")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查pip
    if not check_pip():
        print("请先安装pip")
        sys.exit(1)
    
    # 核心依赖包
    core_packages = [
        ("torch", "torch==2.1.2"),
        ("torchaudio", "torchaudio==2.1.2"),
        ("numpy", "numpy"),
        ("Pillow", "Pillow"),
        ("tqdm", "tqdm"),
    ]
    
    print("\n安装核心依赖...")
    for package, install_name in core_packages:
        install_package(package, install_name)
    
    # 视频处理依赖
    video_packages = [
        ("decord", "decord"),
        ("ffmpeg-python", "ffmpeg-python"),
    ]
    
    print("\n安装视频处理依赖...")
    for package, install_name in video_packages:
        install_package(package, install_name)
    
    # AI模型依赖
    ai_packages = [
        ("transformers", "transformers"),
        ("easyocr", "easyocr"),
        ("spacy", "spacy"),
        ("faiss-cpu", "faiss-cpu"),
    ]
    
    print("\n安装AI模型依赖...")
    for package, install_name in ai_packages:
        install_package(package, install_name)
    
    # 安装spacy模型
    print("\n安装spacy模型...")
    success, stdout, stderr = run_command("python -m spacy download en_core_web_sm")
    if success:
        print("✅ spacy模型安装成功")
    else:
        print("❌ spacy模型安装失败")
    
    # 检查CUDA
    print("\n检查CUDA...")
    check_cuda()
    
    # 创建必要的目录
    print("\n创建必要的目录...")
    directories = ["restore", "restore/audio", "restore/video", "results", "test_output"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    print("\n" + "=" * 60)
    print("环境配置完成！")
    print("=" * 60)
    print("\n下一步操作：")
    print("1. 安装LLaVA-NeXT:")
    print("   git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git")
    print("   cd LLaVA-NeXT")
    print("   pip install -e \".[train]\"")
    print("   cd ..")
    print("\n2. 安装APE (用于目标检测):")
    print("   git clone https://github.com/shenyunhang/APE.git")
    print("   cd APE")
    print("   pip install -r requirements.txt")
    print("   python -m pip install -e .")
    print("   cd ..")
    print("\n3. 运行测试:")
    print("   python test_pipeline.ipynb")
    print("\n4. 启动APE服务:")
    print("   cd APE/demo")
    print("   python ape_service.py")
    print("\n5. 运行主程序:")
    print("   cd evals")
    print("   python generate_videomme.py")

if __name__ == "__main__":
    main() 