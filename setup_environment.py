#!/usr/bin/env python3
"""
Video-RAG Pipeline 环境配置脚本
自动创建分离的conda环境并安装所需的依赖包
"""

import subprocess
import sys
import os
import platform
import argparse
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

def check_conda():
    """检查conda是否可用"""
    print("检查conda...")
    success, stdout, stderr = run_command("conda --version")
    if success:
        print("✅ conda可用")
        return True
    else:
        print("❌ conda不可用，请先安装Anaconda或Miniconda")
        return False

def create_conda_env(env_name, python_version):
    """创建conda环境"""
    print(f"创建conda环境: {env_name} (Python {python_version})")
    success, stdout, stderr = run_command(f"conda create -n {env_name} python={python_version} -y")
    if success:
        print(f"✅ 环境 {env_name} 创建成功")
        return True
    else:
        print(f"❌ 环境 {env_name} 创建失败: {stderr}")
        return False

def install_in_conda_env(env_name, package, install_name=None):
    """在指定conda环境中安装包"""
    if install_name is None:
        install_name = package
    
    print(f"在环境 {env_name} 中安装 {package}...")
    success, stdout, stderr = run_command(f"conda run -n {env_name} pip install {install_name}")
    if success:
        print(f"✅ {package} 安装成功")
        return True
    else:
        print(f"❌ {package} 安装失败: {stderr}")
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

def setup_llava_env():
    """设置LLaVA-NeXT环境"""
    env_name = "llava-next"
    python_version = "3.10"
    
    print(f"\n{'='*60}")
    print(f"设置LLaVA-NeXT环境 ({env_name})")
    print(f"{'='*60}")
    
    # 创建环境
    if not create_conda_env(env_name, python_version):
        return False
    
    # 基础依赖
    base_packages = [
        ("torch", "torch==2.1.2"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio==2.1.2"),
        ("numpy", "numpy"),
        ("Pillow", "Pillow"),
        ("tqdm", "tqdm"),
        ("transformers", "transformers"),
        ("decord", "decord"),
        ("ffmpeg-python", "ffmpeg-python"),
        ("easyocr", "easyocr"),
        ("spacy", "spacy"),
        ("faiss-cpu", "faiss-cpu"),
    ]
    
    print(f"\n在 {env_name} 环境中安装依赖...")
    for package, install_name in base_packages:
        if not install_in_conda_env(env_name, package, install_name):
            print(f"⚠️ {package} 安装失败，继续安装其他包...")
    
    # 安装requirements.txt
    print(f"\n在 {env_name} 环境中安装requirements.txt...")
    run_command(f"conda run -n {env_name} pip install -r requirements.txt")
    
    # 安装spacy模型
    print(f"\n在 {env_name} 环境中安装spacy模型...")
    run_command(f"conda run -n {env_name} python -m spacy download en_core_web_sm")
    
    # 安装LLaVA-NeXT
    print(f"\n在 {env_name} 环境中安装LLaVA-NeXT...")
    run_command("git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git")
    run_command(f"cd LLaVA-NeXT && conda run -n {env_name} pip install -e \".[train]\" && cd ..")
    
    # 复制文件
    print("\n复制vidrag_pipeline文件...")
    run_command("cp -r vidrag_pipeline/* LLaVA-NeXT/ 2>/dev/null || xcopy /E /I vidrag_pipeline LLaVA-NeXT")
    
    print(f"✅ {env_name} 环境配置完成!")
    return True

def setup_ape_env():
    """设置APE环境"""
    env_name = "ape"
    python_version = "3.8"
    
    print(f"\n{'='*60}")
    print(f"设置APE环境 ({env_name})")
    print(f"{'='*60}")
    
    # 创建环境
    if not create_conda_env(env_name, python_version):
        return False
    
    # 安装APE
    print(f"\n在 {env_name} 环境中安装APE...")
    run_command("git clone https://github.com/shenyunhang/APE.git")
    run_command(f"cd APE && conda run -n {env_name} pip install -r requirements.txt && cd ..")
    run_command(f"cd APE && conda run -n {env_name} python -m pip install -e . && cd ..")
    
    # 复制文件
    print("\n复制ape_tools文件...")
    run_command("cp -r ape_tools/* APE/demo/ 2>/dev/null || xcopy /E /I ape_tools APE\\demo")
    
    print(f"✅ {env_name} 环境配置完成!")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Video-RAG Pipeline 环境配置脚本")
    parser.add_argument("--env", choices=["all", "llava", "ape"], default="all",
                       help="要配置的环境 (默认: all)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Video-RAG Pipeline 分离环境配置脚本")
    print("=" * 60)
    
    # 检查conda
    if not check_conda():
        print("请先安装Anaconda或Miniconda")
        sys.exit(1)
    
    # 创建必要的目录
    print("\n创建必要的目录...")
    directories = ["restore", "restore/audio", "restore/video", "results", "test_output"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    # 根据参数配置环境
    if args.env in ["all", "llava"]:
        setup_llava_env()
    
    if args.env in ["all", "ape"]:
        setup_ape_env()
    
    print("\n" + "=" * 60)
    print("环境配置完成！")
    print("=" * 60)
    print("\n使用方法：")
    print("1. 启动APE服务 (终端1):")
    print("   conda activate ape")
    print("   cd APE/demo")
    print("   python ape_service.py")
    print("\n2. 运行主程序 (终端2):")
    print("   conda activate llava-next")
    print("   cd evals")
    print("   python generate_videomme.py")
    print("\n注意: 需要在两个不同的终端窗口中分别激活对应的环境")

if __name__ == "__main__":
    main() 