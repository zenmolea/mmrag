#!/usr/bin/env python3
"""
简单的Video-MME数据集下载脚本
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_video_mme():
    """下载Video-MME数据集"""
    
    # 创建数据目录
    data_dir = Path("video-mme")
    data_dir.mkdir(exist_ok=True)
    
    print("开始下载Video-MME数据集...")
    
    try:
        # 使用huggingface datasets库下载
        from datasets import load_dataset
        
        print("正在从Hugging Face下载数据集...")
        dataset = load_dataset("THUDM/video-mme", cache_dir=str(data_dir))
        
        print(f"数据集下载完成！保存在: {data_dir}")
        print(f"数据集包含: {len(dataset)} 个分割")
        
        # 显示数据集信息
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 个样本")
            
    except ImportError:
        print("huggingface datasets库未安装，正在安装...")
        os.system("pip install datasets")
        
        from datasets import load_dataset
        print("正在从Hugging Face下载数据集...")
        dataset = load_dataset("THUDM/video-mme", cache_dir=str(data_dir))
        
        print(f"数据集下载完成！保存在: {data_dir}")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("请检查网络连接或手动下载数据集")

if __name__ == "__main__":
    download_video_mme()
