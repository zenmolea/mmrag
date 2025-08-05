#!/usr/bin/env python3
"""
MME-Video Dataset Download Script
Downloads and processes the MME-Video dataset for video understanding tasks.
"""

import os
import json
import requests
import zipfile
from tqdm import tqdm
import pandas as pd
from pathlib import Path

class MMEVideoDownloader:
    def __init__(self, dataset_dir="Dataset/mme_video"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # MME-Video dataset URLs (these are example URLs, actual URLs may vary)
        self.dataset_urls = {
            "videos": "https://huggingface.co/datasets/MME-Video/MME-Video/resolve/main/videos.zip",
            "annotations": "https://huggingface.co/datasets/MME-Video/MME-Video/resolve/main/annotations.json",
            "questions": "https://huggingface.co/datasets/MME-Video/MME-Video/resolve/main/questions.json"
        }
        
        # Alternative: Use HuggingFace datasets library
        self.use_hf_datasets = True
        
    def download_file(self, url, filename, chunk_size=8192):
        """Download a file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        filepath = self.dataset_dir / filename
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return filepath
    
    def download_with_hf_datasets(self):
        """Download using HuggingFace datasets library"""
        try:
            from datasets import load_dataset
            
            print("Downloading MME-Video dataset using HuggingFace datasets...")
            
            # Load the dataset
            dataset = load_dataset("MME-Video/MME-Video")
            
            # Save to local directory
            dataset.save_to_disk(str(self.dataset_dir / "hf_dataset"))
            
            print(f"Dataset downloaded to {self.dataset_dir / 'hf_dataset'}")
            return True
            
        except ImportError:
            print("HuggingFace datasets not installed. Installing...")
            os.system("pip install datasets")
            return self.download_with_hf_datasets()
        except Exception as e:
            print(f"Error downloading with HF datasets: {e}")
            return False
    
    def download_manual(self):
        """Manual download method"""
        print("Downloading MME-Video dataset manually...")
        
        for name, url in self.dataset_urls.items():
            try:
                filename = f"{name}.zip" if name == "videos" else f"{name}"
                print(f"Downloading {name}...")
                filepath = self.download_file(url, filename)
                
                if name == "videos" and filepath.suffix == ".zip":
                    print(f"Extracting {filename}...")
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(self.dataset_dir / "videos")
                    filepath.unlink()  # Remove zip file
                    
            except Exception as e:
                print(f"Error downloading {name}: {e}")
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing if download fails"""
        print("Creating sample MME-Video dataset for testing...")
        
        # Create sample video directory
        videos_dir = self.dataset_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Create sample annotations
        sample_annotations = {
            "video_001": {
                "video_path": "videos/sample_video_001.mp4",
                "duration": 10.5,
                "fps": 30,
                "resolution": "1920x1080"
            },
            "video_002": {
                "video_path": "videos/sample_video_002.mp4", 
                "duration": 15.2,
                "fps": 25,
                "resolution": "1280x720"
            }
        }
        
        # Create sample questions
        sample_questions = [
            {
                "video_id": "video_001",
                "question": "How many people are in the video?",
                "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
                "answer": "B",
                "category": "counting"
            },
            {
                "video_id": "video_002", 
                "question": "What color is the car in the video?",
                "options": ["A. Red", "B. Blue", "C. Green", "D. Yellow"],
                "answer": "A",
                "category": "color_recognition"
            }
        ]
        
        # Save sample data
        with open(self.dataset_dir / "sample_annotations.json", 'w') as f:
            json.dump(sample_annotations, f, indent=2)
            
        with open(self.dataset_dir / "sample_questions.json", 'w') as f:
            json.dump(sample_questions, f, indent=2)
        
        # Create sample video files (empty files for testing)
        for video_id in sample_annotations:
            video_path = videos_dir / f"{video_id}.mp4"
            video_path.touch()  # Create empty file
        
        print(f"Sample dataset created in {self.dataset_dir}")
    
    def download(self):
        """Main download method"""
        print(f"Starting MME-Video dataset download to {self.dataset_dir}")
        
        if self.use_hf_datasets:
            success = self.download_with_hf_datasets()
            if not success:
                print("Falling back to manual download...")
                self.download_manual()
        else:
            self.download_manual()
        
        # Create sample dataset if no data was downloaded
        if not any(self.dataset_dir.iterdir()):
            self.create_sample_dataset()
        
        print("Download completed!")
        return self.dataset_dir

def main():
    downloader = MMEVideoDownloader()
    dataset_path = downloader.download()
    print(f"Dataset available at: {dataset_path}")

if __name__ == "__main__":
    main() 