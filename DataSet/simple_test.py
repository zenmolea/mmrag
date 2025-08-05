#!/usr/bin/env python3
"""
Simple Test Script for MME-Video Dataset
Basic functionality test without complex dependencies.
"""

import os
import sys
from pathlib import Path
import json

print("Starting simple test...")

# Add the parent directory to the path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Python path: {sys.path}")

# Test 1: Check if we can import basic modules
print("\n=== Test 1: Basic Imports ===")
try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

# Test 2: Check if pipeline files exist
print("\n=== Test 2: Pipeline Files ===")
pipeline_files = [
    "vidrag_pipeline/video_pipeline.py",
    "vidrag_pipeline/audio_pipeline.py", 
    "vidrag_pipeline/multimodal_rag_pipeline.py",
    "vidrag_pipeline/process_tool.py"
]

for file_path in pipeline_files:
    full_path = parent_dir / file_path
    if full_path.exists():
        print(f"✓ {file_path} exists")
    else:
        print(f"✗ {file_path} not found")

# Test 3: Try to import pipeline modules
print("\n=== Test 3: Pipeline Imports ===")
try:
    from vidrag_pipeline.process_tool import process_video
    print("✓ process_tool imported successfully")
except Exception as e:
    print(f"✗ process_tool import failed: {e}")

try:
    from vidrag_pipeline.video_pipeline import VideoPipeline
    print("✓ VideoPipeline imported successfully")
except Exception as e:
    print(f"✗ VideoPipeline import failed: {e}")

try:
    from vidrag_pipeline.audio_pipeline import AudioPipeline
    print("✓ AudioPipeline imported successfully")
except Exception as e:
    print(f"✗ AudioPipeline import failed: {e}")

# Test 4: Create dataset structure
print("\n=== Test 4: Dataset Structure ===")
dataset_dir = Path("mme_video")
dataset_dir.mkdir(exist_ok=True)
print(f"✓ Created dataset directory: {dataset_dir}")

videos_dir = dataset_dir / "videos"
videos_dir.mkdir(exist_ok=True)
print(f"✓ Created videos directory: {videos_dir}")

# Create a test video file
test_video_path = videos_dir / "test_video.mp4"
test_video_path.touch()
print(f"✓ Created test video file: {test_video_path}")

# Create test data
test_data = {
    "test_video": {
        "question": "How many objects are in the video?",
        "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
        "answer": "B"
    }
}

test_file = dataset_dir / "test_data.json"
with open(test_file, 'w') as f:
    json.dump(test_data, f, indent=2)
print(f"✓ Created test data file: {test_file}")

# Test 5: List directory contents
print("\n=== Test 5: Directory Contents ===")
print("Current directory contents:")
for item in current_dir.iterdir():
    print(f"  {item.name}")

print("\nDataset directory contents:")
if dataset_dir.exists():
    for item in dataset_dir.iterdir():
        print(f"  {item.name}")

print("\n=== Test Complete ===")
print("Simple test finished successfully!") 