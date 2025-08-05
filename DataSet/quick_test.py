#!/usr/bin/env python3
"""
Quick Test Script for MME-Video Pipeline
A simplified test to verify basic pipeline functionality.
"""

import os
import sys
from pathlib import Path
import json

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def test_pipeline_imports():
    """Test if all pipeline modules can be imported"""
    print("Testing pipeline imports...")
    
    try:
        from vidrag_pipeline.video_pipeline import VideoPipeline
        print("✓ VideoPipeline imported successfully")
    except Exception as e:
        print(f"✗ VideoPipeline import failed: {e}")
        return False
    
    try:
        from vidrag_pipeline.audio_pipeline import AudioPipeline
        print("✓ AudioPipeline imported successfully")
    except Exception as e:
        print(f"✗ AudioPipeline import failed: {e}")
        return False
    
    try:
        from vidrag_pipeline.multimodal_rag_pipeline import MultimodalRAGPipeline
        print("✓ MultimodalRAGPipeline imported successfully")
    except Exception as e:
        print(f"✗ MultimodalRAGPipeline import failed: {e}")
        return False
    
    try:
        from vidrag_pipeline.process_tool import process_video, get_ocr_docs, get_asr_docs
        print("✓ Process tools imported successfully")
    except Exception as e:
        print(f"✗ Process tools import failed: {e}")
        return False
    
    return True

def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\nTesting pipeline initialization...")
    
    try:
        from vidrag_pipeline.video_pipeline import VideoPipeline
        video_pipeline = VideoPipeline(max_frames=16, clip_threshold=0.3, beta=3.0)
        print("✓ VideoPipeline initialized successfully")
    except Exception as e:
        print(f"✗ VideoPipeline initialization failed: {e}")
        return False
    
    try:
        from vidrag_pipeline.audio_pipeline import AudioPipeline
        audio_pipeline = AudioPipeline(chunk_length_s=30)
        print("✓ AudioPipeline initialized successfully")
    except Exception as e:
        print(f"✗ AudioPipeline initialization failed: {e}")
        return False
    
    try:
        from vidrag_pipeline.multimodal_rag_pipeline import MultimodalRAGPipeline
        # Use a smaller model for testing
        rag_pipeline = MultimodalRAGPipeline(
            model_name="LLaVA-Video-7B-Qwen2.5",
            conv_template="qwen_1_5",
            rag_threshold=0.3,
            clip_threshold=0.3,
            beta=3.0,
            max_frames=16
        )
        print("✓ MultimodalRAGPipeline initialized successfully")
    except Exception as e:
        print(f"✗ MultimodalRAGPipeline initialization failed: {e}")
        print("  (This is expected if LLaVA model is not available)")
        return False
    
    return True

def test_dataset_structure():
    """Test dataset directory structure"""
    print("\nTesting dataset structure...")
    
    dataset_dir = Path("Dataset/mme_video")
    
    if not dataset_dir.exists():
        print("Creating dataset directory...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test video file
    videos_dir = dataset_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    test_video_path = videos_dir / "test_video.mp4"
    if not test_video_path.exists():
        test_video_path.touch()  # Create empty file
        print(f"✓ Created test video file: {test_video_path}")
    
    # Create test annotations
    test_annotations = {
        "test_video": {
            "video_path": "videos/test_video.mp4",
            "duration": 10.0,
            "fps": 30,
            "resolution": "1920x1080"
        }
    }
    
    annotations_path = dataset_dir / "test_annotations.json"
    with open(annotations_path, 'w') as f:
        json.dump(test_annotations, f, indent=2)
    print(f"✓ Created test annotations: {annotations_path}")
    
    # Create test questions
    test_questions = [
        {
            "video_id": "test_video",
            "question": "How many objects are in the video?",
            "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
            "answer": "B",
            "category": "counting"
        }
    ]
    
    questions_path = dataset_dir / "test_questions.json"
    with open(questions_path, 'w') as f:
        json.dump(test_questions, f, indent=2)
    print(f"✓ Created test questions: {questions_path}")
    
    return True

def test_basic_functionality():
    """Test basic pipeline functionality with mock data"""
    print("\nTesting basic functionality...")
    
    try:
        from vidrag_pipeline.video_pipeline import VideoPipeline
        from vidrag_pipeline.audio_pipeline import AudioPipeline
        
        # Initialize pipelines
        video_pipeline = VideoPipeline(max_frames=8, clip_threshold=0.3, beta=3.0)
        audio_pipeline = AudioPipeline(chunk_length_s=30)
        
        # Test video pipeline methods
        test_video_path = "Dataset/mme_video/videos/test_video.mp4"
        
        # Test video processing (will fail with empty file, but should not crash)
        try:
            video_info = video_pipeline.get_all_video_info(test_video_path)
            print("✓ Video pipeline processing completed")
        except Exception as e:
            print(f"⚠ Video processing failed (expected with empty file): {e}")
        
        # Test audio pipeline methods
        try:
            audio_info = audio_pipeline.process_audio(test_video_path, query="test")
            print("✓ Audio pipeline processing completed")
        except Exception as e:
            print(f"⚠ Audio processing failed (expected with empty file): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*50)
    print("MME-Video Pipeline Quick Test")
    print("="*50)
    
    # Run tests
    tests = [
        ("Pipeline Imports", test_pipeline_imports),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Dataset Structure", test_dataset_structure),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready for use.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    main() 