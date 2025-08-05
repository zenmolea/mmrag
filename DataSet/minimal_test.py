#!/usr/bin/env python3
"""
Minimal Test Script for MME-Video Dataset
Basic functionality test that can run without full environment setup.
"""

import os
import sys
from pathlib import Path
import json

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*50)
    print(title)
    print("="*50)

def test_basic_imports():
    """Test basic Python imports"""
    print_header("Basic Python Imports")
    
    basic_packages = [
        ("os", "os"),
        ("sys", "sys"),
        ("json", "json"),
        ("pathlib", "pathlib")
    ]
    
    success = True
    for package, import_name in basic_packages:
        try:
            __import__(import_name)
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            print(f"✗ {package} import failed: {e}")
            success = False
    
    return success

def test_file_structure():
    """Test if required files exist"""
    print_header("File Structure Check")
    
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    required_files = [
        "download_mme_video.py",
        "test_mme_video.py",
        "quick_test.py",
        "simple_test.py",
        "verify_environment.py",
        "README.md",
        "environment_setup.md"
    ]
    
    pipeline_files = [
        "vidrag_pipeline/video_pipeline.py",
        "vidrag_pipeline/audio_pipeline.py",
        "vidrag_pipeline/multimodal_rag_pipeline.py",
        "vidrag_pipeline/process_tool.py"
    ]
    
    success = True
    
    print("Checking Dataset files:")
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} not found")
            success = False
    
    print("\nChecking Pipeline files:")
    for file_path in pipeline_files:
        full_path = parent_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            success = False
    
    return success

def test_directory_creation():
    """Test directory creation functionality"""
    print_header("Directory Creation Test")
    
    try:
        # Create test directories
        test_dirs = [
            "test_output",
            "test_output/videos",
            "test_output/data"
        ]
        
        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
        
        # Create test files
        test_files = [
            ("test_output/test_video.mp4", ""),
            ("test_output/data/test_questions.json", '{"test": "data"}'),
            ("test_output/data/test_annotations.json", '{"test": "data"}')
        ]
        
        for file_path, content in test_files:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✓ Created file: {file_path}")
        
        # Clean up
        import shutil
        shutil.rmtree("test_output")
        print("✓ Cleaned up test files")
        
        return True
        
    except Exception as e:
        print(f"✗ Directory creation test failed: {e}")
        return False

def test_json_operations():
    """Test JSON read/write operations"""
    print_header("JSON Operations Test")
    
    try:
        # Test data
        test_data = {
            "video_id": "test_video_001",
            "question": "How many objects are in the video?",
            "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
            "answer": "B",
            "category": "counting"
        }
        
        # Write JSON
        with open("temp_test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        print("✓ JSON write successful")
        
        # Read JSON
        with open("temp_test.json", 'r') as f:
            loaded_data = json.load(f)
        print("✓ JSON read successful")
        
        # Verify data
        if loaded_data == test_data:
            print("✓ JSON data integrity verified")
        else:
            print("✗ JSON data integrity check failed")
            return False
        
        # Clean up
        os.remove("temp_test.json")
        print("✓ Cleaned up test file")
        
        return True
        
    except Exception as e:
        print(f"✗ JSON operations test failed: {e}")
        return False

def test_path_operations():
    """Test pathlib operations"""
    print_header("Path Operations Test")
    
    try:
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        
        print(f"Current directory: {current_dir}")
        print(f"Parent directory: {parent_dir}")
        
        # Test path joining
        test_path = current_dir / "test_file.txt"
        print(f"✓ Path joining: {test_path}")
        
        # Test path existence
        if not test_path.exists():
            print("✓ Path existence check working")
        else:
            print("⚠ Test file already exists")
        
        # Test directory listing
        files = list(current_dir.glob("*.py"))
        print(f"✓ Found {len(files)} Python files in current directory")
        
        return True
        
    except Exception as e:
        print(f"✗ Path operations test failed: {e}")
        return False

def create_sample_dataset():
    """Create a sample dataset for testing"""
    print_header("Sample Dataset Creation")
    
    try:
        # Create dataset directory
        dataset_dir = Path("mme_video_sample")
        dataset_dir.mkdir(exist_ok=True)
        
        videos_dir = dataset_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Create sample video file (empty)
        test_video_path = videos_dir / "sample_video.mp4"
        test_video_path.touch()
        
        # Create sample questions
        sample_questions = [
            {
                "video_id": "sample_video",
                "question": "How many people are in the video?",
                "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
                "answer": "B",
                "category": "counting"
            },
            {
                "video_id": "sample_video",
                "question": "What color is the car?",
                "options": ["A. Red", "B. Blue", "C. Green", "D. Yellow"],
                "answer": "A",
                "category": "color_recognition"
            }
        ]
        
        # Create sample annotations
        sample_annotations = {
            "sample_video": {
                "video_path": "videos/sample_video.mp4",
                "duration": 10.5,
                "fps": 30,
                "resolution": "1920x1080"
            }
        }
        
        # Save files
        with open(dataset_dir / "questions.json", 'w') as f:
            json.dump(sample_questions, f, indent=2)
        
        with open(dataset_dir / "annotations.json", 'w') as f:
            json.dump(sample_annotations, f, indent=2)
        
        print("✓ Sample dataset created successfully")
        print(f"  Location: {dataset_dir}")
        print(f"  Files created:")
        print(f"    - {test_video_path}")
        print(f"    - {dataset_dir / 'questions.json'}")
        print(f"    - {dataset_dir / 'annotations.json'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample dataset creation failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("MME-Video Minimal Test Suite")
    print("="*60)
    print("This test runs basic functionality checks without requiring")
    print("the full environment setup.")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Structure", test_file_structure),
        ("Directory Creation", test_directory_creation),
        ("JSON Operations", test_json_operations),
        ("Path Operations", test_path_operations),
        ("Sample Dataset", create_sample_dataset)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print_header("Test Summary")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed!")
        print("\nNext steps:")
        print("1. Follow environment_setup.md to configure full environment")
        print("2. Run: python verify_environment.py")
        print("3. Run: python download_mme_video.py")
        print("4. Run: python test_mme_video.py")
    else:
        print(f"\n⚠ {total - passed} tests failed.")
        print("Please check the failed tests above.")
    
    print(f"\nSample dataset created in: mme_video_sample/")
    print("You can examine the files to understand the expected format.")
    
    return passed == total

if __name__ == "__main__":
    main() 