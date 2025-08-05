#!/usr/bin/env python3
"""
Environment Verification Script for MME-Video Dataset
Checks if all required dependencies are properly installed.
"""

import sys
import os
from pathlib import Path
import subprocess

def check_python_version():
    """Check Python version"""
    print("=== Python Version Check ===")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python version too old. Need Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is not installed")
        return False

def check_torch():
    """Check PyTorch installation"""
    print("\n=== PyTorch Check ===")
    if not check_package("torch"):
        return False
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA is not available (will use CPU)")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False

def check_transformers():
    """Check Transformers installation"""
    print("\n=== Transformers Check ===")
    if not check_package("transformers"):
        return False
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"✗ Transformers check failed: {e}")
        return False

def check_video_processing():
    """Check video processing libraries"""
    print("\n=== Video Processing Check ===")
    
    packages = [
        ("decord", "decord"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("easyocr", "easyocr")
    ]
    
    success = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            success = False
    
    return success

def check_audio_processing():
    """Check audio processing libraries"""
    print("\n=== Audio Processing Check ===")
    
    packages = [
        ("torchaudio", "torchaudio"),
        ("ffmpeg-python", "ffmpeg")
    ]
    
    success = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            success = False
    
    return success

def check_utility_packages():
    """Check utility packages"""
    print("\n=== Utility Packages Check ===")
    
    packages = [
        ("tqdm", "tqdm"),
        ("requests", "requests"),
        ("pandas", "pandas"),
        ("numpy", "numpy")
    ]
    
    success = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            success = False
    
    return success

def check_ffmpeg():
    """Check FFmpeg installation"""
    print("\n=== FFmpeg Check ===")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ FFmpeg is installed")
            # Extract version from output
            lines = result.stdout.split('\n')
            if lines:
                print(f"FFmpeg version: {lines[0]}")
            return True
        else:
            print("✗ FFmpeg is not working properly")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg is not installed")
        return False
    except Exception as e:
        print(f"✗ FFmpeg check failed: {e}")
        return False

def check_pipeline_files():
    """Check if pipeline files exist"""
    print("\n=== Pipeline Files Check ===")
    
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    pipeline_files = [
        "vidrag_pipeline/video_pipeline.py",
        "vidrag_pipeline/audio_pipeline.py",
        "vidrag_pipeline/multimodal_rag_pipeline.py",
        "vidrag_pipeline/process_tool.py"
    ]
    
    success = True
    for file_path in pipeline_files:
        full_path = parent_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} not found")
            success = False
    
    return success

def check_llava():
    """Check LLaVA installation"""
    print("\n=== LLaVA Check ===")
    
    try:
        # Try to import LLaVA modules
        import llava
        print("✓ LLaVA is installed")
        return True
    except ImportError:
        print("✗ LLaVA is not installed")
        print("  Install with: pip install -e /path/to/LLaVA")
        return False
    except Exception as e:
        print(f"✗ LLaVA check failed: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\n=== Disk Space Check ===")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        # Convert to GB
        free_gb = free // (1024**3)
        total_gb = total // (1024**3)
        
        print(f"Total disk space: {total_gb} GB")
        print(f"Available disk space: {free_gb} GB")
        
        if free_gb >= 50:
            print("✓ Sufficient disk space available")
            return True
        else:
            print("⚠ Low disk space. Need at least 50GB")
            return False
    except Exception as e:
        print(f"✗ Disk space check failed: {e}")
        return False

def check_memory():
    """Check available memory"""
    print("\n=== Memory Check ===")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        # Convert to GB
        total_gb = memory.total // (1024**3)
        available_gb = memory.available // (1024**3)
        
        print(f"Total memory: {total_gb} GB")
        print(f"Available memory: {available_gb} GB")
        
        if total_gb >= 16:
            print("✓ Sufficient total memory")
            return True
        else:
            print("⚠ Low memory. Need at least 16GB")
            return False
    except ImportError:
        print("⚠ psutil not installed, skipping memory check")
        return True
    except Exception as e:
        print(f"✗ Memory check failed: {e}")
        return False

def main():
    """Main verification function"""
    print("="*60)
    print("MME-Video Environment Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_torch),
        ("Transformers", check_transformers),
        ("Video Processing", check_video_processing),
        ("Audio Processing", check_audio_processing),
        ("Utility Packages", check_utility_packages),
        ("FFmpeg", check_ffmpeg),
        ("Pipeline Files", check_pipeline_files),
        ("LLaVA", check_llava),
        ("Disk Space", check_disk_space),
        ("Memory", check_memory)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"✗ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Environment is ready for MME-Video.")
        print("\nNext steps:")
        print("1. Run: python download_mme_video.py")
        print("2. Run: python test_mme_video.py")
    else:
        print(f"\n⚠ {total - passed} checks failed.")
        print("Please fix the failed checks before proceeding.")
        print("\nSee environment_setup.md for installation instructions.")
    
    return passed == total

if __name__ == "__main__":
    main() 