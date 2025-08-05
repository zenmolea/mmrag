# MME-Video Dataset

This directory contains scripts for downloading, processing, and testing the MME-Video dataset with our refactored Video-RAG pipeline.

## Overview

MME-Video is a comprehensive video understanding dataset that includes:
- Video files with various content types
- Multiple-choice questions about video content
- Ground truth answers for evaluation
- Multiple categories (counting, color recognition, action recognition, etc.)

## Files

### Core Scripts

- **`download_mme_video.py`** - Downloads the MME-Video dataset
- **`test_mme_video.py`** - Comprehensive testing script for the dataset
- **`quick_test.py`** - Quick validation script for pipeline functionality

### Generated Files

- **`mme_video/`** - Dataset directory (created after download)
- **`test_report.json`** - Test results (generated after testing)

## Installation

### Quick Start (No Environment Setup)
If you want to test basic functionality without setting up the full environment:

```bash
cd Dataset
python minimal_test.py
```

This will create a sample dataset and test basic file operations.

### Full Environment Setup
For complete functionality, follow the detailed setup guide:

1. **Read the setup guide:**
```bash
# View the environment setup guide
cat environment_setup.md
```

2. **Install required dependencies:**
```bash
pip install datasets tqdm requests pandas
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers decord opencv-python pillow easyocr ffmpeg-python
```

3. **Verify environment:**
```bash
cd Dataset
python verify_environment.py
```

4. **Ensure the main pipeline is available:**
```bash
# Make sure you're in the project root directory
cd /path/to/Video-RAG
```

## Usage

### 1. Download the Dataset

```bash
cd Dataset
python download_mme_video.py
```

This will:
- Download the MME-Video dataset from HuggingFace
- Create a local copy in `mme_video/` directory
- Generate sample data if download fails

### 2. Quick Test

```bash
python quick_test.py
```

This will:
- Test pipeline imports
- Test pipeline initialization
- Create test dataset structure
- Test basic functionality

### 3. Full Dataset Test

```bash
python test_mme_video.py
```

This will:
- Load the MME-Video dataset
- Test all three pipelines (Video, Audio, RAG)
- Generate comprehensive test report
- Save results to `test_report.json`

## Dataset Structure

After download, the dataset will have the following structure:

```
Dataset/
├── mme_video/
│   ├── videos/                 # Video files
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   │   └── ...
│   ├── hf_dataset/            # HuggingFace format (if downloaded)
│   ├── sample_annotations.json # Sample annotations
│   ├── sample_questions.json  # Sample questions
│   └── test_report.json       # Test results
├── download_mme_video.py
├── test_mme_video.py
├── quick_test.py
└── README.md
```

## Sample Data Format

### Questions Format
```json
[
  {
    "video_id": "video_001",
    "question": "How many people are in the video?",
    "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
    "answer": "B",
    "category": "counting"
  }
]
```

### Annotations Format
```json
{
  "video_001": {
    "video_path": "videos/video_001.mp4",
    "duration": 10.5,
    "fps": 30,
    "resolution": "1920x1080"
  }
}
```

## Testing Categories

The MME-Video dataset includes various testing categories:

1. **Counting** - Count objects, people, or events
2. **Color Recognition** - Identify colors of objects
3. **Action Recognition** - Recognize actions or movements
4. **Spatial Relations** - Understand spatial relationships
5. **Temporal Relations** - Understand temporal sequences
6. **Object Recognition** - Identify specific objects
7. **Scene Understanding** - Understand overall scene context

## Pipeline Integration

The dataset is designed to work with our refactored pipeline:

- **VideoPipeline** - Processes video frames, OCR, and object detection
- **AudioPipeline** - Handles audio extraction and ASR
- **MultimodalRAGPipeline** - Integrates all modalities for question answering

## Troubleshooting

### Common Issues

1. **Download fails:**
   - Check internet connection
   - Verify HuggingFace access
   - Use sample data for testing

2. **Pipeline import errors:**
   - Ensure you're in the correct directory
   - Check that all dependencies are installed
   - Verify the pipeline files exist

3. **Model loading errors:**
   - Ensure LLaVA model is available
   - Check GPU memory requirements
   - Use CPU mode if needed

### Error Messages

- **"No module named 'vidrag_pipeline'"** - Run from project root
- **"Model not found"** - Download required models first
- **"CUDA out of memory"** - Reduce batch size or use CPU

## Performance Notes

- **Video Processing**: Depends on video length and frame count
- **Audio Processing**: ASR can be slow for long videos
- **RAG Pipeline**: Most computationally intensive step
- **Memory Usage**: Can be high with large models

## Customization

You can customize the testing by modifying:

- `max_test_samples` in test scripts
- Pipeline parameters (thresholds, model names)
- Dataset paths and formats
- Test categories and questions

## Contributing

To add new test cases or improve the dataset handling:

1. Modify the download script for new data sources
2. Add new test categories to the test scripts
3. Update the README with new features
4. Test thoroughly before committing

## License

This dataset handling code follows the same license as the main project. The MME-Video dataset itself has its own license terms. 