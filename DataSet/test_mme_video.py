#!/usr/bin/env python3
"""
MME-Video Dataset Test Script
Tests the MME-Video dataset with our refactored pipeline.
"""

import os
import json
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add the parent directory to the path to import our pipeline modules
sys.path.append(str(Path(__file__).parent.parent))

from vidrag_pipeline.multimodal_rag_pipeline import MultimodalRAGPipeline
from vidrag_pipeline.video_pipeline import VideoPipeline
from vidrag_pipeline.audio_pipeline import AudioPipeline

class MMEVideoTester:
    def __init__(self, dataset_dir="Dataset/mme_video", max_test_samples=5):
        self.dataset_dir = Path(dataset_dir)
        self.max_test_samples = max_test_samples
        
        # Initialize pipelines
        self.video_pipeline = VideoPipeline(max_frames=32, clip_threshold=0.3, beta=3.0)
        self.audio_pipeline = AudioPipeline(chunk_length_s=30)
        self.rag_pipeline = MultimodalRAGPipeline(
            model_name="LLaVA-Video-7B-Qwen2.5",
            conv_template="qwen_1_5",
            rag_threshold=0.3,
            clip_threshold=0.3,
            beta=3.0,
            max_frames=32
        )
        
        # Test results
        self.results = []
        
    def load_dataset(self):
        """Load MME-Video dataset"""
        print("Loading MME-Video dataset...")
        
        # Try to load from HuggingFace format first
        hf_dataset_path = self.dataset_dir / "hf_dataset"
        if hf_dataset_path.exists():
            try:
                from datasets import load_from_disk
                dataset = load_from_disk(str(hf_dataset_path))
                print(f"Loaded dataset with {len(dataset)} samples")
                return dataset
            except Exception as e:
                print(f"Error loading HF dataset: {e}")
        
        # Try to load from sample files
        sample_questions_path = self.dataset_dir / "sample_questions.json"
        sample_annotations_path = self.dataset_dir / "sample_annotations.json"
        
        if sample_questions_path.exists() and sample_annotations_path.exists():
            with open(sample_questions_path, 'r') as f:
                questions = json.load(f)
            with open(sample_annotations_path, 'r') as f:
                annotations = json.load(f)
            
            # Convert to dataset format
            dataset = {
                'train': [],
                'test': []
            }
            
            for i, q in enumerate(questions):
                video_id = q['video_id']
                video_path = self.dataset_dir / annotations[video_id]['video_path']
                
                if video_path.exists():
                    sample = {
                        'video_path': str(video_path),
                        'question': q['question'],
                        'options': q['options'],
                        'answer': q['answer'],
                        'category': q['category'],
                        'video_id': video_id
                    }
                    dataset['test'].append(sample)
            
            print(f"Loaded sample dataset with {len(dataset['test'])} test samples")
            return dataset
        
        # Create minimal test dataset
        print("Creating minimal test dataset...")
        return self.create_minimal_test_dataset()
    
    def create_minimal_test_dataset(self):
        """Create a minimal test dataset for testing"""
        videos_dir = self.dataset_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Create a test video file (empty for now)
        test_video_path = videos_dir / "test_video.mp4"
        test_video_path.touch()
        
        test_samples = [
            {
                'video_path': str(test_video_path),
                'question': 'How many objects are visible in the video?',
                'options': ['A. 1', 'B. 2', 'C. 3', 'D. 4'],
                'answer': 'B',
                'category': 'counting',
                'video_id': 'test_video'
            },
            {
                'video_path': str(test_video_path),
                'question': 'What is the main color in the video?',
                'options': ['A. Red', 'B. Blue', 'C. Green', 'D. Yellow'],
                'answer': 'A',
                'category': 'color_recognition',
                'video_id': 'test_video'
            }
        ]
        
        return {'test': test_samples}
    
    def test_video_pipeline(self, video_path):
        """Test video pipeline functionality"""
        print(f"Testing video pipeline with: {video_path}")
        
        try:
            # Test basic video processing
            video_info = self.video_pipeline.get_all_video_info(video_path)
            
            # Test OCR extraction
            ocr_docs = self.video_pipeline.extract_ocr()
            
            # Test detection (if APE service is available)
            try:
                det_docs, det_top_idx = self.video_pipeline.retrieve_detection(
                    request_det=["person", "car"], 
                    request_type=["location", "number"]
                )
            except Exception as e:
                print(f"Detection test failed (expected if APE service not running): {e}")
                det_docs, det_top_idx = [], []
            
            return {
                'success': True,
                'frames_count': len(video_info['frames']) if video_info['frames'] is not None else 0,
                'ocr_docs_count': len(ocr_docs),
                'det_docs_count': len(det_docs),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'frames_count': 0,
                'ocr_docs_count': 0,
                'det_docs_count': 0,
                'error': str(e)
            }
    
    def test_audio_pipeline(self, video_path):
        """Test audio pipeline functionality"""
        print(f"Testing audio pipeline with: {video_path}")
        
        try:
            # Test audio processing
            audio_info = self.audio_pipeline.process_audio(
                video_path=video_path,
                query="What is being said in the video?",
                threshold=0.3
            )
            
            return {
                'success': True,
                'transcription_length': len(audio_info['full_transcription']),
                'asr_docs_count': len(audio_info['asr_docs']),
                'retrieved_docs_count': len(audio_info['retrieved_docs']),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'transcription_length': 0,
                'asr_docs_count': 0,
                'retrieved_docs_count': 0,
                'error': str(e)
            }
    
    def test_rag_pipeline(self, sample):
        """Test RAG pipeline functionality"""
        print(f"Testing RAG pipeline with question: {sample['question']}")
        
        try:
            # Test full RAG pipeline
            answer = self.rag_pipeline.answer_question(
                video_path=sample['video_path'],
                question=sample['question']
            )
            
            # Get processing info
            processing_info = self.rag_pipeline.get_processing_info()
            
            return {
                'success': True,
                'answer': answer,
                'video_info_available': processing_info['video_info'] is not None,
                'audio_info_available': processing_info['audio_info'] is not None,
                'retrieval_request_available': processing_info['retrieval_request'] is not None,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'answer': None,
                'video_info_available': False,
                'audio_info_available': False,
                'retrieval_request_available': False,
                'error': str(e)
            }
    
    def run_tests(self):
        """Run all tests"""
        print("Starting MME-Video dataset tests...")
        
        # Load dataset
        dataset = self.load_dataset()
        
        if not dataset or 'test' not in dataset or len(dataset['test']) == 0:
            print("No test samples found!")
            return
        
        # Limit test samples
        test_samples = dataset['test'][:self.max_test_samples]
        
        print(f"Running tests on {len(test_samples)} samples...")
        
        for i, sample in enumerate(tqdm(test_samples, desc="Testing samples")):
            print(f"\n--- Test Sample {i+1}/{len(test_samples)} ---")
            print(f"Video: {sample['video_path']}")
            print(f"Question: {sample['question']}")
            print(f"Expected Answer: {sample['answer']}")
            
            # Test video pipeline
            video_result = self.test_video_pipeline(sample['video_path'])
            
            # Test audio pipeline
            audio_result = self.test_audio_pipeline(sample['video_path'])
            
            # Test RAG pipeline
            rag_result = self.test_rag_pipeline(sample)
            
            # Store results
            result = {
                'sample_id': i,
                'video_id': sample['video_id'],
                'question': sample['question'],
                'expected_answer': sample['answer'],
                'video_pipeline': video_result,
                'audio_pipeline': audio_result,
                'rag_pipeline': rag_result
            }
            
            self.results.append(result)
            
            # Print summary
            print(f"Video Pipeline: {'✓' if video_result['success'] else '✗'}")
            print(f"Audio Pipeline: {'✓' if audio_result['success'] else '✗'}")
            print(f"RAG Pipeline: {'✓' if rag_result['success'] else '✗'}")
            if rag_result['success']:
                print(f"Generated Answer: {rag_result['answer']}")
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*50)
        print("MME-Video Dataset Test Report")
        print("="*50)
        
        if not self.results:
            print("No test results available!")
            return
        
        total_samples = len(self.results)
        video_success = sum(1 for r in self.results if r['video_pipeline']['success'])
        audio_success = sum(1 for r in self.results if r['audio_pipeline']['success'])
        rag_success = sum(1 for r in self.results if r['rag_pipeline']['success'])
        
        print(f"Total Test Samples: {total_samples}")
        print(f"Video Pipeline Success Rate: {video_success}/{total_samples} ({video_success/total_samples*100:.1f}%)")
        print(f"Audio Pipeline Success Rate: {audio_success}/{total_samples} ({audio_success/total_samples*100:.1f}%)")
        print(f"RAG Pipeline Success Rate: {rag_success}/{total_samples} ({rag_success/total_samples*100:.1f}%)")
        
        # Detailed results
        print("\nDetailed Results:")
        for result in self.results:
            print(f"\nSample {result['sample_id']+1}:")
            print(f"  Question: {result['question']}")
            print(f"  Expected: {result['expected_answer']}")
            
            if result['rag_pipeline']['success']:
                print(f"  Generated: {result['rag_pipeline']['answer']}")
                print(f"  Match: {'✓' if result['rag_pipeline']['answer'] == result['expected_answer'] else '✗'}")
            else:
                print(f"  Error: {result['rag_pipeline']['error']}")
        
        # Save results
        report_path = self.dataset_dir / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")

def main():
    """Main test function"""
    tester = MMEVideoTester(max_test_samples=3)
    tester.run_tests()
    tester.generate_report()

if __name__ == "__main__":
    main() 