#!/usr/bin/env python3
"""
Pipeline 测试脚本

这个脚本用于测试三个pipeline的基本功能
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_pipeline import VideoPipeline
from audio_pipeline import AudioPipeline
from multimodal_rag_pipeline import MultimodalRAGPipeline

class TestVideoPipeline(unittest.TestCase):
    """测试VideoPipeline"""
    
    def setUp(self):
        """设置测试环境"""
        self.video_pipeline = VideoPipeline(max_frames=8, clip_threshold=0.3, beta=3.0)
        
    @patch('video_pipeline.VideoReader')
    def test_process_video(self, mock_video_reader):
        """测试视频处理功能"""
        # 模拟VideoReader
        mock_vr = Mock()
        mock_vr.__len__.return_value = 100
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.get_batch.return_value.asnumpy.return_value = np.random.rand(8, 336, 336, 3)
        mock_video_reader.return_value = mock_vr
        
        frames, frame_time, video_time = self.video_pipeline.process_video(
            "/test/video.mp4", fps=1, force_sample=True
        )
        
        self.assertEqual(len(frames), 8)
        self.assertIsInstance(frame_time, str)
        self.assertIsInstance(video_time, float)
    
    def test_extract_ocr(self):
        """测试OCR提取功能"""
        # 模拟frames
        self.video_pipeline.frames = np.random.rand(4, 336, 336, 3).astype(np.uint8)
        
        # 模拟EasyOCR结果
        mock_ocr_result = [("test", "Hello World", 0.8)]
        
        with patch.object(self.video_pipeline.ocr_reader, 'readtext', return_value=mock_ocr_result):
            ocr_docs = self.video_pipeline.extract_ocr()
            self.assertIsInstance(ocr_docs, list)
    
    def test_prepare_video_tensor(self):
        """测试视频张量准备功能"""
        # 模拟frames
        self.video_pipeline.raw_video = [np.random.rand(336, 336, 3).astype(np.uint8) for _ in range(4)]
        
        with patch.object(self.video_pipeline.clip_processor, '__call__') as mock_processor:
            mock_processor.return_value = {"pixel_values": np.random.rand(1, 3, 336, 336)}
            
            self.video_pipeline.prepare_video_tensor()
            self.assertIsNotNone(self.video_pipeline.video_tensor)

class TestAudioPipeline(unittest.TestCase):
    """测试AudioPipeline"""
    
    def setUp(self):
        """设置测试环境"""
        self.audio_pipeline = AudioPipeline(chunk_length_s=30)
    
    @patch('audio_pipeline.ffmpeg')
    def test_extract_audio(self, mock_ffmpeg):
        """测试音频提取功能"""
        # 模拟ffmpeg
        mock_ffmpeg.input.return_value.output.return_value.run.return_value = None
        
        result = self.audio_pipeline.extract_audio("/test/video.mp4", "/test/audio.wav")
        self.assertTrue(result)
    
    @patch('audio_pipeline.torchaudio.load')
    def test_chunk_audio(self, mock_load):
        """测试音频分块功能"""
        # 模拟音频数据
        mock_speech = np.random.rand(2, 48000)  # 2秒音频，48kHz
        mock_sr = 48000
        mock_load.return_value = (mock_speech, mock_sr)
        
        with patch('audio_pipeline.torchaudio.transforms.Resample') as mock_resample:
            mock_resample.return_value.return_value = np.random.rand(32000)  # 重采样后
            
            chunks = self.audio_pipeline.chunk_audio("/test/audio.wav")
            self.assertIsInstance(chunks, list)
    
    def test_load_cached_asr(self):
        """测试ASR缓存加载功能"""
        # 模拟缓存文件
        test_content = "Hello world\nThis is a test\n"
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.readlines.return_value = test_content.split('\n')
            
            with patch('os.path.exists', return_value=True):
                docs = self.audio_pipeline.load_cached_asr("/test/video.mp4")
                self.assertEqual(len(docs), 2)

class TestMultimodalRAGPipeline(unittest.TestCase):
    """测试MultimodalRAGPipeline"""
    
    def setUp(self):
        """设置测试环境"""
        # 模拟模型加载
        with patch('multimodal_rag_pipeline.load_pretrained_model') as mock_load:
            mock_load.return_value = (Mock(), Mock(), Mock(), 4096)
            self.rag_pipeline = MultimodalRAGPipeline(
                max_frames=8,
                rag_threshold=0.3,
                clip_threshold=0.3,
                beta=3.0
            )
    
    def test_generate_retrieval_request(self):
        """测试检索请求生成功能"""
        question = "How many people are in the video?"
        
        # 模拟LLaVA推理
        mock_response = '{"ASR": "people count", "DET": ["people"], "TYPE": ["number"]}'
        
        with patch.object(self.rag_pipeline, 'llava_inference', return_value=mock_response):
            request = self.rag_pipeline.generate_retrieval_request(question)
            self.assertIsInstance(request, dict)
            self.assertIn('ASR', request)
            self.assertIn('DET', request)
            self.assertIn('TYPE', request)
    
    def test_build_question_prompt(self):
        """测试问题提示构建功能"""
        question = "What color is the car?"
        
        # 模拟视频和音频信息
        self.rag_pipeline.video_info = {
            'det_docs': ['Frame 1: red car in center'],
            'det_top_idx': [0],
            'retrieved_ocr_docs': ['RED CAR']
        }
        self.rag_pipeline.audio_info = {
            'retrieved_docs': ['The car is red']
        }
        self.rag_pipeline.max_frames = 8
        
        prompt = self.rag_pipeline.build_question_prompt(question)
        self.assertIsInstance(prompt, str)
        self.assertIn(question, prompt)
        self.assertIn('red car', prompt)

def run_basic_tests():
    """运行基本功能测试"""
    print("开始运行Pipeline测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestVideoPipeline))
    test_suite.addTest(unittest.makeSuite(TestAudioPipeline))
    test_suite.addTest(unittest.makeSuite(TestMultimodalRAGPipeline))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print(f"\n测试结果: {result.testsRun} 个测试运行")
    print(f"失败: {len(result.failures)} 个")
    print(f"错误: {len(result.errors)} 个")
    
    return len(result.failures) == 0 and len(result.errors) == 0

def test_pipeline_integration():
    """测试pipeline集成功能"""
    print("\n测试Pipeline集成功能...")
    
    try:
        # 模拟创建pipeline
        with patch('multimodal_rag_pipeline.load_pretrained_model') as mock_load:
            mock_load.return_value = (Mock(), Mock(), Mock(), 4096)
            
            pipeline = MultimodalRAGPipeline(max_frames=4)
            
            # 测试基本属性
            self.assertIsNotNone(pipeline.video_pipeline)
            self.assertIsNotNone(pipeline.audio_pipeline)
            self.assertEqual(pipeline.max_frames, 4)
            
            print("✓ Pipeline集成测试通过")
            return True
            
    except Exception as e:
        print(f"✗ Pipeline集成测试失败: {e}")
        return False

def main():
    """主函数"""
    print("Video-RAG Pipeline 测试")
    print("=" * 40)
    
    # 运行单元测试
    unit_tests_passed = run_basic_tests()
    
    # 运行集成测试
    integration_tests_passed = test_pipeline_integration()
    
    # 总结
    print("\n" + "=" * 40)
    print("测试总结:")
    print(f"单元测试: {'通过' if unit_tests_passed else '失败'}")
    print(f"集成测试: {'通过' if integration_tests_passed else '失败'}")
    
    if unit_tests_passed and integration_tests_passed:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查代码")
        return 1

if __name__ == "__main__":
    exit(main()) 