import os
import easyocr
import numpy as np
from PIL import Image
import json

class OCRPipeline:
    def __init__(self, languages=['en'], confidence_threshold=0.5):
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        
        self.reader = easyocr.Reader(languages)
    
    def extract_text_from_frame(self, frame):
        try:
            ocr_results = self.reader.readtext(frame)
            det_info = ""
            
            for result in ocr_results:
                text = result[1]
                confidence = result[2]
                
                if confidence > self.confidence_threshold:
                    det_info += f"{text}; "
            
            return det_info.strip()
            
        except Exception as e:
            print(f"单帧OCR处理失败: {e}")
            return ""
    
    def get_ocr_docs_with_frame_info(self, frames):
        """获取OCR文档，包含帧信息"""
        ocr_docs_with_frames = []
        
        try:
            for frame_idx, img in enumerate(frames):
                ocr_results = self.reader.readtext(img)
                det_info = ""
                
                for result in ocr_results:
                    text = result[1]
                    confidence = result[2]
                    
                    if confidence > self.confidence_threshold:
                        det_info += f"{text}; "
                
                if det_info.strip():
                    ocr_doc = {
                        'text': det_info.strip(),
                        'frame_index': frame_idx,
                        'confidence': confidence if ocr_results else 0.0
                    }
                    ocr_docs_with_frames.append(ocr_doc)
            
            return ocr_docs_with_frames
            
        except Exception as e:
            print(f"OCR处理失败: {e}")
            return []
    
    def get_ocr_docs(self, frames, text_set=None):
        """保持向后兼容的原始方法"""
        if text_set is None:
            text_set = []
        
        ocr_docs = []
        
        try:
            for img in frames:
                ocr_results = self.reader.readtext(img)
                det_info = ""
                
                for result in ocr_results:
                    text = result[1]
                    confidence = result[2]
                    
                    if confidence > self.confidence_threshold and text not in text_set:
                        det_info += f"{text}; "
                        text_set.append(text)
                
                if len(det_info) > 0:
                    ocr_docs.append(det_info)
            
            return ocr_docs
            
        except Exception as e:
            print(f"OCR处理失败: {e}")
            return []
    
    def process_frames_ocr(self, frames, output_dir=None, save_frames=False):
        try:
            # 获取带帧信息的OCR文档
            ocr_docs_with_frames = self.get_ocr_docs_with_frame_info(frames)
            
            total_frames = len(frames)
            frames_with_text = len(ocr_docs_with_frames)
            total_text_segments = sum(len(doc['text'].split(';')) for doc in ocr_docs_with_frames if doc['text'])
            
            frame_paths = []
            if save_frames and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    img = Image.fromarray(frame)
                    frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                    img.save(frame_path)
                    frame_paths.append(frame_path)
            
            ocr_output_path = None
            if output_dir and ocr_docs_with_frames:
                ocr_output_path = os.path.join(output_dir, "ocr_results.txt")
                try:
                    with open(ocr_output_path, 'w', encoding='utf-8') as f:
                        for doc in ocr_docs_with_frames:
                            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"保存OCR结果失败: {e}")
            
            return {
                'ocr_docs': ocr_docs_with_frames,
                'frame_paths': frame_paths,
                'ocr_output_path': ocr_output_path,
                'stats': {
                    'total_frames': total_frames,
                    'frames_with_text': frames_with_text,
                    'total_text_segments': total_text_segments,
                    'text_detection_rate': frames_with_text / total_frames if total_frames > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"OCR处理失败: {e}")
            return {
                'ocr_docs': [],
                'frame_paths': [],
                'ocr_output_path': None,
                'stats': {
                    'total_frames': len(frames),
                    'frames_with_text': 0,
                    'total_text_segments': 0,
                    'text_detection_rate': 0
                }
            }
    
    def extract_text_with_bbox(self, frame):
        try:
            ocr_results = self.reader.readtext(frame)
            text_bbox_list = []
            
            for result in ocr_results:
                bbox = result[0]
                text = result[1]
                confidence = result[2]
                
                if confidence > self.confidence_threshold:
                    text_bbox_list.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })
            
            return text_bbox_list
            
        except Exception as e:
            print(f"文本和边界框提取失败: {e}")
            return []
    
    def filter_text_by_keywords(self, ocr_docs, keywords):
        if not keywords:
            return ocr_docs
        
        filtered_docs = []
        
        try:
            for doc in ocr_docs:
                doc_lower = doc.lower()
                for keyword in keywords:
                    if keyword.lower() in doc_lower:
                        filtered_docs.append(doc)
                        break
            
            return filtered_docs
            
        except Exception as e:
            print(f"关键词过滤失败: {e}")
            return ocr_docs
    
    def get_text_statistics(self, ocr_docs):
        try:
            all_text = ' '.join(ocr_docs)
            words = all_text.split()
            
            stats = {
                'total_documents': len(ocr_docs),
                'total_characters': len(all_text),
                'total_words': len(words),
                'average_words_per_doc': len(words) / len(ocr_docs) if ocr_docs else 0,
                'unique_words': len(set(words)),
                'most_common_words': self._get_most_common_words(words, top_n=10)
            }
            
            return stats
            
        except Exception as e:
            print(f"统计信息计算失败: {e}")
            return {}
    
    def _get_most_common_words(self, words, top_n=10):
        try:
            from collections import Counter
            
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            filtered_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]
            
            word_counts = Counter(filtered_words)
            return word_counts.most_common(top_n)
            
        except Exception as e:
            print(f"常见单词统计失败: {e}")
            return []
