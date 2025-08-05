import os
import numpy as np
import socket
import pickle
import ast
from PIL import Image
from decord import VideoReader, cpu
from tools.scene_graph import generate_scene_graph_description

class VideoProcessPipeline:
    def __init__(self, max_frames_num=64, fps=1, output_dir="restore"):
        self.max_frames_num = max_frames_num
        self.fps = fps
        self.output_dir = output_dir
    
    def process_video(self, video_path, max_frames_num=None, fps=None, force_sample=False):
        if max_frames_num is None:
            max_frames_num = self.max_frames_num
        if fps is None:
            fps = self.fps
            
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3)), "0.00s", 0
        
        try:
            vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
            total_frame_num = len(vr)
            video_time = total_frame_num / vr.get_avg_fps()
            fps = round(vr.get_avg_fps() / fps)
            frame_idx = [i for i in range(0, len(vr), fps)]
            frame_time = [i / fps for i in frame_idx]
            
            if len(frame_idx) > max_frames_num or force_sample:
                sample_fps = max_frames_num
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            
            frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            
            return spare_frames, frame_time_str, video_time
            
        except Exception as e:
            print(f"视频处理失败: {e}")
            return np.zeros((1, 336, 336, 3)), "0.00s", 0
    
    def save_frames(self, frames, file_name, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
            
        file_paths = []
        try:
            save_dir = os.path.join(output_dir, file_name)
            os.makedirs(save_dir, exist_ok=True)
            
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                file_path = os.path.join(save_dir, f'frame_{i}.png')
                img.save(file_path)
                file_paths.append(file_path)
            
            return file_paths
            
        except Exception as e:
            print(f"保存帧失败: {e}")
            return []
    
    def get_det_docs(self, frames, prompt, file_name, server_host='0.0.0.0', server_port=9999):
        try:
            prompt = ",".join(prompt)
            frames_path = self.save_frames(frames, file_name)
            res = []
            
            if len(frames) > 0:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((server_host, server_port))
                data = (frames_path, prompt)
                client_socket.send(pickle.dumps(data))
                result_data = client_socket.recv(4096)
                try:
                    res = pickle.loads(result_data)
                except:
                    res = []
                client_socket.close()
            
            return res
            
        except Exception as e:
            print(f"目标检测失败: {e}")
            return []
    
    def det_preprocess(self, det_docs, location=False, relation=False, number=False):
        scene_descriptions = []
        
        try:
            for det_doc_per_frame in det_docs:
                objects = []
                scene_description = ""
                
                if len(det_doc_per_frame) > 0:
                    for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
                        if ":" in objs:
                            obj_name = objs.split(":")[0].strip()
                            obj_bbox = objs.split(":")[1].strip()
                            try:
                                obj_bbox = ast.literal_eval(obj_bbox)
                                objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})
                            except:
                                continue
                    
                    if objects:
                        scene_description = generate_scene_graph_description(
                            objects, location, relation, number
                        )
                
                scene_descriptions.append(scene_description)
            
            return scene_descriptions
            
        except Exception as e:
            print(f"检测文档预处理失败: {e}")
            return [""] * len(det_docs)
    
    def extract_video_info(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            duration = total_frames / fps
            width, height = vr[0].shape[1], vr[0].shape[0]
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 0
            }
            
        except Exception as e:
            print(f"视频信息提取失败: {e}")
            return {}
    
    def process_video_with_detection(self, video_path, prompt, file_name, 
                                   max_frames_num=None, fps=None, force_sample=False,
                                   location=False, relation=False, number=False):
        try:
            frames, frame_time, video_time = self.process_video(
                video_path, max_frames_num, fps, force_sample
            )
            
            det_docs = self.get_det_docs(frames, prompt, file_name)
            
            scene_descriptions = self.det_preprocess(det_docs, location, relation, number)
            
            video_info = self.extract_video_info(video_path)
            
            return {
                'frames': frames,
                'frame_time': frame_time,
                'video_time': video_time,
                'det_docs': det_docs,
                'scene_descriptions': scene_descriptions,
                'video_info': video_info,
                'frame_count': len(frames),
                'detection_count': len([d for d in det_docs if d])
            }
            
        except Exception as e:
            print(f"视频处理流程失败: {e}")
            return {
                'frames': [],
                'frame_time': "0.00s",
                'video_time': 0,
                'det_docs': [],
                'scene_descriptions': [],
                'video_info': {},
                'frame_count': 0,
                'detection_count': 0
            }
    
    def batch_process_videos(self, video_paths, prompts, file_names, **kwargs):
        results = []
        
        for i, (video_path, prompt, file_name) in enumerate(zip(video_paths, prompts, file_names)):
            print(f"处理视频 {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
            
            result = self.process_video_with_detection(
                video_path, prompt, file_name, **kwargs
            )
            results.append(result)
        
        return results
    
    def get_frame_statistics(self, frames):
        try:
            if not frames:
                return {}
            
            heights = [frame.shape[0] for frame in frames]
            widths = [frame.shape[1] for frame in frames]
            channels = [frame.shape[2] if len(frame.shape) > 2 else 1 for frame in frames]
            
            brightness_values = []
            for frame in frames:
                if len(frame.shape) == 3:
                    gray = np.mean(frame, axis=2)
                else:
                    gray = frame
                brightness_values.append(np.mean(gray))
            
            return {
                'total_frames': len(frames),
                'avg_height': np.mean(heights),
                'avg_width': np.mean(widths),
                'avg_channels': np.mean(channels),
                'avg_brightness': np.mean(brightness_values),
                'std_brightness': np.std(brightness_values),
                'min_brightness': np.min(brightness_values),
                'max_brightness': np.max(brightness_values)
            }
            
        except Exception as e:
            print(f"帧统计信息计算失败: {e}")
            return {}
