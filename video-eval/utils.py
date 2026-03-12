import os
import json
import cv2
import base64
import yaml
import os
import random
from typing import List
import sys
import os
import datetime

__all__ = ["load_config", "get_total_frames", "load_json", "save_json"]
def load_config(filepath="config.yaml"):
    """Loads the YAML configuration file securely."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Falling back to environment variables.")
        return {}
        
    with open(filepath, 'r') as file:
        try:
            return yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {}

def load_json(file_path):
    """Loads a JSON file safely, catching errors from empty or corrupted files."""
    try:
        # Check if the file is completely empty (0 bytes)
        if os.path.getsize(file_path) == 0:
            print(f"\nWarning: JSON file is empty: {file_path}. Skipping.")
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except json.JSONDecodeError as e:
        print(f"\nWarning: Corrupted JSON syntax in {file_path} - {e}. Skipping.")
        return {}
    except Exception as e:
        print(f"\nWarning: Failed to read {file_path} - {e}. Skipping.")
        return {}

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
def get_total_frames_and_fps(video_path):
    """
    Extracts the total number of frames and the ORIGINAL frame rate.
    """
    if not os.path.exists(video_path):
        return 0, 10.0
    
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback if OpenCV fails to read the FPS metadata
    if orig_fps == 0 or orig_fps is None: 
        orig_fps = 30.0 
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return total_frames, orig_fps

def get_10fps_frame_map(total_frames: int, orig_fps: float) -> List[int]:
    """
    Mathematically determines which original frames correspond to a 10 FPS timeline.
    """
    duration = total_frames / orig_fps if orig_fps > 0 else 0
    num_10fps_frames = int(duration * 10)
    
    fps10_indices = []
    for i in range(num_10fps_frames):
        # Find which original frame belongs at this 1/10th of a second
        orig_frame_idx = int((i / 10.0) * orig_fps)
        fps10_indices.append(min(orig_frame_idx, total_frames - 1))
        
    if not fps10_indices and total_frames > 0:
        fps10_indices = [0]
        
    return fps10_indices

def get_frame_indices(total_frames: int, orig_fps: float, mode: str = 'window', num_frames: int = 32, stride: int = 2, split: str = 'test') -> List[int]:
    """
    1. Resamples the video down to 10 FPS.
    2. Applies windowed or 'all' sampling logic to that new 10 FPS timeline.
    """
    # Step 1: Get the 10 FPS timeline map
    fps10_indices = get_10fps_frame_map(total_frames, orig_fps)
    total_10fps_frames = len(fps10_indices)
    
    if total_10fps_frames == 0:
        return [0] * num_frames if mode == 'window' else [0]

    if mode == 'all':
        return fps10_indices # Returns all frames resampled to 10fps
        
    elif mode == 'window':
        # Step 2: Apply your window logic on the 10 FPS timeline
        window_span = (num_frames - 1) * stride + 1
        
        if split == 'train':
            start_frame = random.randint(0, max(0, total_10fps_frames - window_span))
        else:
            start_frame = max(0, (total_10fps_frames - window_span) // 2)

        final_original_indices = []
        for i in range(num_frames):
            idx_10fps = start_frame + (i * stride)
            
            # Clamp to the end of our 10 FPS list if the video is too short
            idx_10fps_clamped = min(idx_10fps, total_10fps_frames - 1)
            
            # Map back to the original video's physical frame number
            final_original_indices.append(fps10_indices[idx_10fps_clamped])
            
        return final_original_indices
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

def create_sampled_video(input_path: str, output_path: str, indices: List[int], target_fps: int = 10):
    """
    Writes the selected frames to a new temporary video encoded at exactly 10 FPS.
    """
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(target_fps), (width, height))
    
    max_idx = max(indices) if indices else 0
    current_frame = 0
    grabbed_frames = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in indices:
            grabbed_frames[current_frame] = frame
            
        current_frame += 1
        if current_frame > max_idx:
            break
            
    cap.release()
    
    for idx in indices:
        if idx in grabbed_frames:
            out.write(grabbed_frames[idx])
        elif grabbed_frames:
            out.write(list(grabbed_frames.values())[-1])
            
    out.release()