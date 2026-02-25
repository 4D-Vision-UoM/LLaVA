import os
import json
import cv2
import base64
import yaml
import os
        
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

def get_total_frames(video_path):
    if not os.path.exists(video_path):
        return 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

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