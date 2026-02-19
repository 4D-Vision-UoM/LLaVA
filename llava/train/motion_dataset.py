"""
Motion-based Dataset for LLaVA Training

Integrates HumanML motion data with LLaVA's training pipeline.
Instead of loading images, this loads 4D motion sequences (frames, points, xyz+features).
"""

import os
import json
import copy
import logging
import random
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

# Import LLaVA preprocessing functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.train.train import preprocess_multimodal, preprocess

logger = logging.getLogger(__name__)


class MotionLazySupervisedDataset(Dataset):
    """
    Motion-based dataset for LLaVA fine-tuning.
    
    Loads motion sequences from PCD files and formats them for LLaVA training.
    Compatible with LLaVA's training pipeline but uses motion data instead of images.
    """

    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 data_args,
                 data_split: str = 'train',
                 num_frames: int = 32,
                 num_points: int = 2048,
                 seed: int = 42):
        """
        Args:
            data_path: Path to JSON file or directory with motion data
            tokenizer: LLaVA tokenizer
            data_args: Data arguments from LLaVA
            num_frames: Number of frames to sample per sequence
            num_points: Number of points to sample per frame
            augment: Whether to apply augmentations
            seed: Random seed
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_frames = num_frames
        self.num_points = num_points
        self.stride = 2  # Fixed temporal sampling stride
        self.seed = seed
        
        self.categories = ['Action', 'Body-Spatial', 'Temporal']
        
        # Set data_root for compatibility with _discover_samples logic
        self.data_root = Path(data_path) if not os.path.isfile(data_path) else None
        self.split = data_split  # Use provided data_split
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Discover motion sequences
        logger.info(f"Building motion dataset from: {data_path}")
        
        # Validate data path exists
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Check if data_path is a JSON file or directory
        if os.path.isfile(data_path) and data_path.endswith('.json'):
            # Load from JSON file (standard LLaVA format)
            self.list_data_dict = json.load(open(data_path, "r"))
            self.use_json_format = True
            logger.info(f"Loaded {len(self.list_data_dict)} samples from JSON file")
        else:
            # Discover from directory structure (HumanML format)
            logger.info(f"Discovering sequences from directory: {data_path}/{self.split}/Env1")
            self.list_data_dict = self._discover_and_convert(data_path)
            self.use_json_format = False
        
        if len(self.list_data_dict) == 0:
            raise ValueError(f"No samples found! Dataset is empty. Check your data path: {data_path}")
        
        logger.info(f"✓ Motion dataset initialized with {len(self.list_data_dict)} samples")

    def _discover_and_convert(self, data_path: str) -> List[Dict]:
        """
        Recursively finds all 'report.json' files in the directory,
        loads VQA data, and converts to LLaVA format.
        """
        data_root = Path(data_path)
        
        # Find split directory with Env1
        split_dir = data_root / self.split / 'Env1'

        # Find all report.json files
        report_files = sorted(list(split_dir.rglob('report.json')))
        
        logger.info(f"Found {len(report_files)} sequences in {split_dir}")
        print(f"Found {len(report_files)} sequences in {split_dir}")
        
        samples = []
        for report_path in report_files:
            seq_dir = report_path.parent
            
            # Find all PCD frames
            frame_files = sorted(list(seq_dir.glob('frame_*.pcd')))
            if not frame_files:
                continue

            # Parse frame numbers for sorting/sampling
            # Assuming format "frame_XXX.pcd"
            frames = []
            for f in frame_files:
                try:
                    num = int(f.stem.split('_')[1])
                    frames.append((num, str(f)))
                except (IndexError, ValueError):
                    pass
            
            frames.sort(key=lambda x: x[0])
            
            if not frames:
                continue
            
            # Check for VQA JSON
            vqa_json = seq_dir / f"{seq_dir.name}_qna.json"
            if not vqa_json.exists():
                continue
            
            # Load VQA data
            try:
                with open(vqa_json, 'r') as f:
                    vqa_data = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse VQA JSON {vqa_json}: {e}")
                continue
            
            # Get Q&A pairs
            qa_pairs = vqa_data.get('qa_pairs', [])
            if not qa_pairs:
                continue
            
            # Validate Q&A pairs
            valid_qa_pairs = []
            for qa_pair in qa_pairs:
                question = qa_pair.get('question', '')
                answer = qa_pair.get('answer', '')
                if question and answer:
                    valid_qa_pairs.append(qa_pair)
            
            if not valid_qa_pairs:
                continue
            
            # Create ONE sample per sequence, storing all Q&A pairs
            # Q&A selection happens at runtime in __getitem__
            sample = {
                "id": seq_dir.name,
                "motion": str(seq_dir),  # Path to sequence directory
                "qa_pairs": valid_qa_pairs,  # Store all Q&A pairs
                "frames": frames,
                "total_frames": len(frames),
            }
            
            samples.append(sample)
        
        # Calculate total Q&A pairs for logging
        total_qa_pairs = sum(len(s['qa_pairs']) for s in samples)
        logger.info(f"Created {len(samples)} sequences with {total_qa_pairs} total Q&A pairs from {len(report_files)} report files")
        
        if len(samples) == 0:
            logger.error(f"No samples found in {split_dir}")
            logger.error(f"Checked for: report.json files and corresponding _qna.json files")
            raise ValueError(f"No training samples found in {split_dir}. "
                           "Please check that the data directory contains sequences with report.json "
                           "and *_qna.json files.")
        print(f"✓ Discovered {len(samples)} sequences with {total_qa_pairs} Q&A pairs (train: random selection, test: first Q&A)")
        return samples

    def _load_pcd(self, pcd_path: str) -> np.ndarray:
        """
        Pure Python PCD loader to avoid Open3D segfaults.
        Returns (N, 9) array: [x, y, z, r, g, b, nx, ny, nz]
        """
        points, colors, normals = [], [], []
        
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('DATA ascii'):
                    header_end = i + 1
                    break
            
            # Default values
            default_color = [0.5, 0.5, 0.5] 
            default_normal = [0.0, 0.0, 0.0]

            for line in lines[header_end:]:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                # Parse XYZ
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                
                # Parse RGB (Attempt packed int or float columns)
                c = default_color
                if len(parts) >= 4:
                    try:
                        # Try packed RGB (common in PCL)
                        packed_rgb = int(float(parts[3]))
                        r = ((packed_rgb >> 16) & 255) / 255.0
                        g = ((packed_rgb >> 8) & 255) / 255.0
                        b = (packed_rgb & 255) / 255.0
                        c = [r, g, b]
                    except:
                        pass
                colors.append(c)
                
                # Parse Normals (Attempt columns 4,5,6)
                n = default_normal
                if len(parts) >= 7:
                    try:
                        n = [float(parts[4]), float(parts[5]), float(parts[6])]
                    except:
                        pass
                normals.append(n)

            # Convert to numpy
            pts = np.array(points, dtype=np.float32)
            cols = np.array(colors, dtype=np.float32)
            nrms = np.array(normals, dtype=np.float32)
            
            if len(pts) == 0: 
                return np.zeros((0, 9), dtype=np.float32)
            
            return np.concatenate([pts, cols, nrms], axis=1)

        except Exception as e:
            # Return empty on failure to avoid crashing training
            return np.zeros((0, 9), dtype=np.float32)

    def _process_points(self, point_data: np.ndarray) -> np.ndarray:
        """
        Samples or Pads points to self.num_points.
        Input: (N, 9). Output: (num_points, 9)
        """
        N = point_data.shape[0]
        T = self.num_points
        
        if N == 0:
            return np.zeros((T, 9), dtype=np.float32)
        
        if N >= T:
            # Random subsample without replacement
            indices = np.random.choice(N, T, replace=False)
            return point_data[indices]
        else:
            # Pad with zeros
            padding = np.zeros((T - N, 9), dtype=np.float32)
            return np.concatenate([point_data, padding], axis=0)

    def _sample_window_indices(self, total_frames: int) -> List[int]:
        """
        Implements Windowed Augmentation.
        - Training: Random window start position
        - Test/Val: Center window start position
        - Uses fixed stride=2 for temporal sampling
        - Clips that are too short are padded by clamping to last frame
        """
        # Calculate the total span required: (num_frames - 1) * stride + 1
        window_span = (self.num_frames - 1) * self.stride + 1
        
        if self.split == 'train':
            # RANDOM START for training
            if total_frames > window_span:
                start_frame = random.randint(0, total_frames - window_span)
            else:
                # Clip is too short, start from beginning
                start_frame = 0
        else:
            # CENTERED START for test/val
            if total_frames > window_span:
                start_frame = (total_frames - window_span) // 2
            else:
                # Clip is too short, start from beginning
                start_frame = 0

        # Sample frames with stride
        indices = []
        for i in range(self.num_frames):
            idx = start_frame + (i * self.stride)
            # Clamp to last frame if we exceed total_frames
            indices.append(min(idx, total_frames - 1))
        return indices

    def _augment_rotation(self, seq_data: np.ndarray) -> np.ndarray:
        """
        Rotates XYZ and Normals around Y axis by 0, 90, 180, or 270 degrees.
        seq_data: (T, N, 9) -> [x,y,z, r,g,b, nx,ny,nz]
        """
        k = random.choice([0, 1, 2, 3])
        if k == 0: return seq_data

        theta = k * (math.pi / 2)
        c, s = math.cos(theta), math.sin(theta)
        rot_mat = np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]], dtype=np.float32)

        # Apply to XYZ (indices 0,1,2)
        xyz = seq_data[:, :, 0:3]
        shape_orig = xyz.shape
        # Reshape to (Total_Points, 3) for matmul
        seq_data[:, :, 0:3] = (xyz.reshape(-1, 3) @ rot_mat.T).reshape(shape_orig)

        # Apply to Normals (indices 6,7,8)
        nrm = seq_data[:, :, 6:9]
        seq_data[:, :, 6:9] = (nrm.reshape(-1, 3) @ rot_mat.T).reshape(shape_orig)

        return seq_data

    def _load_motion_sequence(self, motion_info) -> torch.Tensor:
        """
        Load and process motion sequence.
        Returns: (T, N, 9) tensor
        """
        
        all_frames = motion_info['frames']
        
        # Sample frame indices
        indices = self._sample_window_indices(motion_info['total_frames'])
        selected_frames = [all_frames[i] for i in indices]
        
        # 3. Load & Process PCDs
        seq_frames_processed = [] # Will hold (N, 9) arrays
        
        for _, file_path in selected_frames:
            raw_data = self._load_pcd(file_path)
            processed_data = self._process_points(raw_data)
            seq_frames_processed.append(processed_data)
        
        # Stack to (T, N, 9) numpy array
        seq_data_np = np.stack(seq_frames_processed, axis=0)

        # 4. Rotation Augmentation
        if self.split == 'train':
            seq_data_np = self._augment_rotation(seq_data_np)

        # 5. Global Normalization
        # Normalize coords (0:3) using min/max of the *entire sequence*
        xyz = seq_data_np[:, :, 0:3]
        min_xyz = np.min(xyz, axis=(0, 1), keepdims=True)
        max_xyz = np.max(xyz, axis=(0, 1), keepdims=True)
        range_xyz = max_xyz - min_xyz
        range_xyz[range_xyz == 0] = 1.0 # Prevent div/0
        
        seq_data_np[:, :, 0:3] = (xyz - min_xyz) / range_xyz
        
        # 6. Convert to tensor - return as [T, N, 9]
        seq_data_tensor = torch.from_numpy(seq_data_np)  # [T, N, 9]
        
        return seq_data_tensor

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            motion_tokens = 197 if 'motion' in sample else 0  # Approx tokens from MoPa
            # Estimate length using first Q&A pair if qa_pairs exist
            if 'qa_pairs' in sample:
                qa_pair = sample['qa_pairs'][0]
                text_len = len(qa_pair.get('question', '').split()) + len(qa_pair.get('answer', '').split())
            elif 'conversations' in sample:
                text_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            else:
                text_len = 0
            length_list.append(text_len + motion_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # Estimate length using first Q&A pair if qa_pairs exist
            if 'qa_pairs' in sample:
                qa_pair = sample['qa_pairs'][0]
                cur_len = len(qa_pair.get('question', '').split()) + len(qa_pair.get('answer', '').split())
            elif 'conversations' in sample:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            else:
                cur_len = 0
            cur_len = cur_len if 'motion' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Returns a training sample in LLaVA format.
        
        For training: randomly selects one Q&A pair per sequence
        For testing: always selects the first Q&A pair for consistency
        """
        sample = self.list_data_dict[i]
        
        # Select Q&A pair based on split
        if 'qa_pairs' in sample:
            qa_pairs = sample['qa_pairs']
            if self.split == 'train':
                # Random selection for training (different each epoch)
                qa_pair = random.choice(qa_pairs)
            else:
                # First Q&A for test/val (consistent evaluation)
                qa_pair = qa_pairs[0]
            
            # Create conversations from selected Q&A pair
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            conversations = [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
            # Temporarily add conversations to sample
            sample = copy.deepcopy(sample)
            sample['conversations'] = conversations
        
        sources = [sample]
        
        # Load motion sequence if present
        if 'motion' in sources[0]:
            motion_path = self.list_data_dict[i]['motion']
            
            # Load motion data
            if 'frames' in self.list_data_dict[i]:
                # Use frame info from discovery
                motion_info = {
                    'frames': self.list_data_dict[i]['frames'],
                    'total_frames': self.list_data_dict[i]['total_frames']
                }
            else:
                # Just use path
                motion_info = motion_path
            
            motion_sequence = self._load_motion_sequence(motion_info)
            
            # Process conversations
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        # Tokenize conversations
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('motion' in self.list_data_dict[i]))
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                           labels=data_dict["labels"][0])
        
        # Add motion data (renamed to 'image' for LLaVA compatibility)
        if 'motion' in self.list_data_dict[i]:
            data_dict['image'] = motion_sequence  # Shape: (T, N, 9)
        elif self.data_args.is_multimodal:
            # Dummy motion data if needed
            data_dict['image'] = torch.zeros(self.num_frames, self.num_points, 9)
        
        return data_dict


def make_motion_supervised_data_module(tokenizer, data_args) -> Dict:
    """
    Create motion-based dataset and collator for LLaVA training.
    
    This replaces make_supervised_data_module for motion-based training.
    """
    from llava.train.train import DataCollatorForSupervisedDataset
    
    train_dataset = MotionLazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        num_frames=32,
        num_points=2048,
        data_split='train',
    )
    
    eval_dataset = MotionLazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        num_frames=32,
        num_points=2048,
        data_split='test',
    )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
