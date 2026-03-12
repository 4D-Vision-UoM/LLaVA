import torch
import importlib
import yaml
from pathlib import Path
from MoPa.mopa_pipeline import PipelineMoPa


import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """
    YAML configuration management for training.
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)



# Configuration paths
ckpt_arch = 'MoPa/ckpt/HumanML_MoPa_32_frames_64batch'
config_path = f'{ckpt_arch}/config.yaml'
ckpt_path = f'{ckpt_arch}/checkpoints/best.pth'

# Load configuration

config = ConfigManager.load_config(config_path)

# Dynamically load pipeline class
model_config = config['model']

# Get pipeline arguments from config
pipeline_args = model_config['pipeline_args']

# Create pipeline instance
model = PipelineMoPa(feature_mode='all',**pipeline_args)

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location='cpu')

# Inspect checkpoint structure
print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Load only the model state dict (not text encoder)
if 'model_state_dict' in checkpoint:
    print("\nLoading model_state_dict...")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    if missing_keys:
        print(f"⚠ Missing keys ({len(missing_keys)}):")
        for key in missing_keys[:5]:  # Show first 5
            print(f"    - {key}")
        if len(missing_keys) > 5:
            print(f"    ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"⚠ Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys[:5]:  # Show first 5
            print(f"    - {key}")
        if len(unexpected_keys) > 5:
            print(f"    ... and {len(unexpected_keys) - 5} more")
    
    if not missing_keys and not unexpected_keys:
        print("✓ All keys matched perfectly!")
    
    print(f"✓ Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    # Fallback: try loading directly
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print("✓ Loaded model weights (direct)")

# Freeze all model parameters
print("\nFreezing model parameters...")
for param in model.parameters():
    param.requires_grad = False

# Verify frozen
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*50}")
print(f"Model Status:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen: {trainable_params == 0}")
print(f"{'='*50}")

# Set to eval mode
model.eval()
print("\n✓ Model loaded, frozen, and set to eval mode")


from torch.utils.data import DataLoader
from MoPa.humanML import HumanMLDataset, collate_fn

# Get data config from loaded checkpoint config
data_config = config['data']
seq_config = data_config['sequence_config']

# Create evaluation dataset
print("Creating evaluation dataset...")
eval_dataset = HumanMLDataset(
    data_root=data_config['data_root'],
    seed=seq_config['seed'],
    split="test",
    num_frames=data_config['num_frames'],
    num_points=data_config.get('num_points', None)
)

# Create evaluation dataloader
eval_loader = DataLoader(
    eval_dataset,
    batch_size=16,  # Smaller batch for evaluation
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)



import torch

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model on device: {device}")

# Get one batch
print("\nGetting one batch...")
sample_batch = next(iter(eval_loader))
clips, captions = sample_batch

# Move to device
clips = clips.to(device)

print(f"Batch shapes:")
print(f"  Clips: {clips.shape}")
print(f"  Captions: {len(captions)} text samples")
print(f"\nSample caption: '{captions[0]}'")

# Run inference
print("\nRunning inference...")
with torch.no_grad():
    motion_features = model(clips)

print(f"\nOutput shape: {motion_features.shape}")
print(f"Output dtype: {motion_features.dtype}")
print(f"Output device: {motion_features.device}")
print(f"Output range: [{motion_features.min():.4f}, {motion_features.max():.4f}]")
print(f"Output mean: {motion_features.mean():.4f}")
print(f"Output std: {motion_features.std():.4f}")

print("\n✓ Successfully ran one batch through the model!")