#!/usr/bin/env python3
"""
Simple and direct fine-tuning script for LLaVA with dummy data.
This script directly calls the training functions without subprocess.
"""

import os
import torch
from llava.train.train import train

def main():
    # Set paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "playground/data/dummy_finetune_data.json")
    image_folder = os.path.join(script_dir, "images")
    output_dir = os.path.join(script_dir, "checkpoints/test-finetune-simple")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Simple LLaVA Fine-tuning Test")
    print("=" * 80)
    print(f"Data path: {data_path}")
    print(f"Image folder: {image_folder}")
    print(f"Output directory: {output_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Prepare training arguments as a list (simulating command line arguments)
    training_args = [
        "--model_name_or_path", "liuhaotian/llava-v1.5-7b",
        "--version", "v1",
        "--data_path", data_path,
        "--image_folder", image_folder,
        "--vision_tower", "openai/clip-vit-large-patch14-336",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", "True" if torch.cuda.is_bf16_supported() else "False",
        "--fp16", "False" if torch.cuda.is_bf16_supported() else "True",
        "--output_dir", output_dir,
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "2",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "50",
        "--save_total_limit", "1",
        "--learning_rate", "2e-5",
        "--weight_decay", "0.0",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True" if torch.cuda.is_available() else "False",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "2",
        "--lazy_preprocess", "True",
        "--report_to", "none",
    ]
    
    # Enable LoRA fine-tuning for memory efficiency
    training_args.extend([
        "--lora_enable", "True",
        "--lora_r", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--lora_bias", "none",
    ])
    
    print("\nStarting training...")
    print("=" * 80)
    
    try:
        # Set sys.argv to simulate command line arguments
        import sys
        sys.argv = ["train"] + training_args
        
        # Call the training function
        train()
        
        print("\n" + "=" * 80)
        print("✓ Training completed successfully!")
        print(f"✓ Model saved to: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Training failed with error:")
        print(f"  {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
