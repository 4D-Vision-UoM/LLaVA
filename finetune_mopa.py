

import os
import json
import torch
from llava.train.train import train


def main():
    # Set paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ####################################################################################
    # Path to pretrained 4D motion encoder checkpoint (MoPa)
    ####################################################################################
    mopa_checkpoint = os.path.join(script_dir, "MoPa/ckpt/HumanML_MoPa_32_frames_64batch")

    ####################################################################################
    # Data path
    ####################################################################################
    data_path = os.path.join(script_dir, "data/v4.4_new_sample/v4.4-humanML3d-2136-video")  # Root directory for HumanML data
    
    ####################################################################################
    # Output path for fine-tuned model
    ####################################################################################
    output_dir = os.path.join(script_dir, "checkpoints/llava-mopa-projection_10_epoch")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("LLaVA Training with MoPa Motion Encoder")
    print("=" * 80)
    print(f"MoPa checkpoint: {mopa_checkpoint}")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Prepare training arguments
    training_args = [
        "--model_name_or_path", "liuhaotian/llava-v1.5-7b",
        "--version", "v1",
        "--data_path", data_path,
        
        # *** KEY: Use motion data instead of images ***
        "--use_motion_data", "True",
        
        ####################################################################################
        # Path to pretrained 4D motion encoder checkpoint (MoPa)
        ####################################################################################
        # *** KEY: Use MoPa motion encoder ***
        "--vision_tower", mopa_checkpoint,
        
        # Vision tower configuration
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-1",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        
        ####################################################################################
        # Model Freezing - tune_mm_mlp_adapter will freeze backbone regardless
        # vision model is always frozen
        ####################################################################################
        # Freeze LLM, train only projection layer
        "--freeze_backbone", "True",
        "--tune_mm_mlp_adapter", "True",
        
        # Training settings
        "--bf16", "True" if torch.cuda.is_bf16_supported() else "False",
        "--fp16", "False" if torch.cuda.is_bf16_supported() else "True",
        "--output_dir", output_dir,
        "--num_train_epochs", "10",
        "--per_device_train_batch_size", "2",
        "--per_device_eval_batch_size", "2",
        "--gradient_accumulation_steps", "2",
        "--evaluation_strategy", "epoch",  # Changed from "no" to "epoch"
        "--save_strategy", "epoch",  # Changed to match eval strategy
        "--save_total_limit", "1",
        "--learning_rate", "1e-3",
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
    
    print("\nStarting training with MoPa motion encoder...")
    print("=" * 80)
    
    
    # Set sys.argv to simulate command line arguments
    import sys
    sys.argv = ["train"] + training_args
    
    # Call the training function
    train()
    
    print("\n" + "=" * 80)
    print("✓ Training completed successfully!")
    print(f"✓ Model saved to: {output_dir}")
    print("=" * 80)
    
        

if __name__ == "__main__":
    main()
