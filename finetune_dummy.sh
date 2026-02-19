#!/bin/bash

# Simple fine-tuning script for testing with dummy data
# This uses minimal settings for quick testing

set -e

echo "========================================="
echo "LLaVA Projection Layer Fine-tuning Test"
echo "(Text and Vision frozen - Projection only)"
echo "========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set paths
DATA_PATH="$SCRIPT_DIR/playground/data/dummy_finetune_data.json"
IMAGE_FOLDER="$SCRIPT_DIR/images"
OUTPUT_DIR="$SCRIPT_DIR/checkpoints/test-finetune-projection-only"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Data path: $DATA_PATH"
echo "Image folder: $IMAGE_FOLDER"
echo "Output directory: $OUTPUT_DIR"
echo "========================================="

# Run training with projection layer only (freeze LLM and vision model)
python -m llava.train.train \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --freeze_backbone True \
    --tune_mm_mlp_adapter True \
    --fp16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none

echo "========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================="
