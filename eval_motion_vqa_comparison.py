"""Evaluate fine-tuned model on motion VQA dataset"""
import os
import torch
import sys
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import ModelArguments
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.train.motion_dataset import MotionLazySupervisedDataset
from transformers import AutoConfig

# Load model
model_path = "checkpoints/HumanML_MoPa_finetuned_gemini_10epoch"
model_base = "liuhaotian/llava-v1.5-7b"
output_dir = "output"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading config...")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
vision_tower_path = config.mm_vision_tower

# Create Model 1: WITH fine-tuned projector
print("Creating model 1: WITH fine-tuned projector...")
model_finetuned = LlavaLlamaForCausalLM.from_pretrained(
    model_base,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model_finetuned.config.use_cache = True

model_args_finetuned = ModelArguments()
model_args_finetuned.vision_tower = vision_tower_path
model_args_finetuned.mm_vision_select_layer = config.mm_vision_select_layer
model_args_finetuned.mm_vision_select_feature = getattr(config, 'mm_vision_select_feature', 'patch')
model_args_finetuned.mm_patch_merge_type = getattr(config, 'mm_patch_merge_type', 'flat')
model_args_finetuned.mm_projector_type = config.mm_projector_type
# model_args_finetuned.pretrain_mm_mlp_adapter = f"{model_path}/mm_projector.bin"

model_finetuned.get_model().initialize_vision_modules(model_args=model_args_finetuned, fsdp=None)
model_finetuned = model_finetuned.cuda()
model_finetuned.eval()

for p in model_finetuned.get_model().mm_projector.parameters():
    p.data = p.data.half()

print("✓ Model 1 (fine-tuned) loaded\n")

# Create Model 2: WITHOUT fine-tuned projector (from scratch)
print("Creating model 2: WITHOUT fine-tuned projector (from scratch)...")
model_scratch = LlavaLlamaForCausalLM.from_pretrained(
    model_base,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model_scratch.config.use_cache = True

model_args_scratch = ModelArguments()
model_args_scratch.vision_tower = vision_tower_path
model_args_scratch.mm_vision_select_layer = config.mm_vision_select_layer
model_args_scratch.mm_vision_select_feature = getattr(config, 'mm_vision_select_feature', 'patch')
model_args_scratch.mm_patch_merge_type = getattr(config, 'mm_patch_merge_type', 'flat')
model_args_scratch.mm_projector_type = config.mm_projector_type
# DO NOT load pretrain_mm_mlp_adapter - keep projector randomly initialized

model_scratch.get_model().initialize_vision_modules(model_args=model_args_scratch, fsdp=None)
model_scratch = model_scratch.cuda()
model_scratch.eval()

for p in model_scratch.get_model().mm_projector.parameters():
    p.data = p.data.half()

print("✓ Model 2 (from scratch) loaded\n")
print("✓ Both models loaded and ready for evaluation\n")

# Load test dataset
class DataArgs:
    def __init__(self):
        self.data_path = "data"
        self.vqa_path = "data/gemini-flash"
        self.motion_path = "data/v4.3-wall-humanML3d-2136"
        self.is_multimodal = True
        self.image_aspect_ratio = 'pad'
        self.image_grid_pinpoints = None

data_args = DataArgs()
test_dataset = MotionLazySupervisedDataset(
    data_path=data_args.data_path,
    tokenizer=tokenizer,
    data_args=data_args,
    data_split='test',
    num_frames=32,
    num_points=2048,
    vqa_path=data_args.vqa_path,
    motion_path=data_args.motion_path,
)

print(f"Loaded {len(test_dataset)} test samples\n")

# Prepare output file
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/{model_path.split('/')[-1]}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
print(f"Results will be saved to: {output_file}\n")

results = []

# Process all samples
for sample_idx in tqdm(range(len(test_dataset.list_data_dict)), desc="Evaluating samples"):
    sample = test_dataset.list_data_dict[sample_idx]
    
    # Load motion sequence
    motion_info = {
        'frames': sample['frames'],
        'total_frames': sample['total_frames']
    }
    motion_tensor = test_dataset._load_motion_sequence(motion_info)
    motion_tensor = motion_tensor.unsqueeze(0).half().cuda()
    
    # Process all QA pairs for this sample
    for qa_idx, qa_pair in enumerate(sample['qa_pairs']):
        question = qa_pair['question']
        answer = qa_pair['answer']
        
        # Build prompt
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # Generate prediction from fine-tuned model
        with torch.inference_mode():
            output_ids_finetuned = model_finetuned.generate(
                input_ids,
                images=motion_tensor,
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        prediction_finetuned = tokenizer.decode(output_ids_finetuned[0], skip_special_tokens=True).strip()
        
        # Generate prediction from scratch model
        with torch.inference_mode():
            output_ids_scratch = model_scratch.generate(
                input_ids,
                images=motion_tensor,
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        prediction_from_scratch = tokenizer.decode(output_ids_scratch[0], skip_special_tokens=True).strip()
        
        # Store result with both predictions
        result = {
            'sample_idx': sample_idx,
            'qa_idx': qa_idx,
            'motion_id': sample.get('id', f"sample_{sample_idx}"),
            'question': question,
            'ground_truth': answer,
            'prediction_finetuned': prediction_finetuned,
            'prediction_from_scratch': prediction_from_scratch,
            'total_frames': sample['total_frames']
        }
        results.append(result)
        
        # Write to file in real-time
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Clear CUDA cache periodically
    if (sample_idx + 1) % 10 == 0:
        torch.cuda.empty_cache()

# Calculate statistics
total = len(results)

print(f"\n✓ Evaluation complete!")
print(f"✓ Processed {total} QA pairs from {len(test_dataset.list_data_dict)} samples")
print(f"✓ Results saved to: {output_file}")
