"""Compare model performance with vs without fine-tuned projection weights"""
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
model_path = "checkpoints/llava-mopa-projection_10_epoch"
model_base = "liuhaotian/llava-v1.5-7b"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading config...")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
vision_tower_path = config.mm_vision_tower

print("Creating model with INITIALIZED projector...")
model_initialized = LlavaLlamaForCausalLM.from_pretrained(
    model_base,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model_initialized.config.use_cache = True

model_args_initialized = ModelArguments()
model_args_initialized.vision_tower = vision_tower_path
model_args_initialized.mm_vision_select_layer = config.mm_vision_select_layer
model_args_initialized.mm_vision_select_feature = getattr(config, 'mm_vision_select_feature', 'patch')
model_args_initialized.mm_patch_merge_type = getattr(config, 'mm_patch_merge_type', 'flat')
model_args_initialized.mm_projector_type = config.mm_projector_type
model_args_initialized.pretrain_mm_mlp_adapter = None  # No pretrained weights - random initialization

model_initialized.get_model().initialize_vision_modules(model_args=model_args_initialized, fsdp=None)
model_initialized = model_initialized.cuda()
model_initialized.eval()

for p in model_initialized.get_model().mm_projector.parameters():
    p.data = p.data.half()

print("✓ Model with initialized projector ready")

print("\nCreating model with FINE-TUNED projector...")
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
model_args_finetuned.pretrain_mm_mlp_adapter = f"{model_path}/mm_projector.bin"  # Load fine-tuned weights

model_finetuned.get_model().initialize_vision_modules(model_args=model_args_finetuned, fsdp=None)
model_finetuned = model_finetuned.cuda()
model_finetuned.eval()

for p in model_finetuned.get_model().mm_projector.parameters():
    p.data = p.data.half()

print("✓ Model with fine-tuned projector ready")
print("✓ Both models loaded and ready for comparison\n")

# Load test dataset
class DataArgs:
    def __init__(self):
        self.data_path = "data"
        self.vqa_path = "data/llama3-8b-instruct"
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
output_file = f"eval_motion_vqa_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
print(f"Results will be saved to: {output_file}\n")

results = []

# Process all samples
for sample_idx in tqdm(range(len(test_dataset.list_data_dict)), desc="Evaluating samples"):
    sample = test_dataset.list_data_dict[sample_idx]
    
    # Load REAL motion
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
        
        # Generate with FINE-TUNED projector
        with torch.inference_mode():
            output_ids_finetuned = model_finetuned.generate(
                input_ids,
                images=motion_tensor,
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        prediction_finetuned = tokenizer.decode(output_ids_finetuned[0], skip_special_tokens=True).strip()
        
        # Generate with INITIALIZED projector
        with torch.inference_mode():
            output_ids_initialized = model_initialized.generate(
                input_ids,
                images=motion_tensor,
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        prediction_initialized = tokenizer.decode(output_ids_initialized[0], skip_special_tokens=True).strip()
        
        # Store result with both predictions
        result = {
            'sample_idx': sample_idx,
            'qa_idx': qa_idx,
            'motion_id': sample.get('id', f"sample_{sample_idx}"),
            'question': question,
            'ground_truth': answer,
            'prediction_finetuned': prediction_finetuned,
            'prediction_initialized': prediction_initialized,
            'predictions_match': prediction_finetuned == prediction_initialized,
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
matching = sum(1 for r in results if r['predictions_match'])
different = total - matching

print(f"\n✓ Evaluation complete!")
print(f"✓ Processed {total} QA pairs from {len(test_dataset.list_data_dict)} samples")
print(f"\n📊 Statistics:")
print(f"   - Predictions identical (fine-tuned vs initialized): {matching}/{total} ({100*matching/total:.1f}%)")
print(f"   - Predictions different (fine-tuned vs initialized): {different}/{total} ({100*different/total:.1f}%)")
print(f"\n✓ Results saved to: {output_file}")

if matching > total * 0.8:
    print(f"\n⚠️  WARNING: {100*matching/total:.1f}% of predictions are identical!")
    print("   This suggests fine-tuning the projector had MINIMAL impact.")
    print("   The model may be relying primarily on language priors from the base LLM.")
else:
    print(f"\n✓ Good: Only {100*matching/total:.1f}% of predictions are identical.")
    print("   The fine-tuned projector appears to provide meaningful improvements.")
