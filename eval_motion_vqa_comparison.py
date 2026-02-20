"""Compare model performance with real vs random motion data"""
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

print("Loading base model...")
model = LlavaLlamaForCausalLM.from_pretrained(
    model_base,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.config.use_cache = True

print("Initializing vision modules...")
model_args = ModelArguments()
model_args.vision_tower = vision_tower_path
model_args.mm_vision_select_layer = config.mm_vision_select_layer
model_args.mm_vision_select_feature = getattr(config, 'mm_vision_select_feature', 'patch')
model_args.mm_patch_merge_type = getattr(config, 'mm_patch_merge_type', 'flat')
model_args.mm_projector_type = config.mm_projector_type
model_args.pretrain_mm_mlp_adapter = None

model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

print("Loading projector weights...")
mm_projector_weights = torch.load(f"{model_path}/mm_projector.bin", map_location='cpu')
print(f"Found {len(mm_projector_weights)} weight tensors")

mm_projector_weights_filtered = {
    k.replace('model.mm_projector.', 'mm_projector.'): v 
    for k, v in mm_projector_weights.items() 
    if 'mm_projector' in k
}
print(f"Filtered to {len(mm_projector_weights_filtered)} projector weights")

model.get_model().load_state_dict(mm_projector_weights_filtered, strict=False)

model = model.cuda()
model.eval()

# Convert projector to fp16
for p in model.get_model().mm_projector.parameters():
    p.data = p.data.half()

print("✓ Model loaded\n")

# Load test dataset
class DataArgs:
    def __init__(self):
        self.is_multimodal = True
        self.image_aspect_ratio = 'pad'
        self.image_grid_pinpoints = None

data_args = DataArgs()
test_dataset = MotionLazySupervisedDataset(
    data_path="data/v4.4_new_sample/v4.4-humanML3d-2136-video",
    tokenizer=tokenizer,
    data_args=data_args,
    data_split='test',
    num_frames=32,
    num_points=2048,
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
    real_motion_tensor = test_dataset._load_motion_sequence(motion_info)
    real_motion_tensor = real_motion_tensor.unsqueeze(0).half().cuda()
    
    # Create RANDOM motion with same shape
    random_motion_tensor = torch.randn_like(real_motion_tensor)
    
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
        
        # Generate with REAL motion
        with torch.inference_mode():
            output_ids_real = model.generate(
                input_ids,
                images=real_motion_tensor,
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        prediction_real = tokenizer.decode(output_ids_real[0], skip_special_tokens=True).strip()
        
        # Generate with RANDOM motion
        with torch.inference_mode():
            output_ids_random = model.generate(
                input_ids,
                images=random_motion_tensor,
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        prediction_random = tokenizer.decode(output_ids_random[0], skip_special_tokens=True).strip()
        
        # Store result with both predictions
        result = {
            'sample_idx': sample_idx,
            'qa_idx': qa_idx,
            'motion_id': sample.get('id', f"sample_{sample_idx}"),
            'question': question,
            'ground_truth': answer,
            'prediction_real_motion': prediction_real,
            'prediction_random_motion': prediction_random,
            'predictions_match': prediction_real == prediction_random,
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
print(f"   - Predictions identical (real vs random): {matching}/{total} ({100*matching/total:.1f}%)")
print(f"   - Predictions different (real vs random): {different}/{total} ({100*different/total:.1f}%)")
print(f"\n✓ Results saved to: {output_file}")

if matching > total * 0.8:
    print(f"\n⚠️  WARNING: {100*matching/total:.1f}% of predictions are identical!")
    print("   This suggests the model is NOT using motion information effectively.")
    print("   It's likely relying on language priors from the base LLM.")
else:
    print(f"\n✓ Good: Only {100*matching/total:.1f}% of predictions are identical.")
    print("  The model appears to use motion information to generate answers.")
