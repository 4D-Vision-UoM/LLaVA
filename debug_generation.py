"""Quick debug script to check model generation"""
import torch
import sys
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
print(f"Keys: {list(mm_projector_weights.keys())[:5]}")

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

# Load one test sample
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

# Test first sample
sample = test_dataset.list_data_dict[0]
qa_pair = sample['qa_pairs'][0]
question = qa_pair['question']
answer = qa_pair['answer']

print(f"Question: {question}")
print(f"Ground truth: {answer}\n")

# Build prompt
qs = DEFAULT_IMAGE_TOKEN + '\n' + question
conv = conv_templates['v1'].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

print(f"Full prompt:\n{prompt}\n")

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
print(f"Input IDs shape: {input_ids.shape}")
print(f"Input IDs: {input_ids[0][:20].tolist()}...\n")

# Load motion
motion_info = {
    'frames': sample['frames'],
    'total_frames': sample['total_frames']
}
motion_tensor = test_dataset._load_motion_sequence(motion_info)
motion_tensor = motion_tensor.unsqueeze(0).half().cuda()
print(f"Motion tensor shape: {motion_tensor.shape}\n")

# Generate
print("Generating...")
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=motion_tensor,
        do_sample=False,
        max_new_tokens=128,
        use_cache=True,
    )

print(f"Output IDs shape: {output_ids.shape}")
print(f"Output IDs: {output_ids[0].tolist()}\n")

# Decode
full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print(f"Full output (with special tokens):\n{full_output}\n")

clean_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Clean output (without special tokens):\n{clean_output}\n")

# Just the generated part
generated_ids = output_ids[0][input_ids.shape[1]:]
print(f"Generated IDs only: {generated_ids.tolist()}")
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Generated text only:\n'{generated_text}'")
