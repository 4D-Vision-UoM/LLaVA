import torch
import os
from transformers import AutoModelForCausalLM, AutoProcessor
from .base_model import BaseVideoModel

class VideoLLaMAModel(BaseVideoModel):
    def __init__(self, model_name="DAMO-NLP-SG/VideoLLaMA3-7B", device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading model '{model_name}' on {self.device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map={"": self.device},
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print("✓ VideoLLaMA loaded successfully!\n")
        except Exception as e:
            print(f"✗ Failed to load VideoLLaMA: {e}")
            exit(1)

    def analyze_video(self, video_path: str, prompt: str) -> str:
        if not os.path.exists(video_path):
            return "Error: Video file not found."

        conversation = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 10, "max_frames": 180}},
                    {"type": "text", "text": prompt},
                ]
            },
        ]

        try:
            inputs = self.processor(
                conversation=conversation, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)
            response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            if "assistant\n" in response:
                response = response.split("assistant\n")[-1]
            return response.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"