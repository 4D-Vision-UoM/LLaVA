from abc import ABC, abstractmethod
import os
import json
from openai import OpenAI
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    @abstractmethod
    def generate_json(self, prompt: str) -> dict:
        pass

class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = "openai/gpt-3.5-turbo"):
        """
        Connects to OpenRouter.ai, allowing you to route requests to hundreds of models.
        Make sure to prefix the model name with the provider (e.g., 'openai/gpt-4-turbo').
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            print("WARNING: OPENROUTER_API_KEY environment variable not set.")
        print(f"Using Model Version: {model}")    
        # Point the OpenAI client to the OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = model

    def generate_json(self, prompt: str) -> dict:
        # We wrap the paper's prompt in a rigid system instruction just in case
        # you route to a non-OpenAI model that needs extra nudging for JSON.
        system_content = "You are a strict, automated evaluation script. You must ONLY output a valid JSON object in this format: {\"score\": int, \"reasoning\": \"string\"}. Do not include markdown formatting or conversational text."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                # OpenRouter supports this for OpenAI models (and many others)
                response_format={"type": "json_object"},
                temperature=0.0 # Deterministic grading
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Fallback regex extraction in case a non-OpenAI model hallucinates markdown tags
            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = result_text[start_idx:end_idx]
                print(f"Raw LLM Output: {result_text}")
                return json.loads(json_str)
            else:
                raise ValueError(f"No JSON object found. Raw output: {result_text}")
                
        except json.JSONDecodeError as e:
            return {"score": 0, "reasoning": f"Failed to parse OpenRouter JSON: {str(e)}"}
        except Exception as e:
            return {"score": 0, "reasoning": f"Error calling OpenRouter API: {str(e)}"}

class TinyLlamaProvider(BaseLLMProvider):
    def __init__(self, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Loads TinyLlama for ultra-fast, low-memory local evaluation.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "model_weights/llm/tiny_llama"
        print(f"Loading {model_id} onto {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load in bfloat16 to save memory and speed up inference
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto" 
        )

    def generate_json(self, prompt: str) -> dict:
        # A rigid system prompt is crucial for 1B parameter models
        messages = [
            {
                "role": "system", 
                "content": "You are a strict evaluation system. You output ONLY valid JSON. Do not include markdown formatting, explanations, or conversational text. Return exactly this format: {\"score\": int, \"reasoning\": \"string\"}"
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Apply TinyLlama's specific chat template (<|system|>, <|user|>, <|assistant|>)
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        try:
            # Generate the response deterministically
            generated_ids = self.model.generate(
                **model_inputs, 
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id # Prevents padding warnings
            )
            
            # Isolate the newly generated tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # --- AGGRESSIVE JSON EXTRACTION ---
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError(f"No JSON object found. Raw output: {response_text}")
                
        except Exception as e:
            return {"score": 0, "reasoning": f"TinyLlama execution error: {str(e)}"}


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = "gemini-3-flash-preview"):
        """
        Connects to Google's Gemini API.
        Recommended models: 'gemini-1.5-pro' (best reasoning) or 'gemini-1.5-flash' (fastest/cheapest).
        """
        api_key = os.environ.get("GEMINI_API_KEY") 
        print(f"using Gemini API key: {api_key[:4]}...{api_key[-4:]}")
        # Configure the API key from the parameter or environment variable
        genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name=model)

    def generate_json(self, prompt: str) -> dict:
        try:
            # Gemini has native support for forcing JSON output
            config = GenerationConfig(
                temperature=0.0, # Keep it deterministic for evaluation
                response_mime_type="application/json" 
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=config
            )
            
            return json.loads(response.text)
            
        except json.JSONDecodeError as e:
            return {"score": 0, "reasoning": f"Failed to parse Gemini JSON: {str(e)}. Raw output: {response.text}"}
        except Exception as e:
            return {"score": 0, "reasoning": f"Error calling Gemini API: {str(e)}"}