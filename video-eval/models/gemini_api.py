import os
import time
import google.generativeai as genai
from .base_model import BaseVideoModel

class GeminiVideoModel(BaseVideoModel):
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"✓ Gemini API initialized with model: {model_name}")

    def analyze_video(self, video_path: str, prompt: str) -> str:
        try:
            # Upload the video using the File API
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing if needed
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
                
            response = self.model.generate_content([video_file, prompt])
            
            # Clean up the file from Google's servers
            genai.delete_file(video_file.name)
            
            return response.text.strip()
        except Exception as e:
            return f"Gemini Error: {str(e)}"