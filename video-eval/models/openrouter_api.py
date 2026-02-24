import os
import requests
import base64
from .base_model import BaseVideoModel

class OpenRouterVideoModel(BaseVideoModel):
    def __init__(self, model_name="google/gemini-1.5-pro"):
        """
        Initializes the OpenRouter model.
        Make sure you have set the OPENROUTER_API_KEY environment variable.
        """
        self.model_name = model_name
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: OPENROUTER_API_KEY environment variable is not set.")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        print(f"✓ OpenRouter API initialized with model: {self.model_name}")

    def analyze_video(self, video_path: str, prompt: str) -> str:
        if not os.path.exists(video_path):
            return "Error: Video file not found."

        try:
            # 1. Read and encode the local MP4 file to Base64
            with open(video_path, "rb") as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
            
            # OpenRouter requires the data URI format for local files
            data_url = f"data:video/mp4;base64,{encoded_string}"

            # 2. Build the payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that analyzes videos. Answer the user's questions based on the video content directly and concisely."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": data_url}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            # 3. Send the request
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            # 4. Handle the response safely
            if response.status_code != 200:
                 return f"Error: OpenRouter returned {response.status_code} - {response.text}"
                 
            response_data = response.json()
            return response_data['choices'][0]['message']['content'].strip()

        except Exception as e:
            return f"Error processing with OpenRouter: {str(e)}"