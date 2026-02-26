import os
import cv2
import base64
import requests
from pathlib import Path

# Prevent OpenCV from spawning its own background threads inside our ThreadPool
cv2.setNumThreads(0)

DETAILED_CAPTION_GENERATION_TEMPLATE = """
### CONTEXT
I am providing {n} consecutive frames from a video derived from the HumanML dataset.
Reference Ground Truth: "{humanML_groundtruth_caption}"

### TASK
Generate a single, detailed English caption that objectively narrates the movement shown in the frames.

### GUIDELINES
1. **Visual Priority:** Use the Reference Ground Truth ONLY as a high-level guide for the context of the action. You must verify that the action actually occurs in the specific frames provided. If the visual contradicts the text, trust the text.
2. **Subject Description:** Refer to the subject simply as "the person". Do not describe their body type, age, face, or clothing unless it is mechanically relevant to the interaction (e.g., "holding a skirt").
3. **Motion Focus:** Focus 90 percent of the caption on the mechanics of the movement (limbs, posture, speed, trajectory).
4. **Chronology:** Describe the sequence of movements in strict chronological order.
5. **Constraints:**
   - NO background description.
   - NO symbolic interpretation (e.g., do not say "he looks sad," say "he walks with a slumped posture").
   - NO distinct physical features (hair, eyes, skin).

### OUTPUT
Provide only the caption text.
"""

QnA_GENERATION_TEMPLATE = """
You are given a detailed caption describing a motion sequence from a dataset.
Your task is to generate natural Question-Answer (QA) pairs derived *strictly* from the information in the caption.

### CATEGORIES
1. **Action:** Focus on the specific movement mechanics, verbs, or velocity (e.g., "How does the person move their legs?").
2. **Body-Spatial:** Focus on the position of body parts relative to each other or the ground (e.g., "Are the hands above the head?", "Is the person touching the floor?").
3. **Temporal:** Focus on the sequence of events (e.g., "What happens immediately after the jump?", "Does the person stand up before or after waving?").

### RULES
- **Filter:** If the caption does not contain information for a specific category, omit that category. Do NOT hallucinate details.
- **Independence:** The answer must stand alone (avoid pronouns like "he" or "it" where possible; refer to "the person" or "the movement").
- **Paraphrasing:** Do not copy-paste the caption. Rephrase naturally.

### INPUT CAPTION
'{generated_caption}'

### OUTPUT FORMAT
Provide the output in valid JSON format only. Do not add markdown backticks.
{{
  'qa_pairs': [
    {{
      'type': 'Action',
      'question': '...',
      'answer': '...'
    }},
    {{
      'type': 'Body-Spatial',
      'question': '...',
      'answer': '...'
    }}
  ]
}}"""

class OpenRouterClient:
    """Handles communication with the OpenRouter API."""
    def __init__(self):
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        self.model_id = "google/gemini-3-flash-preview"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # NEW: Use a Session to reuse TCP connections and prevent socket exhaustion
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def extract_frames_as_base64(self, video_path: Path, num_frames: int = 16) -> list:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Error: Video {video_path} has no frames or could not be read")
            return []

        num_frames = min(num_frames, total_frames)
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        frames_base64 = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                max_dim = 512
                if max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
                _, buffer = cv2.imencode('.jpg', frame)
                frames_base64.append(base64.b64encode(buffer).decode('utf-8'))
                
        cap.release()
        return frames_base64

    def generate_detailed_caption(self, video_path: Path, gt_caption: str) -> str:
        frames_base64 = self.extract_frames_as_base64(video_path)
        if not frames_base64:
            return ""

        content = [{"type": "text", "text": DETAILED_CAPTION_GENERATION_TEMPLATE.format(n=len(frames_base64), humanML_groundtruth_caption=gt_caption)}]
        
        for base64_image in frames_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        data = {"model": self.model_id, "messages": [{"role": "user", "content": content}]}

        try:
            # NEW: Call using self.session instead of requests
            response = self.session.post(self.api_url, json=data, timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error generating caption: {e}")
            return ""

    def generate_qna_pairs(self, detailed_caption: str) -> str:
        if not detailed_caption:
            return ""

        prompt = QnA_GENERATION_TEMPLATE.format(generated_caption=detailed_caption)
        data = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}]}

        try:
            # NEW: Call using self.session instead of requests
            response = self.session.post(self.api_url, json=data, timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error generating QnA: {e}")
            return ""