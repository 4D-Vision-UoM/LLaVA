import os 
import yaml 
import cv2
import base64
import requests
import subprocess
import json
from pathlib import Path 

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

# Path setup
# Assuming the script is run from project root or tools/ folder.
# We derive paths relative to this file to be robust.
current_file_path = Path(__file__).resolve()
# If this file is in tools/, the project root is parent.parent
project_root = current_file_path.parent.parent

data_dir = project_root / "data" / "v4.4-humanML3d-2136-video" / "train" / "Env1"
config_path = project_root / "configs" / "openrouter" / "OpenRouter_API.yaml"

# Load the Openrouter API key
if not config_path.exists():
    # Fallback to check if running from project root and config path is relative
    if Path("configs/openrouter/OpenRouter_API.yaml").exists():
        config_path = Path("configs/openrouter/OpenRouter_API.yaml")
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")

print(f"Loading config from {config_path}")
with open(config_path, "r") as f: 
    config = yaml.safe_load(f)
    # Check for both cases
    api_key = config.get("OPENROUTER_API_KEY") or config.get("openrouter_api_key")
    if not api_key:
        raise ValueError(f"OPENROUTER_API_KEY not found in config file {config_path}")
    os.environ["OPENROUTER_API_KEY"] = api_key

# Get video directories
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory not found at {data_dir}")

# List directories in data_dir
video_dirs = [d for d in data_dir.iterdir() if d.is_dir() and "sequence" in d.name]
video_dirs = sorted(video_dirs)[:1000]  # Take first 10

if not video_dirs:
    print(f"No video directories found in {data_dir}")

def generate_detailed_caption(video_path, humanML_groundtruth_caption, num_frames=16):
    # Extract frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return ""
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Error: Video {video_path} has no frames or could not be read")
        return ""

    if num_frames > total_frames:
        num_frames = total_frames
        
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        print(f"Error: No frames extracted from {video_path}")
        return ""

    # Prepare content for OpenRouter
    content = [{"type": "text", "text": DETAILED_CAPTION_GENERATION_TEMPLATE.format(n=len(frames), humanML_groundtruth_caption=humanML_groundtruth_caption)}]
    for frame in frames:
        # Resize frame if too large to save tokens/bandwidth
        height, width = frame.shape[:2]
        max_dim = 512
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    model_id = "google/gemini-3-flash-preview" 
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": content}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code != 200:
             print(f"Error generating caption API error {response.status_code}: {response.text}")
             return ""
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating caption exception: {e}")
        return ""

def generate_qna_pairs(detailed_caption): 
    if not detailed_caption:
        return ""
        
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"
    }

    prompt = QnA_GENERATION_TEMPLATE.format(generated_caption=detailed_caption)

    data = {
        "model": "google/gemini-3-flash-preview", 
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code != 200:
             print(f"Error generating QnA API error {response.status_code}: {response.text}")
             return ""
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating QnA exception: {e}")
        return ""

# Process videos
log_json_dict = {}
log_dir = Path("/data/dulangaw/4d-fyp/sasika/Pipeline/data")
log_json_path = log_dir / "log.json"

if log_json_path.exists():
    with open(log_json_path, 'r') as f:
        log_json_dict = json.load(f)

for i, video_dir in enumerate(video_dirs): 
    # Keep a log json file 
    if video_dir.name in log_json_dict:
        if log_json_dict[video_dir.name].get("success") == True:
            continue

    print(f"Processing directory: {video_dir.name}")

    # Report path 
    report_path = data_dir / video_dir.name / "report.json"
    if not report_path.exists():
        print(f"No report found in {video_dir}")
        log_json_dict[video_dir.name] = {"success": False, "reason": "no report"}
        with open(log_json_path, 'w') as f:
            json.dump(log_json_dict, f, indent=2)
        continue
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    humanML_groundtruth_caption = report.get("description")
    if not humanML_groundtruth_caption:
        print(f"No 'description' field in {report_path}, skipping.")
        log_json_dict[video_dir.name] = {"success": False, "reason": "no description"}
        with open(log_json_path, 'w') as f:
            json.dump(log_json_dict, f, indent=2)
        continue

    # Find .webm file in the directory
    webm_files = list(video_dir.glob("*.webm"))
    if not webm_files:
        print(f"No .webm file found in {video_dir}")
        log_json_dict[video_dir.name] = {"success": False, "reason": "no webm"}
        with open(log_json_path, 'w') as f:
            json.dump(log_json_dict, f, indent=2)
        continue
    
    video_path_obj = webm_files[0]
    print(f"Found video file: \n\t{video_path_obj}")

    # Convert the webm file to mp4 using ffmpeg
    mp4_video_path = video_path_obj.with_suffix(".mp4")
    if not mp4_video_path.exists():
        print(f"Converting {video_path_obj.name} to .mp4...")
        try:
            subprocess.run([
                "ffmpeg", "-i", str(video_path_obj),
                "-c:v", "libx264", "-c:a", "aac",
                "-strict", "experimental", str(mp4_video_path),
                "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Conversion successful: {mp4_video_path}")
        except FileNotFoundError:
            print("ffmpeg not found. Please install ffmpeg or add it to PATH.")
            log_json_dict[video_dir.name] = {"success": False, "reason": "ffmpeg not found"}
            with open(log_json_path, 'w') as f:
                json.dump(log_json_dict, f, indent=2)
            continue
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert video: {e}")
            log_json_dict[video_dir.name] = {"success": False, "reason": "ffmpeg conversion failed"}
            with open(log_json_path, 'w') as f:
                json.dump(log_json_dict, f, indent=2)
            continue
    else:
        print(f"MP4 already exists: \n\t{mp4_video_path}")

    # Generate detailed captions using the converted mp4 file
    detailed_caption = generate_detailed_caption(mp4_video_path, humanML_groundtruth_caption)
    
    if not detailed_caption:
        print("Skipping QnA generation due to empty caption.")
        log_json_dict[video_dir.name] = {"success": False, "reason": "empty caption"}
        with open(log_json_path, 'w') as f:
            json.dump(log_json_dict, f, indent=2)
        continue

    # Generate QnA pairs for the video using the detailed caption
    qna_pairs = generate_qna_pairs(detailed_caption)

    # Save outputs
    # Keep output naming consistent
    txt_output_path = video_path_obj.with_suffix(".txt")
    qna_output_path = video_path_obj.parent / (video_path_obj.stem + "_qna.json")

    print("===============================")
    print("[Ground truth caption]", "\n\t", humanML_groundtruth_caption)
    print("-------------------------------")
    print("\n[Generated Detailed Caption]", "\n\t", detailed_caption)
    print("-------------------------------")
    # print("\n[Generated QnA Pairs]", "\n\t", qna_pairs)
    print("Report path: \n\t", report_path)
    print("===============================")

    with open(txt_output_path, "w") as f: 
        f.write(detailed_caption)
        print(f"Saved caption to \n\t{txt_output_path}")

    with open(qna_output_path, "w") as f: 
        f.write(qna_pairs)
        print(f"Saved QnA to \n\t{qna_output_path}")
    
    log_json_dict[video_dir.name] = {"success": True}
    with open(log_json_path, 'w') as f:
        json.dump(log_json_dict, f, indent=2)
    print(f"Log saved for {video_dir.name}")