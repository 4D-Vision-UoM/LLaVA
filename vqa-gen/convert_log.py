import json
from pathlib import Path

def convert_json_to_jsonl():
    # Define your paths based on your previous setup
    log_dir = Path("/data/dulangaw/4d-fyp/isuranga/LLaVA/vqa-gen/logs")
    old_json_path = log_dir / "train_log.json"
    new_jsonl_path = log_dir / "train_log.jsonl"

    if not old_json_path.exists():
        print(f"Could not find the old log file at {old_json_path}")
        return

    print(f"Reading old JSON from: {old_json_path}")
    
    # 1. Load the old monolithic dictionary
    with open(old_json_path, 'r') as f:
        old_log_data = json.load(f)

    print(f"Found {len(old_log_data)} entries. Converting to JSONL...")

    # 2. Write each sequence as a brand new line in the .jsonl file
    with open(new_jsonl_path, 'w') as f:
        for sequence_name, status in old_log_data.items():
            # Create a single mini-dictionary for the line
            line_dict = {sequence_name: status}
            
            # Convert to a string and write it with a newline
            f.write(json.dumps(line_dict) + '\n')

    print(f"Success! Converted log saved to: {new_jsonl_path}")
    print("You can now safely delete or rename the old log.json file.")

if __name__ == "__main__":
    convert_json_to_jsonl()