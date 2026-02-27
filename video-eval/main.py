from models import get_model
from pipeline import process_vqa_dataset
import os 

def main():
    # --- Run Configuration ---

    PROVIDER = "openrouter"  # "videollama" or "gemini" 
    
    # --- Paths Configuration ---
    VQA_BASE_DIR = "data/gemini/test"
    VIDEO_BASE_DIR = "data/v4.4-humanML3d-2136-video/mp4_files/test"
    SCRIPT_DIR = "video-eval"
    OUTPUT_DIR = f"{SCRIPT_DIR}/outputs/{PROVIDER}-allframes"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_JSON_PATH = f"{OUTPUT_DIR}/predictions_output.json"
    FAILURES_JSON_PATH = f"{OUTPUT_DIR}/predictions_failures.json"
    

    MAX_TEST_SAMPLES =None # Set to an integer to test a small batch, or None for all
    
    # --- NEW: Retry Configuration ---
    # Set to True if you ONLY want to process items listed in predictions_failures.json
    RETRY_FAILED = False 
    # --- Run Configuration ---
     # Switched to OpenRouter
    
    # Choose your OpenRouter model. Here are a few great video options:
    # "google/gemini-pro-1.5"
    # "google/gemini-flash-1.5"
    # "openai/gpt-4o"
    MODEL_NAME = "google/gemini-3-flash-preview"
    # NEW: Toggle your sampling mode here!
    # "window" -> 32 frames with stride
    # "all" -> Passes the full original video
    SAMPLING_MODE = "all"

    print(f"Initializing pipeline with provider: {PROVIDER.upper()}")
    
    # Initialize the selected model
    if PROVIDER == "videollama":
        model = get_model("videollama", model_name="DAMO-NLP-SG/VideoLLaMA3-7B")
    elif PROVIDER == "gemini":
        model = get_model("gemini", model_name="gemini-1.5-flash")
    elif PROVIDER == "openrouter":
        model = get_model("openrouter", model_name=MODEL_NAME)
    
    # Run the Pipeline
    process_vqa_dataset(
        vqa_dir=VQA_BASE_DIR, 
        video_dir=VIDEO_BASE_DIR, 
        output_path=OUTPUT_JSON_PATH,
        failures_path=FAILURES_JSON_PATH, # Pass the failure log path
        model=model, 
        max_samples=MAX_TEST_SAMPLES,
        sampling_mode=SAMPLING_MODE,
        retry_failed=RETRY_FAILED,         # Pass the retry flag
        max_workers=10,    
    )

if __name__ == "__main__":
    main()
    