from models import get_model
from pipeline import process_vqa_dataset
import os 

def main():
    # --- Run Configuration ---
    # Change 'provider' to "gemini" to use the Gemini API plugin instead
    PROVIDER = "videollama" 
    
    # --- Paths Configuration ---
    VQA_BASE_DIR = "data/All-Datasets/Vqa-Datasets/v4.4-wall-humanML3d-2136-video/vqa-video/gemini-flash/test"
    VIDEO_BASE_DIR = "data/All-Datasets/Vqa-Datasets/v4.4-wall-humanML3d-2136-video/video/mp4_files/test"
    SCRIPT_DIR = "video-eval"
    OUTPUT_DIR = f"{SCRIPT_DIR}/outputs/{PROVIDER}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_JSON_PATH = f"{OUTPUT_DIR}/predictions_output.json"
    FAILURES_JSON_PATH = f"{OUTPUT_DIR}/predictions_failures.json"
    
    # --- Run Configuration ---
    PROVIDER = "videollama"  # "videollama" or "gemini"
    MAX_TEST_SAMPLES =None # Set to an integer to test a small batch, or None for all
    
    # --- NEW: Retry Configuration ---
    # Set to True if you ONLY want to process items listed in predictions_failures.json
    RETRY_FAILED = False 
    

    print(f"Initializing pipeline with provider: {PROVIDER.upper()}")
    
    # Initialize the selected model
    if PROVIDER == "videollama":
        model = get_model("videollama", model_name="DAMO-NLP-SG/VideoLLaMA3-7B")
    elif PROVIDER == "gemini":
        model = get_model("gemini", model_name="gemini-1.5-flash")
    
    # Run the Pipeline
    process_vqa_dataset(
        vqa_dir=VQA_BASE_DIR, 
        video_dir=VIDEO_BASE_DIR, 
        output_path=OUTPUT_JSON_PATH,
        failures_path=FAILURES_JSON_PATH, # Pass the failure log path
        model=model, 
        max_samples=MAX_TEST_SAMPLES,
        retry_failed=RETRY_FAILED         # Pass the retry flag
    )

if __name__ == "__main__":
    main()
    