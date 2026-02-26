from pathlib import Path
from config_manager import ConfigManager
from pipeline import PipelineOrchestrator

def main():
    # 1. Setup Configuration & API Keys
    config = ConfigManager()
    config.load_api_key()

    # 2. Define Split
    CURRENT_SPLIT = "train"
     
    # 3. Define Paths
    # Update EXTERNAL_VIDEOS_DIR to the root of your train/test/val folders
    EXTERNAL_VIDEOS_DIR = "data/v4.4-humanML3d-2136-video/mp4_files" 
    
    DATA_DIR = f"data/v4.4-humanML3d-2136-video/{CURRENT_SPLIT}/Env1"
    LOG_DIR = "vqa-gen/logs"
    OUTPUT_BASE_DIR = "vqa-gen/generated_outputs"


    # 4. Initialize and Run Pipeline
    print(f"Starting pipeline for split: {CURRENT_SPLIT}")
    orchestrator = PipelineOrchestrator(
        data_dir=DATA_DIR, 
        external_videos_dir=EXTERNAL_VIDEOS_DIR, 
        log_dir=LOG_DIR,
        split=CURRENT_SPLIT,
        output_base_dir=OUTPUT_BASE_DIR
    )
    
    orchestrator.run(
        limit=None, 
        max_workers=10,        # Number of parallel processes (don't set too high or API will rate limit you)
        retry_failed=True     # Set to True to re-run items marked as 'success: false' in log.json
    )

if __name__ == "__main__":
    main()