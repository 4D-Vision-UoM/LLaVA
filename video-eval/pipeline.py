import os
from tqdm import tqdm
from utils import get_total_frames, load_json, save_json

def process_vqa_dataset(vqa_dir, video_dir, output_path, failures_path, model, max_samples=None, retry_failed=False):
    """
    Processes the VQA dataset with support for resuming, tracking failures, and retrying.
    Assigns a uniform sample_idx for all questions belonging to the same sequence.
    """
    
    # --- 1. Load Existing State (Resume Functionality) ---
    results = load_json(output_path) if os.path.exists(output_path) else []
    completed_tasks = {(res["motion_id"], res["qa_idx"]) for res in results}
    
    failures = load_json(failures_path) if os.path.exists(failures_path) else []
    failure_dict = {(f["motion_id"], f["qa_idx"]): f for f in failures}

    print(f"State loaded: {len(results)} completed, {len(failure_dict)} failed.")

    # --- 2. Determine Processing Targets ---
    sequence_folders = sorted([f for f in os.listdir(vqa_dir) if f.startswith("sequence_")])
    if max_samples is not None:
        sequence_folders = sequence_folders[:max_samples]
        
    # Track sample_idx based on the sequence index using enumerate
    for sample_idx, seq_folder in enumerate(tqdm(sequence_folders, desc="Processing Sequences")):
        motion_id = seq_folder
        json_path = os.path.join(vqa_dir, motion_id, f"{motion_id}_vqa_pairs.json")
        video_path = os.path.join(video_dir, f"{motion_id}.mp4")
        
        if not os.path.exists(json_path) or not os.path.exists(video_path):
            continue
            
        total_frames = get_total_frames(video_path)
        vqa_data = load_json(json_path)
        qa_pairs = vqa_data.get("qa_pairs", [])
        
        sequence_updated = False 
        
        for qa_idx, qa in enumerate(qa_pairs):
            task_key = (motion_id, qa_idx)
            
            # --- Skipping Logic ---
            if retry_failed:
                if task_key not in failure_dict:
                    continue
            else:
                if task_key in completed_tasks:
                    continue

            # --- Process Video ---
            # Safely grab the question and answer to prevent KeyErrors
            question = qa.get("question", qa.get("q", ""))
            ground_truth = qa.get("answer", qa.get("a", ""))
            
            if not question:
                print(f"\nWarning: Missing question in {motion_id} at index {qa_idx}. Skipping.")
                continue
            
            prediction = model.analyze_video(video_path, question)
            
            is_error = prediction.startswith("Error:") or "Error processing video" in prediction
            
            task_info = {
                "sample_idx": sample_idx,  # Uses the sequence index
                "qa_idx": qa_idx,          # 0, 1, 2 for each question in this specific video
                "motion_id": motion_id,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "total_frames": total_frames
            }

            if is_error:
                failure_dict[task_key] = task_info
            else:
                results.append(task_info)
                completed_tasks.add(task_key)
                
                if task_key in failure_dict:
                    del failure_dict[task_key]
            
            sequence_updated = True

        # --- 3. Checkpoint Saving ---
        if sequence_updated:
            save_json(results, output_path)
            save_json(list(failure_dict.values()), failures_path)

    print(f"\nPipeline finished. Total successful: {len(results)} | Total currently failed: {len(failure_dict)}")