import os
import concurrent.futures
from threading import Lock
from tqdm import tqdm
from utils import get_total_frames, load_json, save_json

def process_single_task(task_args):
    """Worker function for parallel processing."""
    video_path, question, ground_truth, motion_id, sample_idx, qa_idx, total_frames, model = task_args
    
    # Run the model inference
    prediction = model.analyze_video(video_path, question)
    
    # Determine if an error occurred during processing
    is_error = prediction.startswith("Error:") or "Error processing video" in prediction
    
    # Construct the result dictionary using the passed-in sample_idx
    task_info = {
        "sample_idx": sample_idx,
        "qa_idx": qa_idx,
        "motion_id": motion_id,
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "total_frames": total_frames
    }
    
    return is_error, task_info

def process_vqa_dataset(vqa_dir, video_dir, output_path, failures_path, model, max_samples=None, retry_failed=False, max_workers=5):
    """
    Processes the VQA dataset in parallel.
    Assigns a uniform sample_idx for all questions belonging to the same sequence.
    """
    
    # --- 1. Load Existing State (Resume Functionality) ---
    results = load_json(output_path) if os.path.exists(output_path) else []
    completed_tasks = {(res["motion_id"], res["qa_idx"]) for res in results}
    
    failures = load_json(failures_path) if os.path.exists(failures_path) else []
    failure_dict = {(f["motion_id"], f["qa_idx"]): f for f in failures}

    print(f"State loaded: {len(results)} completed, {len(failure_dict)} failed.")

    # --- 2. Determine Processing Targets & Queue Tasks ---
    sequence_folders = sorted([f for f in os.listdir(vqa_dir) if f.startswith("sequence_")])
    if max_samples is not None:
        sequence_folders = sequence_folders[:max_samples]
        
    tasks_to_run = []
    
    # Use enumerate here to establish the correct sample_idx for each sequence folder
    for sample_idx, seq_folder in enumerate(sequence_folders):
        motion_id = seq_folder
        json_path = os.path.join(vqa_dir, motion_id, f"{motion_id}_vqa_pairs.json")
        video_path = os.path.join(video_dir, f"{motion_id}.mp4")
        
        if not os.path.exists(json_path) or not os.path.exists(video_path):
            continue
            
        total_frames = get_total_frames(video_path)
        vqa_data = load_json(json_path)
        qa_pairs = vqa_data.get("qa_pairs", [])
        
        for qa_idx, qa in enumerate(qa_pairs):
            task_key = (motion_id, qa_idx)
            
            # --- Skipping Logic ---
            if retry_failed:
                if task_key not in failure_dict:
                    continue
            else:
                if task_key in completed_tasks:
                    continue

            # Safely grab the question and answer
            question = qa.get("question", qa.get("q", ""))
            ground_truth = qa.get("answer", qa.get("a", ""))
            
            if not question:
                print(f"\nWarning: Missing question in {motion_id} at index {qa_idx}. Skipping.")
                continue
                
            # Add the task to the queue, making sure to pass sample_idx!
            tasks_to_run.append(
                (video_path, question, ground_truth, motion_id, sample_idx, qa_idx, total_frames, model)
            )

    print(f"Tasks queued for processing: {len(tasks_to_run)}")
    if len(tasks_to_run) == 0:
        return

    # --- 3. Execute in Parallel ---
    write_lock = Lock() # Lock prevents threads from overwriting each other when saving to disk
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all queued tasks
        future_to_task = {executor.submit(process_single_task, task): task for task in tasks_to_run}
        
        # Display progress bar as tasks complete
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks_to_run), desc="Processing Questions"):
            try:
                is_error, task_info = future.result()
                task_key = (task_info["motion_id"], task_info["qa_idx"])
                
                with write_lock:
                    if is_error:
                        failure_dict[task_key] = task_info
                    else:
                        results.append(task_info)
                        completed_tasks.add(task_key)
                        
                        # Remove from failures if it previously failed but now succeeded
                        if task_key in failure_dict:
                            del failure_dict[task_key]
                            
                    # Save checkpoints incrementally safely
                    save_json(results, output_path)
                    save_json(list(failure_dict.values()), failures_path)
                    
            except Exception as exc:
                print(f"\nTask generated an exception: {exc}")

    print(f"\nPipeline finished. Total successful: {len(results)} | Total currently failed: {len(failure_dict)}")