import os
import concurrent.futures
from threading import Lock
from tqdm import tqdm
import uuid
from utils import get_total_frames_and_fps, load_json, save_json, get_frame_indices, create_sampled_video

def process_single_task(task_args):
    # Make sure to unpack orig_fps here!
    video_path, question, ground_truth, motion_id, sample_idx, qa_idx, total_frames, orig_fps, model, sampling_mode = task_args
    
    # 1. Get the indices based on the selected mode (now using orig_fps)
    indices = get_frame_indices(total_frames, orig_fps, mode=sampling_mode, num_frames=32, stride=2, split='test')
    
    temp_video_path = None
    
    try:
        if sampling_mode == 'all':
            # Create a 10FPS temporary video of ALL the resampled frames
            unique_id = uuid.uuid4().hex[:8]
            temp_video_path = f"./temp_{motion_id}_{qa_idx}_{unique_id}.mp4"
            create_sampled_video(video_path, temp_video_path, indices, target_fps=10)
            prediction = model.analyze_video(temp_video_path, question)
        else:
            # Create the 32-frame Windowed temporary video
            unique_id = uuid.uuid4().hex[:8]
            temp_video_path = f"./temp_{motion_id}_{qa_idx}_{unique_id}.mp4"
            create_sampled_video(video_path, temp_video_path, indices, target_fps=10)
            prediction = model.analyze_video(temp_video_path, question)
            
    except Exception as e:
        prediction = f"Error generating or analyzing video: {str(e)}"
    finally:
        # 5. Clean up the temporary video file if we created one
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    is_error = prediction.startswith("Error:") or "Error processing" in prediction
    
    task_info = {
        "sample_idx": sample_idx,
        "qa_idx": qa_idx,
        "motion_id": motion_id,
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "total_frames": total_frames,
        # "sampling_mode": sampling_mode, # Track which mode was used
        # "sampled_indices": indices      # Track the exact frames used
    }
    
    return is_error, task_info

def process_vqa_dataset(vqa_dir, video_dir, output_path, failures_path, model, max_samples=None, retry_failed=False, max_workers=5,sampling_mode='window',):
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
        
        total_frames, orig_fps = get_total_frames_and_fps(video_path)
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
                (video_path, question, ground_truth, motion_id, sample_idx, qa_idx, total_frames,orig_fps, model, sampling_mode)
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