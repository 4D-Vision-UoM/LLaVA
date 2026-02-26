import json
import time
import threading
import concurrent.futures
from pathlib import Path
from tqdm import tqdm  # NEW: Import tqdm
from video_locator import VideoLocator
from api_client import OpenRouterClient

class PipelineOrchestrator:
    def __init__(self, data_dir: Path, external_videos_dir: Path, output_base_dir: Path, log_dir: Path, split: str):
        self.data_dir = Path(data_dir)
        self.output_base_dir = Path(output_base_dir)
        self.log_path = Path(log_dir) / f"{split}_log.jsonl"
        self.split = split
        
        self.locator = VideoLocator(external_videos_dir)
        self.api_client = OpenRouterClient()
        
        self.log_lock = threading.Lock()
        self.log_data = self._load_log()
        
        # Batch writing variables
        self.log_buffer = []
        self.batch_size = 10
        
        self.api_max_retries = 3

    def _load_log(self) -> dict:
        """Reads the JSONL file line by line to build the skip-list."""
        log_dict = {}
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        log_dict.update(entry)
        return log_dict

    def _flush_buffer(self):
        """Writes everything in the buffer to disk and clears it. MUST be called inside a lock."""
        if not self.log_buffer:
            return
            
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'a') as f:
                # writelines is extremely fast for writing a list of strings
                f.writelines(self.log_buffer)
            self.log_buffer.clear()
        except Exception as e:
            tqdm.write(f"  [Error] Could not append to log.jsonl: {e}")

    def _mark_status(self, sequence_name: str, success: bool, reason: str = None):
        status = {"success": success}
        if reason:
            status["reason"] = reason
            
        # 1. Prepare the JSONL string (must end with a newline)
        log_entry = {sequence_name: status}
        log_string = json.dumps(log_entry) + '\n'

        # 2. Safely update memory and buffer
        with self.log_lock:
            self.log_data[sequence_name] = status
            self.log_buffer.append(log_string)
            
            # 3. If buffer reaches 10, flush it to the hard drive
            if len(self.log_buffer) >= self.batch_size:
                self._flush_buffer()

    def _process_sequence(self, video_dir: Path):
        """Worker function that handles a single sequence."""
        sequence_name = video_dir.name

        report_path = video_dir / "report.json"
        if not report_path.exists():
            tqdm.write(f"[{sequence_name}] No report found.")
            self._mark_status(sequence_name, False, "no report")
            return
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        gt_caption = report.get("description")
        if not gt_caption:
            tqdm.write(f"[{sequence_name}] No 'description' field in report.")
            self._mark_status(sequence_name, False, "no description")
            return

        video_path = self.locator.find_video(sequence_name, split=self.split)
        if not video_path:
            self._mark_status(sequence_name, False, "missing external video")
            return

        detailed_caption = ""
        for attempt in range(self.api_max_retries):
            detailed_caption = self.api_client.generate_detailed_caption(video_path, gt_caption)
            if detailed_caption:
                break
            tqdm.write(f"[{sequence_name}] API failed, retrying caption... ({attempt + 1}/{self.api_max_retries})")
            time.sleep(2)

        if not detailed_caption:
            tqdm.write(f"[{sequence_name}] Failed to generate caption after {self.api_max_retries} attempts.")
            self._mark_status(sequence_name, False, "caption generation failed")
            return

        qna_pairs = ""
        for attempt in range(self.api_max_retries):
            qna_pairs = self.api_client.generate_qna_pairs(detailed_caption)
            if qna_pairs:
                break
            tqdm.write(f"[{sequence_name}] API failed, retrying QnA... ({attempt + 1}/{self.api_max_retries})")
            time.sleep(2)

        if not qna_pairs:
            tqdm.write(f"[{sequence_name}] Failed to generate QnA after {self.api_max_retries} attempts.")
            self._mark_status(sequence_name, False, "qna generation failed")
            return

        sequence_output_dir = self.output_base_dir / self.split / sequence_name
        sequence_output_dir.mkdir(parents=True, exist_ok=True)

        txt_output_path = sequence_output_dir / f"{sequence_name}.txt"
        qna_output_path = sequence_output_dir / f"{sequence_name}_qna.json"

        with open(txt_output_path, "w") as f: 
            f.write(detailed_caption)

        with open(qna_output_path, "w") as f: 
            f.write(qna_pairs)
        
        self._mark_status(sequence_name, True)

    def run(self, limit: int = 1000, max_workers: int = 4, retry_failed: bool = False):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {self.data_dir}")

        all_video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and "sequence" in d.name])[:limit]
        
        dirs_to_process = []
        for video_dir in all_video_dirs:
            seq_name = video_dir.name
            status = self.log_data.get(seq_name, {})
            
            if status.get("success") is True:
                continue
                
            if status.get("success") is False and not retry_failed:
                continue
                
            dirs_to_process.append(video_dir)

        if not dirs_to_process:
            print("No new or failed sequences left to process!")
            return

        print(f"Found {len(dirs_to_process)} sequences to process using {max_workers} threads...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_sequence, d): d for d in dirs_to_process}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Videos"):
                try:
                    future.result()
                except Exception as e:
                    seq_dir = futures[future]
                    tqdm.write(f"FATAL ERROR in thread for {seq_dir.name}: {e}")
                    
        # CRITICAL: Flush any remaining items in the buffer at the very end
        with self.log_lock:
            self._flush_buffer()
        print("Pipeline execution complete. All logs saved.")