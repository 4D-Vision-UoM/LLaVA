import json
import os
import concurrent.futures
import threading
from tqdm import tqdm
from metrics import BleuMetric, RougeMetric, MeteorMetric, BertScoreMetric, SimCSEMetric, LLMJudgeMetric

class PipelineEvaluator:
    def __init__(self, run_llm_judge=False, max_workers=10):
        self.bleu = BleuMetric()
        self.rouge = RougeMetric()
        self.meteor = MeteorMetric()       
        self.bert_score = BertScoreMetric() 
        self.simcse = SimCSEMetric()       
        
        self.run_llm_judge = run_llm_judge
        if self.run_llm_judge:
            self.llm_judge = LLMJudgeMetric()
            
        self.max_workers = max_workers
        self.local_metrics_lock = threading.Lock()

    def evaluate_file(self, 
                      input_filepath: str, 
                      output_filepath: str, 
                      aggregation_filepath: str,
                      failed_filepath: str = "failed_llm_evals.json",
                      test_mode: bool = False,                        
                      test_samples: int = 10,
                      save_interval: int = 50):
        
        with open(input_filepath, 'r') as f:
            data = json.load(f)

        if test_mode:
            print(f"\n[TEST MODE ENABLED] Only evaluating the first {test_samples} samples.\n")
            data = data[:test_samples]

        results = []
        failed_llm_samples = [] 

        print(f"Starting parallel evaluation with {self.max_workers} workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_item, item): item for item in data}
            
            completed = 0
            progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(data), desc="Evaluating")
            
            for future in progress_bar:
                completed += 1
                try:
                    item, has_failed = future.result()
                    sample_idx = item.get('sample_idx')
                    
                    if has_failed:
                        tqdm.write(f"  -> [WARNING] LLM Judge parse failed for idx {sample_idx}.")
                        failed_llm_samples.append(item)

                    results.append(item)

                    if completed % save_interval == 0:
                        tqdm.write(f"--- Checkpoint Reached ({completed} items). Saving to disk... ---")
                        self._save_state(results, failed_llm_samples, output_filepath, failed_filepath, aggregation_filepath)

                except Exception as exc:
                    tqdm.write(f"Sample generated an exception: {exc}")

        print("\nEvaluation complete. Executing final save...")
        self._save_state(results, failed_llm_samples, output_filepath, failed_filepath, aggregation_filepath)

    def _process_item(self, item: dict) -> tuple:
        ground_truth = item.get("ground_truth", "")
        question = item.get("question", "")
        prediction = item.get("prediction", "")
        
        eval_result = self._evaluate_single(question, ground_truth, prediction)
        
        has_failed = False
        
        if self.run_llm_judge:
            judge_res = eval_result.get("llm_judge", {})
            if judge_res.get("score", 0) == 0 and "error" in judge_res.get("reasoning", "").lower():
                has_failed = True
        
        item["evaluation"] = eval_result
        return item, has_failed

    def _save_state(self, results, failed_samples, out_file, fail_file, agg_file):
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        if self.run_llm_judge and failed_samples:
            with open(fail_file, 'w') as f:
                json.dump(failed_samples, f, indent=4)

        self._recalculate_aggregations(results, agg_file)

    def retry_failed(self, 
                     failed_filepath: str, 
                     main_output_filepath: str, 
                     main_aggregation_filepath: str):
        
        if not os.path.exists(failed_filepath) or not os.path.exists(main_output_filepath):
            print("Error: Could not find the failed file or the main output file.")
            return

        with open(failed_filepath, 'r') as f:
            failed_data = json.load(f)
            
        with open(main_output_filepath, 'r') as f:
            main_results = json.load(f)

        print(f"\n[RETRY MODE] Attempting to fix {len(failed_data)} failed evaluations in parallel...")
        
        still_failed = []
        fixed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_item, item) for item in failed_data]
            progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Retrying Fails")
            
            for future in progress_bar:
                item, has_failed = future.result()
                sample_idx = item.get("sample_idx")
                
                if has_failed:
                    tqdm.write(f"     [FAILED AGAIN] idx: {sample_idx}")
                    still_failed.append(item)
                else:
                    tqdm.write(f"     [SUCCESS] idx: {sample_idx}")
                    fixed_count += 1
                    
                    found_in_main = False
                    for main_item in main_results:
                        if main_item.get("sample_idx") == sample_idx:
                            main_item["evaluation"] = item["evaluation"]
                            found_in_main = True
                            break
                            
                    if not found_in_main:
                        main_results.append(item)

        with open(main_output_filepath, 'w') as f:
            json.dump(main_results, f, indent=4)
        with open(failed_filepath, 'w') as f:
            json.dump(still_failed, f, indent=4)

        self._recalculate_aggregations(main_results, main_aggregation_filepath)
        print(f"\nRetry complete. Fixed {fixed_count} items. {len(still_failed)} items still failing.")

    def _evaluate_single(self, question: str, reference: str, hypothesis: str) -> dict:
        if not hypothesis or not reference:
            return {"error": "Missing prediction or ground truth"}

        with self.local_metrics_lock:
            metrics_result = {
                "bleu": self.bleu.compute(reference, hypothesis),
                "rouge": self.rouge.compute(reference, hypothesis),
                "meteor": self.meteor.compute(reference, hypothesis),      
                "bert_score": self.bert_score.compute(reference, hypothesis), 
                "simcse_cosine": self.simcse.compute(reference, hypothesis) 
            }

        if self.run_llm_judge:
            metrics_result["llm_judge"] = self.llm_judge.compute(question, reference, hypothesis)

        return metrics_result

    def _accumulate_metrics(self, totals_dict: dict, current_metrics: dict):
        for key, value in current_metrics.items():
            if isinstance(value, dict):
                if key not in totals_dict:
                    totals_dict[key] = {}
                self._accumulate_metrics(totals_dict[key], value)
            elif isinstance(value, (int, float)):
                if key not in totals_dict:
                    totals_dict[key] = 0.0
                totals_dict[key] += value

    def _average_metrics(self, totals_dict: dict, count: int) -> dict:
        if count == 0:
            return {}
        avg_dict = {}
        for key, value in totals_dict.items():
            if isinstance(value, dict):
                avg_dict[key] = self._average_metrics(value, count)
            elif isinstance(value, (int, float)):
                avg_dict[key] = value / count
        return avg_dict

    def _recalculate_aggregations(self, all_results: list, aggregation_filepath: str):
        """
        Calculates averages strictly for valid evaluations.
        """
        totals = {}
        count = 0
        
        for item in all_results:
            eval_res = item.get("evaluation", {})
            
            is_valid = "error" not in eval_res
            if self.run_llm_judge and is_valid:
                if eval_res.get("llm_judge", {}).get("score", 0) == 0:
                    is_valid = False
                    
            if is_valid:
                self._accumulate_metrics(totals, eval_res)
                count += 1

        aggregated_results = {
            "averages": self._average_metrics(totals, count),
            "total_successful_samples": count
        }
        
        with open(aggregation_filepath, 'w') as f:
            json.dump(aggregated_results, f, indent=4)
        
        tqdm.write(f"Aggregations updated. (Valid items: {count})")