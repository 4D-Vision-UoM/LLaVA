import json
import os
import concurrent.futures
import threading
from tqdm import tqdm
from metrics import BleuMetric, RougeMetric, MeteorMetric, BertScoreMetric, SimCSEMetric, LLMJudgeMetric

class PipelineEvaluator:
    def __init__(self, run_llm_judge=False, max_workers=2):
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
                      save_interval: int = 2):
        
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
                    item, ft_failed, init_failed = future.result()
                    sample_idx = item.get('sample_idx')
                    
                    if ft_failed or init_failed:
                        tqdm.write(f"  -> [WARNING] LLM Judge parse failed for idx {sample_idx}.")
                        failed_llm_samples.append(item)

                    results.append(item)

                    if completed % save_interval == 0:
                        tqdm.write(f"--- Checkpoint Reached ({completed} items). Saving to disk... ---")
                        # Notice we no longer pass running totals. It recalculates purely from the valid results.
                        self._save_state(results, failed_llm_samples, output_filepath, failed_filepath, aggregation_filepath)

                except Exception as exc:
                    tqdm.write(f"Sample generated an exception: {exc}")

        print("\nEvaluation complete. Executing final save...")
        self._save_state(results, failed_llm_samples, output_filepath, failed_filepath, aggregation_filepath)

    def _process_item(self, item: dict) -> tuple:
        ground_truth = item.get("ground_truth", "")
        question = item.get("question", "")
        
        finetuned_eval = self._evaluate_single(question, ground_truth, item.get("prediction_finetuned", ""))
        initialized_eval = self._evaluate_single(question, ground_truth, item.get("prediction_initialized", ""))
        
        ft_failed = False
        init_failed = False
        
        if self.run_llm_judge:
            ft_judge = finetuned_eval.get("llm_judge", {})
            init_judge = initialized_eval.get("llm_judge", {})
            
            ft_failed = ft_judge.get("score", 0) == 0 and "error" in ft_judge.get("reasoning", "").lower()
            init_failed = init_judge.get("score", 0) == 0 and "error" in init_judge.get("reasoning", "").lower()
        
        item["evaluations"] = {
            "finetuned_model": finetuned_eval,
            "initialized_model": initialized_eval
        }
        
        return item, ft_failed, init_failed

    def _save_state(self, results, failed_samples, out_file, fail_file, agg_file):
        """Saves current state and triggers a strict recalculation of averages."""
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        if self.run_llm_judge and failed_samples:
            with open(fail_file, 'w') as f:
                json.dump(failed_samples, f, indent=4)

        # Centralized aggregation call
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
                item, ft_failed, init_failed = future.result()
                sample_idx = item.get("sample_idx")
                
                if ft_failed or init_failed:
                    tqdm.write(f"     [FAILED AGAIN] idx: {sample_idx}")
                    still_failed.append(item)
                else:
                    tqdm.write(f"     [SUCCESS] idx: {sample_idx}")
                    fixed_count += 1
                    
                    # Ensure the successful item is appended or updated in the main results
                    found_in_main = False
                    for main_item in main_results:
                        if main_item.get("sample_idx") == sample_idx:
                            main_item["evaluations"] = item["evaluations"]
                            found_in_main = True
                            break
                            
                    if not found_in_main:
                        main_results.append(item)

        # Save merged results and the remaining failures
        with open(main_output_filepath, 'w') as f:
            json.dump(main_results, f, indent=4)
        with open(failed_filepath, 'w') as f:
            json.dump(still_failed, f, indent=4)

        # Triggers the exact same aggregation math used in the main run
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
        The single source of truth for all averages. 
        It STRICTLY filters out failed LLM parses from the final counts and totals.
        """
        totals = {"finetuned_model": {}, "initialized_model": {}}
        counts = {"finetuned_model": 0, "initialized_model": 0}
        
        for item in all_results:
            evals = item.get("evaluations", {})
            ft_eval = evals.get("finetuned_model", {})
            init_eval = evals.get("initialized_model", {})
            
            # --- STRICT SUCCESS FILTERING ---
            # 1. Check Finetuned
            ft_valid = "error" not in ft_eval
            if self.run_llm_judge and ft_valid:
                # If the score is 0, it failed the parse and is NOT valid for aggregation
                if ft_eval.get("llm_judge", {}).get("score", 0) == 0:
                    ft_valid = False
                    
            if ft_valid:
                self._accumulate_metrics(totals["finetuned_model"], ft_eval)
                counts["finetuned_model"] += 1

            # 2. Check Initialized
            init_valid = "error" not in init_eval
            if self.run_llm_judge and init_valid:
                if init_eval.get("llm_judge", {}).get("score", 0) == 0:
                    init_valid = False
                    
            if init_valid:
                self._accumulate_metrics(totals["initialized_model"], init_eval)
                counts["initialized_model"] += 1

        aggregated_results = {
            "finetuned_model_averages": self._average_metrics(totals["finetuned_model"], counts["finetuned_model"]),
            "initialized_model_averages": self._average_metrics(totals["initialized_model"], counts["initialized_model"]),
            "total_successful_samples": {
                "finetuned_model": counts["finetuned_model"],
                "initialized_model": counts["initialized_model"]
            }
        }
        
        with open(aggregation_filepath, 'w') as f:
            json.dump(aggregated_results, f, indent=4)
        
        # We output this so you can verify exactly how many successful items were just written to the file
        tqdm.write(f"Aggregations updated. (Valid FT: {counts['finetuned_model']}, Valid Init: {counts['initialized_model']})")