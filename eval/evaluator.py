from metrics import BleuMetric, RougeMetric, MeteorMetric, BertScoreMetric, SimCSEMetric, LLMJudgeMetric
import json

class PipelineEvaluator:
    def __init__(self, run_llm_judge=False):
        self.bleu = BleuMetric()
        self.rouge = RougeMetric()
        self.meteor = MeteorMetric()       
        self.bert_score = BertScoreMetric() 
        self.simcse = SimCSEMetric()       
        
        self.run_llm_judge = run_llm_judge
        if self.run_llm_judge:
            self.llm_judge = LLMJudgeMetric()
            
    def evaluate_file(self, 
                      input_filepath: str, 
                      output_filepath: str, 
                      aggregation_filepath: str,
                      failed_filepath: str = "failed_llm_evals.json", # NEW: Path for failed parses
                      test_mode: bool = False,                        # NEW: Test mode flag
                      test_samples: int = 5):                         # NEW: Number of samples to test
        
        with open(input_filepath, 'r') as f:
            data = json.load(f)

        # --- NEW: Test Mode Logic ---
        if test_mode:
            print(f"\n[TEST MODE ENABLED] Only evaluating the first {test_samples} samples.\n")
            data = data[:test_samples]

        results = []
        failed_llm_samples = [] # NEW: List to track failed LLM responses
        
        # Dictionaries to track the sum of metrics and the total valid counts
        totals = {
            "finetuned_model": {},
            "initialized_model": {}
        }
        counts = {
            "finetuned_model": 0,
            "initialized_model": 0
        }

        for item in data:
            print(f"Evaluating sample_idx: {item.get('sample_idx')}...")
            
            ground_truth = item.get("ground_truth", "")
            question = item.get("question", "")
            
            # 1. Evaluate both prediction types
            finetuned_eval = self._evaluate_single(question, ground_truth, item.get("prediction_finetuned", ""))
            initialized_eval = self._evaluate_single(question, ground_truth, item.get("prediction_initialized", ""))
            
            # --- NEW: Failed LLM Parse Detection Logic ---
            if self.run_llm_judge:
                ft_judge = finetuned_eval.get("llm_judge", {})
                init_judge = initialized_eval.get("llm_judge", {})
                
                # Check if the provider returned a 0 score and an error message string
                ft_failed = ft_judge.get("score") == 0 and "error" in ft_judge.get("reasoning", "").lower()
                init_failed = init_judge.get("score") == 0 and "error" in init_judge.get("reasoning", "").lower()
                
                if ft_failed or init_failed:
                    print(f"  -> [WARNING] LLM Judge parse failed. Saving to {failed_filepath}")
                    failed_llm_samples.append(item)
                    # We continue appending it to results, but you can choose to skip it if you prefer
            
            # 2. Accumulate scores for averaging later
            if "error" not in finetuned_eval:
                self._accumulate_metrics(totals["finetuned_model"], finetuned_eval)
                counts["finetuned_model"] += 1
                
            if "error" not in initialized_eval:
                self._accumulate_metrics(totals["initialized_model"], initialized_eval)
                counts["initialized_model"] += 1

            # 3. Attach individual results to the original item
            item["evaluations"] = {
                "finetuned_model": finetuned_eval,
                "initialized_model": initialized_eval
            }
            results.append(item)

        # Save the detailed individual results
        with open(output_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nDetailed evaluation complete. Results saved to {output_filepath}")

        # --- NEW: Save Failed Samples ---
        if self.run_llm_judge and failed_llm_samples:
            with open(failed_filepath, 'w') as f:
                json.dump(failed_llm_samples, f, indent=4)
            print(f"Tracked {len(failed_llm_samples)} failed LLM evaluations and saved them to {failed_filepath}")

        # 4. Calculate averages and save to the aggregation file
        aggregated_results = {
            "finetuned_model_averages": self._average_metrics(totals["finetuned_model"], counts["finetuned_model"]),
            "initialized_model_averages": self._average_metrics(totals["initialized_model"], counts["initialized_model"]),
            "total_samples_evaluated": {
                "finetuned_model": counts["finetuned_model"],
                "initialized_model": counts["initialized_model"]
            }
        }
        
        with open(aggregation_filepath, 'w') as f:
            json.dump(aggregated_results, f, indent=4)
        print(f"Aggregated averages saved to {aggregation_filepath}")

    def _evaluate_single(self, question: str, reference: str, hypothesis: str) -> dict:
        if not hypothesis or not reference:
            return {"error": "Missing prediction or ground truth"}

        metrics_result = {
            "bleu": self.bleu.compute(reference, hypothesis),
            "rouge": self.rouge.compute(reference, hypothesis),
            "meteor": self.meteor.compute(reference, hypothesis),      
            "bert_score": self.bert_score.compute(reference, hypothesis), 
            "simcse_cosine": self.simcse.compute(reference, hypothesis) 
        }

        if self.run_llm_judge:
            metrics_result["llm_judge"] = self.llm_judge.compute(question, reference, hypothesis)
        
        # print(f"Metrics for current prediction: {metrics_result}")  # Debugging line to see metrics for each prediction    

        return metrics_result

    def _accumulate_metrics(self, totals_dict: dict, current_metrics: dict):
        """Recursively adds numeric values from current_metrics into totals_dict."""
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
        """Recursively divides accumulated metrics by the total count."""
        if count == 0:
            return {}
            
        avg_dict = {}
        for key, value in totals_dict.items():
            if isinstance(value, dict):
                avg_dict[key] = self._average_metrics(value, count)
            elif isinstance(value, (int, float)):
                avg_dict[key] = value / count
        return avg_dict
    
    def retry_failed(self, 
                     failed_filepath: str, 
                     main_output_filepath: str, 
                     main_aggregation_filepath: str):
        """
        Retries failed LLM evaluations and merges the successful ones 
        back into the main results and aggregation files.
        """
        import os
        
        if not os.path.exists(failed_filepath) or not os.path.exists(main_output_filepath):
            print("Error: Could not find the failed file or the main output file.")
            return

        # 1. Load the failed samples and the main successful results
        with open(failed_filepath, 'r') as f:
            failed_data = json.load(f)
            
        with open(main_output_filepath, 'r') as f:
            main_results = json.load(f)

        print(f"\n[RETRY MODE] Attempting to fix {len(failed_data)} failed evaluations...")
        
        still_failed = []
        fixed_count = 0

        # 2. Retry the evaluation for each failed item
        for item in failed_data:
            sample_idx = item.get("sample_idx")
            print(f"  -> Retrying sample_idx: {sample_idx}...")
            
            ground_truth = item.get("ground_truth", "")
            question = item.get("question", "")
            
            # Evaluate again
            finetuned_eval = self._evaluate_single(question, ground_truth, item.get("prediction_finetuned", ""))
            initialized_eval = self._evaluate_single(question, ground_truth, item.get("prediction_initialized", ""))
            
            # Check if it failed AGAIN
            ft_failed = finetuned_eval.get("llm_judge", {}).get("score") == 0
            init_failed = initialized_eval.get("llm_judge", {}).get("score") == 0
            
            if ft_failed or init_failed:
                print(f"     [FAILED AGAIN] Could not parse LLM output.")
                still_failed.append(item)
            else:
                print(f"     [SUCCESS] Evaluated correctly.")
                fixed_count += 1
                
                # 3. Find the matching item in the main_results array and update it
                for main_item in main_results:
                    if main_item.get("sample_idx") == sample_idx:
                        main_item["evaluations"] = {
                            "finetuned_model": finetuned_eval,
                            "initialized_model": initialized_eval
                        }
                        break

        # 4. Save the updated main results back to disk
        with open(main_output_filepath, 'w') as f:
            json.dump(main_results, f, indent=4)
            
        # 5. Overwrite the failed file with only the ones that STILL failed
        with open(failed_filepath, 'w') as f:
            json.dump(still_failed, f, indent=4)

        # 6. Recalculate the grand totals from the newly updated main_results
        self._recalculate_aggregations(main_results, main_aggregation_filepath)
        
        print(f"\nRetry complete. Fixed {fixed_count} items. {len(still_failed)} items still failing.")


    def _recalculate_aggregations(self, all_results: list, aggregation_filepath: str):
        """Helper to recalculate total averages after a retry merge."""
        totals = {"finetuned_model": {}, "initialized_model": {}}
        counts = {"finetuned_model": 0, "initialized_model": 0}
        
        for item in all_results:
            evals = item.get("evaluations", {})
            ft_eval = evals.get("finetuned_model", {})
            init_eval = evals.get("initialized_model", {})
            
            # Only count them if they actually have valid (non-error) LLM Judge scores
            if "error" not in ft_eval and ft_eval.get("llm_judge", {}).get("score", 0) > 0:
                self._accumulate_metrics(totals["finetuned_model"], ft_eval)
                counts["finetuned_model"] += 1
                
            if "error" not in init_eval and init_eval.get("llm_judge", {}).get("score", 0) > 0:
                self._accumulate_metrics(totals["initialized_model"], init_eval)
                counts["initialized_model"] += 1

        aggregated_results = {
            "finetuned_model_averages": self._average_metrics(totals["finetuned_model"], counts["finetuned_model"]),
            "initialized_model_averages": self._average_metrics(totals["initialized_model"], counts["initialized_model"]),
            "total_samples_evaluated": {
                "finetuned_model": counts["finetuned_model"],
                "initialized_model": counts["initialized_model"]
            }
        }
        
        with open(aggregation_filepath, 'w') as f:
            json.dump(aggregated_results, f, indent=4)
        print(f"Recalculated grand averages and updated {aggregation_filepath}")