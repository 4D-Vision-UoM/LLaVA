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
            
    def evaluate_file(self, input_filepath: str, output_filepath: str, aggregation_filepath: str):
        with open(input_filepath, 'r') as f:
            data = json.load(f)

        results = []
        
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
        print(f"Detailed evaluation complete. Results saved to {output_filepath}")

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

        return metrics_result

    # --- NEW HELPER METHODS FOR AGGREGATION ---

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