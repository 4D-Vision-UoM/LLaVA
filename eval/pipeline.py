from llm_providers import TinyLlamaProvider,GeminiProvider
from metrics import LLMJudgeMetric
from evaluator import PipelineEvaluator

def main():
    # Make sure you set your API key if you are using the LLM judge!
    # os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"
    base_dir = "eval/ours"
    input_file = f"{base_dir}/eval_motion_vqa_comparison_20260220_133507.json"   # Your input JSON file
    output_file = f"{base_dir}/results.json" # Where the evaluated data will be saved
    aggregation_filepath=f"{base_dir}/aggregated_metrics.json" # Where the average metrics will be saved
    
    # gemini_provider = GeminiProvider(model="gemini-3-flash-preview")
    model_provider = TinyLlamaProvider(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # 2. Inject it into the Judge
    evaluator = PipelineEvaluator(run_llm_judge=True)
    evaluator.llm_judge = LLMJudgeMetric(provider=model_provider)
    
# --- Mode 1: First Time Run ---
    evaluator.evaluate_file(
        input_filepath=input_file, 
        output_filepath=output_file, 
        aggregation_filepath=aggregation_filepath,
        failed_filepath=f"{base_dir}/failed_llm_evals.json",
        test_mode=True
    )

    # --- Mode 2: Retry Failed Runs ---
    # Just comment out Mode 1 and uncomment this. 
    # It reads the failed file, evaluates them, merges them into detailed_results.json, 
    # and recalculates aggregated_metrics.json!
    # evaluator.retry_failed(
    #     failed_filepath="failed_llm_evals.json",
    #     main_output_filepath="detailed_results.json",
    #     main_aggregation_filepath="aggregated_metrics.json"
    # )

if __name__ == "__main__":
    main()