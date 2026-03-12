from llm_providers import TinyLlamaProvider,GeminiProvider,OpenRouterProvider
from metrics import LLMJudgeMetric
from evaluator import PipelineEvaluator
from utils import load_config, setup_global_logging
import yaml
import os 
        
        
def main():
# 1. Load the configuration
    config = load_config("config/openai_config.yaml")
    base_dir = "eval/Psttransformer/vqa-video/10epoch"
    input_file = f"{base_dir}/HumanML_PSTTransformer_finetuned_gemini_10epoch_evaluation_20260224_015316.json"   # Your input JSON file
    output_file = f"{base_dir}/results.json" # Where the evaluated data will be saved
    aggregation_filepath=f"{base_dir}/aggregated_metrics.json" # Where the average metrics will be saved
    failed_filepath=f"{base_dir}/failed_llm_evals.json" # Where any failed LLM evaluations will be saved for retrying
    
    setup_global_logging(base_dir)  # Initialize logging to file and console
    
    # Safely extract the keys (returns None if the key doesn't exist)
    openrouter_key = config.get("OPENROUTER_API_KEY", {})
    target_model = config.get("TARGET_MODEL", "openai/gpt-4o")
    
    # gemini_provider = GeminiProvider(model="gemini-3-flash-preview")
    model_provider = OpenRouterProvider(api_key=openrouter_key, model=target_model)
    
    # 2. Inject it into the Judge
    evaluator = PipelineEvaluator(run_llm_judge=True, max_workers=15)
    evaluator.llm_judge = LLMJudgeMetric(provider=model_provider)
    
# --- Mode 1: First Time Run ---
    evaluator.evaluate_file(
        input_filepath=input_file, 
        output_filepath=output_file, 
        aggregation_filepath=aggregation_filepath,
        failed_filepath=failed_filepath,
        test_mode=False
    )

    # --- Mode 2: Retry Failed Runs ---
    # Just comment out Mode 1 and uncomment this. 
    # It reads the failed file, evaluates them, merges them into detailed_results.json, 
    # and recalculates aggregated_metrics.json!
    # evaluator.retry_failed(
    #     failed_filepath=failed_filepath,
    #     main_output_filepath=output_file,
    #     main_aggregation_filepath=aggregation_filepath
    # )

if __name__ == "__main__":
    main()
    
    
    
