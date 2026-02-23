from llm_providers import TinyLlamaProvider,GeminiProvider,OpenRouterProvider
from metrics import LLMJudgeMetric
from evaluator import PipelineEvaluator
import yaml
import os

 
def load_config(filepath="config.yaml"):
    """Loads the YAML configuration file securely."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Falling back to environment variables.")
        return {}
        
    with open(filepath, 'r') as file:
        try:
            return yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {}   
        
        
def main():
# 1. Load the configuration
    config = load_config("config/openai_config.yaml")
    base_dir = "eval/ours/vqa-video"
    input_file = f"{base_dir}/HumanML_MoPa_finetuned_gemini_10epoch_evaluation.json"   # Your input JSON file
    output_file = f"{base_dir}/results.json" # Where the evaluated data will be saved
    aggregation_filepath=f"{base_dir}/aggregated_metrics.json" # Where the average metrics will be saved
    failed_filepath=f"{base_dir}/failed_llm_evals.json" # Where any failed LLM evaluations will be saved for retrying
    
    # Safely extract the keys (returns None if the key doesn't exist)
    openrouter_key = config.get("OPENROUTER_API_KEY", {})
    target_model = config.get("TARGET_MODEL", "openai/gpt-4o")
    
    # gemini_provider = GeminiProvider(model="gemini-3-flash-preview")
    model_provider = OpenRouterProvider(api_key=openrouter_key, model=target_model)
    
    # 2. Inject it into the Judge
    evaluator = PipelineEvaluator(run_llm_judge=True, max_workers=10)
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
    
    
    
