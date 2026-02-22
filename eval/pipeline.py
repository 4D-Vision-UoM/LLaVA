import os
from evaluator import PipelineEvaluator

def main():
    # Make sure you set your API key if you are using the LLM judge!
    # os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"
    
    input_file = "eval/eval_motion_vqa_comparison_20260220_133507.json"   # Your input JSON file
    output_file = "eval/results.json" # Where the evaluated data will be saved
    aggregation_filepath="eval/aggregated_metrics.json" # Where the average metrics will be saved
    
    # Set run_llm_judge=True when you are ready to spend API credits
    # It is recommended to run with False first to ensure BLEU/ROUGE work.
    evaluator = PipelineEvaluator(run_llm_judge=False) 
    
    evaluator.evaluate_file(input_filepath=input_file, output_filepath=output_file, aggregation_filepath=aggregation_filepath)

if __name__ == "__main__":
    main()