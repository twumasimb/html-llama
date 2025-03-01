# generate_webpages.py
import argparse
import logging
from tqdm import tqdm
from resources import ACCESSIBILITY_PROMPT_TEMPLATE
from utils import load_json, llm_generate, save_json

def generate_webpages(prompts, num_candidates, output_file):
    import re
    results = []
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating webpages")):
        candidate_list = []
        for i in range(num_candidates):
            solution_prompt = f"Generate accessible webpage HTML code for the following prompt:\n{prompt} and the following accessibility prompt:\n{ACCESSIBILITY_PROMPT_TEMPLATE}"
            candidate = llm_generate(solution_prompt, model="deepseek-r1")
            
            # Improved regex to clean the code - removes backticks and language identifier
            pattern = r"```(?:html|)\n?(.*?)```"
            match = re.search(pattern, candidate, re.DOTALL)
            if match:
                cleaned_candidate = match.group(1).strip()
            else:
                # If no backticks found, just strip any backticks that might be at start/end
                cleaned_candidate = re.sub(r'^```|```$', '', candidate).strip()
                
            candidate_list.append(cleaned_candidate)
        results.append({
            "prompt": prompt,
            "candidates": candidate_list
        })
        if (idx+1) % 100 == 0:
            save_json(results, output_file)
    save_json(results, output_file)
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate webpages using CodeLLama-2-7B based on prompts")
    parser.add_argument("--prompts", type=str, default="prompts_dedup.json", help="JSON file with deduplicated prompts")
    parser.add_argument("--output", type=str, default="webpages.json", help="Output JSON file for generated webpages")
    parser.add_argument("--num_candidates", type=int, default=10, help="Number of candidate solutions per prompt")
    parser.add_argument("--model", type=str, default="meta-llama/CodeLlama-2-7b-hf", help="LLM model name for webpage generation")
    parser.add_argument("--device", type=int, default=0, help="Device id: -1 for CPU, 0 for GPU")
    args = parser.parse_args()
    
    prompts = load_json(args.prompts)
    # We don't need to load the model as llm_generate connects to Ollama API directly
    generate_webpages(prompts, args.num_candidates, args.output)
    
if __name__ == "__main__":
    main()
