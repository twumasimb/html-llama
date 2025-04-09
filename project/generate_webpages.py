# generate_webpages.py
import argparse
import logging
import json
import os
import re
from tqdm import tqdm
from resources import ACCESSIBILITY_PROMPT_TEMPLATE
from utils import load_json, llm_generate, save_json

def generate_webpages(prompts, output_file):
    # Initialize the output file with an empty array if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    results = []
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating webpages")):
        solution_prompt = f"Generate accessible webpage HTML code for the following prompt:\n{prompt} and the following accessibility prompt:\n{ACCESSIBILITY_PROMPT_TEMPLATE}"
        candidate = llm_generate(solution_prompt, model="deepseek-r1:32b")
        print(candidate) #For debugging
        if candidate is not None:
            # Find the code created
            code_pattern = r"```(?:html|)\n?(.*?)```"
            code_match = re.search(code_pattern, candidate, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                # If no backticks found, just strip any backticks that might be at start/end
                code = re.sub(r'^```|```$', '', candidate).strip()
            
            # Find reasoning    
            reasoning_pattern = r"<think>(.*?)</think>"
            reasoning_match = re.search(reasoning_pattern, candidate, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()

        # Create entry for current prompt
        current_entry = {
            "prompt": prompt,
            "reasoning": reasoning,
            "code": code
        }
        
        # Append to in-memory results
        results.append(current_entry)
        
        # Update the JSON file with the new entry
        with open(output_file, 'r') as f:
            current_results = json.load(f)
        
        current_results.append(current_entry)
        
        with open(output_file, 'w') as f:
            json.dump(current_results, f, indent=4)
        
        if (idx+1) % 100 == 0:
            logging.info(f"Generated candidates for {idx+1}/{len(prompts)} prompts")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate webpages using CodeLLama-2-7B based on prompts")
    parser.add_argument("--prompts", type=str, default="prompts_dedup.json", help="JSON file with deduplicated prompts")
    parser.add_argument("--output", type=str, default="generated_dataset/webpages.json", help="Output JSON file for generated webpages")
    parser.add_argument("--device", type=int, default=0, help="Device id: -1 for CPU, 0 for GPU")
    args = parser.parse_args()
    
    prompts = load_json(args.prompts)
    # We don't need to load the model as llm_generate connects to Ollama API directly
    generate_webpages(prompts, args.output)
    
if __name__ == "__main__":
    main()
