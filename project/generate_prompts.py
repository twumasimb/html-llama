# generate_prompts.py
import argparse
import random
import logging
import json
import os
from tqdm import tqdm
from resources import webpage_purposes
from utils import save_json, llm_generate

def generate_prompts(num_prompts, output_file):
    """
    Generate a specified number of prompts using LLaMa 2-7B.
    Prompts focus on accessible webpage ideas.
    """
    # Initialize the output file with an empty array if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    prompts = []
    
    for i in tqdm(range(num_prompts), desc="Generating prompts"):
        random_purpose = random.choice(webpage_purposes)
        
        base_prompt = (
        f"""
            I want to create webpage idea prompts. Here's how:

1. I'll give you a purpose from my list (like "3D modeling gallery")
2. You'll create a short prompt asking for HTML code for that purpose
3. Keep prompts under 100 words

Example:
Purpose: "3D modeling gallery"
Output prompt: "Write HTML code for a webpage that displays a 3D modeling gallery."

Your job is to take any purpose I give you and turn it into a clear prompt asking for HTML code for that specific type of webpage.

Does this make sense? If yes, I'll start giving you purposes from my list.

Purpose: "{random_purpose}"
        """
    )

        generated = llm_generate(base_prompt, model="deepseek-r1")
        if generated is None:
            logging.warning(f"Failed to generate prompt {i+1}, skipping...")
            continue
        
        # Use the first line of the generated text as the prompt.
        prompt_line = generated.strip().split('\n')[0]
        prompts.append(prompt_line)
        
        # Append the new prompt to the existing JSON file
        with open(output_file, 'r') as f:
            current_prompts = json.load(f)
        
        current_prompts.append(prompt_line)
        
        with open(output_file, 'w') as f:
            json.dump(current_prompts, f, indent=4)
        
        if (i+1) % 1000 == 0:
            logging.info(f"Generated {i+1}/{num_prompts} prompts")
    
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Generate webpage prompts using LLaMa 2-70B")
    parser.add_argument("--num_prompts", type=int, default=100000, help="Number of prompts to generate")
    parser.add_argument("--output", type=str, default="prompts.json", help="Output JSON file for prompts")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b", help="LLM model name")
    parser.add_argument("--device", type=int, default=0, help="Device id: -1 for CPU, 0 for GPU")
    args = parser.parse_args()
    
    generate_prompts(args.num_prompts, args.output)

if __name__ == "__main__":
    main()
