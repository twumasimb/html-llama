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

        prompt_template = """
Generate a unique and concise prompt related to web development. 
The prompt should be a specific task or question that a developer might ask when building a website, specifically frontend development, web design, and frameworks.

Each prompt should be:
- Clear and specific
- Concise (no more than 2-3 sentences)
- Relevant to web development
- End with a question or task to solve

Examples of prompts:
- "Create a website for a bookkeeping business"
- "Write HTML and CSS code for a restaurant website"
- "Generate code for a portfolio webpage using HTML, CSS and JavaScript"
- "Let's create a website for an e-commerce business that shows the list of items available and the prices. The website should be interactive and nicely colored"
- "Generate code for a kids toy website, where kids can share their toys"

Generate the prompt in the same format as the examples above. Be creative to include any possible use case. Focus on single webpage apps
"""

        generated = llm_generate(prompt_template, model="deepseek-r1:14b")
        if generated is None:
            logging.warning(f"Failed to generate prompt {i+1}, skipping...")
            continue
        
        # Use the first line of the generated text as the prompt.
        # prompt_line = generated.strip().split('\n')[0]
        prompts.append(generated)
        
        # Append the new prompt to the existing JSON file
        with open(output_file, 'r') as f:
            current_prompts = json.load(f)
        
        current_prompts.append(generated)
        
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
