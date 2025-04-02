# main.py
import argparse
import logging
import os
import subprocess
from tqdm import tqdm
from utils import load_json, save_json, accessibility_test

def run_generate_prompts(num_prompts, output, model, device):
    subprocess.run([
        "python", "generate_prompts.py",
        "--num_prompts", str(num_prompts),
        "--output", output,
        "--model", model,
        "--device", str(device)
    ], check=True)

def run_deduplicate_prompts(input_file, output_file):
    subprocess.run([
        "python", "deduplicate_prompts.py",
        "--input", input_file,
        "--output", output_file
    ], check=True)

def run_generate_webpages(prompts_file, output, num_candidates, device):
    subprocess.run([
        "python", "generate_webpages.py",
        "--prompts", prompts_file,
        "--output", output, 
        "--num_candidates", str(num_candidates),
        "--device", str(device)
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline for accessible webpage generation")
    parser.add_argument("--num_prompts", type=int, default=100000, help="Number of prompts to generate")
    parser.add_argument("--prompts_file", type=str, default="prompts.json", help="File to store generated prompts")
    parser.add_argument("--dedup_file", type=str, default="prompts_dedup.json", help="File to store deduplicated prompts")
    parser.add_argument("--webpages_file", type=str, default="webpages.json", help="File to store generated webpages")
    parser.add_argument("--final_dataset", type=str, default="final_dataset.json", help="File to store final dataset")
    parser.add_argument("--num_candidates", type=int, default=10, help="Number of candidate solutions per prompt")
    parser.add_argument("--llama_model", type=str, default="meta-llama/Llama-2-70b", help="Model for prompt generation")
    parser.add_argument("--code_llama_model", type=str, default="meta-llama/CodeLlama-7b-hf", help="Model for webpage generation")
    parser.add_argument("--device", type=int, default=-1, help="Device id: -1 for CPU, 0 for GPU")
    parser.add_argument("--accessibility_threshold", type=float, default=0.7, help="Threshold for accessibility score (0-1)")
    args = parser.parse_args()
    
    logging.info("Starting the end-to-end pipeline...")
    
    # Step 1: Generate prompts if the prompts file does not exist.
    if not os.path.exists(args.prompts_file):
        logging.info("Generating prompts...")
        run_generate_prompts(args.num_prompts, args.prompts_file, args.llama_model, args.device)
    
    # Step 2: Deduplicate prompts.
    logging.info("Deduplicating prompts...")
    run_deduplicate_prompts(args.prompts_file, args.dedup_file)
    
    # Step 3: Generate webpages for each deduplicated prompt.
    logging.info("Generating webpages...")
    run_generate_webpages(args.dedup_file, args.webpages_file, args.num_candidates, args.device)
    
    # Step 4: Select the best candidate with progress bar
    logging.info("Selecting best candidate that passes accessibility tests...")
    webpages_data = load_json(args.webpages_file)
    final_dataset = []
    for entry in tqdm(webpages_data, desc="Processing candidates"):
        prompt = entry["prompt"]
        candidates = entry["candidates"]
        best_candidate = None
        best_score = 0
        
        for candidate in candidates:
            score = accessibility_test(candidate)
            if score >= args.accessibility_threshold and score > best_score:
                best_candidate = candidate
                best_score = score
        
        if best_candidate:
            final_dataset.append({
                "prompt": prompt,
                "solution": best_candidate,
                "accessibility_score": best_score
            })
    save_json(final_dataset, args.final_dataset)
    logging.info(f"Final dataset saved to {args.final_dataset} with {len(final_dataset)} entries")

if __name__ == "__main__":
    main()
