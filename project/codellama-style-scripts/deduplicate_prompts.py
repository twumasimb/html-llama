# deduplicate_prompts.py
import argparse
from utils import load_json, save_json, deduplicate_prompts

def main():
    parser = argparse.ArgumentParser(description="Deduplicate prompts in a JSON file")
    parser.add_argument("--input", type=str, default="prompts.json", help="Input JSON file with prompts")
    parser.add_argument("--output", type=str, default="prompts_dedup.json", help="Output JSON file for deduplicated prompts")
    args = parser.parse_args()
    
    prompts = load_json(args.input)
    unique_prompts = deduplicate_prompts(prompts)
    save_json(unique_prompts, args.output)
    
if __name__ == "__main__":
    main()
