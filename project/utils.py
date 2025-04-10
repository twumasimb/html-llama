# utils.py
import re
import json
import signal
import random
import logging
import requests
from tqdm import tqdm
from openai import OpenAI
from typing import Optional
from transformers import pipeline
from typing import List, Dict, Union
from contextlib import contextmanager
from datasets import Dataset, DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Timeout handling
class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timing out function executions."""
    def timeout_handler(signum, frame):
        raise TimeoutException("Function execution timed out")
    
    # Set the timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the original handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def save_json(data, filepath):
    """Save data as JSON."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Data saved to {filepath}")

def load_json(filepath):
    """Load JSON data."""
    with open(filepath, "r") as f:
        return json.load(f)

def deduplicate_prompts(prompts, batch_size=1000):
    """Remove duplicate prompts using a memory-efficient approach.
    
    Args:
        prompts: List of prompts to deduplicate
        batch_size: Number of prompts to process at once
        
    Returns:
        List of unique prompts
    """
    seen = set()
    unique_prompts = []
    
    # Process in batches to reduce memory usage
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        for prompt in batch:
            # If prompts are dictionaries/complex objects, use a hashable representation
            # For example, if prompt is a dict: prompt_hash = str(sorted(prompt.items()))
            prompt_hash = prompt
            
            if prompt_hash not in seen:
                seen.add(prompt_hash)
                unique_prompts.append(prompt)
                
    return unique_prompts

def accessibility_test(html_code):
    """
    A comprehensive accessibility test that checks for key accessibility attributes.
    Returns a score between 0 and 1 where higher values indicate better accessibility.
    """
    # Track number of checks and passed checks for scoring
    total_checks = 0
    passed_checks = 0
    
    # Check for ARIA attributes
    aria_attrs = re.findall(r'aria-[a-zA-Z]+="[^"]+"', html_code)
    has_aria = len(aria_attrs) > 0
    total_checks += 1
    passed_checks += int(has_aria)
    
    # Check for role attributes
    role_attrs = re.findall(r'role="[^"]+"', html_code)
    has_roles = len(role_attrs) > 0
    total_checks += 1
    passed_checks += int(has_roles)
    
    # Check for image alt text
    img_tags = re.findall(r'<img[^>]*>', html_code)
    if img_tags:
        missing_alt = any('alt=' not in img.lower() for img in img_tags)
        has_alt = not missing_alt
        total_checks += 1
        passed_checks += int(has_alt)
    
    # Check for form labels
    input_tags = re.findall(r'<input[^>]*>', html_code)
    if input_tags:
        labels = re.findall(r'<label[^>]*>', html_code)
        has_labels = len(labels) > 0
        total_checks += 1
        passed_checks += int(has_labels)
    
    # Calculate score
    score = passed_checks / total_checks if total_checks > 0 else 0.0
    
    return score

def llm_generate(prompt: str, model: str, timeout_seconds: int = 600) -> Optional[str]:
    """Generate response using local LLM API via Ollama with timeout"""
    try:
        with timeout(timeout_seconds):
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt}
            )
            
            # Concatenate all response chunks
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
            

            # Use external API
            # full_response = get_completion(prompt)

            # Extract response after </think> tag if present
            if '</think>' in full_response:
                full_response = full_response.split('</think>')[-1].strip()
                # Split by newlines and take everything after the first empty line
                if '\n\n' in full_response:
                    full_response = full_response.split('\n\n', 1)[1].strip()
                
            return full_response
            
    except TimeoutException:
        logging.error(f"LLM generation timed out after {timeout_seconds} seconds")
        return None
    except Exception as e:
        logging.error(f"Error generating with CodeLlama: {str(e)}")
        return None
    

def create_and_split_dataset(
    data: List[Dict[str, str]], 
    train_ratio: float = 0.8, 
    seed: int = 42,
    output_dir: str = 'dataset',
    save_jsonl: bool = True
) -> DatasetDict:
    """
    Create and split a dataset into train and test sets.
    
    Args:
        data: List of dictionaries containing 'prompt' and 'response' keys
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
        output_dir: Directory to save the final dataset (default: 'dataset')
        save_jsonl: Whether to save train/test splits as JSONL files (default: True)
    
    Returns:
        DatasetDict containing train and test datasets
    """
    # Create a copy of the data to avoid modifying the original
    combined_data = [
        {"prompt": item["prompt"], "code": item["code"]} for item in data
    ]
    
    # Shuffle the data
    random.seed(seed)
    random.shuffle(combined_data)
    
    # Split the data
    split_idx = int(len(combined_data) * train_ratio)
    train_data = combined_data[:split_idx]
    test_data = combined_data[split_idx:]
    
    # Save to JSONL files if requested
    if save_jsonl:
        for split_name, split_data in [("train", train_data), ("validation", test_data)]:
            with open(f'{split_name}.jsonl', 'w') as f:
                for item in split_data:
                    json.dump(item, f)
                    f.write('\n')
            print(f"Created {split_name}.jsonl with {len(split_data)} examples")
    
    # Create dataset dictionary
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'test': Dataset.from_list(test_data)
    })
    
    # Print dataset statistics
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['validation'])}")
    
    # Save the dataset
    dataset.save_to_disk(output_dir)
    
    return dataset