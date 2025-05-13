# utils.py
import json
import re
import random
import logging
import signal
import requests
from transformers import pipeline
from tqdm import tqdm
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

def deduplicate_prompts(prompts):
    """Remove duplicate prompts."""
    unique_prompts = list(set(prompts))
    return unique_prompts

def accessibility_test(html_code):
    """
    A simple accessibility test that checks for key attributes.
    In a production scenario, use an accessibility testing tool.
    Returns True if both an aria-* attribute and a role attribute are found.
    """
    aria_check = re.search(r'aria-[a-zA-Z]+=', html_code) is not None
    role_check = re.search(r'role="[^"]+"', html_code) is not None
    return aria_check and role_check


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
            
            # # Extract response after </think> tag if present
            # if '</think>' in full_response:
            #     full_response = full_response.split('</think>')[-1].strip()
            #     # Split by newlines and take everything after the first empty line
            #     if '\n\n' in full_response:
            #         full_response = full_response.split('\n\n', 1)[1].strip()
                
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
    if train_ratio > 0:
        # Split the data
        split_idx = int(len(combined_data) * train_ratio)
        train_data = combined_data[split_idx:]
        test_data = combined_data[:split_idx]
        
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
        print(f"Test dataset size: {len(dataset['test'])}")
    
    else:
        dataset = DatasetDict({
            'train': Dataset.from_list(combined_data)
        })
    
    # Save the dataset
    dataset.save_to_disk(output_dir)
    
    return dataset



import re

def clean_html_string(input_string):
    # Find the doctype tag
    doctype_index = input_string.find('<!DOCTYPE html>')
    
    if doctype_index == -1:
        # If the doctype tag doesn't exist in the string
        cleaned_string = input_string
    else:
        # Return only the part of the string starting from the doctype tag
        cleaned_string = input_string[doctype_index:]
    
    # Remove everything after </html> tag if it exists
    html_end_pattern = r'(</html>)[\s\S]*$'
    cleaned_string = re.sub(html_end_pattern, r'\1', cleaned_string)
    
    return cleaned_string

# Example usage:
# html_string = "<!DOCTYPE html><html><body>Hello world!</body></html>"
# cleaned_string = clean_html_string(html_string)
# print(cleaned_string)
# Output: <!DOCTYPE html><html><body>Hello world!</body></html>

def write_html_to_file(html_content: str, output_path: str) -> None:
    """
    Writes HTML content to a file while preserving formatting.
    
    Args:
        html_content (str): The HTML content to write
        output_path (str): Path where the HTML file should be saved
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully wrote HTML to {output_path}")
    except Exception as e:
        print(f"Error writing HTML file: {str(e)}")