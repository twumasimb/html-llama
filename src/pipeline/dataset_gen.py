# src/pipeline/dataset_gen.py
import json
import os
from typing import List, Dict
from tqdm import tqdm
import gc
import time
import psutil
from ..generators.prompts import generate_prompt
from ..generators.responses import generate_response, clear_model_cache
from ..tools.a11y_checker import validate_html
from ..tools.html_validator import validate_html_structure
import logging

# Configure logging
logging.basicConfig(
    filename='./src/logs/dataset_gen.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DatasetGenerator:
    def __init__(self, output_file: str = "dataset.json"):
        self.output_file = output_file
        self.dataset = []
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.generation_timeout = 30  # seconds
        # Initialize empty dataset file if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                json.dump([], f)

    def _check_resource_usage(self):
        """Monitor system resources"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f}MB")
        return mem_info.rss / 1024 / 1024 > 1000  # Return True if memory usage > 1GB

    def save_item(self, item: Dict) -> None:
        """Save a single item to the dataset file"""
        try:
            with open(self.output_file, 'r+') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                
                data.append(item)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
                
            logging.info(f"Successfully saved item to {self.output_file}")
        except Exception as e:
            logging.error(f"Error saving item: {str(e)}")

    def generate(self, size: int) -> None:
        successful_pairs = 0
        attempts = 0
        
        pbar = tqdm(total=size, desc=f"Generated pairs: {successful_pairs}")
        while successful_pairs < size and attempts < size * 2:
            try:
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logging.warning("Too many consecutive failures, resetting model state...")
                    clear_model_cache()
                    time.sleep(2)
                    self.consecutive_failures = 0

                if attempts % 50 == 0:  # Check resources periodically
                    if self._check_resource_usage():
                        logging.warning("High memory usage detected, forcing cleanup...")
                        gc.collect()
                        time.sleep(5)

                logging.info(f"Attempt {attempts+1}, Successful pairs: {successful_pairs}")
                
                prompt, features = generate_prompt()
                html_response = generate_response(
                    prompt, 
                    features, 
                    use_api=True,
                    timeout_seconds=self.generation_timeout
                )
                
                if not html_response or len(html_response.strip()) == 0:
                    self.consecutive_failures += 1
                    logging.warning(f"No response generated. Consecutive failures: {self.consecutive_failures}")
                    attempts += 1
                    continue
                
                try:
                    # Validate HTML structure
                    passed_structure, structure_score = validate_html_structure(html_response)
                    if not passed_structure:
                        logging.info(f"Failed HTML structure validation (Score: {structure_score})")
                        attempts += 1
                        continue

                    # Validate accessibility
                    passed_a11y, violations, a11y_score = validate_html(html_response)
                    if not passed_a11y:
                        logging.info(f"Failed accessibility validation (Score: {a11y_score})")
                        logging.info(f"Violations: {violations}")
                        attempts += 1
                        continue
                    
                    # Calculate combined score
                    total_score = (structure_score + a11y_score) / 2
                    
                    item = {
                        "prompt": prompt,
                        "response": html_response,
                        "features": features,
                        "structure_score": structure_score,
                        "accessibility_score": a11y_score,
                        "total_score": total_score,
                        "violations": violations
                    }
                    
                    self.dataset.append(item)
                    self.save_item(item)  # Save immediately after successful validation
                    
                    self.consecutive_failures = 0  # Reset on success
                    successful_pairs += 1
                    pbar.update(1)
                    pbar.set_description(f"Generated pairs: {successful_pairs}")
                    logging.info(f"Successfully added and saved pair {successful_pairs} (Score: {total_score})")
                    
                except Exception as e:
                    logging.info(f"Validation error: {str(e)}")
                    attempts += 1
                    continue
                    
                attempts += 1
                
            except Exception as e:
                self.consecutive_failures += 1
                logging.error(f"Error during generation (failures: {self.consecutive_failures}): {str(e)}")
                clear_model_cache()
                attempts += 1
                time.sleep(1)  # Brief pause before retry

        pbar.close()
    
    def save(self) -> None:
        """Final save - ensures all data is properly saved"""
        print(f"Performing final save of {len(self.dataset)} pairs to {self.output_file}")
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.dataset, f, indent=2)
            logging.info(f"Final save completed successfully")
        except Exception as e:
            logging.error(f"Error during final save: {str(e)}")

# def main(size: int = 5):
#     generator = DatasetGenerator()
#     generator.generate(size)
#     generator.save()

# if __name__ == "__main__":
#     main()