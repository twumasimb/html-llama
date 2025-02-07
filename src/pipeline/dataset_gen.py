# src/pipeline/dataset_gen.py
import json
from typing import List, Dict
from ..generators.prompts import generate_prompt
from ..generators.responses import generate_response
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

    def generate(self, size: int) -> None:
        successful_pairs = 0
        attempts = 0
        
        while successful_pairs < size and attempts < size * 2:
            try:
                print(f"Attempt {attempts+1}, Successful pairs: {successful_pairs}")
                
                prompt, features = generate_prompt()
                html_response = generate_response(prompt, features)
                
                if not html_response or len(html_response.strip()) == 0:
                    print("No valid HTML response generated")
                    attempts += 1
                    continue
                
                try:
                    # Validate HTML structure
                    passed_structure, structure_score = validate_html_structure(html_response)
                    if not passed_structure:
                        print(f"Failed HTML structure validation (Score: {structure_score})")
                        attempts += 1
                        continue

                    # Validate accessibility
                    passed_a11y, violations, a11y_score = validate_html(html_response)
                    if not passed_a11y:
                        print(f"Failed accessibility validation (Score: {a11y_score})")
                        print(f"Violations: {violations}")
                        attempts += 1
                        continue
                    
                    # Calculate combined score
                    total_score = (structure_score + a11y_score) / 2
                    
                    self.dataset.append({
                        "prompt": prompt,
                        "response": html_response,
                        "features": features,
                        "structure_score": structure_score,
                        "accessibility_score": a11y_score,
                        "total_score": total_score,
                        "violations": violations
                    })
                    
                    successful_pairs += 1
                    print(f"Successfully added pair {successful_pairs} (Score: {total_score})")
                    
                except Exception as e:
                    print(f"Validation error: {str(e)}")
                    attempts += 1
                    continue
                    
                attempts += 1
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                attempts += 1
    
    def save(self) -> None:
        print(f"Saving {len(self.dataset)} pairs to {self.output_file}")
        with open(self.output_file, 'w') as f:
            json.dump(self.dataset, f, indent=2)

# def main(size: int = 5):
#     generator = DatasetGenerator()
#     generator.generate(size)
#     generator.save()

# if __name__ == "__main__":
#     main()