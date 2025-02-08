import json
import requests
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_with_codellama(prompt: str) -> Optional[str]:
    """Generate response using local CodeLlama API"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "codellama", "prompt": prompt}
        )
        
        # Concatenate all response chunks
        full_response = ""
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'response' in json_response:
                    full_response += json_response['response']
        
        return full_response
    except Exception as e:
        print(f"Error generating with CodeLlama: {str(e)}")
        return None
    

def load_and_generate_with_codellama(prompt: str) -> Optional[str]:
    """Load CodeLlama model and tokenizer, and generate response"""
    try:
        model_name = "codellama"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Check if GPU is available and move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error loading and generating with CodeLlama: {str(e)}")
        return None

def extract_code(response: str) -> Optional[str]:
    """Extract code between ``` ``` tags"""
    if not response:
        return None
    
    try:
        start_idx = response.index("```") + 3
        end_idx = response.rindex("```")
        # Skip language identifier if present
        if "\n" in response[start_idx:start_idx+10]:
            start_idx = response[start_idx:].index("\n") + start_idx + 1
        return response[start_idx:end_idx].strip()
    except ValueError:
        return None
