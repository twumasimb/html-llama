# src/generators/responses.py
from typing import Dict
from ..utils.codellama import generate_with_codellama, extract_code, load_and_generate_with_codellama
import gc
import torch
from threading import Timer
import signal
from contextlib import contextmanager
import logging

class TimeoutException(Exception): pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Generation timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def clear_model_cache():
    """Clear CUDA cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def create_html_prompt(prompt: str, features: Dict) -> str:
    """Create a prompt following CodeLlama's format"""
    component_type = features.get("component_type", "")
    required_elements = features.get("required_elements", [])
    
    template = """[INST] Write HTML code for the following component that follows accessibility best practices and WCAG guidelines. The output code needs to implement {component_type} functionality and include these elements: {elements}. Please wrap your code answer using ```:

{prompt}
[/INST]"""

    return template.format(
        component_type=component_type,
        elements=", ".join(required_elements),
        prompt=prompt
    )

def generate_response(prompt: str, features: Dict, use_api: bool = True, timeout_seconds: int = 30) -> str:
    """Generate accessible HTML response using CodeLlama"""
    try:
        with timeout(timeout_seconds):
            # Create formatted prompt
            formatted_prompt = create_html_prompt(prompt, features)
            
            # Generate response using CodeLlama
            if use_api:
                response = generate_with_codellama(formatted_prompt)
            else:
                response = load_and_generate_with_codellama(formatted_prompt)
            if not response:
                return ""
            
            # Extract HTML code from response
            html_code = extract_code(response)
            
            # Clear cache after generation
            clear_model_cache()
            
            return html_code if html_code else ""
            
    except TimeoutException:
        logging.error("Model generation timed out")
        clear_model_cache()
        return ""
    except Exception as e:
        logging.error(f"Error in generation: {str(e)}")
        clear_model_cache()
        return ""

# Remove or comment out COMPONENT_TEMPLATES and other template-based code