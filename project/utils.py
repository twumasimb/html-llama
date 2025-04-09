# utils.py
import json
import re
import logging
from transformers import pipeline
from tqdm import tqdm
import requests
import signal
from typing import Optional
from contextlib import contextmanager

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

# def llm_generate(prompt: str, model: str, timeout_seconds: int = 60) -> Optional[str]:
#     """Generate response using Hyperbolic API with timeout"""
#     url = "https://api.hyperbolic.xyz/v1/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJyZXNlYXJjaHNlcnZpY2VzMzAxQGdtYWlsLmNvbSIsImlhdCI6MTczODg1ODc5MX0.Delzv2NEwLoC4rVtbymBhWQNwahQ0yQPgjWyZ0J0rbE"
#     }
#     data = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         "model": model or "deepseek-ai/DeepSeek-R1",  # Use provided model or default to DeepSeek-R1
#         "max_tokens": 10000,
#         "temperature": 0.1,
#         "top_p": 0.9
#     }
    
#     try:
#         with timeout(timeout_seconds):
#             logging.info(f"Sending request to Hyperbolic API with model: {data['model']}")
#             response = requests.post(url, headers=headers, json=data)
            
#             # Log the raw response for debugging
#             logging.debug(f"Raw API response: {response.text}")
            
#             # Check if response is valid JSON
#             try:
#                 response_json = response.json()
#             except json.JSONDecodeError as e:
#                 logging.error(f"Failed to parse API response as JSON: {str(e)}")
#                 logging.error(f"Response status code: {response.status_code}")
#                 logging.error(f"Response content: {response.text}")
#                 return None
            
#             # Check if the response contains the expected data
#             if response.status_code == 200 and 'choices' in response_json and response_json['choices']:
#                 if 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
#                     return response_json['choices'][0]['message']['content']
#                 else:
#                     logging.error(f"Unexpected API response structure: {response_json}")
#                     return None
#             else:
#                 error_message = "Unknown error"
#                 if 'error' in response_json:
#                     if isinstance(response_json['error'], dict) and 'message' in response_json['error']:
#                         error_message = response_json['error']['message']
#                     elif isinstance(response_json['error'], str):
#                         error_message = response_json['error']
                
#                 logging.error(f"API error (status {response.status_code}): {error_message}")
#                 logging.error(f"Full response: {response_json}")
#                 return None
                
#     except TimeoutException:
#         logging.error(f"LLM generation timed out after {timeout_seconds} seconds")
#         return None
#     except Exception as e:
#         logging.error(f"Error generating with Hyperbolic API: {str(e)}")
#         logging.exception("Stack trace:")  # This will log the full stack trace
#         return None