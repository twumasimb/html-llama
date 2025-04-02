# utils.py
import re
import re
import json
import signal
import logging
import requests
from tqdm import tqdm
from openai import OpenAI
from typing import Optional
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple
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

# def accessibility_test(html_code):
#     """
#     A comprehensive accessibility test that checks for key accessibility attributes.
#     Returns a score between 0 and 1 where higher values indicate better accessibility.
#     """
#     # Track number of checks and passed checks for scoring
#     total_checks = 0
#     passed_checks = 0
    
#     # Check for ARIA attributes
#     aria_attrs = re.findall(r'aria-[a-zA-Z]+="[^"]+"', html_code)
#     has_aria = len(aria_attrs) > 0
#     total_checks += 1
#     passed_checks += int(has_aria)
    
#     # Check for role attributes
#     role_attrs = re.findall(r'role="[^"]+"', html_code)
#     has_roles = len(role_attrs) > 0
#     total_checks += 1
#     passed_checks += int(has_roles)
    
#     # Check for image alt text
#     img_tags = re.findall(r'<img[^>]*>', html_code)
#     if img_tags:
#         missing_alt = any('alt=' not in img.lower() for img in img_tags)
#         has_alt = not missing_alt
#         total_checks += 1
#         passed_checks += int(has_alt)
    
#     # Check for form labels
#     input_tags = re.findall(r'<input[^>]*>', html_code)
#     if input_tags:
#         labels = re.findall(r'<label[^>]*>', html_code)
#         has_labels = len(labels) > 0
#         total_checks += 1
#         passed_checks += int(has_labels)
    
#     # Calculate score
#     score = passed_checks / total_checks if total_checks > 0 else 0.0
    
#     return score

def accessibility_test(html_code: str) -> Tuple[float, Dict[str, List[str]]]:
    """
    Comprehensive accessibility test following WCAG principles.
    Returns a tuple of (score, issues dictionary) where score is between 0 and 1.
    """
    soup = BeautifulSoup(html_code, 'html.parser')
    issues = {
        "perceivable": [],
        "operable": [],
        "understandable": [],
        "robust": []
    }
    total_checks = 0
    passed_checks = 0

    # Perceivable Checks
    def check_perceivable():
        nonlocal total_checks, passed_checks
        
        # Check images for alt text
        images = soup.find_all('img')
        total_checks += 1
        if all('alt' in img.attrs for img in images):
            passed_checks += 1
        else:
            issues["perceivable"].append("Some images missing alt text")

        # Check iframes for title/alt
        iframes = soup.find_all('iframe')
        total_checks += 1
        if all('title' in frame.attrs or 'alt' in frame.attrs for frame in iframes):
            passed_checks += 1
        else:
            issues["perceivable"].append("Some iframes missing title/alt attributes")

        # Check form labels
        inputs = soup.find_all(['input', 'select', 'textarea'])
        if inputs:
            total_checks += 1
            labeled_inputs = 0
            for input_elem in inputs:
                id_attr = input_elem.get('id')
                if id_attr and soup.find('label', attrs={'for': id_attr}):
                    labeled_inputs += 1
                elif input_elem.parent.name == 'label':
                    labeled_inputs += 1
            if labeled_inputs == len(inputs):
                passed_checks += 1
            else:
                issues["perceivable"].append("Some form elements missing proper labels")

        # Check for deprecated formatting
        deprecated_tags = soup.find_all(['b', 'i', 'font'])
        total_checks += 1
        if not deprecated_tags:
            passed_checks += 1
        else:
            issues["perceivable"].append("Using deprecated formatting tags")

    # Operable Checks
    def check_operable():
        nonlocal total_checks, passed_checks
        
        # Check for autoplay media
        media = soup.find_all(['audio', 'video'])
        total_checks += 1
        if not any('autoplay' in elem.attrs for elem in media):
            passed_checks += 1
        else:
            issues["operable"].append("Media elements with autoplay detected")

        # Check mouse events have keyboard equivalents
        elements_with_mouse = soup.find_all(
            lambda tag: any(attr for attr in tag.attrs if attr.startswith('onmouse'))
        )
        total_checks += 1
        if all(any(attr.startswith('onkey') for attr in elem.attrs) 
               for elem in elements_with_mouse):
            passed_checks += 1
        else:
            issues["operable"].append("Mouse events without keyboard equivalents")

        # Check meta refresh
        meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
        total_checks += 1
        if not meta_refresh:
            passed_checks += 1
        else:
            issues["operable"].append("Automatic page refresh detected")

    # Understandable Checks
    def check_understandable():
        nonlocal total_checks, passed_checks
        
        # Check document language
        html_tag = soup.find('html')
        total_checks += 1
        if html_tag and 'lang' in html_tag.attrs:
            passed_checks += 1
        else:
            issues["understandable"].append("Missing document language")

        # Check form input purpose
        inputs = soup.find_all('input')
        total_checks += 1
        if all('type' in input_elem.attrs for input_elem in inputs):
            passed_checks += 1
        else:
            issues["understandable"].append("Form inputs missing type attribute")

    # Robust Checks
    def check_robust():
        nonlocal total_checks, passed_checks
        
        # Check for unique IDs
        ids = [tag.get('id') for tag in soup.find_all(attrs={'id': True})]
        total_checks += 1
        if len(ids) == len(set(ids)):
            passed_checks += 1
        else:
            issues["robust"].append("Duplicate ID attributes found")

        # Check for ARIA roles
        elements_with_roles = soup.find_all(attrs={'role': True})
        total_checks += 1
        valid_roles = {'button', 'link', 'menuitem', 'tab', 'search', 'navigation'}
        if all(elem.get('role') in valid_roles for elem in elements_with_roles):
            passed_checks += 1
        else:
            issues["robust"].append("Invalid ARIA roles detected")

    # Run all checks
    check_perceivable()
    check_operable()
    check_understandable()
    check_robust()

    # Calculate final score
    score = passed_checks / total_checks if total_checks > 0 else 0.0
    
    return score, issues

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
                
            return full_response
            
    except TimeoutException:
        logging.error(f"LLM generation timed out after {timeout_seconds} seconds")
        return None
    except Exception as e:
        logging.error(f"Error generating with CodeLlama: {str(e)}")
        return None
    

def get_completion(prompt_template: str) -> str:
    
    client = OpenAI(
      base_url="https://integrate.api.nvidia.com/v1",
      api_key="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"
    )

    completion = client.chat.completions.create(
      model="deepseek-ai/deepseek-r1-distill-qwen-32b",
      messages=[{"role": "user", "content": f"{prompt_template}"}],
      temperature=0.6,
      top_p=0.7,
      max_tokens=4096,
      stream=True
    )

    result = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
    return result




######## New code, check it out
from bs4 import BeautifulSoup
import re

class AccessibilityInspector:
    def __init__(self, html_code):
        self.html_code = html_code
        self.score = 100
        self.max_score = 100

    # Method to check for perceivable content (WCAG 2.2 P)
    def check_perceivable_content(self):
        soup = BeautifulSoup(self.html_code, 'html.parser')
        images = soup.find_all('img')
        for image in images:
            if not image.get('alt'):
                self.score -= 5
                print(f"Image without alt text: {image.get('src')}")

        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            if not heading.text.strip():
                self.score -= 5
                print(f"Empty heading: {heading.name}")

    # Method to check for operable content (WCAG 2.2 O)
    def check_operable_content(self):
        soup = BeautifulSoup(self.html_code, 'html.parser')
        buttons = soup.find_all('button')
        for button in buttons:
            if not button.get('aria-label') and not button.text.strip():
                self.score -= 5
                print(f"Button without label or text: {button}")

        links = soup.find_all('a')
        for link in links:
            if not link.get('href'):
                self.score -= 5
                print(f"Link without href: {link}")

    # Method to check for ARIA roles and states
    def check_aria(self):
        soup = BeautifulSoup(self.html_code, 'html.parser')
        elements_with_roles = soup.find_all(lambda tag: tag.has_attr('role'))
        for element in elements_with_roles:
            role = element.get('role')
            if role not in ['button', 'link', 'menu', 'menuitem', 'progressbar']:
                self.score -= 5
                print(f"Unknown ARIA role: {role} on {element.name}")

        elements_with_states = soup.find_all(lambda tag: tag.has_attr('aria-expanded') or tag.has_attr('aria-checked'))
        for element in elements_with_states:
            state = element.get('aria-expanded') or element.get('aria-checked')
            if state not in ['true', 'false']:
                self.score -= 5
                print(f"Invalid ARIA state: {state} on {element.name}")

    # Method to calculate the final score
    def calculate_score(self):
        self.score = max(self.score, 0)
        self.score = min(self.score, self.max_score)
        return self.score

    # Main method to run the inspector
    def run_inspector(self):
        self.check_perceivable_content()
        self.check_operable_content()
        self.check_aria()
        return self.calculate_score()

# # Example usage
# html_code = """
# <html>
#   <body>
#     <h1>Welcome to our website</h1>
#     <img src="image.jpg" alt="An image on our website">
#     <button aria-label="Click me">Click me</button>
#     <a href="#">Link to somewhere</a>
#     <div role="menu">
#       <div role="menuitem">Menu item 1</div>
#       <div role="menuitem">Menu item 2</div>
#     </div>
#   </body>
# </html>
# """

# inspector = AccessibilityInspector(html_code)
# score = inspector.run_inspector()
# print(f"Accessibility score: {score}")
