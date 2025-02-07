import json
import requests
from typing import Optional

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
