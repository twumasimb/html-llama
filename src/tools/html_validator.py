# src/tools/html_validator.py
from html.parser import HTMLParser
from typing import List, Dict, Tuple
import re

class HTMLValidationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.errors = []
        self.score = 100  # Add score tracking
        
    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        
        # Check required attributes
        if tag == 'img' and not any(attr[0] == 'alt' for attr in attrs):
            self.errors.append({
                'tag': tag,
                'error': 'Missing alt attribute',
                'line': self.getpos()[0]
            })
            self.score -= 10  # Reduce score for each error
        
        if tag == 'a' and not any(attr[0] == 'href' for attr in attrs):
            self.errors.append({
                'tag': tag,
                'error': 'Missing href attribute',
                'line': self.getpos()[0]
            })
            self.score -= 5  # Reduce score for each error

    def handle_endtag(self, tag):
        if self.tags and self.tags[-1] == tag:
            self.tags.pop()
        else:
            self.errors.append({
                'tag': tag,
                'error': 'Mismatched closing tag',
                'line': self.getpos()[0]
            })

def validate_html_structure(html: str) -> Tuple[bool, int]:
    """Validate HTML structure and return (passed, score)"""
    if not html:
        return False, 0
        
    parser = HTMLValidationParser()
    try:
        parser.feed(html)
        passed = len(parser.errors) == 0 and len(parser.tags) == 0
        # Ensure score is between 0 and 100
        score = max(0, min(100, parser.score))
        return passed, score
    except Exception as e:
        print(f"HTML validation error: {str(e)}")
        return False, 0

def get_validation_errors(html: str) -> List[Dict]:
    """Get list of HTML validation errors"""
    parser = HTMLValidationParser()
    try:
        parser.feed(html)
        return parser.errors
    except Exception as e:
        return [{'error': str(e), 'line': 0}]