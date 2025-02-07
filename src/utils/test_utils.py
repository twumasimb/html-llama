# src/utils/test_utils.py
from typing import Dict, List
from bs4 import BeautifulSoup

def validate_aria_labels(soup: BeautifulSoup) -> List[Dict]:
    """Validate ARIA labels and roles"""
    violations = []
    
    # Check interactive elements
    for element in soup.find_all(['button', 'a', 'input', 'select']):
        if not (element.get('aria-label') or element.get('aria-labelledby')):
            violations.append({
                'element': element.name,
                'issue': 'missing aria-label or aria-labelledby',
                'wcag': '4.1.2'
            })
    
    # Check landmarks
    for element in soup.find_all(['main', 'nav', 'aside', 'header', 'footer']):
        if not element.get('role'):
            violations.append({
                'element': element.name,
                'issue': 'missing role attribute',
                'wcag': '1.3.1'
            })
    
    return violations

def validate_color_contrast(style: str) -> List[Dict]:
    """Validate color contrast ratios"""
    # In a real implementation, this would use a color contrast calculation library
    violations = []
    return violations

def validate_keyboard_navigation(html: str) -> List[Dict]:
    """Validate keyboard navigation possibilities"""
    violations = []
    soup = BeautifulSoup(html, 'html.parser')
    
    # Check tabindex
    for element in soup.find_all(attrs={'tabindex': True}):
        if int(element['tabindex']) < 0:
            violations.append({
                'element': element.name,
                'issue': 'negative tabindex',
                'wcag': '2.1.1'
            })
    
    return violations