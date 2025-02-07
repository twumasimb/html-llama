from typing import Dict, List, Tuple
import re
from bs4 import BeautifulSoup
from html.parser import HTMLParser

class AccessibilityChecker:
    def __init__(self):
        self.violations = []
        self.score = 100

    def check_html(self, html: str) -> Tuple[List[Dict], int]:
        """Check HTML for accessibility and return (violations, score)"""
        if not html:
            return [], 0
            
        soup = BeautifulSoup(html, 'html.parser')
        self.violations = []
        self.score = 100
        
        self._check_images(soup)
        self._check_forms(soup)
        self._check_landmarks(soup)
        self._check_headings(soup)
        
        return self.violations, max(0, self.score)

    def _check_images(self, soup: BeautifulSoup):
        """Check image accessibility"""
        for img in soup.find_all('img'):
            if not img.get('alt'):
                self.violations.append({
                    'element': 'img',
                    'issue': 'missing alt attribute',
                    'wcag': '1.1.1'
                })

    def _check_forms(self, soup: BeautifulSoup):
        """Check form accessibility"""
        for input_elem in soup.find_all('input'):
            if not input_elem.get('id'):
                self.violations.append({
                    'element': 'input',
                    'issue': 'missing id attribute',
                    'wcag': '1.3.1'
                })
            
            # Check for associated label
            input_id = input_elem.get('id')
            if input_id and not soup.find('label', attrs={'for': input_id}):
                self.violations.append({
                    'element': 'input',
                    'issue': 'missing associated label',
                    'wcag': '1.3.1'
                })

    def _check_landmarks(self, soup: BeautifulSoup):
        """Check landmark accessibility"""
        if not soup.find('main'):
            self.violations.append({
                'element': 'main',
                'issue': 'missing main landmark',
                'wcag': '1.3.1'
            })

    def _check_headings(self, soup: BeautifulSoup):
        """Check heading hierarchy"""
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headings and not soup.find('h1'):
            self.violations.append({
                'element': 'h1',
                'issue': 'missing main heading',
                'wcag': '1.3.1'
            })

def validate_html(html: str) -> Tuple[bool, List[Dict], int]:
    """Validate HTML accessibility and return (passed, violations, score)"""
    checker = AccessibilityChecker()
    violations, score = checker.check_html(html)
    passed = score >= 70
    return passed, violations, score