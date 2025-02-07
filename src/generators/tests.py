# src/generators/tests.py
from typing import Dict, List
from bs4 import BeautifulSoup

class TestGenerator:
    def generate_tests(self, html: str, features: Dict) -> List[str]:
        """Generate accessibility tests for HTML component"""
        tests = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Generate form tests
        if soup.find('form'):
            tests.extend(self._generate_form_tests(soup))
            
        # Generate navigation tests    
        if soup.find('nav'):
            tests.extend(self._generate_nav_tests(soup))
            
        # Generate heading tests
        if soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tests.extend(self._generate_heading_tests(soup))
            
        return tests

    def _generate_form_tests(self, soup: BeautifulSoup) -> List[str]:
        tests = []
        
        # Test form labels
        tests.append("""
def test_form_labels():
    form = soup.find('form')
    inputs = form.find_all(['input', 'select', 'textarea'])
    for input in inputs:
        input_id = input.get('id')
        assert input_id, "Input element missing id"
        label = soup.find('label', attrs={'for': input_id})
        assert label, f"No label found for input {input_id}"
        assert label.text.strip(), f"Empty label for input {input_id}"
""")

        # Test required attributes
        tests.append("""
def test_required_attributes():
    form = soup.find('form')
    assert form.get('role') == 'form', "Form missing role attribute"
    assert form.get('aria-label'), "Form missing aria-label"
""")

        return tests

    def _generate_nav_tests(self, soup: BeautifulSoup) -> List[str]:
        tests = []
        
        # Test navigation structure
        tests.append("""
def test_nav_structure():
    nav = soup.find('nav')
    assert nav.get('role') == 'navigation', "Nav missing role attribute"
    assert nav.get('aria-label'), "Nav missing aria-label"
""")

        # Test navigation links
        tests.append("""
def test_nav_links():
    nav = soup.find('nav')
    links = nav.find_all('a')
    for link in links:
        assert link.get('href'), "Link missing href attribute"
        assert link.text.strip(), "Empty link text"
""")

        return tests

    def _generate_heading_tests(self, soup: BeautifulSoup) -> List[str]:
        tests = []
        
        # Test heading hierarchy
        tests.append("""
def test_heading_hierarchy():
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    levels = [int(h.name[1]) for h in headings]
    assert 1 in levels, "Missing h1 heading"
    for i in range(len(levels)-1):
        assert levels[i+1] - levels[i] <= 1, "Invalid heading hierarchy"
""")

        return tests