import pytest
from utils import accessibility_test, deduplicate_prompts

def test_accessibility_check():
    # Test valid HTML
    valid_html = '''
    <div role="navigation" aria-label="Main menu">
        <button role="menuitem" aria-expanded="false">Menu</button>
    </div>
    '''
    assert accessibility_test(valid_html) == True
    
    # Test invalid HTML
    invalid_html = '<div>No accessibility attributes</div>'
    assert accessibility_test(invalid_html) == False

def test_deduplicate_prompts():
    prompts = ["Create a login form", "Create a signup form", "Create a login form"]
    deduped = deduplicate_prompts(prompts)
    assert len(deduped) == 2
    assert "Create a login form" in deduped
