import random
from typing import List, Dict, Tuple

COMPONENT_TYPES = [
    "form", "navigation", "modal", "card", "table", 
    "header", "footer", "sidebar", "article", "search",
    "accordion", "tabs", "carousel", "alert", "breadcrumb",
    "pagination", "dropdown", "tooltip", "progress", "rating",
    "timeline", "filter", "gallery", "comments", "notification",
    "menu", "stepper", "datepicker", "slider", "upload"
]

COMPONENT_FEATURES = {
    "form": {
        "purpose": "collects user information",
        "description": "input fields for name, email, and message",
        "required_elements": ["input", "label", "button"],
        "component_type": "form"
    },
    "navigation": {
        "purpose": "allows site navigation",
        "description": "links to main sections",
        "required_elements": ["nav", "ul", "li", "a"],
        "component_type": "navigation"
    },
    "modal": {
        "purpose": "shows additional information",
        "description": "dialog with title and content",
        "required_elements": ["div", "h2", "button"],
        "component_type": "modal"
    },
    "card": {
        "purpose": "displays content summary",
        "description": "title, image and description",
        "required_elements": ["div", "h3", "img", "p"],
        "component_type": "card"
    },
    "table": {
        "purpose": "displays structured data",
        "description": "rows and columns with data",
        "required_elements": ["table", "tr", "th", "td"],
        "component_type": "table"
    },
    "header": {
        "purpose": "shows site branding and main navigation",
        "description": "logo, navigation menu, and actions",
        "required_elements": ["header", "nav", "img", "button"],
        "component_type": "header"
    },
    "footer": {
        "purpose": "provides site information and links",
        "description": "copyright, links, and contact info",
        "required_elements": ["footer", "div", "p", "a"],
        "component_type": "footer"
    },
    "sidebar": {
        "purpose": "shows additional navigation or content",
        "description": "secondary navigation and widgets",
        "required_elements": ["aside", "nav", "div", "ul"],
        "component_type": "sidebar"
    },
    "article": {
        "purpose": "displays content article",
        "description": "title, content, and metadata",
        "required_elements": ["article", "h1", "p", "time"],
        "component_type": "article"
    },
    "search": {
        "purpose": "enables content search",
        "description": "search input and results",
        "required_elements": ["form", "input", "button", "div"],
        "component_type": "search"
    },
    "accordion": {
        "purpose": "shows collapsible content",
        "description": "expandable sections with content",
        "required_elements": ["div", "button", "div", "h3"],
        "component_type": "accordion"
    },
    "tabs": {
        "purpose": "organizes content in tabs",
        "description": "tab list and content panels",
        "required_elements": ["div", "button", "div", "ul"],
        "component_type": "tabs"
    },
    "carousel": {
        "purpose": "displays sliding content",
        "description": "image slides with navigation",
        "required_elements": ["div", "button", "img", "div"],
        "component_type": "carousel"
    },
    "alert": {
        "purpose": "shows important messages",
        "description": "status message with optional actions",
        "required_elements": ["div", "p", "button", "span"],
        "component_type": "alert"
    },
    "breadcrumb": {
        "purpose": "shows navigation path",
        "description": "hierarchical navigation links",
        "required_elements": ["nav", "ol", "li", "a"],
        "component_type": "breadcrumb"
    },
    "pagination": {
        "purpose": "enables content page navigation",
        "description": "numbered pages with previous/next controls",
        "required_elements": ["nav", "ul", "li", "button", "a"],
        "component_type": "pagination"
    },
    "dropdown": {
        "purpose": "shows selectable options menu",
        "description": "button trigger with options list",
        "required_elements": ["div", "button", "ul", "li"],
        "component_type": "dropdown"
    },
    "tooltip": {
        "purpose": "displays additional information on hover",
        "description": "trigger element with popup content",
        "required_elements": ["button", "div", "span"],
        "component_type": "tooltip"
    },
    "progress": {
        "purpose": "shows completion status",
        "description": "progress bar with percentage",
        "required_elements": ["div", "progress", "span"],
        "component_type": "progress"
    },
    "rating": {
        "purpose": "allows user rating input",
        "description": "interactive star rating system",
        "required_elements": ["div", "button", "span", "input"],
        "component_type": "rating"
    }
}

CONTEXT_TEMPLATES = [
    "Create a {component} that {purpose}",
    "Generate HTML for a {component} with {features}",
    "Build a {component} that includes {features}",
    "Write code for a {component} {purpose}"
]

def generate_prompt(component_type: str = None) -> Tuple[str, Dict]:
    if not component_type:
        component_type = random.choice(list(COMPONENT_FEATURES.keys()))
    
    features = COMPONENT_FEATURES.get(component_type)
    if not features:
        raise ValueError(f"Unknown component type: {component_type}")
    
    template = random.choice(CONTEXT_TEMPLATES)
    
    prompt = template.format(
        component=component_type,
        purpose=features["purpose"],
        features=features["description"]
    )
    
    return prompt, features

def generate_features_for_component(component_type: str) -> Dict:
    """Generate required features based on component type"""
    features = {
        "form": {
            "purpose": "collects user information",
            "description": "input fields for name, email, and message",
            "required_elements": ["input", "label", "button"]
        },
        "navigation": {
            "purpose": "allows site navigation",
            "description": "links to main sections",
            "required_elements": ["nav", "ul", "li", "a"]
        }
        # Add more component types
    }
    
    return features.get(component_type, {})

def generate_dataset(size: int) -> List[Dict]:
    """Generate dataset of given size"""
    dataset = []
    for _ in range(size):
        prompt, features = generate_prompt()
        dataset.append({
            "prompt": prompt,
            "expected_features": features
        })
    return dataset