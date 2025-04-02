# test_accessibility.py
import argparse
from utils import accessibility_test

def main():
    parser = argparse.ArgumentParser(description="Test webpage for WCAG 2.1 and ARIA accessibility compliance")
    parser.add_argument("--html", type=str, required=True, help="Path to the HTML file to test")
    args = parser.parse_args()
    
    with open(args.html, "r") as f:
        html_code = f.read()
    
    if accessibility_test(html_code):
        print("The webpage/component passed accessibility tests.")
    else:
        print("The webpage/component FAILED accessibility tests.")
    
if __name__ == "__main__":
    main()
