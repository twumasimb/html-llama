import argparse
from utils import load_json, accessibility_test
from tqdm import tqdm

def validate_dataset(filepath):
    """Validate final dataset for accessibility compliance."""
    data = load_json(filepath)
    valid_entries = 0
    issues = []
    
    for idx, entry in enumerate(tqdm(data, desc="Validating entries")):
        if not accessibility_test(entry["solution"]):
            issues.append(f"Entry {idx}: Failed accessibility test")
            continue
        if len(entry["solution"]) < 50:
            issues.append(f"Entry {idx}: Solution too short")
            continue
        valid_entries += 1
    
    print(f"\nValidation Results:")
    print(f"Total entries: {len(data)}")
    print(f"Valid entries: {valid_entries}")
    print(f"Success rate: {(valid_entries/len(data))*100:.2f}%")
    
    if issues:
        print("\nIssues found:")
        for issue in issues[:10]:
            print(issue)
        if len(issues) > 10:
            print(f"...and {len(issues)-10} more issues")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="final_dataset.json", help="Dataset to validate")
    args = parser.parse_args()
    validate_dataset(args.dataset)

if __name__ == "__main__":
    main()
