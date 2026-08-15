import json
import os
from pathlib import Path

def get_type_str(val):
    if isinstance(val, dict):
        return "dict"
    elif isinstance(val, list):
        return f"list(len={len(val)})"
    else:
        return type(val).__name__

def analyze_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n--- File: {filepath.name} ---")
        if isinstance(data, list):
            print(f"Top-level type: JSON Array (length={len(data)})")
            if data and isinstance(data[0], dict):
                print("First record keys and types:")
                for k, v in data[0].items():
                    print(f"  - {k}: {get_type_str(v)}")
        elif isinstance(data, dict):
            print(f"Top-level type: JSON Object")
            print("Keys and types:")
            for k, v in data.items():
                print(f"  - {k}: {get_type_str(v)}")
        else:
            print(f"Top-level type: {type(data).__name__}")
    except Exception as e:
        print(f"Error reading {filepath.name}: {e}")

if __name__ == '__main__':
    raw_dir = Path("data/raw/hancock")
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.json'):
                analyze_json_file(Path(root) / file)
