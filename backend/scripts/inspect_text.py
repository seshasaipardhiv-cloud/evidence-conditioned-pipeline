import os
import re
from collections import Counter
from pathlib import Path

def inspect_text_files():
    path = Path("data/raw/hancock/text")
    if not path.exists():
        print("Path does not exist:", path)
        return

    files = list(path.rglob("*.txt"))
    print(f"Total .txt files: {len(files)}")
    print(f"Sample files: {[f.name for f in files[:10]]}")

    patterns = Counter(re.sub(r'\d+', 'XXX', f.name) for f in files)
    print("\nFilename patterns and counts:")
    for pat, count in patterns.items():
        print(f"  {pat}: {count}")

    # Inspect one file to check encoding and size
    if files:
        sample_file = files[0]
        print(f"\nSize of {sample_file.name}: {os.path.getsize(sample_file)} bytes")
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"Encoding 'utf-8' successful. Read {len(content)} characters.")
        except Exception as e:
            print(f"Failed to read as utf-8: {e}")

if __name__ == "__main__":
    inspect_text_files()
