"""
Cleanup script to remove unnecessary dummy CSV files
"""
import os
from pathlib import Path

files_to_remove = [
    "data/multi_domain/sample_dataset.csv",
    "data/test_nineoz.csv"
]

for file_path in files_to_remove:
    full_path = Path(file_path)
    if full_path.exists():
        try:
            full_path.unlink()
            print(f"✓ Deleted: {file_path}")
        except Exception as e:
            print(f"✗ Failed to delete {file_path}: {e}")
    else:
        print(f"- File not found: {file_path}")

print("\nCleanup complete!")
