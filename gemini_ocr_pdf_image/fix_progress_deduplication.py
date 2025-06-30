#!/usr/bin/env python3
"""
Fix duplicate progress entries and implement proper deduplication logic.
"""

import os
import csv
from pathlib import Path

def deduplicate_progress_file(progress_file_path):
    """Remove duplicate entries from progress file, keeping the latest entry for each page."""
    if not os.path.exists(progress_file_path):
        return
    
    print(f"Deduplicating {progress_file_path}...")
    
    # Read all entries and keep latest for each page
    page_data = {}
    fieldnames = None
    
    with open(progress_file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            page_num = int(row['page_num'])
            # Keep latest entry (overwrites duplicates)
            page_data[page_num] = row
    
    # Write deduplicated data back
    with open(progress_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write in page order
        for page_num in sorted(page_data.keys()):
            writer.writerow(page_data[page_num])
    
    print(f"  Deduplicated to {len(page_data)} unique pages")

def main():
    """Fix all progress files in the logs directory."""
    logs_dir = Path("./logs")
    
    if not logs_dir.exists():
        print("No logs directory found!")
        return
    
    progress_files = list(logs_dir.glob("*_progress.csv"))
    
    if not progress_files:
        print("No progress files found!")
        return
    
    print(f"Found {len(progress_files)} progress files to deduplicate:")
    
    for progress_file in progress_files:
        deduplicate_progress_file(progress_file)
    
    print("\n✅ Deduplication complete!")
    print("Now the fast resume will work correctly.")

if __name__ == "__main__":
    main()