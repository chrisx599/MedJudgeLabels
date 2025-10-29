#!/usr/bin/env python3
"""
Script to convert MedQuAD CSV file to JSONL format.
"""

import csv
import json
import sys
from pathlib import Path


def csv_to_jsonl(csv_path, output_path):
    """
    Convert CSV file to JSONL format.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSONL file
    """
    print(f"Reading CSV from: {csv_path}")
    print(f"Writing JSONL to: {output_path}")
    
    record_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        # Read CSV with proper encoding and dialect detection
        reader = csv.DictReader(infile)
        
        for row in reader:
            # Convert row to dictionary and write as JSON
            record = {}
            for key, value in row.items():
                # Clean up keys (strip whitespace)
                clean_key = key.strip() if key else key
                # Clean up values (handle empty strings)
                clean_value = value.strip() if value else value
                record[clean_key] = clean_value
            
            # Write as JSONL
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            record_count += 1
            
            if record_count % 100 == 0:
                print(f"Processed {record_count} records...")
    
    print(f"\nConversion complete!")
    print(f"Total records converted: {record_count}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    # Default paths
    csv_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/MedQuAD_combined.csv"
    output_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/MedQuAD_combined.jsonl"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Verify input file exists
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    csv_to_jsonl(csv_path, output_path)
