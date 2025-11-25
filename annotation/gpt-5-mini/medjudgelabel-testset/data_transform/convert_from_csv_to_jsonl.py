#!/usr/bin/env python3
"""
Convert CSV file to JSONL format
"""
import csv
import json
import sys

def convert_csv_to_jsonl(input_csv, output_jsonl):
    """Convert CSV file to JSONL format"""
    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
            for row in csv_reader:
                # Convert empty strings to None for cleaner JSON
                cleaned_row = {}
                for key, value in row.items():
                    if value == '':
                        cleaned_row[key] = None
                    elif key == 'harm_types':
                        # Parse harm_types as list if it's not empty
                        if value and value != '[]':
                            # Handle different formats
                            if value.startswith('['):
                                try:
                                    cleaned_row[key] = json.loads(value.replace("'", '"'))
                                except:
                                    cleaned_row[key] = [v.strip() for v in value.strip('[]').split(',') if v.strip()]
                            else:
                                cleaned_row[key] = [value]
                        else:
                            cleaned_row[key] = []
                    else:
                        cleaned_row[key] = value
                
                # Write as JSON line
                jsonl_file.write(json.dumps(cleaned_row, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    input_file = 'pku_mjl_anno_20251031054243.csv'
    output_file = 'pku_mjl_anno_20251031054243.jsonl'
    
    print(f"Converting {input_file} to {output_file}...")
    convert_csv_to_jsonl(input_file, output_file)
    print("Conversion complete!")
