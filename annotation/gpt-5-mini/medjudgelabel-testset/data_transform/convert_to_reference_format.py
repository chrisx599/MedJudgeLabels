#!/usr/bin/env python3
"""
Convert the flat JSONL format to match the reference format with nested annotation
"""
import json

def convert_to_reference_format(input_jsonl, output_jsonl):
    """Convert flat JSONL to nested annotation format"""
    with open(input_jsonl, 'r', encoding='utf-8') as infile:
        with open(output_jsonl, 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line)
                
                # Create the new structured format
                new_record = {
                    "query": data.get("query"),
                    "response": data.get("response"),
                    "annotation": {
                        "binary_harmfulness": data.get("binary_harmfulness"),
                        "severity": data.get("severity"),
                        "harm_types": data.get("harm_types", []),
                        "explanation": data.get("explanation")
                    },
                    "annotation_success": True,  # Assuming all annotations were successful
                    # Keep the mjl fields at the top level
                    "mjl_harmfulness": data.get("mjl_harmfulness"),
                    "mjl_serverity": data.get("mjl_serverity"),
                    "mjl_harmtype": data.get("mjl_harmtype")
                }
                
                # Write the converted record
                outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    input_file = 'pku_mjl_anno_20251031054243.jsonl'
    output_file = 'pku_mjl_anno_20251031054243_formatted.jsonl'
    
    print(f"Converting {input_file} to reference format...")
    convert_to_reference_format(input_file, output_file)
    print(f"Conversion complete! Output saved to {output_file}")