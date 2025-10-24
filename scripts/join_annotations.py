#!/usr/bin/env python3
"""
Script to perform a left join between annotated data (pku_anno.jsonl) and 
original data (pku_data.jsonl) based on query and response fields.

This script adds the original human labels (is_response_safe and response_severity_level)
to the annotated data.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_original_data(pku_data_path):
    """
    Load original data and create a lookup dictionary.
    
    The dictionary maps (query, response) -> {is_response_safe, response_severity_level, response_id}
    Since each record has response_0 and response_1, we create entries for both.
    """
    lookup = defaultdict(list)
    
    with open(pku_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            query = record.get('query', '')
            meta_data = record.get('meta_data', {})
            
            # Create entry for response_0
            response_0 = meta_data.get('response_0', '')
            is_response_0_safe = meta_data.get('is_response_0_safe')
            response_0_severity = meta_data.get('response_0_severity_level')
            
            if response_0:
                lookup[(query, response_0)].append({
                    'is_response_safe': is_response_0_safe,
                    'response_severity_level': response_0_severity,
                    'response_id': 0
                })
            
            # Create entry for response_1
            response_1 = meta_data.get('response_1', '')
            is_response_1_safe = meta_data.get('is_response_1_safe')
            response_1_severity = meta_data.get('response_1_severity_level')
            
            if response_1:
                lookup[(query, response_1)].append({
                    'is_response_safe': is_response_1_safe,
                    'response_severity_level': response_1_severity,
                    'response_id': 1
                })
    
    return lookup


def join_annotations(anno_path, original_path, output_path):
    """
    Perform left join of annotated data with original data.
    
    Args:
        anno_path: Path to pku_anno.jsonl (left table)
        original_path: Path to pku_data.jsonl (right table)
        output_path: Path to output file
    """
    print(f"Loading original data from {original_path}...")
    lookup = load_original_data(original_path)
    print(f"Loaded {len(lookup)} unique (query, response) pairs from original data")
    
    print(f"Processing annotated data from {anno_path}...")
    matched = 0
    unmatched = 0
    
    with open(anno_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            anno_record = json.loads(line.strip())
            query = anno_record.get('query', '')
            response = anno_record.get('meta_data', {}).get('response', '')
            
            # Try to find matching record in original data
            key = (query, response)
            matches = lookup.get(key, [])
            
            if matches:
                # Use the first match (in case there are duplicates)
                match_data = matches[0]
                anno_record['is_response_safe'] = match_data['is_response_safe']
                anno_record['response_severity_level'] = match_data['response_severity_level']
                matched += 1
            else:
                # No match found - add null values
                anno_record['is_response_safe'] = None
                anno_record['response_severity_level'] = None
                unmatched += 1
            
            # Write the updated record
            outfile.write(json.dumps(anno_record, ensure_ascii=False) + '\n')
            
            if line_num % 1000 == 0:
                print(f"Processed {line_num} records... (matched: {matched}, unmatched: {unmatched})")
    
    print(f"\nJoin complete!")
    print(f"Total records processed: {line_num}")
    print(f"Matched: {matched}")
    print(f"Unmatched: {unmatched}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    # Default paths
    anno_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno.jsonl"
    original_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_data.jsonl"
    output_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_enriched.jsonl"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        anno_path = sys.argv[1]
    if len(sys.argv) > 2:
        original_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    
    # Verify input files exist
    if not Path(anno_path).exists():
        print(f"Error: Annotated data file not found: {anno_path}")
        sys.exit(1)
    if not Path(original_path).exists():
        print(f"Error: Original data file not found: {original_path}")
        sys.exit(1)
    
    join_annotations(anno_path, original_path, output_path)
