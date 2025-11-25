import json
import os

# Get the data directory path (one level up from scripts, then into data)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), 'data')

# Define file paths
filter_data_path = os.path.join(data_dir, 'filter_pku_data.jsonl')
input_data_path = os.path.join(data_dir, 'pku_anno_formatted_test.jsonl')
output_data_path = os.path.join(data_dir, 'pku_anno_formatted_test_orig.jsonl')

# Read filter_pku_data.jsonl and create a lookup dictionary keyed by (query, response) tuple
filter_data = {}
with open(filter_data_path, 'r') as f:
    for line in f:
        record = json.loads(line)
        query = record.get('query')
        meta_data = record.get('meta_data', {})
        
        # Get response_0 and response_1
        response_0 = meta_data.get('response_0')
        response_1 = meta_data.get('response_1')
        
        # Extract the severity levels and safety flags
        is_response_0_safe = meta_data.get('is_response_0_safe')
        is_response_1_safe = meta_data.get('is_response_1_safe')
        response_0_severity_level = meta_data.get('response_0_severity_level')
        response_1_severity_level = meta_data.get('response_1_severity_level')
        
        # Create entries for response_0
        if query and response_0:
            key_0 = (query, response_0)
            filter_data[key_0] = {
                'is_response_safe': is_response_0_safe,
                'response_severity_level': response_0_severity_level
            }
        
        # Create entries for response_1
        if query and response_1:
            key_1 = (query, response_1)
            filter_data[key_1] = {
                'is_response_safe': is_response_1_safe,
                'response_severity_level': response_1_severity_level
            }

# Read pku_anno_formatted_test.jsonl and perform left join
with open(input_data_path, 'r') as infile, \
     open(output_data_path, 'w') as outfile:
    for line in infile:
        record = json.loads(line)
        query = record.get('query')
        response = record.get('response')
        
        # Create a key tuple (query, response)
        key = (query, response)
        
        # Perform left join: add fields from filter_data if available
        if key in filter_data:
            record['is_response_safe'] = filter_data[key]['is_response_safe']
            record['response_severity_level'] = filter_data[key]['response_severity_level']
        else:
            # Add fields with None if not found
            record['is_response_safe'] = None
            record['response_severity_level'] = None
        
        # Write the merged record
        outfile.write(json.dumps(record) + '\n')

print("Successfully created pku_anno_formatted_test_orig.jsonl")
