import json
import os
import re
import csv
from pathlib import Path

def extract_answer_from_json(json_content):
    """Extract the letter from <answer> tags, handling both simple letter and letter with additional content."""
    answer_text = json_content.get('answer', '')
    
    # First try to match the pattern with additional content (e.g., "C. 27")
    match = re.search(r'<answer>([A-E])[.\s].*?</answer>', answer_text)
    if match:
        return match.group(1)
    
    # If no match, try to match just the letter
    match = re.search(r'<answer>([A-E])</answer>', answer_text)
    if match:
        return match.group(1)
    
    return None

def process_files(folder_path):
    """Process all JSON files in the folder and create a CSV with results."""
    # Create a list to store results
    results = []
    
    # Get all JSON files in the folder
    folder = Path(folder_path)
    json_files = sorted(folder.glob('*.json'))
    
    # Keep track of processed IDs
    processed_ids = set()
    
    # Process each file
    for json_file in json_files:
        # Extract ID from filename (00001 from 00001_result.json)
        file_id = json_file.name.split('_')[0]
        
        # Check if ID is within our range (00001 to 00251)
        try:
            id_num = int(file_id)
            if 1 <= id_num <= 251:
                # Read and parse JSON file
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_content = json.load(f)
                        
                    # Extract answer
                    answer = extract_answer_from_json(json_content)
                    if answer:
                        results.append([file_id, answer])
                        processed_ids.add(id_num)
                    else:
                        print(f"Warning: No answer found in {json_file}")
                        
                except Exception as e:
                    print(f"Error processing {json_file}: {str(e)}")
        except ValueError:
            print(f"Warning: Invalid ID format in filename {json_file}")
    
    # Sort all results by ID
    results.sort(key=lambda x: x[0])
    
    # Report any missing IDs in the range
    all_expected_ids = set(range(1, 252))
    missing_ids = all_expected_ids - processed_ids
    if missing_ids:
        print(f"Warning: Missing answers for IDs: {sorted(missing_ids)}")
    
    # Write results to CSV
    output_file = 'answers.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])  # Write header
        writer.writerows(results)
    
    print(f"Results written to {output_file}")
    print(f"Processed {len(results)} files")

if __name__ == "__main__":
    # Specify the folder path containing JSON files
    folder_path = "final_answers"
    
    # Process the files
    process_files(folder_path)