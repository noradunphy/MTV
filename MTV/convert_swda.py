import os
import csv
import glob
from collections import defaultdict

def parse_swda_file(filepath):
    """Parse a single SwDA transcript file"""
    utterances = []
    current_conv = os.path.basename(filepath).split('.')[0]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                # Parse the line
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    # Extract utterance index
                    utt_idx = parts[0].strip()
                    # Extract speaker
                    speaker = parts[1].strip()
                    # Extract dialog act tag (usually in the format "da1,da2")
                    act_tags = parts[2].strip().split(',')[0]  # Take first tag if multiple
                    # Extract text (everything after the third pipe)
                    text = '|'.join(parts[3:]).strip()
                    
                    utterances.append({
                        'conversation_no': current_conv,
                        'utterance_idx': utt_idx,
                        'act_tag': act_tags,
                        'caller': 'A' if speaker == 'A' else 'B',
                        'text': text
                    })
    
    return utterances

def convert_swda_to_csv(swda_dir, output_file):
    """Convert all SwDA transcript files to a single CSV"""
    all_utterances = []
    
    # Find all transcript files
    transcript_files = glob.glob(os.path.join(swda_dir, '**', '*.utt'), recursive=True)
    
    print(f"Found {len(transcript_files)} transcript files")
    
    # Process each file
    for filepath in transcript_files:
        utterances = parse_swda_file(filepath)
        all_utterances.extend(utterances)
    
    # Write to CSV
    if all_utterances:
        fieldnames = ['conversation_no', 'utterance_idx', 'act_tag', 'caller', 'text']
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_utterances)
        
        print(f"Successfully converted {len(all_utterances)} utterances to {output_file}")
    else:
        print("No utterances found to convert")

def main():
    # Set paths
    swda_dir = 'data/swda/swda'  # Directory containing the original SwDA files
    output_file = 'data/swda/swda.csv'  # Output CSV file
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert data
    print("Converting SwDA data to CSV format...")
    convert_swda_to_csv(swda_dir, output_file)

if __name__ == '__main__':
    main() 