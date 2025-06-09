import os
import json
import csv
import random
from collections import defaultdict

def load_swda_data(data_dir):
    """
    Load SwDA data from the CSV files.
    Expected columns: conversation_no, utterance_idx, act_tag, caller, text
    """
    conversations = defaultdict(list)
    
    # Read the main data file
    with open(os.path.join(data_dir, 'swda.csv'), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row['conversation_no']
            conversations[conv_id].append({
                'utterance_id': f"{conv_id}_{row['utterance_idx']}",
                'text': row['text'].strip(),
                'dialog_act': row['act_tag'],
                'caller': row['caller']
            })
    
    return conversations

def split_data(conversations, train_ratio=0.8, val_ratio=0.1):
    """Split conversations into train/val/test sets"""
    conv_ids = list(conversations.keys())
    random.shuffle(conv_ids)
    
    n_conv = len(conv_ids)
    train_idx = int(n_conv * train_ratio)
    val_idx = int(n_conv * (train_ratio + val_ratio))
    
    train_convs = conv_ids[:train_idx]
    val_convs = conv_ids[train_idx:val_idx]
    test_convs = conv_ids[val_idx:]
    
    # Convert to required format with history
    def convert_to_json_lines(conv_ids):
        data = []
        for conv_id in conv_ids:
            utterances = conversations[conv_id]
            # Sort utterances by utterance_id to maintain order
            utterances.sort(key=lambda x: int(x['utterance_id'].split('_')[1]))
            
            # Add each utterance with its history
            for i, utt in enumerate(utterances):
                history_text = ""
                if i > 0:
                    history_lines = []
                    for prev_utt in utterances[:i]:
                        history_lines.append(f"{prev_utt['caller']}: {prev_utt['text']}")
                    history_text = "\n".join(history_lines)
                
                data.append({
                    'utterance_id': utt['utterance_id'],
                    'text': utt['text'],
                    'dialog_act': utt['dialog_act'],
                    'caller': utt['caller'],
                    'history': history_text,
                    'response': utt['text']  # The current utterance is the response
                })
        return data
    
    return (convert_to_json_lines(train_convs),
            convert_to_json_lines(val_convs),
            convert_to_json_lines(test_convs))

def save_jsonl(data, output_file):
    """Save data in JSONL format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    # Set paths
    data_dir = 'data/swda'  # Update this to your SwDA data directory
    output_dir = 'data/swda/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading SwDA data...")
    conversations = load_swda_data(data_dir)
    
    # Split data
    print("Splitting data into train/val/test...")
    train_data, val_data, test_data = split_data(conversations)
    
    # Save processed data
    print("Saving processed data...")
    save_jsonl(train_data, os.path.join(output_dir, 'train.json'))
    save_jsonl(val_data, os.path.join(output_dir, 'val.json'))
    save_jsonl(test_data, os.path.join(output_dir, 'test.json'))
    
    print(f"Done! Processed {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")
    
    # Print sample of processed data
    print("\nSample of processed data:")
    print(json.dumps(train_data[0], indent=2))

if __name__ == '__main__':
    main() 