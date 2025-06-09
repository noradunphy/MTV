from datasets import load_dataset
import json
import re
from collections import defaultdict
import os

def clean_swda_text(text):
    """
    Remove SWDA transcription symbols but keep the content inside curly braces.
    For example, '{F uh, }' becomes 'uh, '.
    """
    # Remove hashtags and their content
    text = re.sub(r'#.*?#', '', text)
    # Replace {X content} with just 'content' (keep content, remove code and braces)
    text = re.sub(r'\{[A-Z] ([^}]*)\}', r'\1', text)
    # Remove angle-bracket annotations like <laughter>
    text = re.sub(r'<[^>]*>', '', text)
    # Remove slashes (intonation units) 
    text = text.replace('/', '')
    # Remove plus, minus, brackets, and other common symbols
    text = re.sub(r'[\[\]\+\-\#]', '', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    # Remove curly brackets and parentheses
    text = re.sub(r'[{}()]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove turns that only contain a period
    if text.strip() == '.':
        return ''
    return text.strip()

def load_and_clean_swda_split(split_data):
    """
    Loads the SWDA dataset from HuggingFace and formats each utterance as a dictionary
    """
    # Get the mapping from damsl_act_tag indices to names
    damsl_tag_names = split_data.features['damsl_act_tag'].names

    # Debug: Print first few examples to see act_tag values
    print("\nDebug: First 5 examples of act_tag values:")
    for i, ex in enumerate(split_data):
        if i >= 5:
            break
        damsl_idx = ex['damsl_act_tag']
        damsl_tag = damsl_tag_names[damsl_idx]
        print(f"Example {i}: damsl_act_tag={damsl_idx} -> {damsl_tag}")

    # Group utterances by conversation_no
    conversations = defaultdict(list)
    for ex in split_data:
        conversations[ex['conversation_no']].append(ex)

    # Sort utterances in each conversation by utterance_index
    for conv in conversations.values():
        conv.sort(key=lambda x: x['utterance_index'])

    # Format data
    formatted = []
    for conv_no, utterances in conversations.items():
        for i, utt in enumerate(utterances):
            if i == 0:
                history_text = ""
            else:
                history_lines = []
                for u in utterances[:i]:
                    cleaned = clean_swda_text(u['text'])
                    if cleaned.strip() != "":
                        history_lines.append(u['caller'] + ": " + cleaned)
                history_text = "\n".join(history_lines)

            response = clean_swda_text(utt['text'])
            
            # Convert damsl_act_tag index to actual tag name
            damsl_idx = utt['damsl_act_tag']
            label = damsl_tag_names[damsl_idx]

            if response.strip() != "":
                formatted.append({
                    "text": history_text,
                    "response": response,
                    "dialog_act": label,
                    "utterance_id": f"{conv_no}_{utt['utterance_index']}",
                    "caller": utt['caller']
                })

    return formatted

def save_swda_splits_from_original():
    """
    Save SWDA data splits to JSON files
    """
    ds = load_dataset("swda", trust_remote_code=True)
    print("\nDataset features:", ds['train'].features)
    print("\nFirst example:", ds['train'][0])
    
    os.makedirs("data/swda", exist_ok=True)

    for split in ['test', 'train', 'validation']:
        print(f"Processing {split} split...")
        formatted_data = load_and_clean_swda_split(ds[split])
        with open(f"data/swda/{split}.json", "w") as f:
            json.dump(formatted_data, f)
        print(f"Saved {len(formatted_data)} {split} examples")

if __name__ == "__main__":
    save_swda_splits_from_original()
