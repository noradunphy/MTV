from datasets import load_dataset
from collections import defaultdict, Counter
import random
import re
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
#import pdb

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
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_and_format_swda_hf():
    """
    Loads the SWDA dataset from HuggingFace and formats each utterance as:
    (dialogue history, response, label, utterance_index, conversation_no)
    """
    ds = load_dataset("swda", trust_remote_code=True)
    data = ds['train']
    label_names = ds['train'].features['damsl_act_tag'].names

    # Group utterances by conversation_no (int)
    conversations = defaultdict(list)
    for ex in data:
        conversations[ex['conversation_no']].append(ex)

    # Sort utterances in each conversation by utterance_index (int)
    for conv in conversations.values():
        conv.sort(key=lambda x: x['utterance_index'])

    # Build (history, response, label, utterance_index, conversation_no) tuples
    formatted = []
    for conv_no, utterances in conversations.items():
        for i, utt in enumerate(utterances):
            if i == 0:
                history_text = ""
            else:
                # Only include non-empty cleaned utterances in the history
                history_lines = []
                for u in utterances[:i]:
                    cleaned = clean_swda_text(u['text'])
                    if cleaned.strip() != "":
                        history_lines.append(u['caller'] + ": " + cleaned)
                history_text = "\n".join(history_lines)
            response = clean_swda_text(utt['text'])
            # Robust label extraction
            damsl = utt.get('damsl_act_tag', "")
            if isinstance(damsl, int):
                label = label_names[damsl]
            elif isinstance(damsl, list) and len(damsl) > 0 and isinstance(damsl[0], int):
                label = label_names[damsl[0]]
            elif isinstance(damsl, str):
                label = damsl
            else:
                label = str(damsl) if damsl is not None else ""
            utterance_index = utt['utterance_index']
            formatted.append({
                "history": history_text,
                "response": response,
                "label": label,
                "utterance_index": utterance_index,
                "conversation_no": conv_no,
                "next_speaker": utt['caller'],
            })
    # Remove examples with empty responses
    formatted = [ex for ex in formatted if ex['response'].strip() != ""]
    return formatted


def group_by_label(formatted_data):
    """
    Groups formatted SWDA data by dialog act label.
    Returns a dict: {label: [examples]}
    """
    grouped = defaultdict(list)
    for ex in formatted_data:
        grouped[ex['label']].append(ex)
    return grouped


def format_for_model(ex):
    """
    Formats a single example for model input: just the dialogue history, then the next speaker letter and a colon.
    """
    if ex['history'].strip() == "":
        return f"{ex['next_speaker']}:"
    else:
        return f"{ex['history']}\n{ex['next_speaker']}:"

# Example usage (uncomment to run as script):
if __name__ == "__main__":
    formatted = load_and_format_swda_hf()
    grouped = group_by_label(formatted)

    # Print the number of examples for each label
    '''
    example usage
    '''
    ex = random.choice(formatted)

    print(format_for_model(ex))
    # print("Target:", ex['response'])
    # print("Label:", ex['label'])
    # print("---")
    '''
    end of example usage
    '''

    '''
    # t-SNE plot
    '''
    # # Use a small, fast model for demo; you can use a larger one if you want
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    # # Only use non-empty responses for meaningful embeddings
    # utterances = [ex['response'] for ex in formatted if ex['response'].strip() != ""]
    # labels = [ex['label'] for ex in formatted if ex['response'].strip() != ""]

    # # Compute embeddings
    # embeddings = model.encode(utterances, show_progress_bar=True)

    # tsne = TSNE(n_components=2, random_state=42, perplexity=30, verbose=1)
    # X_2d = tsne.fit_transform(embeddings)

    # # For coloring, map each label to an integer
    # unique_labels = list(set(labels))
    # label_to_int = {l: i for i, l in enumerate(unique_labels)}
    # colors = [label_to_int[l] for l in labels]

    # plt.figure(figsize=(12, 10))
    # scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=colors, cmap='tab20', alpha=0.6, s=10)

    # # Create a legend with the most common labels
    # most_common = [l for l, _ in Counter(labels).most_common(10)]
    # handles = [plt.Line2D([0], [0], marker='o', color='w', label=l,
    #                       markerfacecolor=plt.cm.tab20(label_to_int[l]/len(unique_labels)), markersize=10)
    #            for l in most_common]
    # plt.legend(handles=handles, title='Dialog Act', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.title('t-SNE of SWDA Utterances Colored by Dialog Act')
    # plt.xlabel('t-SNE 1')
    # plt.ylabel('t-SNE 2')
    # plt.tight_layout()
    # plt.savefig("swda_tsne.png", dpi=300, bbox_inches='tight')
    # print("Plot saved as swda_tsne.png") 

    '''
    end of t-SNE plot
    '''

    # Now, print discourse marker only lines with sd label
    # ds = load_dataset("swda", trust_remote_code=True)
    # data = ds['train']
    # label_names = ds['train'].features['damsl_act_tag'].names

    # for ex in data:
    #     text = ex['text'].strip()
    #     if re.fullmatch(r'\{D [^}]*\}', text):
    #         label_idx = ex['damsl_act_tag']
    #         label_abbr = label_names[label_idx] if isinstance(label_idx, int) else label_idx
    #         if label_abbr == 'sd':
    #             print(f"Utterance: {text}")
    #             print(f"Label: {label_abbr}")
    #             print(f"Conversation: {ex['conversation_no']}, Utterance Index: {ex['utterance_index']}")
    #             print('---')

    # formatted = load_and_format_swda_hf()
    # empty_responses = [ex for ex in formatted if ex['response'].strip() == ""]

    # print(f"Number of empty responses: {len(empty_responses)}")