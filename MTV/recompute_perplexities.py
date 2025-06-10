from mtv_utils import *
from models import *
from preprocess import *
from tqdm import tqdm
import torch
import argparse
import numpy as np
from backchannel_classifier import load_classifier as load_backchannel_classifier
from declarative_classifier import load_classifier as load_declarative_classifier
import pdb
import random

def recompute_perplexities(args):
    # Create output file
    output_file = f"recomputed_perplexities_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}.txt"
    with open(output_file, 'w', buffering=1) as f:
        f.write(f"Recomputed Perplexities for {args.model_name} on {args.data_name}\n")
        f.write(f"Target dialogue act: {args.dialogue_act if args.dialogue_act else 'all'}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()

        print(f"[INFO] Loading training data from {args.train_path} for act '{args.dialogue_act}'...")
        train_dataset = open_data(args.data_name, args.train_path, getattr(args, 'dialogue_act', None))
        f.write(f"[INFO] Loaded {len(train_dataset)} training examples.\n")
        f.flush()

        print(f"[INFO] Loading validation data from {args.val_path} for act '{args.dialogue_act}'...")
        val_dataset = open_data(args.data_name, args.val_path, getattr(args, 'dialogue_act', None))
        f.write(f"[INFO] Loaded {len(val_dataset)} validation examples.\n")
        f.flush()

        # Filter for shorter SWDA dialogues if needed
        if args.data_name == "swda":
            train_dataset = [ex for ex in train_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
            val_dataset = [ex for ex in val_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
            f.write(f"[INFO] After filtering: {len(train_dataset)} training examples, {len(val_dataset)} validation examples\n")
            f.flush()

        print("[INFO] Loading model...")
        model_helper = load_model(args.model_name, args.data_name, zero_shot=args.zero_shot)
        f.write(f"[INFO] Model '{args.model_name}' loaded successfully!\n")
        f.flush()
        
        print("[INFO] Loading classifiers...")
        backchannel_classifier = load_backchannel_classifier()
        declarative_classifier = load_declarative_classifier()
        f.write("[INFO] Classifiers loaded!\n")
        f.flush()

        # Load saved activations and bernoullis
        print("[INFO] Loading saved activations and bernoullis...")
        mean_activations = torch.load(args.activation_path)
        intervention_locations = torch.load(args.bernoullis_path)
        f.write(f"[INFO] Loaded activations and {len(intervention_locations)} intervention locations.\n")
        f.flush()

        # Initialize tracking lists
        clean_perplexities = []
        interv_perplexities = []
        target_acts = []
        clean_pred_acts = []
        interv_pred_acts = []

        f.write("\n[INFO] Starting evaluation loop over validation set...\n")
        f.flush()
        
        for idx, item in enumerate(tqdm(val_dataset)):
            text, image_list, target_out, question_id = model_helper.format_func(train_dataset, item, num_shot=args.eval_num_shot)
            new_input = model_helper.insert_image(text, image_list)
            #pdb.set_trace()
            # Get clean and interventsion outputs with perplexities
            zero_activations = torch.zeros_like(mean_activations)
            
            # Create intervention locations for every layer and head
            num_layers = model_helper.model_config["n_layers"]
            num_heads = model_helper.model_config["n_heads"]
            
            all_intervention_locations = []
            for layer in range(num_layers):
                for head in range(num_heads):
                    all_intervention_locations.append((layer, head, -1))  # -1 for last token
            
            clean_out, interv_out, clean_ppl, interv_ppl = fv_intervention_natural_text(
                new_input, 
                model_helper, 
                max_new_tokens=args.max_token, 
                return_item="both",  # Always compute both
                intervention_locations=all_intervention_locations,  # Intervene everywhere
                avg_activations=zero_activations,
                target_output=target_out  # Pass target output for perplexity computation
            )

            # Track perplexities
            if clean_ppl is not None:
                clean_perplexities.append(clean_ppl)
            if interv_ppl is not None:
                interv_perplexities.append(interv_ppl)

            # Get target dialog act
            target_act = item.get('dialog_act', 'o')
            target_acts.append(target_act)

            # Extract first turn response from model outputs
            clean_out = extract_first_turn(clean_out)
            interv_out = extract_first_turn(interv_out)

            # Classify generated responses based on target act
            if target_act == 'sd':
                # Use declarative classifier for sd acts
                clean_is_declarative = declarative_classifier.classify_utterance(clean_out)
                interv_is_declarative = declarative_classifier.classify_utterance(interv_out)
                clean_act = 'sd' if clean_is_declarative else 'o'
                interv_act = 'sd' if interv_is_declarative else 'o'
            else:
                # Use backchannel classifier for other acts
                clean_is_backchannel = backchannel_classifier.classify_utterance(clean_out)
                interv_is_backchannel = backchannel_classifier.classify_utterance(interv_out)
                clean_act = 'b' if clean_is_backchannel else 'o'
                interv_act = 'b' if interv_is_backchannel else 'o'

            clean_pred_acts.append(clean_act)
            interv_pred_acts.append(interv_act)

            # Write detailed output for each example
            f.write(f"\n[DEBUG] Example {idx+1}:\n")
            f.write(f"[DEBUG] Input text: {text}\n")
            f.write(f"[DEBUG] Clean model output: {clean_out}\n")
            f.write(f"[DEBUG] Intervention model output: {interv_out}\n")
            f.write(f"[DEBUG] Target output: {target_out}\n")
            f.write(f"[DEBUG] Target dialog act: {target_act}\n")
            f.write(f"[DEBUG] Clean model dialog act: {clean_act}\n")
            f.write(f"[DEBUG] Intervention model dialog act: {interv_act}\n")
            f.write(f"[DEBUG] Clean model perplexity: {clean_ppl:.2f}\n")
            f.write(f"[DEBUG] Intervention model perplexity: {interv_ppl:.2f}\n")
            f.write("-" * 80 + "\n")
            f.flush()

        # Write summary statistics
        f.write("\n[INFO] Perplexity Statistics:\n")
        if clean_perplexities:
            f.write(f"[INFO] Clean model perplexity - Mean: {np.mean(clean_perplexities):.2f}, Std: {np.std(clean_perplexities):.2f}\n")
        if interv_perplexities:
            f.write(f"[INFO] Intervention model perplexity - Mean: {np.mean(interv_perplexities):.2f}, Std: {np.std(interv_perplexities):.2f}\n")
        if clean_perplexities and interv_perplexities:
            ppl_diff = np.mean(interv_perplexities) - np.mean(clean_perplexities)
            f.write(f"[INFO] Perplexity difference (Intervention - Clean): {ppl_diff:.2f}\n")
        f.flush()

        # Calculate and save confusion matrices
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Get unique acts
        unique_acts = sorted(set(target_acts + clean_pred_acts + interv_pred_acts))
        
        # Create confusion matrices
        clean_cm = confusion_matrix(target_acts, clean_pred_acts, labels=unique_acts)
        interv_cm = confusion_matrix(target_acts, interv_pred_acts, labels=unique_acts)
        
        # Plot confusion matrices
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(clean_cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_acts, yticklabels=unique_acts)
        plt.title('Clean Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(interv_cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_acts, yticklabels=unique_acts)
        plt.title('Intervention Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        confusion_matrix_path = f"recomputed_confusion_matrices_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}.png"
        plt.savefig(confusion_matrix_path)
        f.write(f"[INFO] Confusion matrices saved to '{confusion_matrix_path}'\n")
        f.flush()

    print(f"[INFO] Recomputed perplexities saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="text")
    parser.add_argument("--data_name", type=str, default="swda")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--eval_num_shot", type=int, default=0)
    parser.add_argument("--max_token", type=int, default=10)
    parser.add_argument("--bernoullis_path", type=str, required=True)
    parser.add_argument("--activation_path", type=str, required=True)
    parser.add_argument("--dialogue_act", type=str, default=None)
    parser.add_argument("--zero_shot", type=bool, default=False)

    args = parser.parse_args()
    recompute_perplexities(args) 