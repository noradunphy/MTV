from mtv_utils import *
from models import *
from preprocess import *
from speech_act_classifier import load_classifier
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
import pdb
logging.set_verbosity_error() 
import numpy as np


def eval_cross_act_intervention(args):
    # Create output file
    output_file = f"cross_act_eval_{args.model_name}_{args.data_name}_from_{args.source_act}_to_{args.target_act}.txt"
    with open(output_file, 'w', buffering=1) as f:  # Line buffering for real-time updates
        f.write(f"Cross-Act Intervention Evaluation Results\n")
        f.write(f"Model: {args.model_name} on {args.data_name}\n")
        f.write(f"Source act (intervention locations from): {args.source_act}\n")
        f.write(f"Target act (evaluating on): {args.target_act}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()

        print(f"[INFO] Loading validation data from {args.val_path} for act '{args.target_act}'...")
        val_dataset = open_data(args.data_name, args.val_path, args.target_act)
        f.write(f"[INFO] Loaded {len(val_dataset)} validation examples for target act '{args.target_act}'.\n")
        f.flush()

        # Filter for shorter SWDA dialogues if needed
        if args.data_name == "swda":
            val_dataset = [ex for ex in val_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
            f.write(f"[INFO] After filtering: {len(val_dataset)} validation examples\n")
            f.flush()

        eval_data = val_dataset[:min(50, len(val_dataset))]

        print("[INFO] Loading model...")
        model_helper = load_model(args.model_name, args.data_name, zero_shot=True)  # Force zero-shot mode
        f.write(f"[INFO] Model '{args.model_name}' loaded successfully in zero-shot mode!\n")
        f.flush()
        
        print("[INFO] Loading speech act classifier...")
        classifier, tag2idx = load_classifier()
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}
        f.write("[INFO] Speech act classifier loaded!\n")
        f.flush()

        # Load intervention locations from source act
        print(f"[INFO] Loading intervention locations from {args.source_act}...")
        intervention_locations = torch.load(args.source_bernoullis_path)
        f.write(f"[INFO] Loaded {len(intervention_locations)} intervention locations from source act '{args.source_act}'.\n")
        f.flush()

        # Load mean activations from source act
        print(f"[INFO] Loading mean activations from {args.source_act}...")
        mean_activations = torch.load(args.source_activation_path)
        f.write(f"[INFO] Loaded mean activations from source act '{args.source_act}'.\n")
        f.flush()

        clean_answers = []
        interv_answers = []
        clean_count, interv_count = 0, 0
        target_acts = []
        clean_pred_acts = []
        interv_pred_acts = []
        
        # Add perplexity tracking
        clean_perplexities = []
        interv_perplexities = []

        f.write("\n[INFO] Starting evaluation loop over validation set...\n")
        f.flush()
        for idx, item in enumerate(tqdm(val_dataset)):
            # Use model_helper.format_func() to properly format input, even in zero-shot mode
            text, image_list, target_out, question_id = model_helper.format_func(val_dataset, item, num_shot=0)  # num_shot=0 for zero-shot
            new_input = model_helper.insert_image(text, image_list)
            
            # Debug: Check if target_out is empty
            if not target_out or target_out.strip() == "":
                f.write(f"[WARNING] Empty target output for example {idx+1}. Skipping perplexity computation.\n")
                f.flush()
                target_out = " "  # Use a space as fallback to avoid tokenizer error
            
            clean_out, interv_out, clean_ppl, interv_ppl = fv_intervention_natural_text(
                new_input, 
                model_helper, 
                max_new_tokens=args.max_token, 
                return_item=args.cur_mode, 
                intervention_locations=intervention_locations, 
                avg_activations=mean_activations,
                target_output=target_out
            )

            # Track perplexities if computed
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

            # Classify generated responses
            clean_act_idx = classifier.classify_utterance(clean_out)
            interv_act_idx = classifier.classify_utterance(interv_out)
            clean_act = idx2tag[clean_act_idx]
            interv_act = idx2tag[interv_act_idx]
            clean_pred_acts.append(clean_act)
            interv_pred_acts.append(interv_act)

            f.write(f"\n[DEBUG] Example {idx+1}:\n")
            f.write(f"[DEBUG] Input text: {text}\n")
            f.write(f"[DEBUG] Clean model output: {clean_out}\n")
            f.write(f"[DEBUG] Intervention model output: {interv_out}\n")
            f.write(f"[DEBUG] Target output: {target_out}\n")
            f.write(f"[DEBUG] Target dialog act: {target_act}\n")
            f.write(f"[DEBUG] Clean model dialog act: {clean_act}\n")
            f.write(f"[DEBUG] Intervention model dialog act: {interv_act}\n")
            if clean_ppl is not None:
                f.write(f"[DEBUG] Clean model perplexity: {clean_ppl:.2f}\n")
            if interv_ppl is not None:
                f.write(f"[DEBUG] Intervention model perplexity: {interv_ppl:.2f}\n")
            f.write("-" * 80 + "\n")
            f.flush()

            if args.model_name == "Qwen-VL":
                interv_answers.append({"answer":interv_out, "question_id":question_id})
                clean_answers.append({"answer":clean_out, "question_id":question_id})
            else:
                interv_answers.append({"answer":interv_out, "question_id":question_id})
                clean_answers.append({"answer":clean_out, "question_id":question_id})

            clean_correct = int(clean_act == target_act)
            interv_correct = int(interv_act == target_act)
            clean_count += clean_correct
            interv_count += interv_correct

        f.write(f"\n[INFO] Evaluation complete. Clean correct: {clean_count}, Interv correct: {interv_count}, Total: {len(val_dataset)}\n")
        f.flush()
        
        # Report perplexity statistics
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
        confusion_matrix_path = f"cross_act_confusion_matrices_{args.model_name}_{args.data_name}_from_{args.source_act}_to_{args.target_act}.png"
        plt.savefig(confusion_matrix_path)
        f.write(f"[INFO] Confusion matrices saved to '{confusion_matrix_path}'\n")
        f.flush()

        if args.is_eval:
            if args.data_name == "swda":
                f.write(f"SWDA Intervention Dialog Act Accuracy: {interv_count/len(val_dataset):.4f}\n")
                f.write(f"SWDA Clean Dialog Act Accuracy: {clean_count/len(val_dataset):.4f}\n")
            else:
                if args.data_name in ["flower", "cub", "dtd"]:
                    f.write(f"Intervention Score:{interv_count/len(val_dataset)}\n")
                    f.write(f"Clean Score:{clean_count/len(val_dataset)}\n")
                else:
                    f.write(f"{args.data_name}_{args.experiment_name} Intervention Score:\n")
                    eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_interv.json", interv_answers)
                    f.write(f"{args.data_name}_{args.experiment_name} Clean Score:\n")
                    eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_clean.json", clean_answers)
            f.flush()

    print(f"[INFO] Evaluation results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_name", type=str, default="swda")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--eval_num_shot", type=int, default=0)
    parser.add_argument("--max_token", type=int, default=10)
    parser.add_argument("--source_bernoullis_path", type=str, required=True, help="Path to intervention locations from source act")
    parser.add_argument("--source_activation_path", type=str, required=True, help="Path to mean activations from source act")
    parser.add_argument("--source_act", type=str, required=True, help="Source speech act (where interventions were learned from)")
    parser.add_argument("--target_act", type=str, required=True, help="Target speech act (what we're evaluating on)")
    parser.add_argument("--is_eval", type=bool, default=False)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--zero_shot", type=bool, default=True)  # Default to zero-shot mode

    args = parser.parse_args()

    # If using SWDA, default to text-only model unless specified
    if args.data_name == "swda" and (args.model_name is None or args.model_name.lower() == "none"):
        args.model_name = "text"

    eval_cross_act_intervention(args) 