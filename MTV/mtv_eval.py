from mtv_utils import *
from models import *
from preprocess import *
# Original classifier import
# from speech_act_classifier import load_classifier
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
import pdb
logging.set_verbosity_error() 
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import TextModelHelper
# Import all classifiers
from backchannel_classifier import load_classifier as load_backchannel_classifier
from declarative_classifier import load_classifier as load_declarative_classifier
from statement_opinion_classifier import load_classifier as load_statement_opinion_classifier


def eval_reinforce(args):
    # Create output file
    output_file = f"eval_results_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}.txt"
    with open(output_file, 'w', buffering=1) as f:  # Line buffering for real-time updates
        f.write(f"Evaluation Results for {args.model_name} on {args.data_name}\n")
        f.write(f"Target dialogue act: {args.dialogue_act if args.dialogue_act else 'all'}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()  # Ensure initial header is written

        print(f"[INFO] Loading training data from {args.train_path} for act '{args.dialogue_act}'...")
        train_dataset = open_data(args.data_name, args.train_path, getattr(args, 'dialogue_act', None))
        f.write(f"[INFO] Loaded {len(train_dataset)} training examples.\n")
        f.flush()
        print(f"[INFO] Loading validation data from {args.val_path} for act '{args.dialogue_act}'...")
        val_dataset = open_data(args.data_name, args.val_path, getattr(args, 'dialogue_act', None))
        f.write(f"[INFO] Loaded {len(val_dataset)} validation examples.\n")
        f.flush()

        # Filter for shorter SWDA dialogues
        if args.data_name == "swda":
            train_dataset = [ex for ex in train_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
            val_dataset = [ex for ex in val_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
            f.write(f"[INFO] After filtering: {len(train_dataset)} training examples, {len(val_dataset)} validation examples\n")
            f.flush()

        activation_data = train_dataset
        reinforce_data = random.sample(train_dataset, min(100, len(train_dataset)))
        eval_data = val_dataset[:min(50, len(val_dataset))]

        print("[INFO] Loading model...")
        model_helper = load_model(args.model_name, args.data_name, zero_shot=args.zero_shot)
        f.write(f"[INFO] Model '{args.model_name}' loaded successfully!\n")
        f.flush()
        
        print("[INFO] Loading classifiers...")
        # Load all classifiers
        backchannel_classifier = load_backchannel_classifier()
        declarative_classifier = load_declarative_classifier()
        statement_opinion_classifier = load_statement_opinion_classifier()
        f.write("[INFO] Classifiers loaded!\n")
        f.flush()
        
        if args.cur_mode != "clean":
            print("[INFO] Computing mean activations...")
            mean_activations = get_last_mean_head_activations(activation_data, model_helper, N_TRIALS = args.num_example, shot=args.num_shot)
            f.write("[INFO] Mean activations computed!\n")
            f.flush()

            print("[INFO] Saving activations...")
            activation_save_path = args.activation_path
            if args.dialogue_act is not None:
                activation_save_path = f"{activation_save_path}_{args.dialogue_act}.pt"
            torch.save(mean_activations, activation_save_path)
            mean_activations = torch.load(activation_save_path)
            f.write(f"[INFO] Activations saved and loaded from {activation_save_path}!\n")
            f.flush()

            print("[INFO] Starting reinforcement learning...")
            
            # Debug information
            f.write("\n[DEBUG] Model Configuration:\n")
            f.write(f"[DEBUG] Model type: {type(model_helper).__name__}\n")
            f.write(f"[DEBUG] Model config: {model_helper.model_config}\n")
            f.write(f"[DEBUG] Number of layers: {model_helper.model_config['n_layers']}\n")
            f.write(f"[DEBUG] Number of heads: {model_helper.model_config['n_heads']}\n")
            f.flush()

            f.write("\n[DEBUG] Data Statistics:\n")
            f.write(f"[DEBUG] Number of training examples: {len(reinforce_data)}\n")
            f.write(f"[DEBUG] Number of validation examples: {len(eval_data)}\n")
            f.write("[DEBUG] Example input format:\n")
            sample_item = reinforce_data[0]
            text, image_list, target_out, question_id = model_helper.format_func(train_dataset, sample_item, num_shot=args.num_shot)
            f.write(f"{text}...\n")
            f.write(f"[DEBUG] Has images: {len(image_list) > 0}\n")
            f.write(f"[DEBUG] Target output: {target_out}\n")
            f.flush()

            f.write("\n[DEBUG] Activation Statistics:\n")
            f.write(f"[DEBUG] Mean activation shape: {mean_activations.shape}\n")
            f.write(f"[DEBUG] Mean activation range: [{mean_activations.min():.4f}, {mean_activations.max():.4f}]\n")
            f.write(f"[DEBUG] Mean activation mean: {mean_activations.mean():.4f}\n")
            f.write(f"[DEBUG] Mean activation std: {mean_activations.std():.4f}\n")
            f.flush()

            bernoullis = reinforce(mean_activations, model_helper, reinforce_data, eval_data)
            f.write("[INFO] Reinforcement learning completed!\n")
            f.flush()

            best_heads = (999, None)
            f.write("[INFO] Sampling best intervention locations...\n")
            f.flush()
            for i in range(10):
                f.write(f"[DEBUG] Sampling intervention set {i+1}/10...\n")
                f.flush()
                sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=0, max=1) for bernoulli in bernoullis])
                if args.model_name == "idefics2":
                    sigmoid_tensor = torch.nn.functional.threshold(sigmoid_tensor, 0.8, 0)
                prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
                sampled = prob_dist.sample()
                intervention_locations = reinforce_intervention_location(sampled)
                cur_heads_loss = validate_reinforce(model_helper, bernoullis, 1e-3, mean_activations, train_dataset[:min(50, len(train_dataset))], 0, sampled=sampled)
                f.write(f"[DEBUG] Intervention set {i+1} validation loss: {cur_heads_loss}\n")
                f.flush()
                if cur_heads_loss < best_heads[0]:
                    best_heads = (cur_heads_loss, intervention_locations)
            bernoullis_save_path = args.bernoullis_path
            if args.dialogue_act is not None:
                bernoullis_save_path = f"{bernoullis_save_path}_{args.dialogue_act}.pt"
            torch.save(best_heads[1], bernoullis_save_path)
            intervention_locations = best_heads[1]
            f.write(f"[INFO] Best intervention locations saved to {bernoullis_save_path}.\n")
            f.flush()

            intervention_locations = torch.load(bernoullis_save_path)
            f.write(f"[INFO] Loaded {len(intervention_locations)} intervention locations.\n")
            f.flush()
        else:
            mean_activations = None
            intervention_locations = None
            f.write("[INFO] Running in clean mode: no activations or interventions will be used.\n")
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
            text, image_list, target_out, question_id = model_helper.format_func(train_dataset, item, num_shot=args.eval_num_shot)
            new_input = model_helper.insert_image(text, image_list)
            clean_out, interv_out, clean_ppl, interv_ppl = fv_intervention_natural_text(new_input, model_helper, max_new_tokens=args.max_token, return_item=args.cur_mode, intervention_locations=intervention_locations, avg_activations=mean_activations)

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

            # Classify generated responses based on target act
            if target_act == 'sd':
                # Use declarative classifier for sd acts
                clean_is_declarative = declarative_classifier.classify_utterance(clean_out)
                interv_is_declarative = declarative_classifier.classify_utterance(interv_out)
                clean_act = 'sd' if clean_is_declarative else 'o'
                interv_act = 'sd' if interv_is_declarative else 'o'
            elif target_act == 'sv':
                # Use statement opinion classifier for sv acts
                clean_is_statement_opinion = statement_opinion_classifier.classify_utterance(clean_out)
                interv_is_statement_opinion = statement_opinion_classifier.classify_utterance(interv_out)
                clean_act = 'sv' if clean_is_statement_opinion else 'o'
                interv_act = 'sv' if interv_is_statement_opinion else 'o'
            else:
                # Use backchannel classifier for other acts
                clean_is_backchannel = backchannel_classifier.classify_utterance(clean_out)
                interv_is_backchannel = backchannel_classifier.classify_utterance(interv_out)
                clean_act = 'b' if clean_is_backchannel else 'o'
                interv_act = 'b' if interv_is_backchannel else 'o'

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
        confusion_matrix_path = f"confusion_matrices_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}.png"
        plt.savefig(confusion_matrix_path)
        f.write(f"[INFO] Confusion matrices saved to '{confusion_matrix_path}'\n")
        f.flush()

        if args.is_eval:
            if args.data_name == "swda":
                if args.cur_mode == "interv" or args.cur_mode == "both":
                    f.write(f"SWDA Intervention Dialog Act Accuracy: {interv_count/len(val_dataset):.4f}\n")
                if args.cur_mode == "clean" or args.cur_mode == "both":
                    f.write(f"SWDA Clean Dialog Act Accuracy: {clean_count/len(val_dataset):.4f}\n")
            else:
                if args.cur_mode == "interv" or args.cur_mode == "both":
                    if args.data_name == "flower" or args.data_name =="cub" or args.data_name == "dtd":
                        f.write(f"Intervention Score:{interv_count/len(val_dataset)}\n")
                    else:
                        f.write(f"{args.data_name}_{args.experiment_name} Intervention Score:\n")
                        eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_interv.json", interv_answers)
                if args.cur_mode == "clean" or args.cur_mode == "both":
                    if args.data_name == "flower" or args.data_name =="cub" or args.data_name == "dtd":
                        f.write(f"Clean Score:{clean_count/len(val_dataset)}\n")
                    else:
                        f.write(f"{args.data_name}_{args.experiment_name} Clean Score:\n")
                        eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_clean.json", clean_answers)
            f.flush()

    print(f"[INFO] Evaluation results saved to {output_file}")


def compute_perplexity(model_helper, input_text, target_text):
    if input_text is None or target_text is None:
        return None
        
    # Combine input and target text
    full_text = input_text + " " + target_text
    
    # Tokenize the text
    inputs = model_helper.tokenizer(full_text, return_tensors="pt").to(model_helper.device)
    
    # Compute loss
    with torch.no_grad():
        outputs = model_helper.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    # Compute perplexity
    perplexity = torch.exp(loss).item()
    
    return perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_name", type=str, default="vizwiz")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--num_example", type=int, default=100)
    parser.add_argument("--num_shot", type=int, default=4)
    parser.add_argument("--eval_num_shot", type=int, default=0)
    parser.add_argument("--max_token", type=int, default=10)
    parser.add_argument("--bernoullis_path", type=str, default=None)
    parser.add_argument("--is_eval", type=bool, default=False)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--activation_path", type=str, default=None)
    parser.add_argument("--dialogue_act", type=str, default=None, help="Target dialogue act (e.g., 'sd')")
    parser.add_argument("--zero_shot", type=bool, default=False)

    args = parser.parse_args()

    # If using SWDA, default to text-only model unless specified
    if args.data_name == "swda" and (args.model_name is None or args.model_name.lower() == "none"):
        args.model_name = "text"

    eval_reinforce(args)

