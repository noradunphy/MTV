from mtv_utils import *
from models import *
from preprocess import *
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


def eval_reinforce_resume(args):
    # Create output file
    output_file_json = f"eval_results_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}_maxlen{args.max_dialogue_length}_resume.json"
    
    # Initialize results list to store data for each example
    results = []
    
    print(f"Resuming Evaluation Results for {args.model_name} on {args.data_name}")
    print(f"Target dialogue act: {args.dialogue_act if args.dialogue_act else 'all'}")

    print(f"[INFO] Loading training data from {args.train_path} for act '{args.dialogue_act}'...")
    train_dataset = open_data(args.data_name, args.train_path, getattr(args, 'dialogue_act', None))
    print(f"[INFO] Loaded {len(train_dataset)} training examples.")
    
    print(f"[INFO] Loading validation data from {args.val_path} for act '{args.dialogue_act}'...")
    val_dataset = open_data(args.data_name, args.val_path, getattr(args, 'dialogue_act', None))
    print(f"[INFO] Loaded {len(val_dataset)} validation examples.")

    # Filter for shorter SWDA dialogues
    if args.data_name == "swda":
        train_dataset = [ex for ex in train_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < args.max_dialogue_length]
        val_dataset = [ex for ex in val_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < args.max_dialogue_length]
        print(f"[INFO] After filtering (max length {args.max_dialogue_length}): {len(train_dataset)} training examples, {len(val_dataset)} validation examples")

    print("[INFO] Loading model...")
    model_helper = load_model(args.model_name, args.data_name, zero_shot=args.zero_shot)
    print(f"[INFO] Model '{args.model_name}' loaded successfully!")
    
    # Load the classifier for the target dialogue act
    print("[INFO] Loading classifier...")
    classifier, classify_func = get_classifier(args.dialogue_act, contextual=False)
    print("[INFO] Classifier loaded!")
    
    # Load existing activations and intervention locations
    if args.cur_mode != "clean":
        print(f"[INFO] Loading existing activations from {args.activation_path}...")
        mean_activations = torch.load(args.activation_path)
        print(f"[INFO] Loaded activations with shape: {mean_activations.shape}")
        
        print(f"[INFO] Loading existing intervention locations from {args.bernoullis_path}...")
        intervention_locations = torch.load(args.bernoullis_path)
        print(f"[INFO] Loaded {len(intervention_locations)} intervention locations.")
    else:
        mean_activations = None
        intervention_locations = None
        print("[INFO] Running in clean mode: no activations or interventions will be used.")

    clean_answers = []
    interv_answers = []
    clean_count, interv_count = 0, 0
    target_acts = []
    clean_pred_acts = []
    interv_pred_acts = []
    
    # Add perplexity tracking
    clean_perplexities = []
    interv_perplexities = []

    print("\n[INFO] Starting evaluation loop over validation set...")
    for idx, item in enumerate(tqdm(val_dataset)):
        # Get the current dialogue and next caller for JSON output
        current_dialogue = item.get('text', '')
        next_caller = item.get('caller', '')
        
        text, image_list, target_out, question_id = model_helper.format_func(train_dataset, item, num_shot=args.eval_num_shot)
        new_input = model_helper.insert_image(text, image_list)
        
        # Debug: Check if target_out is empty
        if not target_out or target_out.strip() == "":
            print(f"[WARNING] Empty target output for example {idx+1}. Skipping perplexity computation.")
            target_out = " "  # Use a space as fallback to avoid tokenizer error
        
        clean_out, interv_out, clean_ppl, interv_ppl = fv_intervention_natural_text(new_input, model_helper, max_new_tokens=args.max_token, return_item=args.cur_mode, intervention_locations=intervention_locations, avg_activations=mean_activations, target_output=target_out)

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

        # Use the generalizable classifier function
        clean_act = classify_func(clean_out)
        interv_act = classify_func(interv_out)

        clean_pred_acts.append(clean_act)
        interv_pred_acts.append(interv_act)

        # Store answers for evaluation
        interv_answers.append({"answer": interv_out, "question_id": question_id})
        clean_answers.append({"answer": clean_out, "question_id": question_id})

        clean_correct = int(clean_act == target_act)
        interv_correct = int(interv_act == target_act)
        clean_count += clean_correct
        interv_count += interv_correct
        
        # Calculate perplexity difference
        ppl_diff = interv_ppl - clean_ppl if (clean_ppl is not None and interv_ppl is not None) else None
        
        # Store results for this example
        example_data = {
            "example_id": idx + 1,
            "current_dialogue": current_dialogue,
            "next_caller": next_caller,
            "input_text": text,
            "target_output": target_out,
            "target_dialogue_act": target_act,
            "clean_output": clean_out,
            "clean_dialogue_act": clean_act,
            "clean_correct": clean_correct,
            "clean_perplexity": clean_ppl,
            "intervention_output": interv_out,
            "intervention_dialogue_act": interv_act,
            "intervention_correct": interv_correct,
            "intervention_perplexity": interv_ppl,
            "perplexity_difference": ppl_diff
        }
        
        results.append(example_data)

    print(f"\n[INFO] Evaluation complete. Clean correct: {clean_count}, Interv correct: {interv_count}, Total: {len(val_dataset)}")
    
    # Calculate summary statistics
    summary = {
        "model_name": args.model_name,
        "data_name": args.data_name,
        "target_dialogue_act": args.dialogue_act if args.dialogue_act else "all",
        "evaluation_mode": args.cur_mode,
        "max_dialogue_length": args.max_dialogue_length,
        "num_examples": len(val_dataset),
        "num_shot": args.num_shot,
        "eval_num_shot": args.eval_num_shot,
        "max_tokens": args.max_token,
        "clean_accuracy": clean_count / len(val_dataset) if len(val_dataset) > 0 else 0,
        "intervention_accuracy": interv_count / len(val_dataset) if len(val_dataset) > 0 else 0,
        "clean_perplexity_mean": np.mean(clean_perplexities) if clean_perplexities else None,
        "clean_perplexity_std": np.std(clean_perplexities) if clean_perplexities else None,
        "intervention_perplexity_mean": np.mean(interv_perplexities) if interv_perplexities else None,
        "intervention_perplexity_std": np.std(interv_perplexities) if interv_perplexities else None,
        "perplexity_difference_mean": np.mean(interv_perplexities) - np.mean(clean_perplexities) 
                                      if (clean_perplexities and interv_perplexities) else None
    }

    if args.is_eval:
        if args.data_name == "swda":
            if args.cur_mode == "interv" or args.cur_mode == "both":
                summary["swda_intervention_accuracy"] = interv_count/len(val_dataset)
            if args.cur_mode == "clean" or args.cur_mode == "both":
                summary["swda_clean_accuracy"] = clean_count/len(val_dataset)
        else:
            if args.cur_mode == "interv" or args.cur_mode == "both":
                if args.data_name == "flower" or args.data_name =="cub" or args.data_name == "dtd":
                    summary["intervention_score"] = interv_count/len(val_dataset)
                else:
                    eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_interv.json", interv_answers)
            if args.cur_mode == "clean" or args.cur_mode == "both":
                if args.data_name == "flower" or args.data_name =="cub" or args.data_name == "dtd":
                    summary["clean_score"] = clean_count/len(val_dataset)
                else:
                    eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_clean.json", clean_answers)

    # Save results to JSON
    output_data = {
        "summary": summary,
        "examples": results
    }
    
    with open(output_file_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[INFO] Evaluation results saved to {output_file_json}")


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
    parser.add_argument("--max_dialogue_length", type=int, default=100, help="Maximum dialogue length (in words) for filtering SWDA data")
    parser.add_argument("--bernoullis_path", type=str, required=True, help="Path to existing bernoullis file")
    parser.add_argument("--activation_path", type=str, required=True, help="Path to existing activations file")
    parser.add_argument("--is_eval", type=bool, default=False)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--dialogue_act", type=str, default=None, help="Target dialogue act (e.g., 'sd')")
    parser.add_argument("--zero_shot", type=bool, default=False)

    args = parser.parse_args()

    # If using SWDA, default to text-only model unless specified
    if args.data_name == "swda" and (args.model_name is None or args.model_name.lower() == "none"):
        args.model_name = "text"

    eval_reinforce_resume(args) 