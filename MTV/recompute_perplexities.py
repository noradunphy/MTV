from mtv_utils import *
from models import *
from preprocess import *
from tqdm import tqdm
import torch
import argparse
import numpy as np
import json
import os
import pandas as pd
import pdb
import random

def recompute_perplexities(args):
    # Create output file
    output_file_json = f"recomputed_perplexities_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}.json"
    
    # Initialize results list to store data for each example
    results = []
    
    # Log basic info to console
    print(f"[INFO] Recomputing Perplexities for {args.model_name} on {args.data_name}")
    print(f"[INFO] Target dialogue act: {args.dialogue_act if args.dialogue_act else 'all'}")
    
    print(f"[INFO] Loading training data from {args.train_path} for act '{args.dialogue_act}'...")
    train_dataset = open_data(args.data_name, args.train_path, getattr(args, 'dialogue_act', None))
    print(f"[INFO] Loaded {len(train_dataset)} training examples.")
    
    print(f"[INFO] Loading validation data from {args.val_path} for act '{args.dialogue_act}'...")
    val_dataset = open_data(args.data_name, args.val_path, getattr(args, 'dialogue_act', None))
    print(f"[INFO] Loaded {len(val_dataset)} validation examples.")
    
    # Filter for shorter SWDA dialogues if needed
    if args.data_name == "swda":
        train_dataset = [ex for ex in train_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
        val_dataset = [ex for ex in val_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
        print(f"[INFO] After filtering: {len(train_dataset)} training examples, {len(val_dataset)} validation examples")
    
    print("[INFO] Loading model...")
    model_helper = load_model(args.model_name, args.data_name, zero_shot=args.zero_shot)
    print(f"[INFO] Model '{args.model_name}' loaded successfully!")
    
    # Load the classifier for the target dialogue act
    print("[INFO] Loading classifier...")
    classifier, classify_func = get_classifier(args.dialogue_act or 'b', contextual=False)
    print("[INFO] Classifier loaded!")
    
    # Load saved activations and bernoullis
    print("[INFO] Loading saved activations and bernoullis...")
    mean_activations = torch.load(args.activation_path)
    intervention_locations = torch.load(args.bernoullis_path)
    print(f"[INFO] Loaded activations and {len(intervention_locations)} intervention locations.")
    
    # Print model configuration details
    print(f"[INFO] Model configuration:")
    print(f"Number of attention heads: {model_helper.model_config['n_heads']}")
    print(f"Hidden dimension per head: {model_helper.model_config['resid_dim']//model_helper.model_config['n_heads']}")
    print(f"Total hidden dimension: {model_helper.model_config['resid_dim']}")

    # Print activation dimensions
    print(f"\n[INFO] Activation dimensions:")
    print(f"Mean activations shape: {mean_activations.shape}")
    print(f"Number of intervention locations: {len(intervention_locations)}")

    # Initialize tracking lists
    clean_perplexities = []
    interv_perplexities = []
    target_acts = []
    clean_pred_acts = []
    interv_pred_acts = []
    
    print("[INFO] Starting evaluation loop over validation set...")
    val_dataset = val_dataset[:min(50, len(val_dataset))]
    
    for idx, item in enumerate(tqdm(val_dataset)):
        # Get the current dialogue and next caller for JSON output
        current_dialogue = item.get('text', '')
        next_caller = item.get('caller', '')
        
        # Use ICL for model input
        text, image_list, target_out, question_id = model_helper.format_func(train_dataset, item, num_shot=args.eval_num_shot)
        new_input = model_helper.insert_image(text, image_list)
        
        # Get clean and intervention outputs with perplexities
        print(f"[DEBUG] Target output: {target_out}")
        clean_out, interv_out, clean_ppl, interv_ppl = fv_intervention_natural_text(
            new_input, 
            model_helper, 
            max_new_tokens=args.max_token, 
            return_item="both",  # Always compute both
            intervention_locations=intervention_locations, 
            avg_activations=mean_activations,
            target_output=target_out
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
        
        # Use the generalizable classifier function
        clean_act = classify_func(clean_out)
        interv_act = classify_func(interv_out)
            
        clean_pred_acts.append(clean_act)
        interv_pred_acts.append(interv_act)
        
        # Calculate perplexity difference
        ppl_diff = interv_ppl - clean_ppl if (clean_ppl is not None and interv_ppl is not None) else None
        
        # Store results for this example - only include current dialogue in output
        example_data = {
            "current_dialogue": current_dialogue,
            "next_caller": next_caller,
            "target_output": target_out,
            "target_dialogue_act": target_act,
            "clean_output": clean_out,
            "clean_dialogue_act": clean_act,
            "clean_perplexity": clean_ppl,
            "intervention_output": interv_out,
            "intervention_dialogue_act": interv_act,
            "intervention_perplexity": interv_ppl,
            "perplexity_difference": ppl_diff
        }
        
        results.append(example_data)
    
    # Calculate summary statistics
    summary = {
        "model_name": args.model_name,
        "data_name": args.data_name,
        "target_dialogue_act": args.dialogue_act if args.dialogue_act else "all",
        "num_examples": len(results),
        "clean_perplexity_mean": np.mean(clean_perplexities) if clean_perplexities else None,
        "clean_perplexity_std": np.std(clean_perplexities) if clean_perplexities else None,
        "intervention_perplexity_mean": np.mean(interv_perplexities) if interv_perplexities else None,
        "intervention_perplexity_std": np.std(interv_perplexities) if interv_perplexities else None,
        "perplexity_difference_mean": np.mean(interv_perplexities) - np.mean(clean_perplexities) 
                                      if (clean_perplexities and interv_perplexities) else None
    }
    
    # Save results to JSON
    output_data = {
        "summary": summary,
        "examples": results
    }
    
    with open(output_file_json, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"[INFO] Results saved to {output_file_json}")
    
    # # Also create CSV for easy import to Google Sheets
    # df = pd.DataFrame(results)
    # csv_file = f"recomputed_perplexities_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}.csv"
    # df.to_csv(csv_file, index=False)
    # print(f"[INFO] CSV saved to {csv_file}")
    
    
    
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