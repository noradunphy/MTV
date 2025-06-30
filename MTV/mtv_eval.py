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
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import TextModelHelper
# Import all classifiers
from backchannel_classifier import load_classifier as load_backchannel_classifier
from declarative_classifier import load_classifier as load_declarative_classifier
from statement_opinion_classifier import load_classifier as load_statement_opinion_classifier

def build_reference_sets(train_dataset, dialogue_acts, size_per_act, seed=42):
    """
    Build reference sets of utterances for each dialogue act from training data.
    
    Args:
        train_dataset: List of training examples
        dialogue_acts: List of dialogue acts to collect references for
        size_per_act: Number of reference utterances to sample per act
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping dialogue acts to lists of reference utterances
    """
    random.seed(seed)
    
    # Group training examples by dialogue act
    act_to_examples = {}
    for ex in train_dataset:
        act = ex.get('dialog_act', 'o')
        if act not in act_to_examples:
            act_to_examples[act] = []
        act_to_examples[act].append(ex)
    
    # Sample reference utterances for each act
    ref_sets = {}
    for act in dialogue_acts:
        if act not in act_to_examples or not act_to_examples[act]:
            print(f"[WARNING] No training examples found for act '{act}', using empty reference set")
            ref_sets[act] = []
            continue
            
        # Sample with replacement if we need more than we have
        examples = random.choices(act_to_examples[act], k=size_per_act) if size_per_act > len(act_to_examples[act]) else random.sample(act_to_examples[act], size_per_act)
        # Extract the response; fall back to other fields if empty
        utterances = []
        for ex in examples:
            utt = ex.get('response', '')
            if not utt:
                utt = ex.get('target_out', '') if 'target_out' in ex else ex.get('text', '')
            if utt:
                utterances.append(utt)
        ref_sets[act] = utterances

    # Print reference sets to file
    with open('reference_sets.txt', 'w') as f:
        f.write("\n[INFO] Reference sets:\n")
        for act, utterances in ref_sets.items():
            f.write(f"\n{act} ({len(utterances)} utterances):\n")
            for i, utt in enumerate(utterances, 1):
                f.write(f"  {i}. {utt}\n")
        f.write("\n") # Add blank line after reference sets
        
    return ref_sets

def eval_reinforce(args):
    # Create output file
    resume_suffix = "_resume" if args.resume else ""
    output_file_json = f"eval_results_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}_default_config.json"
    
    # Initialize results list to store data for each example
    results = []
    
    print(f"Evaluation Results for {args.model_name} on {args.data_name}")
    print(f"Target dialogue act: {args.dialogue_act if args.dialogue_act else 'all'}")

    # Open JSON file for incremental writing
    with open(output_file_json, 'w') as f:
        # Write the initial structure with empty results
        json.dump({
            "summary": {},  # Will be updated at the end
            "examples": []
        }, f, indent=2)
        f.write('\n')  # Add newline for easier reading

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

    # Randomly sample validation examples if max_val_examples is specified
    if args.max_val_examples is not None:
        val_dataset = random.sample(val_dataset, min(args.max_val_examples, len(val_dataset)))
        print(f"[INFO] Randomly sampled {len(val_dataset)} validation examples for evaluation")

    # Build reference sets for perplexity computation if requested
    ref_utterances = None
    if args.ppl_reference_size > 0:
        print(f"[INFO] Building reference sets with {args.ppl_reference_size} utterances per dialogue act...")
        dialogue_acts = ['sd', 'sv', 'b', 'aa', '%']  # Standard SWDA acts

        # Load full training data (no act filter) to gather references from all acts
        full_train_dataset = open_data(args.data_name, args.train_path, None)

        if args.data_name == "swda":
            full_train_dataset = [ex for ex in full_train_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < args.max_dialogue_length]

        ref_utterances = build_reference_sets(
            full_train_dataset,
            dialogue_acts,
            args.ppl_reference_size,
            seed=args.seed if hasattr(args, 'seed') else 42
        )
        print(f"[INFO] Built reference sets for {len(ref_utterances)} dialogue acts")
        for act, refs in ref_utterances.items():
            print(f"  • {act}: {len(refs)} references")

    activation_data = train_dataset
    reinforce_data = random.sample(train_dataset, min(100, len(train_dataset)))
    eval_data = val_dataset[:min(50, len(val_dataset))]

    print("[INFO] Loading model...")
    model_helper = load_model(args.model_name, args.data_name, zero_shot=args.zero_shot)
    print(f"[INFO] Model '{args.model_name}' loaded successfully!")
    
    # Load the classifier for the target dialogue act
    print("[INFO] Loading classifier...")
    classifier, classify_func = get_classifier(args.dialogue_act, contextual=False)
    print("[INFO] Classifier loaded!")
    
    if args.cur_mode != "clean":
        if args.resume:
            # Load existing activations and intervention locations
            print(f"[INFO] Resuming: Loading existing activations from {args.activation_path}...")
            mean_activations = torch.load(args.activation_path)
            print(f"[INFO] Loaded activations with shape: {mean_activations.shape}")
            
            print(f"[INFO] Resuming: Loading existing intervention locations from {args.bernoullis_path}...")
            intervention_locations = torch.load(args.bernoullis_path)
            print(f"[INFO] Loaded {len(intervention_locations)} intervention locations.")
        else:
            print("[INFO] Computing mean activations...")
            mean_activations = get_last_mean_head_activations(activation_data, model_helper, N_TRIALS = args.num_example, shot=args.num_shot)
            print("[INFO] Mean activations computed!")

            print("[INFO] Saving activations...")
            activation_save_path = args.activation_path
            if args.dialogue_act is not None:
                activation_save_path = f"{activation_save_path}_{args.dialogue_act}.pt"
            torch.save(mean_activations, activation_save_path)
            mean_activations = torch.load(activation_save_path)
            print(f"[INFO] Activations saved and loaded from {activation_save_path}!")

            print("[INFO] Starting reinforcement learning...")
            
            # Debug information
            print(f"[DEBUG] Model type: {type(model_helper).__name__}")
            print(f"[DEBUG] Number of layers: {model_helper.model_config['n_layers']}")
            print(f"[DEBUG] Number of heads: {model_helper.model_config['n_heads']}")
            print(f"[DEBUG] Mean activation shape: {mean_activations.shape}")

            bernoullis = reinforce(mean_activations, model_helper, reinforce_data, eval_data)
            print("[INFO] Reinforcement learning completed!")

            best_heads = (999, None)
            print("[INFO] Sampling best intervention locations...")
            for i in range(10):
                print(f"[DEBUG] Sampling intervention set {i+1}/10...")
                sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=0, max=1) for bernoulli in bernoullis])
                if args.model_name == "idefics2":
                    sigmoid_tensor = torch.nn.functional.threshold(sigmoid_tensor, 0.8, 0)
                prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
                sampled = prob_dist.sample()
                intervention_locations = reinforce_intervention_location(sampled)
                cur_heads_loss = validate_reinforce(model_helper, bernoullis, 1e-3, mean_activations, train_dataset[:min(50, len(train_dataset))], 0, sampled=sampled)
                print(f"[DEBUG] Intervention set {i+1} validation loss: {cur_heads_loss}")
                if cur_heads_loss < best_heads[0]:
                    best_heads = (cur_heads_loss, intervention_locations)
            bernoullis_save_path = args.bernoullis_path
            if args.dialogue_act is not None:
                bernoullis_save_path = f"{bernoullis_save_path}_{args.dialogue_act}.pt"
            torch.save(best_heads[1], bernoullis_save_path)
            intervention_locations = best_heads[1]
            print(f"[INFO] Best intervention locations saved to {bernoullis_save_path}.")

            intervention_locations = torch.load(bernoullis_save_path)
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
        
        clean_out, interv_out, clean_ppl, interv_ppl = fv_intervention_natural_text(
            new_input, 
            model_helper, 
            max_new_tokens=args.max_token, 
            return_item=args.cur_mode, 
            intervention_locations=intervention_locations, 
            avg_activations=mean_activations, 
            target_output=target_out,
            ref_utterances=ref_utterances,
            skip_generation=args.skip_generation
        )

        # Track perplexities and determine predicted acts
        if isinstance(clean_ppl, dict):
            # Using reference sets - find act with lowest perplexity
            clean_act = min(clean_ppl.items(), key=lambda x: x[1])[0] if clean_ppl else 'o'
            if args.cur_mode in ("interv", "both"):
                interv_act = min(interv_ppl.items(), key=lambda x: x[1])[0] if interv_ppl else 'o'
        else:
            # Original single-target behavior
            if clean_ppl is not None:
                clean_perplexities.append(clean_ppl)
            if interv_ppl is not None:
                interv_perplexities.append(interv_ppl)
                
            # Use classifier for act prediction if not skipping generation
            if not args.skip_generation:
                clean_out = extract_first_turn(clean_out)
                clean_act = classify_func(clean_out)
                
                if args.cur_mode in ("interv", "both"):
                    interv_out = extract_first_turn(interv_out)
                    interv_act = classify_func(interv_out)
            else:
                # When skipping generation, use the gold act (no generation to classify)
                target_act_local = item.get('dialog_act', 'o')
                clean_act = target_act_local
                interv_act = target_act_local

        # Get target dialog act and check correctness
        target_act = item.get('dialog_act', 'o')
        target_acts.append(target_act)
        
        clean_correct = int(clean_act == target_act)
        clean_count += clean_correct
        
        if args.cur_mode in ("interv", "both"):
            interv_correct = int(interv_act == target_act)
            interv_count += interv_correct
        else:
            interv_correct = None
            
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
            "clean_correct": clean_correct
        }
        
        # Add perplexity data based on method used
        if isinstance(clean_ppl, dict):
            example_data.update({
                "clean_perplexities_by_act": clean_ppl,
                "clean_chosen_act": clean_act,
                "clean_chosen_act_perplexity": clean_ppl.get(clean_act)
            })
        else:
            example_data["clean_perplexity"] = clean_ppl
            
        # Add intervention results if applicable
        if args.cur_mode in ("interv", "both"):
            example_data.update({
                "intervention_output": interv_out,
                "intervention_dialogue_act": interv_act,
                "intervention_correct": interv_correct
            })
            
            if isinstance(interv_ppl, dict):
                example_data.update({
                    "intervention_perplexities_by_act": interv_ppl,
                    "intervention_chosen_act": interv_act,
                    "intervention_chosen_act_perplexity": interv_ppl.get(interv_act)
                })
            else:
                example_data["intervention_perplexity"] = interv_ppl
                
            if clean_ppl is not None and interv_ppl is not None:
                if isinstance(clean_ppl, dict) and isinstance(interv_ppl, dict):
                    # Compare chosen act perplexities
                    clean_best_ppl = clean_ppl.get(clean_act)
                    interv_best_ppl = interv_ppl.get(interv_act)
                    if clean_best_ppl is not None and interv_best_ppl is not None:
                        example_data["perplexity_difference"] = interv_best_ppl - clean_best_ppl
                else:
                    example_data["perplexity_difference"] = interv_ppl - clean_ppl
        
        # Store answers for potential downstream eval (e.g., VQA datasets)
        interv_answers.append({"answer": interv_out, "question_id": question_id}) if args.cur_mode in ("interv", "both") else None
        clean_answers.append({"answer": clean_out, "question_id": question_id})
        
        # Write the current example to the JSON file
        with open(output_file_json, 'r+') as f:
            data = json.load(f)
            data["examples"].append(example_data)
            # Debug: confirm example appended
            if (idx % 25) == 0:
                print(f"[DEBUG] Writing example {idx+1}; total stored = {len(data['examples'])}")
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

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
        "ppl_reference_size": args.ppl_reference_size,
        "skip_generation": args.skip_generation,
        "clean_accuracy": clean_count / len(val_dataset) if len(val_dataset) > 0 else 0
    }
    
    if args.cur_mode in ("interv", "both"):
        summary["intervention_accuracy"] = interv_count / len(val_dataset) if len(val_dataset) > 0 else 0
    
    # Add perplexity statistics if using original method
    if not isinstance(clean_ppl, dict) and clean_perplexities:
        summary.update({
            "clean_perplexity_mean": np.mean(clean_perplexities),
            "clean_perplexity_std": np.std(clean_perplexities)
        })
        if args.cur_mode in ("interv", "both") and interv_perplexities:
            summary.update({
                "intervention_perplexity_mean": np.mean(interv_perplexities),
                "intervention_perplexity_std": np.std(interv_perplexities),
                "perplexity_difference_mean": np.mean(interv_perplexities) - np.mean(clean_perplexities)
            })

    # Update the summary in the JSON file
    with open(output_file_json, 'r+') as f:
        data = json.load(f)
        data["summary"] = summary
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    print(f"[INFO] Evaluation results saved to {output_file_json}")

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
    parser.add_argument("--bernoullis_path", type=str, default=None)
    parser.add_argument("--is_eval", type=bool, default=False)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--activation_path", type=str, default=None)
    parser.add_argument("--dialogue_act", type=str, default=None, help="Target dialogue act (e.g., 'sd')")
    parser.add_argument("--zero_shot", type=bool, default=False)
    parser.add_argument("--resume", action="store_true", help="Resume evaluation using existing bernoullis and activations files")
    parser.add_argument("--max_val_examples", type=int, default=250, help="Maximum number of validation examples to evaluate on (randomly sampled)")
    parser.add_argument("--ppl_reference_size", type=int, default=0, help="Number of reference utterances per dialogue act for perplexity computation (0 to use original method)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation and only compute perplexities")

    args = parser.parse_args()

    # If using SWDA, default to text-only model unless specified
    if args.data_name == "swda" and (args.model_name is None or args.model_name.lower() == "none"):
        args.model_name = "text"

    # If resuming, require the file paths
    if args.resume:
        if args.bernoullis_path is None or args.activation_path is None:
            print("ERROR: When using --resume, both --bernoullis_path and --activation_path must be specified")
            exit(1)

    eval_reinforce(args)

