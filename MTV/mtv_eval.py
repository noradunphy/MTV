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
# from backchannel_classifier import load_classifier as load_backchannel_classifier
# from new_classifiers.declarative_classifier import load_classifier as load_declarative_classifier
# from statement_opinion_classifier import load_classifier as load_statement_opinion_classifier


def eval_reinforce(args):
    # Create output file
    resume_suffix = "_resume" if args.resume else ""
    output_file_json = f"eval_results_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}_MCQ.json"
    
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

    print(f"[INFO] Loading full training data...")
    full_train_dataset = open_data(args.data_name, args.train_path, dialogue_act=None)
    full_train_dataset = [ex for ex in full_train_dataset if ex.get('text','').strip()]
    print(f"[INFO] Loaded {len(full_train_dataset)} total training examples.")

    print(f"[INFO] Loading training data for target act '{args.dialogue_act}'...")
    train_dataset = open_data(args.data_name, args.train_path, args.dialogue_act)
    print(f"[INFO] Loaded {len(train_dataset)} training examples for target act.")
    
    print(f"[INFO] Loading validation data from {args.val_path} for act '{args.dialogue_act}'...")
    val_dataset = open_data(args.data_name, args.val_path, getattr(args, 'dialogue_act', None))
    print(f"[INFO] Loaded {len(val_dataset)} validation examples.")

    # Filter for non-empty contexts (for SWDA and DailyDialog)
    if args.data_name in ("swda", "dailydialog"):
        train_dataset = [ex for ex in train_dataset if ex.get('text', '').strip()]
        val_dataset = [ex for ex in val_dataset if ex.get('text', '').strip()]
        print(f"[INFO] After filtering for non-empty contexts: {len(train_dataset)} training examples, {len(val_dataset)} validation examples")

    # Filter for shorter SWDA dialogues
    if args.data_name == "swda":
        train_dataset = [ex for ex in train_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < args.max_dialogue_length]
        val_dataset = [ex for ex in val_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < args.max_dialogue_length]
        print(f"[INFO] After filtering (max length {args.max_dialogue_length}): {len(train_dataset)} training examples, {len(val_dataset)} validation examples")

    # Randomly sample validation examples if max_val_examples is specified
    if args.max_val_examples is not None:
        val_dataset = random.sample(val_dataset, min(args.max_val_examples, len(val_dataset)))
        print(f"[INFO] Randomly sampled {len(val_dataset)} validation examples for evaluation")

    activation_data = train_dataset
    reinforce_data = random.sample(train_dataset, min(100, len(train_dataset)))
    eval_data = val_dataset[:min(50, len(val_dataset))]

    print("[INFO] Loading model...")
    model_helper = load_model(args.model_name, args.data_name, zero_shot=args.zero_shot)
    print(f"[INFO] Model '{args.model_name}' loaded successfully!")
    
    # # Load the classifier for the target dialogue act
    # print("[INFO] Loading classifier...")
    # classifier, classify_func = get_classifier(args.dialogue_act, contextual=False)
    # print("[INFO] Classifier loaded!")
    
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
            mean_activations = get_last_mean_head_activations(activation_data, model_helper, N_TRIALS = args.num_example, shot=args.num_shot, full_dataset=full_train_dataset)
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

            bernoullis = reinforce(mean_activations, model_helper, reinforce_data, eval_data, full_train_dataset)
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
                cur_heads_loss = validate_reinforce(model_helper, bernoullis, 1e-3, mean_activations, train_dataset[:min(50, len(train_dataset))], 0, sampled=sampled, full_dataset=full_train_dataset)
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

    # Track selections for patched-in category
    clean_patch_count = 0
    interv_patch_count = 0

    # Track overall multiple choice accuracy
    clean_correct_total = 0
    interv_correct_total = 0

    # ------------------------------------------------------------------
    # Prepare / reset prompt log file so each run starts fresh
    # ------------------------------------------------------------------
    prompt_log_file = (
        f"prompt_logs_{args.model_name}_{args.data_name}_"
        f"{args.dialogue_act if args.dialogue_act else 'all'}.txt"
    )
    # Opening in write mode truncates the file (or creates it) before we
    # enter the evaluation loop.  We'll continue to append inside the
    # loop for each example.
    with open(prompt_log_file, 'w', encoding='utf-8') as _f:
        header = (
            f"Evaluation run for model={args.model_name} dataset={args.data_name} "
            f"act={args.dialogue_act if args.dialogue_act else 'all'}\n"
            f"{'='*80}\n"
        )
        _f.write(header)

    print("\n[INFO] Starting evaluation loop over validation set...")
    for idx, item in enumerate(tqdm(val_dataset)):
        # Get the current dialogue and next caller for JSON output
        current_dialogue = item.get('text', '')
        next_caller = item.get('caller', '')
        
        # Use the format function (now defaults to multiple choice for SWDA)
        text, image_list, target_out, question_id = model_helper.format_func(
            filtered_dataset=train_dataset,      # filtered dataset for target act examples
            full_dataset=full_train_dataset,
            cur_item=item,
            num_shot=args.eval_num_shot,
            split="test",
            model_helper=model_helper
        )
        
        # Log the prompt to file for inspection
        #prompt_log_file = f"prompt_logs_{args.model_name}_{args.data_name}_{args.dialogue_act if args.dialogue_act else 'all'}.txt"
        with open(prompt_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"EXAMPLE {idx + 1}\n")
            f.write(f"Target Output: {target_out}\n")
            f.write(f"Target Dialogue Act: {item.get('dialog_act', 'unknown')}\n")
            f.write(f"Next Caller: {item.get('caller', 'unknown')}\n")
            f.write(f"{'='*80}\n")
            f.write("PROMPT:\n")
            f.write(text)
            f.write(f"\n{'='*80}\n\n")
            f.flush()  # Ensure prompt is written immediately
        
        new_input = model_helper.insert_image(text, image_list)
        
        
        clean_out, interv_out, clean_ppl, interv_ppl = fv_intervention_natural_text(
            new_input, 
            model_helper, 
            max_new_tokens=args.max_token, 
            return_item=args.cur_mode, 
            intervention_locations=intervention_locations, 
            avg_activations=mean_activations, 
            target_output=target_out,
            skip_generation=args.skip_generation
        )

        # Log the outputs to file
        with open(prompt_log_file, 'a', encoding='utf-8') as f:
            f.write("CLEAN OUTPUT:\n")
            f.write(clean_out)
            f.write(f"\n{'='*80}\n\n")
            f.write("INTERVENTION OUTPUT:\n")
            f.write(interv_out)
            f.write(f"\n{'='*80}\n\n")
            f.flush()  # Ensure outputs are written immediately

        # Track perplexities and determine predicted acts
        if args.data_name == "swda":
            # For SWDA, extract the first character from generated output (should be 1, 2, 3, 4, etc.)
            clean_output = extract_first_turn(clean_out).strip() if clean_out and clean_out.strip() else '1'
            clean_act = clean_output[0] if clean_output and clean_output[0].isdigit() else '1'
            
            if args.cur_mode in ("interv", "both"):
                interv_output = extract_first_turn(interv_out).strip() if interv_out and interv_out.strip() else '1'
                interv_act = interv_output[0] if interv_output and interv_output[0].isdigit() else '1'
        # else:
        #     # Fallback to original behavior for non-SWDA datasets
        #     if not args.skip_generation:
        #         clean_out = extract_first_turn(clean_out)
        #         clean_act = classify_func(clean_out)
                
        #         if args.cur_mode in ("interv", "both"):
        #             interv_out = extract_first_turn(interv_out)
        #             interv_act = classify_func(interv_out)
        #     else:
        #         target_act_local = item.get('dialog_act', 'o')
        #         clean_act = target_act_local
        #         interv_act = target_act_local

        # Get target dialog act and check correctness
        target_act = item.get('dialog_act', 'o')
        target_acts.append(target_act)
        
        # For SWDA, the target is the number, not the act
        if args.data_name == "swda":
            target_number = target_out  # target_out is already the number from format function
            clean_correct = clean_act == target_number
            if args.cur_mode in ("interv", "both"):
                interv_correct = interv_act == target_number
        else:
            # For other datasets, compare acts
            clean_correct = clean_act == target_act
            if args.cur_mode in ("interv", "both"):
                interv_correct = interv_act == target_act
        
        # Store results for this example
        example_data = {
            "example_id": idx + 1,
            "current_dialogue": current_dialogue,
            "next_caller": next_caller,
            "input_text": text,
            "target_output": target_out,
            "target_dialogue_act": target_act,
            "clean_dialogue_act": clean_act,
        }
        
        # Add perplexity data based on method used
        if isinstance(clean_ppl, dict):
            example_data.update({
                "clean_perplexities_by_act": clean_ppl,
                "clean_chosen_act": clean_act,
            })
            
        # Add intervention results if applicable
        if args.cur_mode in ("interv", "both"):
            example_data.update({
                "intervention_dialogue_act": interv_act,
            })
            
            if isinstance(interv_ppl, dict):
                example_data.update({
                    "intervention_perplexities_by_act": interv_ppl,
                    "intervention_chosen_act": interv_act,
                })
        
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

        # Track selections for patched-in category
        # For SWDA, track if the model selects the correct number (which corresponds to the target dialogue act)
        # For other datasets, use the existing logic
        if args.data_name == "swda":
            # In SWDA, the target_number corresponds to the correct dialogue act
            # We want to track if the intervention increases selection of the target dialogue act
            clean_patch_selected = clean_act == target_number
            if args.cur_mode in ("interv", "both"):
                interv_patch_selected = interv_act == target_number
            else:
                interv_patch_selected = False
                
            # Update example data
            example_data["clean_selected_correct"] = clean_patch_selected
            example_data["clean_selected_act"] = clean_act
            if args.cur_mode in ("interv", "both"):
                example_data["interv_selected_correct"] = interv_patch_selected
                example_data["interv_selected_act"] = interv_act
                
            if clean_patch_selected:
                clean_patch_count += 1
            if interv_patch_selected:
                interv_patch_count += 1


        # Update accuracy counts
        if clean_correct:
            clean_correct_total += 1
        if args.cur_mode in ("interv", "both") and interv_correct:
            interv_correct_total += 1

    print(f"\n[INFO] Multiple choice evaluation complete. Total examples: {len(val_dataset)}")
    
    # Calculate summary statistics
    summary = {
        "model_name": args.model_name,
        "data_name": args.data_name,
        "target_dialogue_act": args.dialogue_act if args.dialogue_act else "all",
        "evaluation_mode": args.cur_mode,
        "num_examples": len(val_dataset),
    }
    
    # Add patched category selection stats for multiple-choice
    if args.data_name == "swda":
        # For SWDA, track selection of correct dialogue act (target number)
        summary["clean_correct_act_selection_rate"] = clean_patch_count / len(val_dataset) if len(val_dataset) > 0 else 0
        summary["clean_correct_act_selections"] = clean_patch_count
        summary["total_examples"] = len(val_dataset)
        
        if args.cur_mode in ("interv", "both"):
            summary["intervention_correct_act_selection_rate"] = interv_patch_count / len(val_dataset) if len(val_dataset) > 0 else 0
            summary["intervention_correct_act_selections"] = interv_patch_count
            summary["correct_act_selection_rate_increase"] = summary["intervention_correct_act_selection_rate"] - summary["clean_correct_act_selection_rate"]
            summary["correct_act_selection_absolute_increase"] = interv_patch_count - clean_patch_count
    elif args.dialogue_act is not None:
        # Use existing logic for non-SWDA datasets with specific dialogue act
        update_multiple_choice_summary(
            summary,
            clean_patch_count,
            interv_patch_count,
            len(val_dataset),
            args
        )

    # Add overall accuracy to summary
    if args.data_name == "swda":
        summary["clean_number_accuracy"] = clean_correct_total / len(val_dataset) if len(val_dataset) > 0 else 0
        if args.cur_mode in ("interv", "both"):
            summary["intervention_number_accuracy"] = interv_correct_total / len(val_dataset) if len(val_dataset) > 0 else 0
            summary["number_accuracy_change"] = summary["intervention_number_accuracy"] - summary["clean_number_accuracy"]
    else:
        summary["clean_multiple_choice_accuracy"] = clean_correct_total / len(val_dataset) if len(val_dataset) > 0 else 0
        if args.cur_mode in ("interv", "both"):
            summary["intervention_multiple_choice_accuracy"] = interv_correct_total / len(val_dataset) if len(val_dataset) > 0 else 0
            summary["multiple_choice_accuracy_change"] = summary["intervention_multiple_choice_accuracy"] - summary["clean_multiple_choice_accuracy"]

    # Update the summary in the JSON file
    with open(output_file_json, 'r+') as f:
        data = json.load(f)
        data["summary"] = summary
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

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

