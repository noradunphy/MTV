#!/usr/bin/env python3
"""
Test file to verify that greedy decoding is actually taking the most probable token every time.
This test mirrors the way generate() is called in the actual mtv_eval pipeline.

The test checks that for each generated token position, the argmax of the scores 
corresponds to the index of the token which was actually generated.
"""

import torch
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
from mtv_utils import load_model, fv_intervention_natural_text, open_data
from models import TextModelHelper
from baukit import TraceDict


def verify_greedy_decoding_single_example(model_helper, model_input, max_new_tokens, intervention_fn=None):
    """
    Verify greedy decoding for a single example by checking that argmax of scores 
    corresponds to the generated token at each position.
    
    Args:
        model_helper: ModelHelper instance
        model_input: Tokenized input to model
        max_new_tokens: Maximum tokens to generate
        intervention_fn: Optional intervention function
        
    Returns:
        dict: Results of the verification
    """
    model = model_helper.model
    tokenizer = model_helper.tokenizer
    device = next(model.parameters()).device
    
    # Get the original input length
    input_len = model_input["input_ids"].size(1)
    
    # Generate with scores
    if intervention_fn is not None:
        # Intervention branch - wrap model forward pass with hook
        with TraceDict(model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn):
            with torch.no_grad():
                gen_text, scores = model_helper.generate(
                    model_input,
                    max_new_tokens=max_new_tokens,
                    return_scores=True,
                    return_dict_in_generate=True
                )
    else:
        # Clean branch - no intervention
        with torch.no_grad():
            gen_text, scores = model_helper.generate(
                model_input,
                max_new_tokens=max_new_tokens,
                return_scores=True,
                return_dict_in_generate=True
            )
    
    # Tokenize the generated text to get the actual token IDs
    gen_tokens_enc = tokenizer(gen_text, return_tensors="pt", add_special_tokens=False)
    generated_token_ids = gen_tokens_enc["input_ids"][0]
    
    # Check if we have scores and they match the expected length
    if scores is None:
        return {
            "success": False,
            "error": "No scores returned from generate()",
            "generated_text": gen_text,
            "generated_tokens": generated_token_ids.tolist(),
            "scores_length": 0
        }
    
    # Verify that scores length matches generated tokens length
    if len(scores) != len(generated_token_ids):
        return {
            "success": False,
            "error": f"Scores length ({len(scores)}) doesn't match generated tokens length ({len(generated_token_ids)})",
            "generated_text": gen_text,
            "generated_tokens": generated_token_ids.tolist(),
            "scores_length": len(scores)
        }
    
    # Check each position
    verification_results = []
    all_greedy_correct = True
    
    for pos in range(len(scores)):
        # Get the scores for this position (last dimension of the score tensor)
        position_scores = scores[pos][0, -1, :]  # Shape: [vocab_size]
        
        # Get the argmax (most probable token)
        predicted_token_id = torch.argmax(position_scores).item()
        
        # Get the actual generated token at this position
        actual_token_id = generated_token_ids[pos].item()
        
        # Check if they match
        is_greedy_correct = predicted_token_id == actual_token_id
        
        # Get the probability of the actual token
        actual_token_prob = torch.softmax(position_scores, dim=-1)[actual_token_id].item()
        predicted_token_prob = torch.softmax(position_scores, dim=-1)[predicted_token_id].item()
        
        # Get top-5 tokens for debugging
        top_5_probs, top_5_ids = torch.topk(torch.softmax(position_scores, dim=-1), 5)
        top_5_tokens = [tokenizer.decode([tid]) for tid in top_5_ids.tolist()]
        
        verification_results.append({
            "position": pos,
            "predicted_token_id": predicted_token_id,
            "actual_token_id": actual_token_id,
            "predicted_token": tokenizer.decode([predicted_token_id]),
            "actual_token": tokenizer.decode([actual_token_id]),
            "is_greedy_correct": is_greedy_correct,
            "actual_token_prob": actual_token_prob,
            "predicted_token_prob": predicted_token_prob,
            "top_5_tokens": list(zip(top_5_tokens, top_5_probs.tolist()))
        })
        
        if not is_greedy_correct:
            all_greedy_correct = False
    
    return {
        "success": True,
        "all_greedy_correct": all_greedy_correct,
        "generated_text": gen_text,
        "generated_tokens": generated_token_ids.tolist(),
        "scores_length": len(scores),
        "verification_results": verification_results
    }


def test_greedy_decoding_pipeline(args):
    """
    Test greedy decoding using the same pipeline as mtv_eval.
    """
    print(f"[INFO] Testing greedy decoding for model: {args.model_name}")
    print(f"[INFO] Dataset: {args.data_name}")
    print(f"[INFO] Max tokens: {args.max_token}")
    print(f"[INFO] Number of examples: {args.num_examples}")
    
    # Load model
    print("[INFO] Loading model...")
    model_helper = load_model(args.model_name, args.data_name, zero_shot=args.zero_shot)
    print(f"[INFO] Model '{args.model_name}' loaded successfully!")
    
    # Load data
    print("[INFO] Loading data...")
    train_dataset = open_data(args.data_name, args.train_path, args.dialogue_act)
    val_dataset = open_data(args.data_name, args.val_path, args.dialogue_act)
    
    # Filter by dialogue length if specified
    if args.data_name == "swda" and args.max_dialogue_length:
        train_dataset = [ex for ex in train_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < args.max_dialogue_length]
        val_dataset = [ex for ex in val_dataset if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < args.max_dialogue_length]
    
    # Randomly sample examples
    random.seed(args.seed)
    val_dataset = random.sample(val_dataset, min(args.num_examples, len(val_dataset)))
    print(f"[INFO] Using {len(val_dataset)} validation examples")
    
    # Load intervention data if testing intervention
    intervention_locations = None
    mean_activations = None
    if args.cur_mode in ("interv", "both"):
        print("[INFO] Loading intervention data...")
        if args.activation_path:
            mean_activations = torch.load(args.activation_path, map_location='cpu')
            # Sample intervention locations (this is a simplified version)
            intervention_locations = [(0, 0, -1)]  # Example: layer 0, head 0, last token
        else:
            print("[WARNING] No activation path provided, skipping intervention tests")
            args.cur_mode = "clean"
    
    # Results storage
    clean_results = []
    intervention_results = []
    
    print(f"\n[INFO] Starting greedy decoding verification...")
    for idx, item in enumerate(tqdm(val_dataset)):
        # Format input the same way as in mtv_eval
        text, image_list, target_out, question_id = model_helper.format_func(train_dataset, item, num_shot=args.eval_num_shot)
        new_input = model_helper.insert_image(text, image_list)
        
        # Debug: Check if target_out is empty
        if not target_out or target_out.strip() == "":
            print(f"[WARNING] Empty target output for example {idx+1}. Skipping.")
            continue
        
        # Test clean generation
        print(f"\n[TEST {idx+1}] Testing CLEAN generation...")
        clean_result = verify_greedy_decoding_single_example(
            model_helper, 
            new_input, 
            max_new_tokens=args.max_token
        )
        clean_results.append(clean_result)
        
        # Test intervention generation if requested
        if args.cur_mode in ("interv", "both") and intervention_locations is not None:
            print(f"[TEST {idx+1}] Testing INTERVENTION generation...")
            
            # Create intervention function
            from mtv_utils import last_replace_activation_w_avg
            interv_fn = last_replace_activation_w_avg(
                layer_head_token_pairs=intervention_locations,
                avg_activations=mean_activations,
                model=model_helper.model,
                model_config=model_helper.model_config,
                batched_input=False,
                last_token_only=True,
                split_idx=model_helper.split_idx
            )
            
            intervention_result = verify_greedy_decoding_single_example(
                model_helper, 
                new_input, 
                max_new_tokens=args.max_token,
                intervention_fn=interv_fn
            )
            intervention_results.append(intervention_result)
        
        # Print summary for this example
        print(f"  Clean: {'✓' if clean_result['success'] and clean_result['all_greedy_correct'] else '✗'}")
        if args.cur_mode in ("interv", "both") and intervention_locations is not None:
            print(f"  Intervention: {'✓' if intervention_result['success'] and intervention_result['all_greedy_correct'] else '✗'}")
        
        # Early stopping for debugging
        if args.max_examples and idx >= args.max_examples - 1:
            break
    
    # Analyze results
    print(f"\n[INFO] Analysis complete!")
    
    # Clean results analysis
    clean_successful = sum(1 for r in clean_results if r['success'])
    clean_greedy_correct = sum(1 for r in clean_results if r['success'] and r['all_greedy_correct'])
    
    print(f"\nCLEAN GENERATION RESULTS:")
    print(f"  Total examples: {len(clean_results)}")
    print(f"  Successful generations: {clean_successful}")
    print(f"  All positions greedy correct: {clean_greedy_correct}")
    print(f"  Greedy accuracy: {clean_greedy_correct/len(clean_results)*100:.2f}%")
    
    # Intervention results analysis
    if args.cur_mode in ("interv", "both") and intervention_locations is not None:
        interv_successful = sum(1 for r in intervention_results if r['success'])
        interv_greedy_correct = sum(1 for r in intervention_results if r['success'] and r['all_greedy_correct'])
        
        print(f"\nINTERVENTION GENERATION RESULTS:")
        print(f"  Total examples: {len(intervention_results)}")
        print(f"  Successful generations: {interv_successful}")
        print(f"  All positions greedy correct: {interv_greedy_correct}")
        print(f"  Greedy accuracy: {interv_greedy_correct/len(intervention_results)*100:.2f}%")
    
    # Save detailed results
    output_file = f"greedy_decoding_test_{args.model_name}_{args.data_name}.json"
    results_summary = {
        "model_name": args.model_name,
        "data_name": args.data_name,
        "max_tokens": args.max_token,
        "num_examples": len(clean_results),
        "clean_results": {
            "total": len(clean_results),
            "successful": clean_successful,
            "greedy_correct": clean_greedy_correct,
            "greedy_accuracy": clean_greedy_correct/len(clean_results)*100 if clean_results else 0
        }
    }
    
    if args.cur_mode in ("interv", "both") and intervention_locations is not None:
        results_summary["intervention_results"] = {
            "total": len(intervention_results),
            "successful": interv_successful,
            "greedy_correct": interv_greedy_correct,
            "greedy_accuracy": interv_greedy_correct/len(intervention_results)*100 if intervention_results else 0
        }
    
    # Add detailed results for failed cases
    failed_clean = [r for r in clean_results if not r['success'] or not r['all_greedy_correct']]
    if failed_clean:
        results_summary["failed_clean_examples"] = failed_clean[:5]  # Limit to first 5 for brevity
    
    if args.cur_mode in ("interv", "both") and intervention_locations is not None:
        failed_interv = [r for r in intervention_results if not r['success'] or not r['all_greedy_correct']]
        if failed_interv:
            results_summary["failed_intervention_examples"] = failed_interv[:5]  # Limit to first 5 for brevity
    
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n[INFO] Detailed results saved to {output_file}")
    
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test greedy decoding behavior")
    parser.add_argument("--model_name", type=str, default="text", help="Name of the model to test")
    parser.add_argument("--data_name", type=str, default="swda", help="Dataset name")
    parser.add_argument("--train_path", type=str, default=None, help="Path to training data")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to test")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum examples to test (for debugging)")
    parser.add_argument("--max_token", type=int, default=10, help="Maximum tokens to generate")
    parser.add_argument("--eval_num_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--max_dialogue_length", type=int, default=100, help="Maximum dialogue length for filtering")
    parser.add_argument("--dialogue_act", type=str, default=None, help="Target dialogue act")
    parser.add_argument("--zero_shot", action="store_true", help="Use zero-shot mode")
    parser.add_argument("--cur_mode", type=str, default="both", choices=["clean", "interv", "both"], help="What to test")
    parser.add_argument("--activation_path", type=str, default=None, help="Path to activation data for intervention")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.data_name == "swda" and args.train_path is None:
        args.train_path = "./data/swda/swda_train.json"
    if args.data_name == "swda" and args.val_path is None:
        args.val_path = "./data/swda/swda_val.json"
    
    results = test_greedy_decoding_pipeline(args)
    
    print(f"\n[SUMMARY] Greedy decoding test completed!")
    print(f"Clean greedy accuracy: {results['clean_results']['greedy_accuracy']:.2f}%")
    if 'intervention_results' in results:
        print(f"Intervention greedy accuracy: {results['intervention_results']['greedy_accuracy']:.2f}%") 