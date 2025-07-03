#!/usr/bin/env python3
"""
Example script demonstrating the updated SWDA multiple choice pipeline.

This script shows how to use the new multiple choice format for SWDA,
where the model is trained to predict letter answers (A, B, C, D) instead
of generating full utterances.
"""

import argparse
import json
import random
from preprocess import open_data, get_format_func

def main():
    parser = argparse.ArgumentParser(description="Example SWDA multiple choice pipeline")
    parser.add_argument("--train_path", type=str, default="data/swda/processed/train.json", 
                       help="Path to training data")
    parser.add_argument("--val_path", type=str, default="data/swda/processed/val.json", 
                       help="Path to validation data")
    parser.add_argument("--dialogue_act", type=str, default=None,
                       help="Target dialogue act to filter for (e.g., 'sd', 'sv', 'b', 'aa')")
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of examples to show")
    parser.add_argument("--num_shot", type=int, default=2,
                       help="Number of few-shot examples")
    
    args = parser.parse_args()
    
    print("SWDA Multiple Choice Pipeline Example")
    print("=" * 50)
    
    # Load data
    print(f"Loading training data from {args.train_path}...")
    train_dataset = open_data("swda", args.train_path, args.dialogue_act)
    print(f"Loaded {len(train_dataset)} training examples")
    
    print(f"Loading validation data from {args.val_path}...")
    val_dataset = open_data("swda", args.val_path, args.dialogue_act)
    print(f"Loaded {len(val_dataset)} validation examples")
    
    # Get the format function (now defaults to multiple choice for SWDA)
    format_func = get_format_func("swda")
    print(f"Using format function: {format_func.__name__}")
    
    # Show examples
    print(f"\nShowing {args.num_examples} examples with {args.num_shot}-shot prompting:")
    print("=" * 50)
    
    for i in range(min(args.num_examples, len(val_dataset))):
        item = val_dataset[i]
        
        # Format the example
        text, image_list, target_letter, question_id = format_func(
            train_dataset, 
            item, 
            num_shot=args.num_shot,
            split="train"
        )
        
        print(f"\nExample {i+1}:")
        print(f"Question ID: {question_id}")
        print(f"Target Dialogue Act: {item.get('dialog_act', 'unknown')}")
        print(f"Target Letter: {target_letter}")
        print(f"Caller: {item.get('caller', 'unknown')}")
        print("\nFormatted Input:")
        print("-" * 30)
        print(text)
        print("-" * 30)
        
        if i < args.num_examples - 1:
            print("\n" + "="*50)
    
    print(f"\nPipeline Summary:")
    print(f"- Dataset: SWDA")
    print(f"- Format: Multiple Choice with Letter Answers")
    print(f"- Few-shot examples: {args.num_shot}")
    print(f"- Target dialogue act filter: {args.dialogue_act if args.dialogue_act else 'All acts'}")
    print(f"- Training examples: {len(train_dataset)}")
    print(f"- Validation examples: {len(val_dataset)}")
    
    print(f"\nTo run the full evaluation pipeline:")
    print(f"python mtv_eval.py --model_name text --data_name swda \\")
    print(f"  --train_path {args.train_path} --val_path {args.val_path} \\")
    print(f"  --num_example 100 --num_shot 4 --eval_num_shot 2 \\")
    print(f"  --max_token 5 --cur_mode both \\")
    print(f"  --bernoullis_path storage/swda_bernoullis.pt \\")
    print(f"  --activation_path storage/swda_activations.pt")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main() 