#!/usr/bin/env python3
"""
Fix accuracy calculation issue in evaluation results.

The problem: When skip_generation is true, the code incorrectly uses the gold dialogue act
for both clean and intervention predictions, making both accuracies identical.

The fix: Use the perplexity-based dialogue act predictions (clean_chosen_act and intervention_chosen_act)
to compute the correct accuracies.
"""

import json
import argparse
from pathlib import Path

def fix_accuracy_calculation(input_file):
    """Fix accuracy calculation in the evaluation results file."""
    
    print(f"Loading results from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Check if this file has the issue (skip_generation = true)
    if not data.get('summary', {}).get('skip_generation', False):
        print("This file doesn't have skip_generation=true, no fix needed.")
        return
    
    print("Found skip_generation=true, fixing accuracy calculation...")
    
    # Count correct predictions using perplexity-based chosen acts
    clean_count = 0
    intervention_count = 0
    total_examples = len(data['examples'])
    
    for example in data['examples']:
        target_act = example['target_dialogue_act']
        
        # Use perplexity-based chosen acts instead of gold acts
        clean_chosen_act = example.get('clean_chosen_act')
        intervention_chosen_act = example.get('intervention_chosen_act')
        
        if clean_chosen_act is not None:
            clean_correct = int(clean_chosen_act == target_act)
            clean_count += clean_correct
            # Update the example data
            example['clean_correct'] = clean_correct
            example['clean_dialogue_act'] = clean_chosen_act
        
        if intervention_chosen_act is not None:
            intervention_correct = int(intervention_chosen_act == target_act)
            intervention_count += intervention_correct
            # Update the example data
            example['intervention_correct'] = intervention_correct
            example['intervention_dialogue_act'] = intervention_chosen_act
    
    # Update summary statistics
    if total_examples > 0:
        clean_accuracy = clean_count / total_examples
        intervention_accuracy = intervention_count / total_examples
        
        data['summary']['clean_accuracy'] = clean_accuracy
        data['summary']['intervention_accuracy'] = intervention_accuracy
        
        print(f"Fixed accuracies:")
        print(f"  Clean accuracy: {clean_accuracy:.4f} ({clean_count}/{total_examples})")
        print(f"  Intervention accuracy: {intervention_accuracy:.4f} ({intervention_count}/{total_examples})")
    else:
        print("No examples found in the file.")
        return
    
    # Save the fixed results
    output_file = input_file.replace('.json', '_fixed.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Fix accuracy calculation in evaluation results")
    parser.add_argument("input_file", help="Path to the evaluation results JSON file")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: File {args.input_file} not found.")
        return
    
    fix_accuracy_calculation(args.input_file)

if __name__ == "__main__":
    main() 