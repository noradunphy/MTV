#!/usr/bin/env python3
"""
Test script to verify the updated MCQ sampling logic.
This script tests that the original target utterance is excluded from options.
"""

import json
import random
from preprocess import format_swda_multiple_choice, open_data

def test_mcq_sampling():
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Testing Updated MCQ Sampling Logic")
    print("=" * 50)
    
    # Load some test data
    try:
        train_dataset = open_data("swda", "data/swda/processed/train.json")
        val_dataset = open_data("swda", "data/swda/processed/val.json")
        print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    except FileNotFoundError:
        print("Test data not found. Creating mock data...")
        # Create mock data for testing
        train_dataset = [
            {"text": "How are you doing?", "response": "I'm doing well, thanks!", "dialog_act": "sd", "caller": "A", "utterance_id": 1},
            {"text": "What's the weather like?", "response": "It's sunny today.", "dialog_act": "sd", "caller": "B", "utterance_id": 2},
            {"text": "Do you like coffee?", "response": "Yes, I love coffee!", "dialog_act": "sd", "caller": "A", "utterance_id": 3},
            {"text": "That's interesting.", "response": "I see.", "dialog_act": "b", "caller": "B", "utterance_id": 4},
            {"text": "Tell me more.", "response": "Go on.", "dialog_act": "b", "caller": "A", "utterance_id": 5},
            {"text": "I agree with you.", "response": "That's right.", "dialog_act": "aa", "caller": "B", "utterance_id": 6},
        ]
        val_dataset = [
            {"text": "What's your favorite color?", "response": "I like blue.", "dialog_act": "sd", "caller": "A", "utterance_id": 7},
        ]
    
    # Test the format function
    test_item = val_dataset[0]
    print(f"\nTest Item:")
    print(f"  Text: {test_item.get('text', 'N/A')}")
    print(f"  Original Response: {test_item.get('response', 'N/A')}")
    print(f"  Dialogue Act: {test_item.get('dialog_act', 'N/A')}")
    
    # Format with multiple choice
    text, image_list, target_letter, question_id = format_swda_multiple_choice(
        train_dataset, 
        test_item, 
        num_shot=2,
        split="train"
    )
    
    print(f"\nFormatted Output:")
    print(f"  Target Letter: {target_letter}")
    print(f"  Question ID: {question_id}")
    print(f"\nFormatted Text:")
    print("-" * 40)
    print(text)
    print("-" * 40)
    
    # Check if the original response is in the options
    original_response = test_item.get('response', '')
    if original_response in text:
        print(f"\n❌ WARNING: Original response '{original_response}' is still in the options!")
        print("This means the exclusion logic is not working properly.")
    else:
        print(f"\n✅ SUCCESS: Original response '{original_response}' is NOT in the options.")
        print("The exclusion logic is working correctly.")
    
    # Parse the options to verify
    lines = text.split('\n')
    options_found = []
    for line in lines:
        if line.startswith('Options:'):
            options_text = line.replace('Options:', '').strip()
            options_found = [opt.strip() for opt in options_text.split(' ') if opt.strip()]
            break
    
    if options_found:
        print(f"\nOptions found: {options_found}")
        print(f"Original response in options: {original_response in options_found}")
    
    # Test multiple examples to ensure consistency
    print(f"\n" + "="*50)
    print("Testing multiple examples...")
    
    for i in range(min(3, len(val_dataset))):
        test_item = val_dataset[i]
        text, _, target_letter, _ = format_swda_multiple_choice(
            train_dataset, 
            test_item, 
            num_shot=1,
            split="train"
        )
        
        original_response = test_item.get('response', '')
        if original_response in text:
            print(f"❌ Example {i+1}: Original response still in options")
        else:
            print(f"✅ Example {i+1}: Original response excluded correctly")
    
    print(f"\nTest completed!")

if __name__ == "__main__":
    test_mcq_sampling() 