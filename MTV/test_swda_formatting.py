#!/usr/bin/env python3
"""
Test script to verify the updated SWDA multiple choice formatting with explicit instructions.
"""

import random
from preprocess import format_swda_multiple_choice
import torch
from mtv_utils import load_model

def create_mock_swda_data():
    """Create mock SWDA data for testing."""
    mock_data = [
        {
            "text": "A: How are you doing today?\nB: I'm doing well, thank you.",
            "response": "I'm doing well, thank you for asking.",
            "caller": "B",
            "dialog_act": "sd",  # Statement-non-opinion
            "utterance_id": 1
        },
        {
            "text": "A: What do you think about the weather?\nB: It's quite nice today.",
            "response": "It's quite nice today.",
            "caller": "B", 
            "dialog_act": "sd",
            "utterance_id": 2
        },
        {
            "text": "A: Do you want to go to the movies?\nB: That sounds great!",
            "response": "That sounds great!",
            "caller": "B",
            "dialog_act": "aa",  # Agreement
            "utterance_id": 3
        },
        {
            "text": "A: I think we should leave early.\nB: I agree with you.",
            "response": "I agree with you.",
            "caller": "B",
            "dialog_act": "aa",
            "utterance_id": 4
        },
        {
            "text": "A: What time is it?\nB: It's 3 o'clock.",
            "response": "It's 3 o'clock.",
            "caller": "B",
            "dialog_act": "qy",  # Yes-no question
            "utterance_id": 5
        }
    ]
    return mock_data

def test_swda_formatting():
    """Test the updated SWDA multiple choice formatting."""
    print("🧪 Testing Updated SWDA Multiple Choice Formatting")
    print("=" * 60)
    
    # Create mock data
    mock_data = create_mock_swda_data()
    
    # Test with different scenarios
    test_cases = [
        {
            "name": "Zero-shot (no examples)",
            "num_shot": 0,
            "target_act": "sd"
        },
        {
            "name": "One-shot (with examples)", 
            "num_shot": 1,
            "target_act": "sd"
        },
        {
            "name": "Two-shot (with examples)",
            "num_shot": 2, 
            "target_act": "aa"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        
        # Filter data for target act
        filtered_data = [ex for ex in mock_data if ex['dialog_act'] == test_case['target_act']]
        
        if not filtered_data:
            print(f"❌ No data found for act '{test_case['target_act']}'")
            continue
            
        # Test the formatting
        try:
            text, image_list, target_number, utt_id = format_swda_multiple_choice(
                filtered_dataset=filtered_data,
                full_dataset=mock_data,
                cur_item=filtered_data[0],  # Use first example
                num_shot=test_case['num_shot'],
                model_helper=None,
                split="test"
            )
            
            print(f"✅ Formatting successful!")
            print(f"Target number: {target_number}")
            print(f"Utterance ID: {utt_id}")
            print(f"Generated prompt length: {len(text)} characters")
            
            # Show the formatted prompt
            print("\nFormatted prompt:")
            print("-" * 40)
            print(text)
            print("-" * 40)
            
            # Check if the explicit instruction is present
            if "Answer with the option's number from the given choice directly" in text:
                print("✅ Explicit instruction found in prompt")
            else:
                print("❌ Explicit instruction NOT found in prompt")
                
        except Exception as e:
            print(f"❌ Error during formatting: {e}")
        

    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")

def test_model_output():
    """Test the model output."""
    print("🧪 Testing Model Output")
    print("=" * 60)
    
    # Create mock data
    mock_data = create_mock_swda_data()
    model_helper = load_model("text", "swda", zero_shot=False)

    for example in mock_data:
        text, image_list, target_number, utt_id = format_swda_multiple_choice(
            filtered_dataset=mock_data,
            full_dataset=mock_data,
            cur_item=example,
            num_shot=4,
            model_helper=model_helper,
            split="test"
        )
        

        # Show the formatted prompt
        print("\nFormatted prompt:")
        print(text)

        # Tokenize the text for the model
        inputs = model_helper.insert_image(text, image_list)
        
        with torch.no_grad():
            response = model_helper.generate(inputs, max_new_tokens=20)
        
        print(f"Response: {response}")

if __name__ == "__main__":
    test_model_output() 