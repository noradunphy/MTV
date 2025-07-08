#!/usr/bin/env python3
"""
Test script to demonstrate the cleaned up formatting for SWDA multiple choice.
This shows how the formatting makes it super clear for the model what to learn.
"""

import random
from preprocess import format_swda_multiple_choice

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
            "text": "A: Have you seen that new movie?\nB: No, I haven't seen it yet.",
            "response": "No, I haven't seen it yet.",
            "caller": "B",
            "dialog_act": "sd",
            "utterance_id": 5
        }
    ]
    return mock_data

def test_clean_formatting():
    """Test the cleaned up formatting."""
    print("🧪 Testing Cleaned Up Formatting for SWDA Multiple Choice")
    print("=" * 80)
    
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
            print("\n" + "="*60)
            print("CLEANED UP FORMAT:")
            print("="*60)
            print(text)
            print("="*60)
            
            # Check for key formatting improvements
            improvements = []
            if "EXAMPLE:" in text:
                improvements.append("✅ Clear 'EXAMPLE:' labels for few-shot examples")
            if "TEST EXAMPLE:" in text:
                improvements.append("✅ Clear 'TEST EXAMPLE:' label for the query")
            if "QUESTION:" in text:
                improvements.append("✅ Clear 'QUESTION:' labels")
            if "OPTIONS:" in text:
                improvements.append("✅ Clear 'OPTIONS:' labels")
            if "ANSWER:" in text:
                improvements.append("✅ Clear 'ANSWER:' labels")
            if "______" in text:
                improvements.append("✅ Clear separators between examples")
            if not "Answer with the option's number" in text:
                improvements.append("✅ Removed redundant instruction text")
                
            print(f"\nFormatting Improvements:")
            for improvement in improvements:
                print(f"  {improvement}")
                
        except Exception as e:
            print(f"❌ Error during formatting: {e}")

def compare_formats():
    """Compare old vs new formats."""
    print("\n📊 Comparing Old vs New Formats")
    print("=" * 80)
    
    # Create a simple example
    example = {
        "text": "A: How are you doing today?\nB: I'm doing well, thank you.",
        "response": "I'm doing well, thank you for asking.",
        "caller": "B",
        "dialog_act": "sd",
        "utterance_id": 1
    }
    
    mock_data = [example]
    
    print("OLD FORMAT (less clear):")
    print("-" * 40)
    old_format = """A: How are you doing today?
B: I'm doing well, thank you.

Given the conversation context above, which response is most appropriate?
B: 1) I'm doing well, thank you for asking. 2) I see. 3) That's interesting. 4) Go on.
Answer with the option's number from the given choice directly. Answer: 1"""
    print(old_format)
    
    print("\nNEW FORMAT (super clear):")
    print("-" * 40)
    
    # Get new format
    new_text, _, _, _ = format_swda_multiple_choice(
        filtered_dataset=mock_data,
        full_dataset=mock_data,
        cur_item=example,
        num_shot=0,
        model_helper=None,
        split="test"
    )
    print(new_text)
    
    print("\nKey Improvements for Model Learning:")
    print("1. ✅ 'TEST EXAMPLE:' - Clear label for what the model should focus on")
    print("2. ✅ 'QUESTION:' - Explicit question label")
    print("3. ✅ 'OPTIONS:' - Clear options section")
    print("4. ✅ 'ANSWER:' - Clear answer prompt")
    print("5. ✅ Removed redundant instruction text")
    print("6. ✅ Consistent capitalization for labels")
    print("7. ✅ Better visual structure for pattern recognition")

def show_few_shot_example():
    """Show how few-shot examples look with the new formatting."""
    print("\n🎯 Few-Shot Example with Clean Formatting")
    print("=" * 80)
    
    mock_data = create_mock_swda_data()
    
    # Get a 2-shot example
    text, _, target_number, _ = format_swda_multiple_choice(
        filtered_dataset=mock_data,
        full_dataset=mock_data,
        cur_item=mock_data[0],
        num_shot=2,
        model_helper=None,
        split="test"
    )
    
    print("This is how the model sees the examples:")
    print("="*60)
    print(text)
    print("="*60)
    
    print(f"\nTarget Answer: {target_number}")
    print("\nWhat the model learns:")
    print("1. 📝 Pattern: EXAMPLE → QUESTION → OPTIONS → ANSWER")
    print("2. 📝 Pattern: TEST EXAMPLE → QUESTION → OPTIONS → ANSWER:")
    print("3. 📝 Numbers 1-4 correspond to specific responses")
    print("4. 📝 The separator '______' indicates example boundaries")
    print("5. 📝 Consistent formatting across all examples")

if __name__ == "__main__":
    test_clean_formatting()
    compare_formats()
    show_few_shot_example() 