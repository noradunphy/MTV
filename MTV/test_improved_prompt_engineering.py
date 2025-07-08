#!/usr/bin/env python3
"""
Test script to demonstrate improved prompt engineering for Llama-3.1-8B-Instruct
on the SWDA multiple choice task.
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
        },
        {
            "text": "A: Have you seen that new movie?\nB: No, I haven't seen it yet.",
            "response": "No, I haven't seen it yet.",
            "caller": "B",
            "dialog_act": "sd",
            "utterance_id": 6
        },
        {
            "text": "A: The food was delicious.\nB: Yes, it was really good.",
            "response": "Yes, it was really good.",
            "caller": "B",
            "dialog_act": "aa",
            "utterance_id": 7
        }
    ]
    return mock_data

def test_improved_prompt_engineering():
    """Test the MTV-compatible prompt engineering for Llama-3.1-8B-Instruct."""
    print("🧪 Testing MTV-Compatible Prompt Engineering for Llama-3.1-8B-Instruct")
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
            print("IMPROVED PROMPT FOR LLAMA-3.1-8B-INSTRUCT:")
            print("="*60)
            print(text)
            print("="*60)
            
            # Check for key improvements
            improvements = []
            if "Given the conversation context above" in text:
                improvements.append("✅ Clear task description")
            if "Answer with the option's number" in text:
                improvements.append("✅ Explicit instruction for number output")
            if "1) " in text and "2) " in text:
                improvements.append("✅ Numbered options format")
            if "Answer:" in text:
                improvements.append("✅ Clear answer prompt")
            if not "<|begin_of_text|>" in text:
                improvements.append("✅ No system prompts (MTV-compatible)")
            if not "**" in text:
                improvements.append("✅ No markdown formatting (consistent with other datasets)")
                
            print(f"\nPrompt Engineering Improvements:")
            for improvement in improvements:
                print(f"  {improvement}")
                
        except Exception as e:
            print(f"❌ Error during formatting: {e}")

def test_model_with_improved_prompts():
    """Test the model with improved prompts."""
    print("\n🧪 Testing Model with Improved Prompts")
    print("=" * 80)
    
    # Create mock data
    mock_data = create_mock_swda_data()
    
    try:
        # Load model
        print("Loading Llama-3.1-8B-Instruct model...")
        model_helper = load_model("text", "swda", zero_shot=False)
        print("✅ Model loaded successfully!")
        
        # Test with a few examples
        for i, example in enumerate(mock_data[:3], 1):
            print(f"\n--- Example {i} ---")
            
            # Format with improved prompts
            text, image_list, target_number, utt_id = format_swda_multiple_choice(
                filtered_dataset=mock_data,
                full_dataset=mock_data,
                cur_item=example,
                num_shot=1,  # Use 1-shot for demonstration
                model_helper=model_helper,
                split="test"
            )
            
            print(f"Target: {target_number}")
            print(f"Dialogue Act: {example['dialog_act']}")
            print(f"Caller: {example['caller']}")
            
            # Show a preview of the prompt
            prompt_preview = text[:500] + "..." if len(text) > 500 else text
            print(f"\nPrompt Preview:\n{prompt_preview}")
            
            # Generate response
            try:
                inputs = model_helper.insert_image(text, image_list)
                
                with torch.no_grad():
                    response = model_helper.generate(inputs, max_new_tokens=10)
                
                print(f"Model Response: {response}")
                
                # Check if response contains a number
                import re
                number_match = re.search(r'\b[1-4]\b', response)
                if number_match:
                    predicted_number = number_match.group()
                    print(f"Extracted Number: {predicted_number}")
                    if predicted_number == target_number:
                        print("✅ Correct prediction!")
                    else:
                        print(f"❌ Incorrect prediction. Expected: {target_number}")
                else:
                    print("❌ No number found in response")
                    
            except Exception as e:
                print(f"❌ Generation error: {e}")
                
    except Exception as e:
        print(f"❌ Model loading error: {e}")

def compare_prompt_formats():
    """Compare old vs new prompt formats."""
    print("\n📊 Comparing Prompt Formats")
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
    
    print("OLD FORMAT (instruction-tuned style - NOT MTV compatible):")
    print("-" * 40)
    old_format = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant that analyzes conversation context and selects the most appropriate response from multiple choice options. Your task is to:

1. Read the conversation context carefully
2. Consider the speaker's role and the flow of the conversation
3. Select the response that best fits the context and maintains natural dialogue flow
4. Answer with ONLY the number (1, 2, 3, or 4) corresponding to your choice

<|eot_id|><|start_header_id|>user<|end_header_id|>

**Conversation Context:**
A: How are you doing today?
B: I'm doing well, thank you.

**Task:** Given the conversation above, which response is most appropriate for B?

**Options:**
1) I'm doing well, thank you for asking.
2) I see.
3) That's interesting.
4) Go on.

**Answer:**"""
    print(old_format)
    
    print("\nNEW FORMAT (MTV-compatible):")
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
    
    print("\nKey Improvements:")
    print("1. ✅ System instruction with clear role definition")
    print("2. ✅ Structured format with markdown headers")
    print("3. ✅ Explicit task description")
    print("4. ✅ Detailed guidance for the model")
    print("5. ✅ Better visual separation of sections")
    print("6. ✅ Llama-3.1-8B-Instruct specific formatting")

if __name__ == "__main__":
    test_improved_prompt_engineering()
    test_model_with_improved_prompts()
    compare_prompt_formats() 