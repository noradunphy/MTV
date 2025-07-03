#!/usr/bin/env python3
"""
Script to compare old vs new MCQ formats and demonstrate token savings.
"""

import json
import random
from preprocess import format_swda_multiple_choice

def old_format_example():
    """Generate an example using the old instruction-tuned style format."""
    return """Chat 1
A: How are you doing today?
B: I'm doing well, thanks.
Options:
A. Thank you for telling me.
B. I need help with my homework.
C. Me too, it's my favorite food.
D. That's great to hear!
Final Response A: D
______

A: What do you think about the weather?
B: It's quite nice today.
Options:
A. I agree, it's beautiful outside.
B. Can you help me with this?
C. That's interesting.
D. I don't know.
Final Response A:"""

def new_format_example():
    """Generate an example using the new base-model optimized format."""
    return """Context: A: How are you doing today?
B: I'm doing well, thanks.
Options: A.Thank you for telling me. B.I need help with my homework. C.Me too, it's my favorite food. D.That's great to hear!
Answer: D

Context: A: What do you think about the weather?
B: It's quite nice today.
Options: A.I agree, it's beautiful outside. B.Can you help me with this? C.That's interesting. D.I don't know.
Answer:"""

def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 characters for English text)."""
    return len(text) // 4

def main():
    print("MCQ Format Comparison for Base Models")
    print("=" * 50)
    
    # Generate examples using the actual format function
    sample_data = [
        {
            "text": "A: How are you doing today?\nB: I'm doing well, thanks.",
            "response": "That's great to hear!",
            "dialog_act": "aa",
            "caller": "A",
            "utterance_id": "test_1"
        },
        {
            "text": "A: What do you think about the weather?\nB: It's quite nice today.",
            "response": "I agree, it's beautiful outside.",
            "dialog_act": "aa",
            "caller": "B",
            "utterance_id": "test_2"
        },
        {
            "text": "A: Can you help me with this problem?\nB: Sure, what is it?",
            "response": "I need help with my homework.",
            "dialog_act": "sd",
            "caller": "A",
            "utterance_id": "test_3"
        }
    ]
    
    # Test the optimized format
    current_item = sample_data[0]
    text, image_list, target_letter, question_id = format_swda_multiple_choice(
        sample_data, 
        current_item, 
        num_shot=2,
        split="train"
    )
    
    print("Optimized Format (Current):")
    print("-" * 30)
    print(text)
    print(f"\nTarget letter: {target_letter}")
    
    # Compare with old format
    old_format = old_format_example()
    new_format = new_format_example()
    
    print("\n" + "=" * 50)
    print("Token Usage Comparison:")
    print("=" * 50)
    
    old_tokens = estimate_tokens(old_format)
    new_tokens = estimate_tokens(new_format)
    savings = ((old_tokens - new_tokens) / old_tokens) * 100
    
    print(f"Old format (instruction-tuned style): {old_tokens} tokens")
    print(f"New format (base-model optimized): {new_tokens} tokens")
    print(f"Token savings: {savings:.1f}%")
    
    print("\n" + "=" * 50)
    print("Key Optimizations:")
    print("=" * 50)
    
    optimizations = [
        "1. Removed verbose formatting ('Chat 1', '______')",
        "2. Inline options format instead of multi-line",
        "3. Simplified 'Answer:' instead of 'Final Response A:'",
        "4. Removed unnecessary line breaks and spacing",
        "5. Consistent pattern across all examples",
        "6. Direct task focus without instruction overhead"
    ]
    
    for opt in optimizations:
        print(opt)
    
    print("\n" + "=" * 50)
    print("Performance Benefits:")
    print("=" * 50)
    
    benefits = [
        f"• {savings:.1f}% fewer tokens per example",
        "• More examples fit in context window",
        "• Faster inference times",
        "• Better pattern recognition for base models",
        "• Reduced cognitive load on the model",
        "• More consistent letter prediction"
    ]
    
    for benefit in benefits:
        print(benefit)
    
    print("\n" + "=" * 50)
    print("Recommended Settings for Llama-3.1-8B:")
    print("=" * 50)
    
    recommendations = [
        "• Temperature: 0.1-0.3 (for consistent letter prediction)",
        "• Max tokens: 5 (single letter output)",
        "• Few-shot examples: 2-4",
        "• Greedy decoding recommended",
        "• Monitor position bias in option selection"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    random.seed(42)
    main() 