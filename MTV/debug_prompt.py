#!/usr/bin/env python3
"""
Debug script to understand why the model is generating multiple-choice responses.
"""

import torch
from mtv_utils import load_model

def debug_prompt_formatting():
    """Debug how prompts are being formatted and sent to the model."""
    
    print("Loading model...")
    model_helper = load_model("text", "swda", zero_shot=False)
    
    # Test different prompt formats
    test_prompts = [
        "If someone says 'I agree with you', what type of dialogue act is this?",
        "What dialogue act is 'I agree with you'?",
        "Classify this utterance: 'I agree with you'",
        "The utterance 'I agree with you' is an example of what dialogue act?",
        "Simply answer: What dialogue act is 'I agree with you'?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Original prompt: {prompt}")
        
        # Show what the model receives
        inputs = model_helper.insert_image(prompt, [])
        print(f"Input keys: {inputs.keys()}")
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        # Decode the input to see what the model actually sees
        decoded_input = model_helper.tokenizer.decode(inputs['input_ids'][0])
        print(f"Decoded input: {repr(decoded_input)}")
        
        # Generate response
        with torch.no_grad():
            response = model_helper.generate(inputs, max_new_tokens=30)
        
        print(f"Response: {response}")
        print("-" * 50)

def test_simple_conversation():
    """Test with a simple conversational prompt."""
    
    print("\n" + "="*50)
    print("Testing simple conversation...")
    
    model_helper = load_model("text", "swda", zero_shot=False)
    
    simple_prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Tell me a joke.",
        "What color is the sky?"
    ]
    
    for prompt in simple_prompts:
        print(f"\nPrompt: {prompt}")
        inputs = model_helper.insert_image(prompt, [])
        
        with torch.no_grad():
            response = model_helper.generate(inputs, max_new_tokens=20)
        
        print(f"Response: {response}")

if __name__ == "__main__":
    debug_prompt_formatting()
    test_simple_conversation() 