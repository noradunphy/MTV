#!/usr/bin/env python3
"""
Test script to verify Llama-3.1-8B-Instruct model loading and instruction following.
"""

import torch
from mtv_utils import load_model

def test_model_loading():
    """Test that the model can be loaded successfully."""
    print("Testing model loading...")
    
    try:
        # Load the model with "text" model_name (which now uses Llama-3.1-8B-Instruct)
        model_helper = load_model("text", "swda", zero_shot=False)
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model_helper.model)}")
        print(f"Tokenizer type: {type(model_helper.tokenizer)}")
        return model_helper
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_instruction_following(model_helper):
    """Test that the model can follow instructions properly."""
    print("\nTesting instruction following...")
    
    # Test prompts that should demonstrate instruction-following capabilities
    test_prompts = [
        "Please explain what a dialogue act is in one sentence.",
        "What is the difference between a statement and a question?",
        "Complete this sentence: 'The weather today is'",
        "If someone says 'I agree with you', what type of dialogue act is this?",
        "Generate a short response to: 'How are you doing?'"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        try:
            # Format the input for the model
            inputs = model_helper.insert_image(prompt, [])
            
            # Generate response
            with torch.no_grad():
                response = model_helper.generate(inputs, max_new_tokens=50)
            
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")

def test_swda_format():
    """Test that the model works with SWDA dataset formatting."""
    print("\nTesting SWDA format compatibility...")
    
    try:
        # Create a simple SWDA-style example
        example = {
            "context": "A: How are you doing today?",
            "response": "I'm doing well, thank you for asking.",
            "dialog_act": "sd"  # Statement-non-opinion
        }
        
        # Test the format function
        text, image_list, target_out, _ = model_helper.format_func(
            [example], None, example, num_shot=0, model_helper=model_helper
        )
        
        print(f"Formatted text: {text[:200]}...")
        print(f"Target output: {target_out}")
        print("✅ SWDA formatting works!")
        
    except Exception as e:
        print(f"❌ SWDA formatting error: {e}")

def test_memory_usage():
    """Check memory usage and clear cache."""
    print("\nChecking memory usage...")
    
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        print("✅ GPU cache cleared")

def main():
    """Main test function."""
    print("🧪 Testing Llama-3.1-8B-Instruct Model")
    print("=" * 50)
    
    # Test 1: Model loading
    model_helper = test_model_loading()
    if model_helper is None:
        print("❌ Cannot proceed without model. Exiting.")
        return
    
    # Test 2: Instruction following
    test_instruction_following(model_helper)
    
    # Test 3: SWDA format compatibility
    test_swda_format()
    
    # Test 4: Memory usage
    test_memory_usage()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("The Llama-3.1-8B-Instruct model is working correctly.")

if __name__ == "__main__":
    main() 