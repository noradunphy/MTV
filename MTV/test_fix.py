#!/usr/bin/env python3
"""
Simple test script to verify that the perplexity computation fixes work.
"""

import torch
from models import TextModelHelper
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_perplexity_computation():
    """Test the perplexity computation with various target texts."""
    
    print("[INFO] Loading test model...")
    # Load a small test model
    model_name = "microsoft/DialoGPT-small"  # Small model for testing
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"[INFO] Model loaded on {device}")
    
    # Create a TextModelHelper instance
    model_helper = TextModelHelper(model, tokenizer, "swda", zero_shot=True)
    
    # Test cases
    test_cases = [
        ("Hello", "Hello there"),  # Normal case
        ("", "Hello there"),       # Empty target_text
        (None, "Hello there"),     # None target_text
        ("Hello", ""),             # Empty target_text
        ("Hello", None),           # None target_text
    ]
    
    for i, (prefix_text, target_text) in enumerate(test_cases):
        print(f"\n[TEST {i+1}] Prefix: '{prefix_text}', Target: '{target_text}'")
        
        try:
            # Tokenize prefix
            prefix_input = tokenizer(prefix_text, return_tensors="pt", padding=True, truncation=True)
            prefix_input = {k: v.to(device) for k, v in prefix_input.items()}
            
            # Test perplexity computation
            ppl = model_helper._compute_autoregressive_perplexity(
                model, tokenizer, prefix_input, target_text, device
            )
            
            print(f"[RESULT] Perplexity: {ppl}")
            
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
    
    print("\n[INFO] Test completed!")

if __name__ == "__main__":
    test_perplexity_computation() 