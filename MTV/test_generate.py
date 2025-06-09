import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import TextModelHelper
from preprocess import get_format_func, open_data
# from speech_act_classifier import load_classifier
from backchannel_classifier import load_classifier

def test_generate(num_shot=4):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize helper
    helper = TextModelHelper(model, tokenizer, "swda", zero_shot=False)  # Set to False to allow N-shot

    # Load binary backchannel classifier
    print("Loading backchannel classifier...")
    classifier = load_classifier()
    print("Classifier loaded!")

    # Load data from actual JSON files
    print("Loading SWDA data...")
    train_data = open_data("swda", "data/swda/train.json")
    val_data = open_data("swda", "data/swda/validation.json")
    
    # Filter for shorter dialogues (like in mtv_eval.py)
    train_data = [ex for ex in train_data if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
    val_data = [ex for ex in val_data if len(ex.get('text', '').split()) + len(ex.get('response', '').split()) < 100]
    
    # Filter for backchannels (dialog_act == 'b')
    backchannel_data = [ex for ex in val_data if ex.get('dialog_act') == 'b']
    
    print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")
    print(f"Found {len(backchannel_data)} backchannel examples")
    
    # Use a backchannel example for testing
    test_example = backchannel_data[8]  # Use first backchannel example
    
    print("\nTest example:")
    print(f"Text: {test_example['text']}")
    print(f"Response: {test_example['response']}")
    print(f"Caller: {test_example['caller']}")
    print(f"Dialog act: {test_example['dialog_act']}")
    
    # Test zero-shot format
    print("\n=== Testing Zero-shot Format ===")
    full_text_zero, image_list_zero, answer_zero, question_id_zero = helper.format_func(val_data, test_example, num_shot=0)
    inputs_zero = helper.insert_image(full_text_zero, image_list_zero)
    output_zero = helper.generate(inputs_zero, max_new_tokens=50)
    
    print(f"\nZero-shot formatted input: {full_text_zero}")
    print(f"Expected output: {answer_zero}")
    print(f"Generated output: {output_zero}")
    
    # Test N-shot format
    print("\n=== Testing N-shot Format ===")
    full_text_nshot, image_list_nshot, answer_nshot, question_id_nshot = helper.format_func(train_data, test_example, num_shot=num_shot)
    inputs_nshot = helper.insert_image(full_text_nshot, image_list_nshot)
    output_nshot = helper.generate(inputs_nshot, max_new_tokens=50)
    
    print(f"\nN-shot formatted input: {full_text_nshot}")
    print(f"Expected output: {answer_nshot}")
    print(f"Generated output: {output_nshot}")
    
    # Verify N-shot format
    print("\n=== Verifying N-shot Format ===")
    if "Chat 1" in full_text_nshot:
        print("✅ Test passed: N-shot format includes 'Chat' headers")
    else:
        print("❌ Test failed: N-shot format missing 'Chat' headers")
        
    if "Final Response" in full_text_nshot:
        print("✅ Test passed: N-shot format includes 'Final Response' markers")
    else:
        print("❌ Test failed: N-shot format missing 'Final Response' markers")
        
    if "______" in full_text_nshot:
        print("✅ Test passed: N-shot format includes separators")
    else:
        print("❌ Test failed: N-shot format missing separators")
    
    # Check if input is not in output for both formats
    print("\n=== Checking Output Format ===")
    if full_text_zero in output_zero:
        print("❌ Test failed: Zero-shot input was found in output")
    else:
        print("✅ Test passed: Zero-shot input was not found in output")
        
    if full_text_nshot in output_nshot:
        print("❌ Test failed: N-shot input was found in output")
    else:
        print("✅ Test passed: N-shot input was not found in output")
    
    # Check if output contains multiple turns for both formats
    if "\n" in output_zero:
        print("❌ Test failed: Zero-shot output contains multiple turns")
    else:
        print("✅ Test passed: Zero-shot output contains only one turn")
        
    if "\n" in output_nshot:
        print("❌ Test failed: N-shot output contains multiple turns")
    else:
        print("✅ Test passed: N-shot output contains only one turn")
    
    # Classify the generated outputs using binary classifier
    print("\n=== Classifier Results ===")
    # Zero-shot classification
    is_backchannel_zero = classifier.classify_utterance(output_zero)
    print(f"Zero-shot predicted: {'Backchannel' if is_backchannel_zero else 'Not a backchannel'}")
    print(f"Expected: Backchannel")
    if is_backchannel_zero:
        print("✅ Test passed: Zero-shot classifier correctly identified as backchannel")
    else:
        print("❌ Test failed: Zero-shot classifier did not identify as backchannel")
        
    # N-shot classification
    is_backchannel_nshot = classifier.classify_utterance(output_nshot)
    print(f"\nN-shot predicted: {'Backchannel' if is_backchannel_nshot else 'Not a backchannel'}")
    print(f"Expected: Backchannel")
    if is_backchannel_nshot:
        print("✅ Test passed: N-shot classifier correctly identified as backchannel")
    else:
        print("❌ Test failed: N-shot classifier did not identify as backchannel")

if __name__ == "__main__":
    test_generate(num_shot=4)  # Test with 4-shot prompting 