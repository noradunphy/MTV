import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
import json
from mtv_utils import ppl_from_scores, load_model, print_token_log_probs
import re


def evaluate_wikitext_full_sequence_generate(model_name="text", num_lines=50):
    """Evaluate perplexity using generate() with single token prefix and full sequence."""
    # ——— Model setup ———
    print(f"Evaluating {model_name} on {num_lines} lines of WikiText-2 using generate()")
    model_helper = load_model(model_name, "wikitext-2-raw-v1", zero_shot=True)
    tokenizer = model_helper.tokenizer

    # ——— Load WikiText-2 validation lines ———
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{num_lines}]")
    print(f"Loaded {len(wiki)} total lines from dataset")
    
    # Filter out empty lines
    lines = [ex["text"].strip() for ex in wiki if ex["text"].strip()]
    print(f"After filtering empty lines: {len(lines)} lines")
    if len(lines) < len(wiki):
        print("First few empty lines that were filtered:")
        for ex in wiki:
            if not ex["text"].strip():
                print(f"  {ex['text']!r}")
                if len(ex["text"]) > 0:
                    print(f"  (contains {len(ex['text'])} whitespace characters)")
                break

    # ——— Open log file ———
    output_file = f"wikitext_ppl_generate_{model_name}.txt"
    print(f"Writing results to {output_file}")
    with open(output_file, "w", buffering=1) as f:
        f.write(f"WikiText-2 Full Sequence PPLs using generate() ({model_name})\n")
        f.write("=" * 60 + "\n\n")

        ppls = []
        print(f"Evaluating {len(lines)} lines")
        for i, line in enumerate(lines, start=1):
            # Use single start token as prefix and full sequence as postfix
            prefix_text = tokenizer.bos_token if tokenizer.bos_token else tokenizer.eos_token
            suffix_text = line
            
            # Tokenize prefix and suffix
            prefix_enc = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
            suffix_enc = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)
            
            # Move to GPU and ensure Long type
            prefix_ids = prefix_enc["input_ids"].to(model_helper.model.device).long()
            suffix_ids = suffix_enc["input_ids"][0]
            
            # Create input dictionary for generate
            prefix_input = {
                "input_ids": prefix_ids,
                "attention_mask": torch.ones_like(prefix_ids, dtype=torch.long)
            }
            
            # Generate with single token prefix
            output, scores = model_helper.generate(
                prefix_input,
                max_new_tokens=len(suffix_ids),
                return_scores=True,
                return_dict_in_generate=True
            )
            
            # Compute perplexity on the full sequence
            ppl, _ = ppl_from_scores(scores, tokenizer, suffix_text)
            ppls.append(ppl)

            print(f"[{i:03d}] {line[:50]!r}…", file=f)
            print(f"Full sequence: {suffix_text}", file=f)
            print_token_log_probs(scores, tokenizer, suffix_text, f=f)
            print(f"[DEBUG] PPL = {ppl:.2f}\n", file=f)
            
        avg_ppl = sum(ppls) / len(ppls)
        f.write(f"\nAverage PPL over {len(ppls)} lines: {avg_ppl:.2f}\n")

    print(f"Generate evaluation complete; logs written to {output_file}")
    return avg_ppl

def evaluate_wikitext_full_sequence_forward(model_name="text", num_lines=50):
    """Evaluate perplexity using forward() on full sequence."""
    # ——— Model setup ———
    print(f"Evaluating {model_name} on {num_lines} lines of WikiText-2 using forward()")
    model_helper = load_model(model_name, "wikitext-2-raw-v1", zero_shot=True)
    tokenizer = model_helper.tokenizer

    # ——— Load WikiText-2 validation lines ———
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{num_lines}]")
    print(f"Loaded {len(wiki)} total lines from dataset")
    
    # Filter out empty lines
    lines = [ex["text"].strip() for ex in wiki if ex["text"].strip()]
    print(f"After filtering empty lines: {len(lines)} lines")
    if len(lines) < len(wiki):
        print("First few empty lines that were filtered:")
        for ex in wiki:
            if not ex["text"].strip():
                print(f"  {ex['text']!r}")
                if len(ex["text"]) > 0:
                    print(f"  (contains {len(ex['text'])} whitespace characters)")
                break

    # ——— Open log file ———
    output_file = f"wikitext_ppl_forward_{model_name}.txt"
    print(f"Writing results to {output_file}")
    with open(output_file, "w", buffering=1) as f:
        f.write(f"WikiText-2 Full Sequence PPLs using forward() ({model_name})\n")
        f.write("=" * 60 + "\n\n")

        ppls = []
        print(f"Evaluating {len(lines)} lines")
        for i, line in enumerate(lines, start=1):
            # Tokenize full sequence
            inputs = tokenizer(line, return_tensors="pt", add_special_tokens=False).to(model_helper.model.device)
            
            # Get logits from forward pass with labels for loss computation
            with torch.no_grad():
                outputs = model_helper.forward(inputs, labels=inputs["input_ids"])
                loss = outputs.loss                  # ⏟  = −(1/N) ∑ log P(tₙ | p, t_{<n})
                ppl = torch.exp(loss).item()
                ppls.append(ppl)

            print(f"[{i:03d}] {line[:50]!r}…", file=f)
            print(f"Full sequence: {line}", file=f)
            print(f"[DEBUG] PPL = {ppl:.2f}\n", file=f)
            
        avg_ppl = sum(ppls) / len(ppls)
        f.write(f"\nAverage PPL over {len(ppls)} lines: {avg_ppl:.2f}\n")

    print(f"Forward evaluation complete; logs written to {output_file}")
    return avg_ppl


import torch
import torch.nn.functional as F

def compute_ppl_via_manual_ce(model, tokenizer, prefix_text, target_text, device="cuda"):
    """
    Compute perplexity using both the model's built-in loss and a manual
    cross-entropy loss on logits to verify they match.
    """
    # 1. Tokenize prefix + target, without special tokens
    prefix_enc = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    target_enc = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
    # 2. Concatenate inputs and build attention mask
    input_ids = torch.cat([prefix_enc["input_ids"], target_enc["input_ids"]], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # 3. Prepare labels: mask out prefix tokens
    labels = input_ids.clone()
    prefix_len = prefix_enc["input_ids"].size(1)
    labels[:, :prefix_len] = -100  # ignore prefix in loss calculation

    # 4. Forward pass with built-in loss
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False)
    loss_builtin = outputs.loss            # = mean(−log P(t_n | context, t_<n))
    ppl_builtin = torch.exp(loss_builtin)  # exp(cross-entropy)

    # 5. Manually compute CE from logits
    logits = outputs.logits                # [1, seq_len, vocab_size]
    # shift so token i’s logit predicts token i+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    loss_manual = F.cross_entropy(flat_logits,
                                 flat_labels,
                                 ignore_index=-100,
                                 reduction="mean")
    ppl_manual = torch.exp(loss_manual)

    return {
        "loss_builtin": loss_builtin.item(),
        "ppl_builtin": ppl_builtin.item(),
        "loss_manual": loss_manual.item(),
        "ppl_manual": ppl_manual.item()
    }

# —— Example usage ——
def compare_ppl_via_manual_ce(model_name, num_examples=30):
    """
    Evaluate perplexity on WikiText-2 using forward() with both built-in and manual CE loss
    """
    from mtv_utils import load_model
    model_helper = load_model(model_name, "wikitext-2-raw-v1", zero_shot=True)
    model = model_helper.model
    tokenizer = model_helper.tokenizer

    # Load WikiText-2 test data
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Open log file
    with open("wikitext_ppl_forward_comparison.txt", "w") as f:
        f.write("WikiText-2 Full Sequence PPLs Comparison\n")
        f.write("=" * 60 + "\n\n")
        
        total_builtin_ppl = 0
        total_manual_ppl = 0
        count = 0
        
        for i, example in enumerate(dataset):
            if i >= num_examples:
                break
                
            text = example['text'].strip()
            if len(text) < 10:  # Skip very short sequences
                continue
                
            # Use BOS token as prefix, full text as target
            prefix_text = tokenizer.bos_token
            target_text = text
            
            # Compute perplexities
            results = compute_ppl_via_manual_ce(model, tokenizer, prefix_text, target_text)
            
            # Log results
            f.write(f"[{i+1:03d}] '{text[:50]}…'\n")
            #f.write(f"Full sequence: {text}\n")
            f.write(f"Built-in PPL = {results['ppl_builtin']:.2f}\n")
            f.write(f"Manual PPL = {results['ppl_manual']:.2f}\n")
            f.write(f"Difference = {results['ppl_builtin'] - results['ppl_manual']:.4f}\n\n")
            
            total_builtin_ppl += results['ppl_builtin']
            total_manual_ppl += results['ppl_manual'] 
            count += 1
            
        # Calculate averages
        avg_builtin_ppl = total_builtin_ppl / count
        avg_manual_ppl = total_manual_ppl / count
        
        f.write("\nSummary:\n")
        f.write(f"Average Built-in PPL: {avg_builtin_ppl:.2f}\n")
        f.write(f"Average Manual PPL: {avg_manual_ppl:.2f}\n")
        f.write(f"Average Difference: {avg_builtin_ppl - avg_manual_ppl:.4f}\n")
        
        return avg_builtin_ppl




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="text", help="Name of the model to evaluate")
    parser.add_argument("--num_examples", type=int, default=30, help="Number of examples to evaluate")
    args = parser.parse_args()
    
    
    # Run both evaluations
    #generate_ppl = evaluate_wikitext_full_sequence_generate(args.model_name, args.num_examples)
    #forward_ppl = evaluate_wikitext_full_sequence_forward(args.model_name, args.num_examples)
    compare_ppl_via_manual_ce(args.model_name, args.num_examples)
    # print("\nSummary:")
    # #print(f"Generate() perplexity: {generate_ppl:.2f}")
    # print(f"Forward() perplexity: {forward_ppl:.2f}")
    # #print(f"Difference (Generate - Forward): {generate_ppl - forward_ppl:.2f}") 