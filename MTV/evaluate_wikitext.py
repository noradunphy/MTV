import torch
from datasets import load_dataset
from mtv_utils import load_model, ppl_from_scores, print_token_log_probs
import re

def split_at_sentence_boundary(text, tokenizer, target_length):
    """Split text at sentence boundary closest to target length."""
    # First try to split at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        prefix = sentences[0]
        suffix = ' '.join(sentences[1:])
        return prefix, suffix
    
    # If no sentence boundaries, split at word boundaries
    words = text.split()
    if len(words) > 1:
        mid = len(words) // 2
        prefix = ' '.join(words[:mid])
        suffix = ' '.join(words[mid:])
        return prefix, suffix
    
    # If single word, split in middle
    return text[:len(text)//2], text[len(text)//2:]

def sanity_check_wikitext(model_name="text", num_lines=50):
    # ——— Model setup (same as in recompute_perplexities) ———
    model_helper = load_model(model_name, "wikitext-2-raw-v1", zero_shot=True)
    
    model_helper.model.to("cuda")
    tokenizer = model_helper.tokenizer

    # ——— Load WikiText-2 validation lines ———
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"validation[:{num_lines}]")
    lines = [ex["text"].strip() for ex in wiki if ex["text"].strip()]

    # ——— Open log file ———
    output_file = f"wikitext_ppl_sanity_{model_name}.txt"
    with open(output_file, "w", buffering=1) as f:
        f.write(f"WikiText-2 Sanity Check PPLs ({model_name})\n")
        f.write("=" * 60 + "\n\n")

        ppls = []
        for i, line in enumerate(lines, start=1):
            # Split line into prefix and suffix at natural boundaries
            prefix_text, suffix_text = split_at_sentence_boundary(line, tokenizer, len(line)//2)
            
            # Tokenize prefix
            prefix_enc = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
            prefix_ids = prefix_enc["input_ids"].to("cuda")
            
            # Tokenize suffix for length reference
            suffix_enc = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)
            suffix_ids = suffix_enc["input_ids"][0]

            # Build context for generation
            context_input = {
                "input_ids": prefix_ids,
                "attention_mask": torch.ones_like(prefix_ids)
            }

            # Generate raw logits for the suffix length
            output, scores = model_helper.generate(
                context_input,
                max_new_tokens=len(suffix_ids),
                return_scores=True,
                return_dict_in_generate=True
            )
            
            # Evaluate perplexity on the suffix only
            ppl, _ = ppl_from_scores(scores, tokenizer, suffix_text)
            ppls.append(ppl)

            print(f"[{i:03d}] {line[:50]!r}…", file=f)
            print(f"Prefix: {prefix_text}", file=f)
            print(f"Suffix: {suffix_text}", file=f)
            print(f"Output: {output}", file=f)
            print_token_log_probs(scores, tokenizer, suffix_text, f)  # Print log probs for suffix only
            print(f"[DEBUG] PPL = {ppl:.2f}\n", file=f)
            
        avg_ppl = sum(ppls) / len(ppls)
        f.write(f"\nAverage PPL over {len(ppls)} lines: {avg_ppl:.2f}\n")

    print(f"Sanity check complete; logs written to {output_file}")

if __name__ == "__main__":
    sanity_check_wikitext("text", num_lines=50)