# Prompt Engineering Guide for MTV with Llama-3.1-8B-Instruct

This guide explains how to optimize prompts for Llama-3.1-8B-Instruct within the MTV framework while maintaining consistency with other datasets and ensuring compatibility with activation learning.

## Key Principles for MTV Compatibility

### 1. **Consistency Across Datasets**
All datasets in MTV use similar prompt structures to ensure the activation learning process works consistently:

- **Flower/CUB/DTD**: Simple multiple choice with `<image>` tags
- **VizWiz/OKVQA**: Question-answer format with `<image>` tags  
- **SWDA**: Multiple choice dialogue act classification

### 2. **Minimal Token Overhead**
Keep prompts concise to focus the model's attention on the task and reduce interference with activation learning.

### 3. **Clear Task Instructions**
Use explicit but concise instructions that guide the model without overwhelming it.

## Optimized SWDA Prompt Structure

### Current Format (MTV-Compatible)
```
A: How are you doing today?
B: I'm doing well, thank you.
Given the conversation context above, which response is most appropriate?
B: 1) I'm doing well, thank you for asking. 2) I see. 3) That's interesting. 4) Go on.
Answer with the option's number from the given choice directly. Answer: 1
```

### Key Features
1. **Direct context presentation** - No extra formatting
2. **Clear question** - Explicit task description
3. **Numbered options** - Easy to parse
4. **Explicit instruction** - "Answer with the option's number"
5. **Consistent structure** - Matches other MTV datasets

## Why This Works Better Than Instruction-Tuned Format

### 1. **Activation Learning Compatibility**
- MTV learns activations from the model's internal representations
- Complex system prompts can interfere with this process
- Simple, direct prompts allow cleaner activation extraction

### 2. **Consistency with Other Datasets**
- Flower: `<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}`
- CUB: `<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}`
- SWDA: `Given the conversation context above, which response is most appropriate?`

### 3. **Token Efficiency**
- Fewer tokens = more focused attention
- Reduces computational overhead
- Better for few-shot learning

## Prompt Engineering Best Practices for MTV

### 1. **Use Explicit Instructions**
```
✅ Good: "Answer with the option's number from the given choice directly."
❌ Bad: "Choose the best option" (ambiguous)
```

### 2. **Maintain Consistent Formatting**
```
✅ Good: "1) option1 2) option2 3) option3 4) option4"
❌ Bad: "A) option1\nB) option2\nC) option3\nD) option4"
```

### 3. **Clear Context Separation**
```
✅ Good: 
Context line 1
Context line 2
Question
Options
Answer:

❌ Bad:
**Context:** [context]
**Question:** [question]
**Options:** [options]
```

### 4. **Avoid System Prompts**
```
❌ Don't use:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant...

✅ Use:
Direct task description
```

## Testing and Validation

### 1. **Check Activation Learning Compatibility**
- Ensure prompts don't interfere with `gather_last_attn_activations()`
- Verify consistent token patterns across examples
- Test with different `num_shot` values

### 2. **Validate Model Responses**
- Check for consistent number extraction
- Verify no extra text generation
- Ensure proper dialogue act classification

### 3. **Compare with Other Datasets**
- Test prompt length consistency
- Verify similar performance patterns
- Check activation similarity

## Implementation Example

```python
def format_swda_multiple_choice_optimized(filtered_dataset, full_dataset, cur_item=None, num_shot=0, model_helper=None, split="train"):
    """
    Optimized format function that's compatible with MTV activation learning.
    """
    # ... setup code ...
    
    prompt = ""
    
    if num_shot > 0:
        # Format few-shot examples
        for example in shot_examples:
            context = example.get("text", "").strip()
            prompt += f"{context}\n"
            prompt += f"Given the conversation context above, which response is most appropriate?\n"
            prompt += f"{example['caller']}: 1) {options[0]} 2) {options[1]} 3) {options[2]} 4) {options[3]}\n"
            prompt += f"Answer with the option's number from the given choice directly. Answer: {correct_number}\n\n"
    
    # Format current query
    context = data.get("text", "").strip()
    prompt += f"{context}\n"
    prompt += f"Given the conversation context above, which response is most appropriate?\n"
    prompt += f"{data['caller']}: 1) {options[0]} 2) {options[1]} 3) {options[2]} 4) {options[3]}\n"
    prompt += "Answer with the option's number from the given choice directly. Answer:"
    
    return prompt, [], str(target_number), data.get("utterance_id", -1)
```

## Performance Optimization Tips

### 1. **Token Length Management**
- Keep prompts under 512 tokens for optimal performance
- Use concise context descriptions
- Minimize redundant instructions

### 2. **Few-Shot Optimization**
- Use 2-4 examples for best results
- Ensure examples are diverse but consistent
- Maintain same format across all examples

### 3. **Error Handling**
- Validate option shuffling
- Check for empty contexts
- Ensure proper number extraction

## Conclusion

The key to successful prompt engineering in MTV is **consistency and simplicity**. By maintaining the same structural patterns across all datasets and avoiding complex system prompts, we ensure:

1. **Compatibility** with activation learning processes
2. **Consistency** across different datasets and models
3. **Efficiency** in token usage and computation
4. **Reliability** in model responses and evaluation

This approach allows MTV to effectively learn and manipulate model activations while maintaining high performance on the SWDA multiple choice task. 