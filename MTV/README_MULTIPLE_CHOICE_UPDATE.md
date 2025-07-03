# SWDA Multiple Choice Pipeline Update

This document describes the updates made to the MTV pipeline to use multiple choice options for the SWDA (Switchboard Dialog Act) dataset, with letter answers (A, B, C, D) as the target instead of generating full utterances.

## Overview of Changes

The pipeline has been updated to:

1. **Use multiple choice format by default for SWDA**: The `get_format_func()` now returns `format_swda_multiple_choice` for SWDA datasets
2. **Generate realistic distractors**: The format function automatically generates multiple choice options with realistic distractors from other dialogue acts
3. **Train towards letter targets**: The model is now trained to predict single letters (A, B, C, D) instead of full utterances
4. **Support few-shot prompting**: The format function supports N-shot prompting with multiple choice examples
5. **Simplified evaluation logic**: Removed complex conditional logic for multiple choice vs. generation

## Key Changes Made

### 1. Updated `preprocess.py`

#### `get_format_func()` function
- Changed the default format function for SWDA from `format_swda_next_utt` to `format_swda_multiple_choice`

#### `format_swda_multiple_choice()` function
- **Enhanced few-shot examples**: Now generates realistic multiple choice options for each few-shot example
- **Automatic distractor generation**: Samples utterances from different dialogue acts to create realistic distractors
- **Randomized option positions**: Shuffles options to prevent position bias
- **Letter-based targets**: Returns single letters (A, B, C, D) as targets instead of full utterances

### 2. Updated `mtv_eval.py`

#### Simplified evaluation logic
- **Removed complex conditional logic**: No longer needs to check for `--multiple_choice` flag
- **Direct format function usage**: Uses the format function directly without additional processing
- **Letter-based accuracy**: Compares predicted letters with target letters for accuracy calculation
- **Updated summary statistics**: Reports letter accuracy instead of multiple choice accuracy for SWDA

#### Removed deprecated code
- Removed `--multiple_choice` command line argument
- Removed `act_to_utterances` building logic
- Removed complex multiple choice candidate generation

### 3. Updated accuracy calculation

#### For SWDA datasets:
- **Target**: Single letter (A, B, C, D) from the format function
- **Prediction**: First character of model output (extracted and converted to uppercase)
- **Accuracy**: Direct letter comparison

#### For other datasets:
- **Target**: Dialogue act classification
- **Prediction**: Classified dialogue act from generated text
- **Accuracy**: Act comparison

## Usage

### Basic Usage

The pipeline now works seamlessly with SWDA using multiple choice by default:

```bash
python mtv_eval.py --model_name text --data_name swda \
  --train_path data/swda/processed/train.json \
  --val_path data/swda/processed/val.json \
  --num_example 100 --num_shot 4 --eval_num_shot 2 \
  --max_token 5 --cur_mode both \
  --bernoullis_path storage/swda_bernoullis.pt \
  --activation_path storage/swda_activations.pt
```

### Testing the Pipeline

Run the test script to verify the multiple choice format:

```bash
python3 test_multiple_choice.py
```

### Example Output

Run the example script to see formatted examples:

```bash
python3 example_swda_multiple_choice.py --num_examples 3 --num_shot 2
```

## Format Example

### Input Format
```
Chat 1
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
Final Response A:
```

### Target Output
- **Target**: `D` (the letter corresponding to the correct option)
- **Model should predict**: `D`

## Benefits

1. **Faster training**: Single token prediction instead of full utterance generation
2. **Better evaluation**: Clear accuracy metrics based on letter prediction
3. **Reduced complexity**: No need for utterance classification or text generation evaluation
4. **Consistent format**: Multiple choice format is more consistent and easier to evaluate
5. **Realistic distractors**: Automatically generated distractors from the dataset

## Backward Compatibility

- **Other datasets**: Unchanged - still use their original format functions
- **Existing code**: The `format_swda_next_utt` function is still available if needed
- **Command line**: Removed `--multiple_choice` flag since it's now the default for SWDA

## Files Modified

1. `preprocess.py` - Updated format function selection and enhanced multiple choice formatting
2. `mtv_eval.py` - Simplified evaluation logic and updated accuracy calculation
3. `test_multiple_choice.py` - New test script (created)
4. `example_swda_multiple_choice.py` - New example script (created)
5. `README_MULTIPLE_CHOICE_UPDATE.md` - This documentation (created)

## Future Improvements

1. **Dynamic option count**: Allow variable number of options (3-5)
2. **Quality-based distractors**: Use semantic similarity to select better distractors
3. **Balanced sampling**: Ensure equal representation of dialogue acts in options
4. **Contextual distractors**: Generate distractors that are contextually appropriate 