# Using SwDA (Switchboard Dialog Act) Dataset with MTV

This guide explains how to set up and use the Switchboard Dialog Act (SwDA) dataset with the MTV framework for dialog act classification.

## Dataset Setup

1. First, download the SwDA dataset. You can get it from:
   - [SwDA Download Link](https://web.stanford.edu/~jurafsky/swb1_dialogact_annot.tar.gz)
   - Or use the NXT version: [NXT-format Switchboard](https://groups.inf.ed.ac.uk/switchboard/)

2. Extract the downloaded dataset and convert it to CSV format with the following columns:
   - conversation_no: Unique identifier for the conversation
   - utterance_idx: Index of the utterance within the conversation
   - act_tag: Dialog act label
   - caller: Speaker identifier (A or B)
   - text: The utterance text

3. Place the CSV file in the following directory structure:
```
MTV/
├── data/
│   └── swda/
│       └── swda.csv
```

4. Run the processing script to convert the data into the required format:
```bash
python process_swda.py
```

This will create three files in `data/swda/processed/`:
- train.json
- val.json
- test.json

## Running MTV with SwDA

Use the following command to run MTV evaluation on SwDA:

```bash
python mtv_eval.py \
    --model_name text \
    --data_name swda \
    --train_path data/swda/processed/train.json \
    --val_path data/swda/processed/val.json \
    --num_example 100 \
    --num_shot 4 \
    --eval_num_shot 0 \
    --max_token 10 \
    --bernoullis_path storage/swda_bernoullis.pt \
    --activation_path storage/swda_activations.pt \
    --is_eval True \
    --result_folder results/swda/ \
    --cur_mode both \
    --experiment_name swda_experiment
```

## Data Format

Each line in the JSON files contains a dictionary with:
- text: The utterance text
- dialog_act: The dialog act label
- utterance_id: Unique identifier for the utterance (conversation_no_utterance_idx)
- caller: Speaker identifier

## Dialog Act Labels

The SwDA dataset uses 42 dialog act tags. Some common ones include:
- sd: Statement-non-opinion
- sv: Statement-opinion
- qy: Yes-No-Question
- qw: Wh-Question
- b: Acknowledge (Backchannel)
- aa: Agree/Accept
- etc.

## Model Details

The framework uses Llama 2 7B Chat as the base model and applies MTV's intervention mechanism to improve dialog act classification performance. The model processes text-only input (no images) and outputs dialog act labels. 