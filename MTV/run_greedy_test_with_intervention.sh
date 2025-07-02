#!/bin/bash

# Script to run greedy decoding verification test with intervention
# This test verifies that greedy decoding is actually taking the most probable token every time
# for both clean and intervention models

export CUDA_VISIBLE_DEVICES=0

echo "Running greedy decoding verification test with intervention..."
echo "This test checks that argmax of scores corresponds to generated tokens at each position."
echo "Testing both clean and intervention generation."

# Check if activation data exists
if [ -f "storage/activations_sd.pt" ]; then
    echo "Found activation data, running both clean and intervention tests..."
    
    python3 test_greedy_decoding.py \
        --model_name text \
        --data_name swda \
        --train_path data/swda/train.json \
        --val_path data/swda/validation.json \
        --num_examples 4 \
        --max_token 100 \
        --eval_num_shot 4 \
        --cur_mode both \
        --activation_path ./storage/activations_sd.pt \
        --seed 42
else
    echo "No activation data found, running clean generation test only..."
    echo "To test intervention, first run the training pipeline to generate activation data."
    
    python3 test_greedy_decoding.py \
        --model_name text \
        --data_name swda \
        --train_path data/swda/train.json \
        --val_path data/swda/validation.json \
        --num_examples 5 \
        --max_token 100 \
        --eval_num_shot 4 \
        --cur_mode clean \
        --seed 42
fi

echo ""
echo "Test completed! Check the output JSON file for detailed results."
echo "If greedy accuracy is 100% for both clean and intervention, then greedy decoding is working correctly." 