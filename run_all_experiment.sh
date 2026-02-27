#!/bin/bash

MODEL_NAME="MedViT_tiny"
DATASET="chestmnist"
EPOCHS=50
SAMPLE=0.1

# Common arguments
COMMON_ARGS="--model_name $MODEL_NAME --dataset $DATASET --sample $SAMPLE --epochs $EPOCHS"

# Function to run experiment
run_experiment() {
    python main.py $COMMON_ARGS "$@"
}

# Run experiments with different loss functions
for loss in ASL BCE CBLoss CBLossOriginal Focal; do
    run_experiment --loss_function $loss
done

# Run experiments with sampler
for loss in BCE DBFocal; do
    run_experiment --loss_function $loss --use_sampler
done