#!/bin/bash

MODEL_NAME="MedViT_tiny"
DATASET="chestmnist"
EPOCHS=50
SAMPLE=0.1

if [ -n "$SAMPLE" ]; then
    SAMPLE_PCT=$(python -c "print(int(float('$SAMPLE') * 100))")
    CONFIG_DIR="sample_${SAMPLE_PCT}pct"
else
    CONFIG_DIR="full"
fi

# Common arguments
COMMON_ARGS="--model_name $MODEL_NAME --dataset $DATASET --sample $SAMPLE --epochs $EPOCHS"

# Function to run experiment
run_experiment() {
    python main.py $COMMON_ARGS "$@"
}

# Run experiments with different loss functions
for loss in ASL BCE CBLoss CBLossOriginal Focal LDACE_CCL; do
    run_experiment --loss_function $loss
    python analyze_single_result.py --csv_path ./results/$loss/$DATASET/$MODEL_NAME/$CONFIG_DIR/model_metrics.csv
done

# Run experiments with sampler
for loss in BCE DBFocal; do
    run_experiment --loss_function $loss --use_sampler
    python analyze_single_result.py --csv_path ./results/$loss/$DATASET/$MODEL_NAME/${CONFIG_DIR}_sampler/model_metrics.csv
done