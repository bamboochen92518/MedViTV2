#!/bin/bash

# ============================================================
# Grouping Experiment Script
# Purpose: Test different label grouping strategies based on 
# imbalance ratio thresholds
# ============================================================

MODEL_NAME="MedViT_tiny"
DATASET="chestmnist"
EPOCHS=50
SAMPLE=0.1

echo "============================================================"
echo "🚀 Starting Label Grouping Experiments"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Sample ratio: $SAMPLE"
echo "Epochs: $EPOCHS"
echo "============================================================"
echo ""

# ============================================================
# Experiment 1: Mean_ir_threshold=1.5, cvir_threshold=0.2
# ============================================================
echo "============================================================"
echo "📊 Experiment 1: mean_ir_threshold=1.5, cvir_threshold=0.2"
echo "============================================================"
echo "  Group 0: Classes [3] (1 classes)"
echo "  Group 1: Classes [2, 0] (2 classes)"
echo "  Group 2: Classes [5, 4, 7, 8] (4 classes)"
echo "  Group 3: Classes [12, 1, 10, 9] (4 classes)"
echo "  Group 4: Classes [11, 6] (2 classes)"
echo "  Group 5: Classes [13] (1 classes)"
echo "------------------------------------------------------------"

echo "Training Group 0: [3]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 3 --label_tail 3 --epochs $EPOCHS

echo "Training Group 1: [2, 0]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 2 --label_tail 0 --epochs $EPOCHS

echo "Training Group 2: [5, 4, 7, 8]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 5 --label_tail 8 --epochs $EPOCHS

echo "Training Group 3: [12, 1, 10, 9]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 12 --label_tail 9 --epochs $EPOCHS

echo "Training Group 4: [11, 6]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 11 --label_tail 6 --epochs $EPOCHS

echo "Training Group 5: [13]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 13 --label_tail 13 --epochs $EPOCHS

echo "✅ Experiment 1 training completed!"
echo ""
echo "📊 Aggregating Experiment 1 results..."
python aggregate_group_results.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --experiment_name "exp1_ir1.5_cvir0.2" \
    --group_configs "3,3" "2,0" "5,8" "12,9" "11,6" "13,13" \
    --sample $SAMPLE
echo ""

# ============================================================
# Experiment 2: Mean_ir_threshold=2.5, cvir_threshold=0.4
# ============================================================
echo "============================================================"
echo "📊 Experiment 2: mean_ir_threshold=2.5, cvir_threshold=0.4"
echo "============================================================"
echo "  Group 0: Classes [3, 2, 0] (3 classes)"
echo "  Group 1: Classes [5, 4, 7, 8, 12, 1, 10, 9] (8 classes)"
echo "  Group 2: Classes [11, 6] (2 classes)"
echo "  Group 3: Classes [13] (1 classes)"
echo "------------------------------------------------------------"

echo "Training Group 0: [3, 2, 0]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 3 --label_tail 0 --epochs $EPOCHS

echo "Training Group 1: [5, 4, 7, 8, 12, 1, 10, 9]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 5 --label_tail 9 --epochs $EPOCHS

echo "Training Group 2: [11, 6]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 11 --label_tail 6 --epochs $EPOCHS

echo "Training Group 3: [13]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 13 --label_tail 13 --epochs $EPOCHS

echo "✅ Experiment 2 training completed!"
echo ""
echo "📊 Aggregating Experiment 2 results..."
python aggregate_group_results.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --experiment_name "exp2_ir2.5_cvir0.4" \
    --group_configs "3,0" "5,9" "11,6" "13,13" \
    --sample $SAMPLE
echo ""

# ============================================================
# Experiment 3: Mean_ir_threshold=3.5, cvir_threshold=0.6
# ============================================================
echo "============================================================"
echo "📊 Experiment 3: mean_ir_threshold=3.5, cvir_threshold=0.6"
echo "============================================================"
echo "  Group 0: Classes [3, 2, 0, 5, 4, 7, 8, 12] (8 classes)"
echo "  Group 1: Classes [1, 10, 9, 11, 6] (5 classes)"
echo "  Group 2: Classes [13] (1 classes)"
echo "------------------------------------------------------------"

echo "Training Group 0: [3, 2, 0, 5, 4, 7, 8, 12]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 3 --label_tail 12 --epochs $EPOCHS

echo "Training Group 1: [1, 10, 9, 11, 6]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 1 --label_tail 6 --epochs $EPOCHS

echo "Training Group 2: [13]"
python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" --sample $SAMPLE --label_head 13 --label_tail 13 --epochs $EPOCHS

echo "✅ Experiment 3 training completed!"
echo ""
echo "📊 Aggregating Experiment 3 results..."
python aggregate_group_results.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --experiment_name "exp3_ir3.5_cvir0.6" \
    --group_configs "3,12" "1,6" "13,13" \
    --sample $SAMPLE
echo ""

# ============================================================
# All Experiments Completed
# ============================================================
echo "============================================================"
echo "✅ All Grouping Experiments Completed Successfully!"
echo "============================================================"
echo "Total experiments run: 3 grouping strategies"
echo "Total groups trained: 13 models"
echo "Results saved in: ./results/"
echo "Aggregated results in: ./results/group_evaluation/"
echo ""
echo "📊 Summary files generated:"
echo "  - exp1_ir1.5_cvir0.2_per_class.csv"
echo "  - exp1_ir1.5_cvir0.2_tier_summary.csv"
echo "  - exp2_ir2.5_cvir0.4_per_class.csv"
echo "  - exp2_ir2.5_cvir0.4_tier_summary.csv"
echo "  - exp3_ir3.5_cvir0.6_per_class.csv"
echo "  - exp3_ir3.5_cvir0.6_tier_summary.csv"
echo "============================================================"
echo ""