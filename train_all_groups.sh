#!/bin/bash

# ============================================================
# Auto Train All Label Groups Shell Script
# ============================================================

# Configuration
MODEL_NAME="MedViT_tiny"
DATASET="chestmnist"
EPOCHS=100
BATCH_SIZE=24
LR=0.0001
PRETRAINED=False

echo "============================================================"
echo "🚀 Starting Sequential Group Training"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Epochs per group: $EPOCHS"
echo "Note: Groups are dynamically computed using DP algorithm"
echo "============================================================"
echo ""

# Train Group 0
echo "============================================================"
echo "📊 Training Group 0 (Classes 3, 2, 0)"
echo "============================================================"
python main.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --group 0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --pretrained $PRETRAINED

if [ $? -ne 0 ]; then
    echo "❌ Group 0 training failed!"
    exit 1
fi
echo "✅ Group 0 training completed!"
echo ""

# Train Group 1
echo "============================================================"
echo "📊 Training Group 1 (Classes 5, 4, 7, 8, 12, 1, 10, 9)"
echo "============================================================"
python main.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --group 1 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --pretrained $PRETRAINED

if [ $? -ne 0 ]; then
    echo "❌ Group 1 training failed!"
    exit 1
fi
echo "✅ Group 1 training completed!"
echo ""

# Train Group 2
echo "============================================================"
echo "📊 Training Group 2 (Classes 11, 6)"
echo "============================================================"
python main.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --group 2 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --pretrained $PRETRAINED

if [ $? -ne 0 ]; then
    echo "❌ Group 2 training failed!"
    exit 1
fi
echo "✅ Group 2 training completed!"
echo ""

# Train Group 3
echo "============================================================"
echo "📊 Training Group 3 (Class 13)"
echo "============================================================"
python main.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --group 3 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --pretrained $PRETRAINED

if [ $? -ne 0 ]; then
    echo "❌ Group 3 training failed!"
    exit 1
fi
echo "✅ Group 3 training completed!"
echo ""

# Complete
echo "============================================================"
echo "✅ All Groups Training Completed Successfully!"
echo "============================================================"
echo "Saved models:"
echo "  - ${MODEL_NAME}_${DATASET}_group0.pth (3 classes)"
echo "  - ${MODEL_NAME}_${DATASET}_group1.pth (8 classes)"
echo "  - ${MODEL_NAME}_${DATASET}_group2.pth (2 classes)"
echo "  - ${MODEL_NAME}_${DATASET}_group3.pth (1 class)"
echo "============================================================"
echo ""
echo "Metrics CSV files:"
echo "  - ${MODEL_NAME}_${DATASET}_group0_metrics.csv"
echo "  - ${MODEL_NAME}_${DATASET}_group1_metrics.csv"
echo "  - ${MODEL_NAME}_${DATASET}_group2_metrics.csv"
echo "  - ${MODEL_NAME}_${DATASET}_group3_metrics.csv"
echo "============================================================"
