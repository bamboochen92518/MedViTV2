#!/bin/bash

# ============================================================
# Label Range Experiment Script
# Testing different label range combinations for each group
# ============================================================

# Sorted label order: [3, 2, 0, 5, 4, 7, 8, 12, 1, 10, 9, 11, 6, 13]
# Group 0: 3, 2, 0 (indices 0-2)
# Group 1: 5, 4, 7, 8, 12, 1, 10, 9 (indices 3-10)
# Group 2: 11, 6 (indices 11-12)
# Group 3: 13 (index 13)

MODEL_NAME="MedViT_tiny"
DATASET="chestmnist"
EPOCHS=50
SAMPLE=0.1

echo "============================================================"
echo "🚀 Starting Label Range Experiments"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Sample ratio: $SAMPLE"
echo "Epochs: $EPOCHS"
echo "============================================================"
echo ""

# ============================================================
# Group 0 Experiments (label 3-0)
# ============================================================
echo "============================================================"
echo "📊 Group 0 Experiments (Core: 3, 2, 0)"
echo "============================================================"

# Define label_heads: all labels before group 0 start (none) + group 0 labels
GROUP0_HEADS=(3 2 0)
# Define label_tails: group 0 labels + all labels after group 0 end
GROUP0_TAILS=(0 5 4 7 8 12 1 10 9 11 6 13)

for head in "${GROUP0_HEADS[@]}"; do
    for tail in "${GROUP0_TAILS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "Training: label_head=$head, label_tail=$tail"
        echo "------------------------------------------------------------"
        python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" \
            --sample $SAMPLE --label_head $head --label_tail $tail --epochs $EPOCHS
        
        if [ $? -ne 0 ]; then
            echo "❌ Training failed for head=$head, tail=$tail"
        else
            echo "✅ Training completed for head=$head, tail=$tail"
        fi
    done
done

# ============================================================
# Group 1 Experiments (label 5-9)
# ============================================================
echo ""
echo "============================================================"
echo "📊 Group 1 Experiments (Core: 5, 4, 7, 8, 12, 1, 10, 9)"
echo "============================================================"

# Define label_heads: all labels before group 1 start + group 1 first label
GROUP1_HEADS=(3 2 0 5)
# Define label_tails: group 1 last label + all labels after group 1 end
GROUP1_TAILS=(9 11 6 13)

for head in "${GROUP1_HEADS[@]}"; do
    for tail in "${GROUP1_TAILS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "Training: label_head=$head, label_tail=$tail"
        echo "------------------------------------------------------------"
        python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" \
            --sample $SAMPLE --label_head $head --label_tail $tail --epochs $EPOCHS
        
        if [ $? -ne 0 ]; then
            echo "❌ Training failed for head=$head, tail=$tail"
        else
            echo "✅ Training completed for head=$head, tail=$tail"
        fi
    done
done

# ============================================================
# Group 2 Experiments (label 11-6)
# ============================================================
echo ""
echo "============================================================"
echo "📊 Group 2 Experiments (Core: 11, 6)"
echo "============================================================"

# Define label_heads: all labels before group 2 start + group 2 first label
GROUP2_HEADS=(3 2 0 5 4 7 8 12 1 10 9 11)
# Define label_tails: group 2 last label + all labels after group 2 end
GROUP2_TAILS=(6 13)

for head in "${GROUP2_HEADS[@]}"; do
    for tail in "${GROUP2_TAILS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "Training: label_head=$head, label_tail=$tail"
        echo "------------------------------------------------------------"
        python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" \
            --sample $SAMPLE --label_head $head --label_tail $tail --epochs $EPOCHS
        
        if [ $? -ne 0 ]; then
            echo "❌ Training failed for head=$head, tail=$tail"
        else
            echo "✅ Training completed for head=$head, tail=$tail"
        fi
    done
done

# ============================================================
# Group 3 Experiments (label 13)
# ============================================================
echo ""
echo "============================================================"
echo "📊 Group 3 Experiments (Core: 13)"
echo "============================================================"

# Define label_heads: all labels before group 3 start + group 3 label
GROUP3_HEADS=(3 2 0 5 4 7 8 12 1 10 9 11 6 13)
# Define label_tails: only group 3 label (last label, nothing after)
GROUP3_TAILS=(13)

for head in "${GROUP3_HEADS[@]}"; do
    for tail in "${GROUP3_TAILS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "Training: label_head=$head, label_tail=$tail"
        echo "------------------------------------------------------------"
        python main.py --model_name "$MODEL_NAME" --dataset "$DATASET" \
            --sample $SAMPLE --label_head $head --label_tail $tail --epochs $EPOCHS
        
        if [ $? -ne 0 ]; then
            echo "❌ Training failed for head=$head, tail=$tail"
        else
            echo "✅ Training completed for head=$head, tail=$tail"
        fi
    done
done

echo ""
echo "============================================================"
echo "✅ All Label Range Experiments Completed!"
echo "============================================================"
echo ""
echo "Summary:"
echo "  Group 0: 3 heads × 12 tails = 36 experiments"
echo "  Group 1: 4 heads × 4 tails = 16 experiments"
echo "  Group 2: 12 heads × 2 tails = 24 experiments"
echo "  Group 3: 14 heads × 1 tail = 14 experiments"
echo "  Total: 90 experiments"
echo ""

# ============================================================
# Evaluate Group Metrics
# ============================================================
echo ""
echo "============================================================"
echo "📊 Evaluating Group Metrics"
echo "============================================================"
echo ""
echo "Running evaluate_group_metrics.py to extract core metrics..."
python evaluate_group_metrics.py

if [ $? -ne 0 ]; then
    echo "❌ Group metrics evaluation failed"
else
    echo "✅ Group metrics evaluation completed"
fi

# ============================================================
# Merge and Analyze Group Results
# ============================================================
echo ""
echo "============================================================"
echo "🔄 Merging Group Evaluations and Analyzing Best Configs"
echo "============================================================"
echo ""
echo "Running merge_group_expanding_algo_evaluations.py..."
python merge_group_expanding_algo_evaluations.py --eval_dir results/grouping_expanding_algo_evaluation

if [ $? -ne 0 ]; then
    echo "❌ Merging and analysis failed"
else
    echo "✅ Merging and analysis completed"
fi

echo ""
echo "============================================================"
echo "🎉 All Tasks Completed!"
echo "============================================================"
echo ""
echo "📊 Results saved to:"
echo "  - Group evaluations: results/grouping_expanding_algo_evaluation/"
echo "    • group0_evaluation.csv"
echo "    • group1_evaluation.csv"
echo "    • group2_evaluation.csv"
echo "    • group3_evaluation.csv"
echo "  - Merged results:"
echo "    • all_groups_merged.csv"
echo "    • best_configs_summary.csv"
echo "    • best_configs_tier_analysis.csv"
echo "    • comprehensive_report.txt"
echo "============================================================"