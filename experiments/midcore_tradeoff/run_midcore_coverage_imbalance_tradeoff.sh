#!/bin/bash

# ============================================================
# Mid-Core Coverage x Imbalance Trade-off Experiment (Experiment A)
#
# Purpose:
#   Decouple two factors that affect mid-core performance:
#     1) Coverage  : how many labels are included in training
#     2) Imbalance : how skewed the label distribution is during training
#
#   We pick three coverage levels from the greedy group experiment:
#     S0 : core-only       label_head=5  label_tail=12   (5 labels)
#     S7 : best trade-off  label_head=3  label_tail=11   (12 labels)
#     S9 : full range      label_head=3  label_tail=13   (14 labels)
#
#   For each coverage level we sweep alpha_balance:
#     0.0 : natural distribution (high imbalance)
#     0.5 : partial inverse-frequency rebalance
#     1.0 : full inverse-frequency rebalance (low imbalance)
#
#   Expected hypothesis-supporting pattern:
#     S9 + balance > S9 + natural   (wide coverage IS useful, was hidden by imbalance)
#     S0 + balance ~= S0 + natural  (narrow coverage bottleneck is information, not imbalance)
#     Balanced curve becomes monotonic in coverage (S0 < S7 < S9 or S0 < S7 ~ S9)
# ============================================================
# Always run from repo root regardless of where the script is invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
MODEL_NAME="MedViT_tiny"
DATASET="chestmnist"
EPOCHS=50
SAMPLE=0.1
LOSS_FUNCTION="default"

if [ -f "./.experiment.env" ]; then
    source ./.experiment.env
fi

VENV_ACTIVATE_PATH="${VENV_ACTIVATE:-../LABenv/bin/activate}"
if [ -f "$VENV_ACTIVATE_PATH" ]; then
    source "$VENV_ACTIVATE_PATH"
    echo "✅ Activated venv: $VENV_ACTIVATE_PATH"
else
    echo "⚠️  Venv activate script not found: $VENV_ACTIVATE_PATH"
    echo "   Continue with current Python environment."
fi

# Coverage settings: "tag head tail"
COVERAGE_SETTINGS=(
    "S0 5 12"
    "S7 3 11"
    "S9 3 13"
)

# Imbalance settings (alpha_balance)
ALPHAS=(0.0 0.5 1.0)

if [ -n "$SAMPLE" ]; then
    SAMPLE_PCT=$(python -c "print(int(float('$SAMPLE') * 100))")
else
    SAMPLE_PCT=100
fi

echo "============================================================"
echo "🚀 Mid-Core Coverage x Imbalance Trade-off Experiment"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS  Sample: $SAMPLE"
echo "Coverage levels: ${COVERAGE_SETTINGS[*]}"
echo "Alpha_balance sweep: ${ALPHAS[*]}"
echo "============================================================"
echo ""

mkdir -p ./results/tradeoff_plots

for cov in "${COVERAGE_SETTINGS[@]}"; do
    TAG=$(echo "$cov" | awk '{print $1}')
    HEAD=$(echo "$cov" | awk '{print $2}')
    TAIL=$(echo "$cov" | awk '{print $3}')

    for alpha in "${ALPHAS[@]}"; do
        BAL_TAG=$(python -c "print('{:.2f}'.format(float('$alpha')).replace('.', 'p'))")
        CONFIG_DIR="class_${HEAD}_to_${TAIL}_sample_${SAMPLE_PCT}pct_clsbal_a${BAL_TAG}"
        SAMPLED_COOCC_PATH="./results/${LOSS_FUNCTION}/${DATASET}/${MODEL_NAME}/${CONFIG_DIR}/sampled_cooccurrence.csv"

        echo "------------------------------------------------------------"
        echo "📊 Coverage=$TAG ($HEAD..$TAIL)  alpha_balance=$alpha"
        echo "------------------------------------------------------------"

        # Force fresh sampled co-occurrence file each run.
        rm -f "$SAMPLED_COOCC_PATH"

        python main.py \
            --model_name "$MODEL_NAME" \
            --dataset "$DATASET" \
            --label_head $HEAD \
            --label_tail $TAIL \
            --sample $SAMPLE \
            --class_balanced_sampling True \
            --class_balanced_alpha $alpha \
            --plot_distribution False \
            --epochs $EPOCHS

        if [ $? -ne 0 ]; then
            echo "❌ Training failed: $TAG alpha=$alpha"
            continue
        fi

        CSV_PATH="./results/${LOSS_FUNCTION}/${DATASET}/${MODEL_NAME}/${CONFIG_DIR}/model_metrics.csv"
        if [ -f "$CSV_PATH" ]; then
            echo "📈 Best epoch summary for $TAG alpha=$alpha"
            python analyze_single_result.py --csv_path "$CSV_PATH"
        else
            echo "⚠️  Metrics CSV not found: $CSV_PATH"
        fi

        echo "✅ Completed: $TAG alpha=$alpha"
        echo ""
    done
done

echo "============================================================"
echo "✅ Coverage x Imbalance Experiment Complete"
echo "============================================================"
echo ""
echo "📈 Generating trade-off plot (coverage_imbalance mode)..."
python experiments/midcore_tradeoff/plot_midcore_tradeoff.py --mode coverage_imbalance
echo "============================================================"
