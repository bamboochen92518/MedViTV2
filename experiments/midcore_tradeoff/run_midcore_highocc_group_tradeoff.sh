#!/bin/bash

# ============================================================
# Mid-Core High-Cooccurrence Group Expansion Experiment
# Purpose:
#   Start from mid core labels only, then greedily expand one side
#   (left/right neighbor in sorted order) by choosing the side that
#   gives larger co-occurrence gain at each step.
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
SORTED_LABELS=(3 2 0 5 4 7 8 12 1 10 9 11 6 13)
SORTED_LABELS_CSV="3,2,0,5,4,7,8,12,1,10,9,11,6,13"
CORE_HEAD=5
CORE_TAIL=12
COOCC_CSV="./results/sampling_plots/chestmnist_sample10_original_cooccurrence.csv"
SUMMARY_CSV="./results/tradeoff_plots/midcore_greedy_group_summary.csv"

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

# Middle core labels (user-defined)
# Core: [5, 4, 7, 8, 12]
# Sorted order: [3, 2, 0, 5, 4, 7, 8, 12, 1, 10, 9, 11, 6, 13]

if [ -n "$SAMPLE" ]; then
    SAMPLE_PCT=$(python -c "print(int(float('$SAMPLE') * 100))")
    CONFIG_SUFFIX="sample_${SAMPLE_PCT}pct"
else
    CONFIG_SUFFIX="full"
fi

echo "============================================================"
echo "🚀 Mid-Core Group Expansion Trade-off Experiment"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Sample ratio: $SAMPLE"
echo "Epochs: $EPOCHS"
echo "Middle core labels: [5,4,7,8,12]"
echo "Sorted labels: [${SORTED_LABELS[*]}]"
echo "============================================================"
echo ""

# Ensure output directory exists
mkdir -p ./results/tradeoff_plots

# Initialize summary CSV
echo "step,label_head,label_tail,added_side,added_label,left_gain,right_gain,mid_core_auc" > "$SUMMARY_CSV"

# Find initial head/tail indices for core range
HEAD_IDX=-1
TAIL_IDX=-1
for i in "${!SORTED_LABELS[@]}"; do
    if [ "${SORTED_LABELS[$i]}" -eq "$CORE_HEAD" ]; then
        HEAD_IDX=$i
    fi
    if [ "${SORTED_LABELS[$i]}" -eq "$CORE_TAIL" ]; then
        TAIL_IDX=$i
    fi
done

if [ $HEAD_IDX -lt 0 ] || [ $TAIL_IDX -lt 0 ] || [ $HEAD_IDX -gt $TAIL_IDX ]; then
    echo "❌ Cannot locate valid core range in SORTED_LABELS"
    exit 1
fi

LAST_IDX=$((${#SORTED_LABELS[@]} - 1))

if [ ! -f "$COOCC_CSV" ]; then
    echo "⚠️  Co-occurrence CSV not found: $COOCC_CSV"
    echo "   Falling back to generated train co-occurrence matrix..."
    python - << 'PY'
import os
import numpy as np
import pandas as pd
import medmnist
from medmnist import INFO

os.makedirs('./results/sampling_plots', exist_ok=True)
info = INFO['chestmnist']
DataClass = getattr(medmnist, info['python_class'])
ds = DataClass(split='train', download=True, as_rgb=True, root='./data', size=224, mmap_mode='r')

labels = []
for i in range(len(ds)):
    _, y = ds[i]
    labels.append(np.array(y).flatten())
Y = np.stack(labels, axis=0).astype(np.int32)
C = Y.T @ Y

cols = [f'Label_{i}' for i in range(C.shape[0])]
df = pd.DataFrame(C, index=cols, columns=cols)
df.to_csv('./results/sampling_plots/chestmnist_train_original_cooccurrence.csv')
print('Generated: ./results/sampling_plots/chestmnist_train_original_cooccurrence.csv')
PY
    COOCC_CSV="./results/sampling_plots/chestmnist_train_original_cooccurrence.csv"
fi

STEP=0
while true; do
    LABEL_HEAD=${SORTED_LABELS[$HEAD_IDX]}
    LABEL_TAIL=${SORTED_LABELS[$TAIL_IDX]}
    EXP_NAME="step_${STEP}_${LABEL_HEAD}_to_${LABEL_TAIL}"

    echo "------------------------------------------------------------"
    echo "📊 Running $EXP_NAME"
    echo "   label_head=$LABEL_HEAD, label_tail=$LABEL_TAIL"
    echo "   current_idx_range=$HEAD_IDX..$TAIL_IDX"
    echo "------------------------------------------------------------"

    python main.py \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET" \
        --sample $SAMPLE \
        --label_head $LABEL_HEAD \
        --label_tail $LABEL_TAIL \
        --epochs $EPOCHS

    if [ $? -ne 0 ]; then
        echo "❌ Training failed: $EXP_NAME"
        continue
    fi

    CSV_PATH="./results/${LOSS_FUNCTION}/${DATASET}/${MODEL_NAME}/class_${LABEL_HEAD}_to_${LABEL_TAIL}_${CONFIG_SUFFIX}/model_metrics.csv"

    if [ -f "$CSV_PATH" ]; then
        echo "📈 Analyzing best epoch for $EXP_NAME"
        python analyze_single_result.py --csv_path "$CSV_PATH"

        MID_CORE_AUC=$(python - << PY
import pandas as pd
df = pd.read_csv('$CSV_PATH')
best = df.iloc[df['auc'].idxmax()]
core = [5, 4, 7, 8, 12]
vals = []
for c in core:
    col = f'AUC_class_{c}'
    if col in df.columns:
        vals.append(float(best[col]))
print(sum(vals) / len(vals) if vals else float('nan'))
PY
)
    else
        echo "⚠️  Metrics CSV not found: $CSV_PATH"
        MID_CORE_AUC="nan"
    fi

    # Stop after full coverage
    if [ $HEAD_IDX -eq 0 ] && [ $TAIL_IDX -eq $LAST_IDX ]; then
        echo "$STEP,$LABEL_HEAD,$LABEL_TAIL,none,none,0,0,$MID_CORE_AUC" >> "$SUMMARY_CSV"
        echo "✅ Full label range reached. Stop."
        break
    fi

    # Decide next expansion side by co-occurrence gain
    DECISION=$(python - << PY
import math
import pandas as pd

co = pd.read_csv('$COOCC_CSV', index_col=0)
co_values = co.values
sorted_labels = [int(x) for x in '$SORTED_LABELS_CSV'.split(',')]
head_idx = $HEAD_IDX
tail_idx = $TAIL_IDX

current = sorted_labels[head_idx:tail_idx+1]
left_label = sorted_labels[head_idx-1] if head_idx > 0 else None
right_label = sorted_labels[tail_idx+1] if tail_idx < len(sorted_labels)-1 else None

def gain(candidate, selected):
    if candidate is None:
        return -1.0
    vals = []
    cii = max(float(co_values[candidate, candidate]), 1.0)
    for l in selected:
        if l == candidate:
            continue
        ljj = max(float(co_values[l, l]), 1.0)
        vals.append(float(co_values[candidate, l]) / math.sqrt(cii * ljj))
    if not vals:
        return -1.0
    return sum(vals) / len(vals)

lg = gain(left_label, current)
rg = gain(right_label, current)

if left_label is None:
    side, lab = 'right', right_label
elif right_label is None:
    side, lab = 'left', left_label
elif lg >= rg:
    side, lab = 'left', left_label
else:
    side, lab = 'right', right_label

print(f"{side},{lab},{lg},{rg}")
PY
)

    ADDED_SIDE=$(echo "$DECISION" | cut -d',' -f1)
    ADDED_LABEL=$(echo "$DECISION" | cut -d',' -f2)
    LEFT_GAIN=$(echo "$DECISION" | cut -d',' -f3)
    RIGHT_GAIN=$(echo "$DECISION" | cut -d',' -f4)

    echo "$STEP,$LABEL_HEAD,$LABEL_TAIL,$ADDED_SIDE,$ADDED_LABEL,$LEFT_GAIN,$RIGHT_GAIN,$MID_CORE_AUC" >> "$SUMMARY_CSV"

    if [ "$ADDED_SIDE" = "left" ]; then
        HEAD_IDX=$((HEAD_IDX - 1))
    else
        TAIL_IDX=$((TAIL_IDX + 1))
    fi

    echo "✅ Completed: $EXP_NAME"
    echo "   Decision: add $ADDED_SIDE label $ADDED_LABEL (left_gain=$LEFT_GAIN, right_gain=$RIGHT_GAIN)"
    echo ""

    STEP=$((STEP + 1))
done

echo "============================================================"
echo "✅ Mid-Core Group Expansion Trade-off Experiment Complete"
echo "============================================================"
echo ""
echo "Suggested next steps:"
echo "  1) Inspect greedy summary: $SUMMARY_CSV"
echo "  2) Plot x=MeanIR, y=CoOccScore"
echo "  3) Compare mid-core AUC along greedy expansion path"
echo ""
echo "📈 Generating trade-off plot (group mode)..."
python experiments/midcore_tradeoff/plot_midcore_tradeoff.py --mode group
echo "============================================================"
