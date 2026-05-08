#!/usr/bin/env python3
"""
Plot mid-core trade-off charts.

Supports two experiment families:
  - group              : run_midcore_highocc_group_tradeoff.sh outputs
                         (x = MeanIR, y = CoOcc info, color = mid-core AUC)
  - coverage_imbalance : run_midcore_coverage_imbalance_tradeoff.sh outputs
                         (x = alpha_balance, y = mid-core AUC, one line per coverage)
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import pandas as pd
from medmnist import INFO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import get_label_range


MID_CORE = [5, 4, 7, 8, 12]
RESULTS_BASE = REPO_ROOT / "results" / "default" / "chestmnist" / "MedViT_tiny"
OUT_DIR = REPO_ROOT / "results" / "tradeoff_plots"
GREEDY_SUMMARY_PATH = OUT_DIR / "midcore_greedy_group_summary.csv"

GROUP_PATTERN = re.compile(r"^class_(\d+)_to_(\d+)_sample_(\d+)pct$")
COVERAGE_IMBALANCE_PATTERN = re.compile(
    r"^class_(\d+)_to_(\d+)_sample_(\d+)pct_clsbal_a([0-9p]+)$"
)

# Coverage tag mapping for the coverage x imbalance experiment.
# Keys are (label_head, label_tail). Must match run_midcore_coverage_imbalance_tradeoff.sh.
COVERAGE_TAGS = {
    (5, 12): "S0 (core only)",
    (3, 11): "S7 (best)",
    (3, 13): "S9 (full)",
}
COVERAGE_ORDER = [(5, 12), (3, 11), (3, 13)]
COVERAGE_COLORS = {
    (5, 12): "#d62728",  # red - narrow
    (3, 11): "#2ca02c",  # green - mid
    (3, 13): "#1f77b4",  # blue - wide
}


def load_chest_train_labels(data_root="./data"):
    data_root = str((REPO_ROOT / "data") if data_root == "./data" else Path(data_root))
    info = INFO["chestmnist"]
    data_class = getattr(medmnist, info["python_class"])
    ds = data_class(split="train", download=True, as_rgb=True, root=data_root, size=224, mmap_mode="r")

    labels = []
    for i in range(len(ds)):
        _, y = ds[i]
        if hasattr(y, "flatten"):
            labels.append(y.flatten())
        else:
            labels.append(np.array(y).flatten())
    return np.stack(labels, axis=0).astype(np.int32)


def compute_stats(train_y):
    # Positive count per label
    counts = train_y.sum(axis=0).astype(float)

    # Co-occurrence counts matrix C_ij: number of samples where both i and j are positive
    co = train_y.T @ train_y
    return counts, co


def mean_ir(counts, labels):
    vals = [counts[l] for l in labels if counts[l] > 0]
    if not vals:
        return float("nan")
    max_count = max(vals)
    irs = [max_count / v for v in vals]
    return float(np.mean(irs))


def coocc_score_normalized(co, labels, core=MID_CORE):
    active_core = [c for c in core if c in labels]
    if not active_core:
        return float("nan")

    scores = []
    for i in active_core:
        for j in labels:
            if i == j:
                continue
            denom = math.sqrt(max(co[i, i], 1) * max(co[j, j], 1))
            scores.append(float(co[i, j]) / denom)

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def coocc_info_score_additive(co, labels, core=MID_CORE):
    """
    Additive co-occurrence information score.

    Design goal: with fixed core and growing label set, score should be non-decreasing
    because every newly added label contributes non-negative pair counts.
    """
    active_core = [c for c in core if c in labels and c < co.shape[0]]
    if not active_core:
        return float("nan")

    total = 0.0
    for i in active_core:
        for j in labels:
            if i == j or j >= co.shape[0]:
                continue
            total += float(co[i, j])

    denom = sum(float(co[i, i]) for i in active_core)
    if denom <= 0:
        return float("nan")
    return total / denom


def read_coocc_matrix(csv_path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    return df.values.astype(float)


def mean_ir_from_coocc_diag(co, labels):
    vals = []
    for l in labels:
        if l < co.shape[0]:
            v = float(co[l, l])
            if v > 0:
                vals.append(v)
    if not vals:
        return float("nan")
    max_count = max(vals)
    irs = [max_count / v for v in vals]
    return float(np.mean(irs))


def read_mid_core_auc(csv_path):
    df = pd.read_csv(csv_path)
    if "auc" not in df.columns:
        return float("nan")
    best_idx = df["auc"].idxmax()
    row = df.iloc[best_idx]

    auc_vals = []
    for c in MID_CORE:
        col = f"AUC_class_{c}"
        if col in df.columns:
            auc_vals.append(float(row[col]))
    if not auc_vals:
        return float("nan")
    return float(np.mean(auc_vals))


def collect_group_rows(counts, co):
    rows = []
    if not GREEDY_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Greedy summary not found: {GREEDY_SUMMARY_PATH}. "
            "Run run_midcore_highocc_group_tradeoff.sh first."
        )

    s = pd.read_csv(GREEDY_SUMMARY_PATH)
    summary_lookup = {
        (int(r["label_head"]), int(r["label_tail"])): {
            "step": int(r["step"]),
            "added_side": r["added_side"],
            "added_label": r["added_label"],
        }
        for _, r in s.iterrows()
    }
    allowed_pairs = set(summary_lookup.keys())

    for d in RESULTS_BASE.iterdir():
        if not d.is_dir():
            continue
        m = GROUP_PATTERN.match(d.name)
        if not m:
            continue

        head = int(m.group(1))
        tail = int(m.group(2))
        if (head, tail) not in allowed_pairs:
            continue
        csv_path = d / "model_metrics.csv"
        if not csv_path.exists():
            continue

        labels = get_label_range("chestmnist", head, tail)
        rows.append(
            {
                "experiment": "group",
                "setting": d.name,
                "head": head,
                "tail": tail,
                "alpha": np.nan,
                "mean_ir": mean_ir(counts, labels),
                "coocc_score": coocc_info_score_additive(co, labels),
                "coocc_score_norm": coocc_score_normalized(co, labels),
                "mid_core_auc": read_mid_core_auc(csv_path),
                "step": summary_lookup[(head, tail)]["step"],
                "added_side": summary_lookup[(head, tail)]["added_side"],
                "added_label": summary_lookup[(head, tail)]["added_label"],
            }
        )
    return rows


def collect_coverage_imbalance_rows(counts, co):
    """Collect rows from run_midcore_coverage_imbalance_tradeoff.sh outputs."""
    rows = []
    if not RESULTS_BASE.exists():
        return rows

    for d in RESULTS_BASE.iterdir():
        if not d.is_dir():
            continue
        m = COVERAGE_IMBALANCE_PATTERN.match(d.name)
        if not m:
            continue
        head = int(m.group(1))
        tail = int(m.group(2))
        alpha_balance = float(m.group(4).replace("p", "."))
        csv_path = d / "model_metrics.csv"
        if not csv_path.exists():
            continue

        labels = get_label_range("chestmnist", head, tail)

        sampled_co_path = d / "sampled_cooccurrence.csv"
        sampled_co = read_coocc_matrix(sampled_co_path)
        if sampled_co is not None:
            row_mean_ir = mean_ir_from_coocc_diag(sampled_co, labels)
            row_coocc = coocc_info_score_additive(sampled_co, labels)
            row_coocc_norm = coocc_score_normalized(sampled_co, labels)
        else:
            row_mean_ir = mean_ir(counts, labels)
            row_coocc = coocc_info_score_additive(co, labels)
            row_coocc_norm = coocc_score_normalized(co, labels)

        rows.append({
            "experiment": "coverage_imbalance",
            "setting": d.name,
            "head": head,
            "tail": tail,
            "alpha_balance": alpha_balance,
            "coverage_tag": COVERAGE_TAGS.get((head, tail), f"{head}-{tail}"),
            "mean_ir": row_mean_ir,
            "coocc_score": row_coocc,
            "coocc_score_norm": row_coocc_norm,
            "mid_core_auc": read_mid_core_auc(csv_path),
        })
    return rows


def _plot_coverage_imbalance_rows(ax, df):
    """Plot one line per coverage level: x = alpha_balance, y = mid_core_auc."""
    df = df.copy()

    handles = []
    for (head, tail) in COVERAGE_ORDER:
        sub = df[(df["head"] == head) & (df["tail"] == tail)].sort_values("alpha_balance")
        if sub.empty:
            continue
        color = COVERAGE_COLORS.get((head, tail), None)
        label = COVERAGE_TAGS.get((head, tail), f"{head}-{tail}")
        line, = ax.plot(
            sub["alpha_balance"],
            sub["mid_core_auc"],
            marker="o",
            markersize=9,
            linewidth=2.0,
            color=color,
            label=label,
        )
        handles.append(line)

        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['mid_core_auc']:.4f}",
                (row["alpha_balance"], row["mid_core_auc"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    ax.legend(handles=handles, title="Coverage", loc="best")
    return df


def _annotate_key_group_points(ax, df):
    start_row = df.loc[df["step"].idxmin()]
    best_row = df.loc[df["mid_core_auc"].idxmax()]
    end_row = df.loc[df["step"].idxmax()]

    annotations = [
        (start_row, "Start\ncore only", (10, -18)),
        (best_row, "Best\ntrade-off", (10, 10)),
        (end_row, "Full\nrange", (10, -18)),
    ]

    seen = set()
    for row, text, offset in annotations:
        key = (float(row["mean_ir"]), float(row["coocc_score"]))
        if key in seen:
            continue
        seen.add(key)
        ax.annotate(
            text,
            (row["mean_ir"], row["coocc_score"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9),
        )


def _plot_group_rows(ax, df):
    df = df.sort_values("step").copy()

    ax.plot(
        df["mean_ir"],
        df["coocc_score"],
        color="0.45",
        linewidth=1.8,
        alpha=0.8,
        zorder=1,
    )

    sc = ax.scatter(
        df["mean_ir"],
        df["coocc_score"],
        c=df["mid_core_auc"],
        cmap="viridis",
        s=150,
        edgecolors="black",
        linewidths=0.7,
        zorder=2,
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"S{int(row['step'])}",
            (row["mean_ir"], row["coocc_score"]),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )

    for idx in range(1, len(df)):
        prev_row = df.iloc[idx - 1]
        row = df.iloc[idx]
        mid_x = (prev_row["mean_ir"] + row["mean_ir"]) / 2
        mid_y = (prev_row["coocc_score"] + row["coocc_score"]) / 2
        added = row["added_label"]
        if pd.notna(added) and str(added) != "none":
            ax.annotate(
                f"+{int(float(added))}",
                (mid_x, mid_y),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="0.25",
            )

    _annotate_key_group_points(ax, df)
    return sc, df


def plot_rows(df, mode):
    if df.empty:
        print(f"No rows found for mode={mode}. Skip plotting.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    if mode == "group":
        sc, df = _plot_group_rows(ax, df)
    elif mode == "coverage_imbalance":
        df = _plot_coverage_imbalance_rows(ax, df)
        sc = None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "coverage_imbalance":
        ax.set_xlabel("alpha_balance (0 = natural, 1 = inverse-freq balanced)")
        ax.set_ylabel("Mid-Core AUC (best epoch)")
        ax.set_title("Mid-Core Trade-off: Coverage x Imbalance")
    else:
        ax.set_xlabel("MeanIR (higher = more imbalanced)")
        ax.set_ylabel("Co-occurrence Information (additive, higher = more context)")
        ax.set_title("Mid-Core Trade-off: Greedy Group Expansion")
    ax.grid(alpha=0.3)

    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Mid-Core AUC (best epoch)")

    png_path = OUT_DIR / f"midcore_tradeoff_{mode}.png"
    csv_path = OUT_DIR / f"midcore_tradeoff_{mode}.csv"

    if mode == "coverage_imbalance":
        df.sort_values(["head", "tail", "alpha_balance"], inplace=True)
    else:
        df.sort_values(["mean_ir", "coocc_score"], inplace=True)
    df.to_csv(csv_path, index=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=250)
    plt.close(fig)

    print(f"Saved data: {csv_path}")
    print(f"Saved plot: {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["group", "coverage_imbalance", "all"],
        default="all",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ChestMNIST training labels to compute IR/co-occurrence...")
    train_y = load_chest_train_labels()
    counts, co = compute_stats(train_y)

    if args.mode in ("group", "all"):
        group_df = pd.DataFrame(collect_group_rows(counts, co))
        plot_rows(group_df, "group")

    if args.mode in ("coverage_imbalance", "all"):
        ci_df = pd.DataFrame(collect_coverage_imbalance_rows(counts, co))
        plot_rows(ci_df, "coverage_imbalance")


if __name__ == "__main__":
    main()
