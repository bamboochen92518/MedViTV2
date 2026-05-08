#!/usr/bin/env python3
"""
Aggregate AUC performance from different methods
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Global grouping mode control
USE_MID = True

# Define class order
HEAD = [3, 2, 0]
MID = [5, 4, 7, 8, 12]
TAIL = [1, 10, 9, 11, 6, 13]
CLASS_ORDER = HEAD + MID + TAIL

# Define file paths and method names
METHODS = {
    'ASL': 'results/ASL/chestmnist/MedViT_tiny/sample_10pct/analysis/per_class_metrics.csv',
    'BCE': 'results/BCE/chestmnist/MedViT_tiny/sample_10pct/analysis/per_class_metrics.csv',
    'CBLoss': 'results/CBLoss/chestmnist/MedViT_tiny/sample_10pct/analysis/per_class_metrics.csv',
    'CBLossOriginal': 'results/CBLossOriginal/chestmnist/MedViT_tiny/sample_10pct/analysis/per_class_metrics.csv',
    'LDACE_CCL': 'results/LDACE_CCL/chestmnist/MedViT_tiny/sample_10pct/analysis/per_class_metrics.csv',
    'Focal': 'results/Focal/chestmnist/MedViT_tiny/sample_10pct/analysis/per_class_metrics.csv',
    'BCE_sampler': 'results/BCE/chestmnist/MedViT_tiny/sample_10pct_sampler/analysis/per_class_metrics.csv',
    'DBFocal_sampler': 'results/DBFocal/chestmnist/MedViT_tiny/sample_10pct_sampler/analysis/per_class_metrics.csv',
    'Group_exp1_ir1.5_cvir0.2': 'results/group_evaluation/exp1_ir1.5_cvir0.2_per_class.csv',
    'Group_exp2_ir2.5_cvir0.4': 'results/group_evaluation/exp2_ir2.5_cvir0.4_per_class.csv',
    'Group_exp3_ir3.5_cvir0.6': 'results/group_evaluation/exp3_ir3.5_cvir0.6_per_class.csv',
    'Expanding_algo': 'results/grouping_expanding_algo_evaluation/merged_best_per_class.csv',
}

def load_auc_data(filepath):
    """Load AUC data from CSV file"""
    df = pd.read_csv(filepath)
    return df.set_index('class_id')['AUC'].to_dict()

def format_display_value(value):
    """Format a value for table display."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)

def underline_text(text):
    """Underline text using combining low line characters."""
    return ''.join(char + '\u0332' for char in text)

def build_grouped_result(result, use_mid):
    """Insert grouped summary rows based on the requested grouping mode."""
    group_specs = [('head', HEAD)]
    if use_mid:
        group_specs.append(('mid', MID))
        group_specs.append(('tail', TAIL))
    else:
        group_specs.append(('tail', MID + TAIL))

    grouped_frames = []
    for group_name, class_ids in group_specs:
        group_df = result.loc[result.index.isin(class_ids)]
        summary_row = group_df.mean().round(4).rename(f'mean_AUC_{group_name}')
        grouped_frames.append(pd.concat([group_df, summary_row.to_frame().T]))

    total_row = result.mean().round(4).rename('mean_AUC_total')
    grouped_frames.append(total_row.to_frame().T)
    return pd.concat(grouped_frames)

def save_table_png(dataframe, output_path):
    """Render a dataframe as a PNG table."""
    numeric_df = dataframe.apply(pd.to_numeric, errors='coerce')
    display_df = dataframe.map(format_display_value)

    num_rows, num_cols = display_df.shape
    fig_width = max(16, num_cols * 1.15)
    fig_height = max(4, num_rows * 0.5 + 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for col_idx, column_name in enumerate(numeric_df.columns):
        column_values = numeric_df[column_name].dropna()
        if column_values.empty:
            continue

        sorted_values = column_values.sort_values(ascending=False).unique()
        best_value = sorted_values[0]
        second_best_value = sorted_values[1] if len(sorted_values) > 1 else None

        for row_idx, row_name in enumerate(numeric_df.index, start=1):
            cell_value = numeric_df.loc[row_name, column_name]
            if pd.isna(cell_value):
                continue

            text = table[(row_idx, col_idx)].get_text()
            if cell_value == best_value:
                text.set_fontweight('bold')
            elif second_best_value is not None and cell_value == second_best_value:
                text.set_text(underline_text(text.get_text()))

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    # Collect AUC data from all methods
    auc_data = {}
    
    for method, filepath in METHODS.items():
        full_path = Path(filepath)
        if full_path.exists():
            auc_data[method] = load_auc_data(full_path)
            print(f"✓ Loaded {method}")
        else:
            print(f"✗ Missing {method}: {filepath}")
    
    # Build result dataframe
    result = pd.DataFrame(auc_data)

    # Reorder rows by CLASS_ORDER
    result = result.reindex(CLASS_ORDER)

    result = build_grouped_result(result, USE_MID)

    # Transpose: methods as rows, class_ids as columns
    result = result.T
    result.index.name = 'method'

    output_suffix = 'use_mid' if USE_MID else 'head_tail_only'

    # Save to CSV
    output_file = f'results/auc_comparison_{output_suffix}.csv'
    result.to_csv(output_file)
    print(f"\n✓ Output saved to {output_file}")

    png_output_file = f'results/auc_comparison_{output_suffix}.png'
    save_table_png(result, png_output_file)
    print(f"✓ Table image saved to {png_output_file}")
    
    # Display preview
    print("\nPreview:")
    print(result.to_string(float_format='%.4f'))

if __name__ == '__main__':
    main()
