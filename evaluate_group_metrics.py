"""
Evaluate Group Metrics Script
Extract core group metrics from training results with different label ranges
"""

import os
import pandas as pd
import numpy as np
from datasets import get_label_groups


def get_group_core_labels():
    """
    Define core labels for each group
    
    Returns:
        dict: {group_id: [core_labels]}
    """
    return {
        0: [3, 2, 0],                          # Group 0 core
        1: [5, 4, 7, 8, 12, 1, 10, 9],        # Group 1 core
        2: [11, 6],                            # Group 2 core
        3: [13]                                # Group 3 core
    }


def extract_core_metrics(csv_path, core_labels):
    """
    Extract AUC and AP metrics for core labels from a training CSV
    
    Args:
        csv_path: Path to the training metrics CSV
        core_labels: List of core label indices to extract
        
    Returns:
        dict: Metrics for the best epoch based on mAP
    """
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # Find the epoch with best mAP
    best_epoch_idx = df['mAP'].idxmax()
    best_epoch_row = df.iloc[best_epoch_idx]
    
    # Extract AUC values for core labels
    core_auc_values = []
    for label in core_labels:
        auc_col = f'AUC_class_{label}'
        if auc_col in df.columns:
            core_auc_values.append(best_epoch_row[auc_col])
        else:
            core_auc_values.append(np.nan)
    
    # Extract AP values for core labels
    core_ap_values = []
    for label in core_labels:
        ap_col = f'AP_class_{label}'
        if ap_col in df.columns:
            core_ap_values.append(best_epoch_row[ap_col])
        else:
            core_ap_values.append(np.nan)
    
    # Calculate mean AUC and mAP for core labels only
    core_mean_auc = np.nanmean(core_auc_values) if core_auc_values else np.nan
    core_mean_ap = np.nanmean(core_ap_values) if core_ap_values else np.nan
    
    metrics = {
        'epoch': int(best_epoch_row['epoch']),
        'train_loss': best_epoch_row['train_loss'],
        'val_accuracy': best_epoch_row['val_accuracy'],
        'f1_score': best_epoch_row['f1_score'],
        'core_mean_auc': core_mean_auc,
        'core_mAP': core_mean_ap,
    }
    
    # Add individual AUC values
    for i, label in enumerate(core_labels):
        metrics[f'AUC_class_{label}'] = core_auc_values[i]
    
    # Add individual AP values
    for i, label in enumerate(core_labels):
        metrics[f'AP_class_{label}'] = core_ap_values[i]
    
    return metrics


def parse_filename_range(filename):
    """
    Parse label range from filename
    
    Args:
        filename: e.g., 'MedViT_tiny_chestmnist_class3to5_metrics.csv'
        
    Returns:
        tuple: (head, tail) or None if not a range file
    """
    import re
    match = re.search(r'class(\d+)to(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def evaluate_group_experiments(group_id, results_dir='results', model_name='MedViT_tiny', dataset='chestmnist'):
    """
    Evaluate all experiments for a specific group
    
    Args:
        group_id: Group ID (0-3)
        results_dir: Directory containing results
        model_name: Model name used in training
        dataset: Dataset name
        
    Returns:
        DataFrame: Summary of all experiments for this group
    """
    core_labels = get_group_core_labels()[group_id]
    
    print(f"\n{'='*60}")
    print(f"Evaluating Group {group_id} (Core labels: {core_labels})")
    print(f"{'='*60}")
    
    # Find all CSV files that might contain this group's training results
    all_results = []
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('_metrics.csv'):
            continue
        
        if not filename.startswith(f'{model_name}_{dataset}_class'):
            continue
        
        # Parse label range from filename
        label_range = parse_filename_range(filename)
        if label_range is None:
            continue
        
        head, tail = label_range
        
        # Check if this training range includes ALL core labels for this group
        from datasets import get_label_range
        try:
            trained_labels = get_label_range(dataset, head, tail)
        except:
            continue
        
        # Check if all core labels are in trained labels
        if not all(label in trained_labels for label in core_labels):
            continue
        
        # Extract metrics
        csv_path = os.path.join(results_dir, filename)
        metrics = extract_core_metrics(csv_path, core_labels)
        
        if metrics is None:
            continue
        
        # Add metadata
        metrics['label_head'] = head
        metrics['label_tail'] = tail
        metrics['num_labels_trained'] = len(trained_labels)
        metrics['trained_labels'] = str(trained_labels)
        
        all_results.append(metrics)
        
        print(f"  ✓ Processed: {filename} (labels {head}-{tail}, {len(trained_labels)} classes)")
    
    if not all_results:
        print(f"  ⚠️  No results found for group {group_id}")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by core_mAP (descending)
    df = df.sort_values('core_mAP', ascending=False)
    
    return df


def main():
    """Main function to evaluate all groups"""
    print("\n" + "="*60)
    print("🎯 Group-wise Metrics Evaluation")
    print("="*60)
    
    results_dir = 'results'
    output_dir = 'results/group_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = 'MedViT_tiny'
    dataset = 'chestmnist'
    
    # Evaluate each group
    for group_id in range(4):
        df = evaluate_group_experiments(group_id, results_dir, model_name, dataset)
        
        if df is not None:
            # Save to CSV
            output_path = os.path.join(output_dir, f'group{group_id}_evaluation.csv')
            df.to_csv(output_path, index=False, float_format='%.4f')
            print(f"  📊 Saved: {output_path}")
            
            # Print top 5 results
            print(f"\n  Top 5 configurations for Group {group_id}:")
            print(f"  {'Rank':<6}{'Range':<12}{'#Labels':<10}{'core_mAP':<12}{'core_AUC':<12}")
            print(f"  {'-'*52}")
            for idx, row in df.head(5).iterrows():
                rank = df.index.get_loc(idx) + 1
                label_range = f"{int(row['label_head'])}-{int(row['label_tail'])}"
                num_labels = int(row['num_labels_trained'])
                core_map = row['core_mAP']
                core_auc = row['core_mean_auc']
                print(f"  {rank:<6}{label_range:<12}{num_labels:<10}{core_map:<12.4f}{core_auc:<12.4f}")
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
