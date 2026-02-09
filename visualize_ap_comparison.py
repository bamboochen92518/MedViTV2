"""
Visualize AP Comparison Script
Compares per-class AP between grouped and non-grouped training
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_metrics_csv(csv_path):
    """Load metrics CSV file"""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found!")
        return None
    return pd.read_csv(csv_path)


def get_class_to_group_mapping():
    """
    Define mapping from class index to group index
    
    Returns:
        dict: {class_idx: group_idx}
    """
    mapping = {}
    
    # Group 0: classes [3, 2, 0]
    for cls in [3, 2, 0]:
        mapping[cls] = 0
    
    # Group 1: classes [5, 4, 7, 8, 12, 1, 10, 9]
    for cls in [5, 4, 7, 8, 12, 1, 10, 9]:
        mapping[cls] = 1
    
    # Group 2: classes [11, 6]
    for cls in [11, 6]:
        mapping[cls] = 2
    
    # Group 3: class [13]
    mapping[13] = 3
    
    return mapping


def visualize_class_ap(class_idx, results_dir='results', output_dir='results/visualize'):
    """
    Create AP comparison plot for a specific class
    
    Args:
        class_idx: Class index (0-13)
        results_dir: Directory containing CSV files
        output_dir: Directory to save output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load non-grouped metrics
    no_group_csv = os.path.join(results_dir, 'MedViT_tiny_chestmnist_metrics.csv')
    df_no_group = load_metrics_csv(no_group_csv)
    
    if df_no_group is None:
        print(f"Skipping class {class_idx}: no-group CSV not found")
        return
    
    # Get class to group mapping
    class_to_group = get_class_to_group_mapping()
    group_idx = class_to_group[class_idx]
    
    # Load grouped metrics
    group_csv = os.path.join(results_dir, f'MedViT_tiny_chestmnist_group{group_idx}_metrics.csv')
    df_group = load_metrics_csv(group_csv)
    
    if df_group is None:
        print(f"Skipping class {class_idx}: group{group_idx} CSV not found")
        return
    
    # Extract AP columns
    ap_col_no_group = f'AP_class_{class_idx}'
    ap_col_group = f'AP_class_{class_idx}'
    
    if ap_col_no_group not in df_no_group.columns:
        print(f"Warning: {ap_col_no_group} not found in no-group CSV")
        return
    
    if ap_col_group not in df_group.columns:
        print(f"Warning: {ap_col_group} not found in group{group_idx} CSV")
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot no-group AP
    epochs_no_group = df_no_group['epoch'].values
    ap_no_group = df_no_group[ap_col_no_group].values
    plt.plot(epochs_no_group, ap_no_group, 
             label='No Grouping', 
             linewidth=2, 
             marker='o', 
             markersize=3,
             alpha=0.8)
    
    # Plot grouped AP
    epochs_group = df_group['epoch'].values
    ap_group = df_group[ap_col_group].values
    plt.plot(epochs_group, ap_group, 
             label=f'With Grouping (Group {group_idx})', 
             linewidth=2, 
             marker='s', 
             markersize=3,
             alpha=0.8)
    
    # Calculate best AP values (instead of final)
    best_ap_no_group = np.max(ap_no_group)
    best_epoch_no_group = epochs_no_group[np.argmax(ap_no_group)]
    best_ap_group = np.max(ap_group)
    best_epoch_group = epochs_group[np.argmax(ap_group)]
    improvement = ((best_ap_group - best_ap_no_group) / best_ap_no_group) * 100
    
    # Add grid and labels
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Average Precision (AP)', fontsize=12, fontweight='bold')
    plt.title(f'Class {class_idx} - AP Comparison\n'
              f'Best AP: No Group = {best_ap_no_group:.4f} (epoch {best_epoch_no_group}), '
              f'Group {group_idx} = {best_ap_group:.4f} (epoch {best_epoch_group}) '
              f'({improvement:+.2f}%)',
              fontsize=13, fontweight='bold')
    
    plt.legend(loc='best', fontsize=11)
    plt.ylim([0, 1.0])
    
    # Add text box with statistics
    textstr = f'Best AP (No Group): {best_ap_no_group:.4f} @ epoch {best_epoch_no_group}\n'
    textstr += f'Best AP (Group {group_idx}): {best_ap_group:.4f} @ epoch {best_epoch_group}\n'
    textstr += f'Mean AP (No Group): {np.mean(ap_no_group):.4f}\n'
    textstr += f'Mean AP (Group {group_idx}): {np.mean(ap_group):.4f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'class{class_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def main():
    """Main function to generate all visualizations"""
    print("\n" + "="*60)
    print("AP Comparison Visualization")
    print("="*60 + "\n")
    
    results_dir = 'results'
    output_dir = 'results/visualize'
    
    # Generate plots for all 14 classes
    for class_idx in range(14):
        print(f"Generating plot for Class {class_idx}...")
        try:
            visualize_class_ap(class_idx, results_dir, output_dir)
        except Exception as e:
            print(f"Error processing class {class_idx}: {e}")
    
    print("\n" + "="*60)
    print(f"✅ All visualizations saved to: {output_dir}")
    print("="*60 + "\n")
    
    # Generate summary statistics
    generate_summary_table(results_dir, output_dir)


def generate_summary_table(results_dir='results', output_dir='results/visualize'):
    """Generate summary table comparing best AP values"""
    print("Generating summary table...")
    
    class_to_group = get_class_to_group_mapping()
    summary_data = []
    
    # Load no-group metrics
    no_group_csv = os.path.join(results_dir, 'MedViT_tiny_chestmnist_metrics.csv')
    df_no_group = load_metrics_csv(no_group_csv)
    
    if df_no_group is None:
        return
    
    # Load all group metrics
    group_dfs = {}
    for group_idx in range(4):
        group_csv = os.path.join(results_dir, f'MedViT_tiny_chestmnist_group{group_idx}_metrics.csv')
        group_dfs[group_idx] = load_metrics_csv(group_csv)
    
    # Get group sizes for weighted average
    group_sizes = {0: 3, 1: 8, 2: 2, 3: 1}  # [3,2,0], [5,4,7,8,12,1,10,9], [11,6], [13]
    
    # Collect data for each class
    for class_idx in range(14):
        group_idx = class_to_group[class_idx]
        
        # Get BEST AP values (max across all epochs) instead of final
        ap_col = f'AP_class_{class_idx}'
        
        if ap_col in df_no_group.columns:
            best_ap_no_group = df_no_group[ap_col].max()
            best_epoch_no_group = df_no_group.loc[df_no_group[ap_col].idxmax(), 'epoch']
        else:
            best_ap_no_group = np.nan
            best_epoch_no_group = np.nan
        
        if group_dfs[group_idx] is not None and ap_col in group_dfs[group_idx].columns:
            best_ap_group = group_dfs[group_idx][ap_col].max()
            best_epoch_group = group_dfs[group_idx].loc[group_dfs[group_idx][ap_col].idxmax(), 'epoch']
        else:
            best_ap_group = np.nan
            best_epoch_group = np.nan
        
        improvement = ((best_ap_group - best_ap_no_group) / best_ap_no_group * 100) if not np.isnan(best_ap_no_group) and best_ap_no_group > 0 else np.nan
        
        summary_data.append({
            'Class': class_idx,
            'Group': group_idx,
            'Best_AP_NoGroup': best_ap_no_group,
            'Epoch_NoGroup': int(best_epoch_no_group) if not np.isnan(best_epoch_no_group) else np.nan,
            'Best_AP_Grouped': best_ap_group,
            'Epoch_Grouped': int(best_epoch_group) if not np.isnan(best_epoch_group) else np.nan,
            'Improvement (%)': improvement
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_path = os.path.join(output_dir, 'ap_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"✓ Summary table saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary Statistics (Based on Best AP per Class):")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    # Calculate overall statistics based on BEST AP
    avg_improvement = summary_df['Improvement (%)'].mean()
    improved_classes = (summary_df['Improvement (%)'] > 0).sum()
    
    # Calculate mAP using BEST AP values
    mAP_no_group = summary_df['Best_AP_NoGroup'].mean()
    mAP_grouped = summary_df['Best_AP_Grouped'].mean()
    mAP_improvement = ((mAP_grouped - mAP_no_group) / mAP_no_group * 100) if mAP_no_group > 0 else 0
    
    print(f"\nOverall Statistics (Based on Best AP):")
    print(f"  mAP (No Group - Best): {mAP_no_group:.4f}")
    print(f"  mAP (Grouped - Best): {mAP_grouped:.4f}")
    print(f"  mAP Improvement: {mAP_improvement:+.2f}%")
    print(f"  Average Improvement per class: {avg_improvement:+.2f}%")
    print(f"  Classes Improved: {improved_classes}/14")
    print(f"  Classes Degraded: {14 - improved_classes}/14")
    
    # ===== New: Calculate weighted AUC statistics =====
    print(f"\n" + "="*60)
    print("Overall Statistics (Based on Best AUC - Weighted by Group Size):")
    print("="*60)
    
    # Get best AUC for no-group training
    if 'auc' in df_no_group.columns:
        best_auc_no_group_overall = df_no_group['auc'].max()
        best_auc_epoch_no_group = df_no_group.loc[df_no_group['auc'].idxmax(), 'epoch']
    else:
        best_auc_no_group_overall = np.nan
        best_auc_epoch_no_group = np.nan
    
    # Calculate weighted AUC for grouped training
    weighted_auc_sum = 0
    total_weight = 0
    group_auc_info = []
    
    for group_idx in range(4):
        if group_dfs[group_idx] is not None and 'auc' in group_dfs[group_idx].columns:
            best_auc_group = group_dfs[group_idx]['auc'].max()
            best_auc_epoch = group_dfs[group_idx].loc[group_dfs[group_idx]['auc'].idxmax(), 'epoch']
            weight = group_sizes[group_idx]
            
            weighted_auc_sum += best_auc_group * weight
            total_weight += weight
            
            group_auc_info.append({
                'group': group_idx,
                'auc': best_auc_group,
                'epoch': int(best_auc_epoch),
                'weight': weight
            })
        else:
            group_auc_info.append({
                'group': group_idx,
                'auc': np.nan,
                'epoch': np.nan,
                'weight': group_sizes[group_idx]
            })
    
    # Calculate weighted average AUC
    weighted_auc_grouped = weighted_auc_sum / total_weight if total_weight > 0 else np.nan
    auc_improvement = ((weighted_auc_grouped - best_auc_no_group_overall) / best_auc_no_group_overall * 100) if not np.isnan(best_auc_no_group_overall) and best_auc_no_group_overall > 0 else np.nan
    
    print(f"  Best AUC (No Group): {best_auc_no_group_overall:.4f} @ epoch {int(best_auc_epoch_no_group) if not np.isnan(best_auc_epoch_no_group) else 'N/A'}")
    print(f"\n  Per-Group Best AUC:")
    for info in group_auc_info:
        if not np.isnan(info['auc']):
            print(f"    Group {info['group']}: {info['auc']:.4f} @ epoch {info['epoch']} (weight: {info['weight']} classes)")
        else:
            print(f"    Group {info['group']}: N/A (weight: {info['weight']} classes)")
    
    print(f"\n  Weighted AUC (Grouped): {weighted_auc_grouped:.4f}")
    print(f"  AUC Improvement: {auc_improvement:+.2f}%")
    print(f"\n  Note: Weighted by number of classes in each group")
    print(f"        Group 0: 3 classes, Group 1: 8 classes, Group 2: 2 classes, Group 3: 1 class")
    print("="*60)


if __name__ == '__main__':
    main()
