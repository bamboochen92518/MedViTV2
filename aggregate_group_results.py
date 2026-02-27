"""
Aggregate Group Results
Combine the best epoch results from multiple group models and calculate
head, middle, tail, and total metrics.

Usage:
    python aggregate_group_results.py --experiment_name exp1 --group_configs "3,3" "2,0" "5,8"
"""

import argparse
import pandas as pd
import numpy as npcor
import os
from pathlib import Path


class GroupResultAggregator:
    """Aggregate results from multiple group models"""
    
    # Fixed class definitions for ChestMNIST
    HEAD_CLASSES = [3, 2, 0]
    MID_CLASSES = [5, 4, 7, 8]
    TAIL_CLASSES = [12, 1, 10, 9, 11, 6, 13]
    
    def __init__(self, model_name: str, dataset: str, experiment_name: str, sample: float = 0.1, loss_function: str = 'default'):
        self.model_name = model_name
        self.dataset = dataset
        self.experiment_name = experiment_name
        self.sample = sample
        self.loss_function = loss_function
        self.results_dir = Path('./results')
        
    def find_best_epoch_for_group(self, label_head: int, label_tail: int) -> dict:
        """
        Find the best epoch for a specific group based on validation AUC
        
        Args:
            label_head: Starting label
            label_tail: Ending label
            
        Returns:
            Dictionary containing best epoch info and metrics
        """
        # Build the correct path: results/{loss}/{dataset}/{model}/class_{head}_to_{tail}_sample_{pct}pct/model_metrics.csv
        sample_pct = int(self.sample * 100)
        config_dir = f'class_{label_head}_to_{label_tail}_sample_{sample_pct}pct'
        
        metrics_file = self.results_dir / self.loss_function / self.dataset / self.model_name / config_dir / 'model_metrics.csv'
        
        if not metrics_file.exists():
            print(f"⚠️  Warning: Metrics file not found: {metrics_file}")
            return None
        
        df = pd.read_csv(metrics_file)
        
        # Find best epoch based on validation AUC
        if 'auc' in df.columns:
            best_idx = df['auc'].idxmax()
        elif 'val_accuracy' in df.columns:
            best_idx = df['val_accuracy'].idxmax()
        else:
            print(f"⚠️  Warning: No validation metric found in {metrics_file}")
            return None
        
        best_epoch = df.loc[best_idx]
        
        return {
            'label_head': label_head,
            'label_tail': label_tail,
            'best_epoch': int(best_epoch['epoch']) if 'epoch' in best_epoch else best_idx + 1,
            'metrics': best_epoch.to_dict(),
            'csv_columns': df.columns.tolist()
        }
    
    def extract_per_class_metrics(self, group_results: list) -> pd.DataFrame:
        """
        Extract per-class AP and AUC from all groups
        
        Args:
            group_results: List of group result dictionaries
            
        Returns:
            DataFrame with per-class metrics
        """
        # Initialize storage for all classes
        all_classes = sorted(self.HEAD_CLASSES + self.MID_CLASSES + self.TAIL_CLASSES)
        per_class_data = {
            'class_id': all_classes,
            'AP': [np.nan] * len(all_classes),
            'AUC': [np.nan] * len(all_classes),
            'tier': [''] * len(all_classes)
        }
        
        # Assign tier labels
        for idx, class_id in enumerate(all_classes):
            if class_id in self.HEAD_CLASSES:
                per_class_data['tier'][idx] = 'head'
            elif class_id in self.MID_CLASSES:
                per_class_data['tier'][idx] = 'mid'
            elif class_id in self.TAIL_CLASSES:
                per_class_data['tier'][idx] = 'tail'
        
        # Extract metrics from each group
        for group_result in group_results:
            if group_result is None:
                continue
            
            metrics = group_result['metrics']
            
            # Look for per-class metrics in the format: AP_class_3, AUC_class_3
            for class_id in all_classes:
                class_idx = all_classes.index(class_id)
                
                # Try to find AP metric
                ap_key = f'AP_class_{class_id}'
                if ap_key in metrics:
                    per_class_data['AP'][class_idx] = metrics[ap_key]
                
                # Try to find AUC metric
                auc_key = f'AUC_class_{class_id}'
                if auc_key in metrics:
                    per_class_data['AUC'][class_idx] = metrics[auc_key]
        
        return pd.DataFrame(per_class_data)
    
    def calculate_tier_metrics(self, per_class_df: pd.DataFrame, group_results: list) -> pd.DataFrame:
        """
        Calculate aggregated metrics for head, mid, tail, and total
        
        Args:
            per_class_df: DataFrame with per-class metrics
            group_results: List of group result dictionaries (for overall AUC)
            
        Returns:
            DataFrame with tier-level metrics
        """
        tier_data = []
        
        # Calculate per-tier metrics
        for tier in ['head', 'mid', 'tail']:
            tier_df = per_class_df[per_class_df['tier'] == tier]
            
            if len(tier_df) > 0:
                tier_data.append({
                    'tier': tier,
                    'num_classes': len(tier_df),
                    'mAP': tier_df['AP'].mean(),
                    'AUC': tier_df['AUC'].mean(),
                    'class_ids': str(tier_df['class_id'].tolist())
                })
        
        # Calculate total (all classes)
        total_mAP = per_class_df['AP'].mean()
        total_AUC = per_class_df['AUC'].mean()
        
        tier_data.append({
            'tier': 'total',
            'num_classes': len(per_class_df),
            'mAP': total_mAP,
            'AUC': total_AUC,
            'class_ids': 'all'
        })
        
        # Create DataFrame
        tier_df = pd.DataFrame(tier_data)
        
        # Reorder columns
        tier_df = tier_df[['tier', 'num_classes', 'mAP', 'AUC', 'class_ids']]
        
        return tier_df
    
    def aggregate_results(self, group_configs: list):
        """
        Main aggregation function
        
        Args:
            group_configs: List of (label_head, label_tail) tuples
        """
        print(f"\n{'='*70}")
        print(f"📊 Aggregating Results for Experiment: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Sample: {self.sample}")
        print(f"Loss: {self.loss_function}")
        print(f"Number of groups: {len(group_configs)}")
        print(f"{'='*70}\n")
        
        # Find best epoch for each group
        group_results = []
        for idx, (label_head, label_tail) in enumerate(group_configs):
            print(f"🔍 Processing Group {idx}: classes [{label_head} to {label_tail}]...")
            result = self.find_best_epoch_for_group(label_head, label_tail)
            if result:
                print(f"   ✅ Best epoch: {result['best_epoch']} (AUC: {result['metrics'].get('auc', 'N/A'):.4f})")
                group_results.append(result)
            else:
                print(f"   ❌ Failed to process Group {idx}")
        
        if not group_results:
            print("\n❌ No valid group results found!")
            return
        
        print(f"\n✅ Successfully processed {len(group_results)} groups\n")
        
        # Extract per-class metrics
        print("📋 Extracting per-class metrics...")
        per_class_df = self.extract_per_class_metrics(group_results)
        
        # Calculate tier metrics
        print("📊 Calculating tier-level metrics...\n")
        tier_df = self.calculate_tier_metrics(per_class_df, group_results)
        
        # Save results
        output_dir = self.results_dir / 'group_evaluation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-class results
        per_class_file = output_dir / f"{self.experiment_name}_per_class.csv"
        per_class_df.to_csv(per_class_file, index=False, float_format='%.4f')
        print(f"✅ Per-class results saved to: {per_class_file}")
        
        # Save tier results
        tier_file = output_dir / f"{self.experiment_name}_tier_summary.csv"
        tier_df.to_csv(tier_file, index=False, float_format='%.4f')
        print(f"✅ Tier summary saved to: {tier_file}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📈 Tier Summary for {self.experiment_name}")
        print(f"{'='*70}")
        print(tier_df.to_string(index=False))
        print(f"{'='*70}\n")
        
        # Save detailed report
        self._save_detailed_report(group_results, per_class_df, tier_df)
        
    def _save_detailed_report(self, group_results: list, per_class_df: pd.DataFrame, tier_df: pd.DataFrame):
        """Save a detailed text report"""
        output_dir = self.results_dir / 'group_evaluation'
        report_file = output_dir / f"{self.experiment_name}_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"Experiment Report: {self.experiment_name}\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Sample: {self.sample}\n")
            f.write(f"Loss: {self.loss_function}\n")
            f.write(f"Number of groups: {len(group_results)}\n\n")
            
            # Best epochs per group
            f.write(f"{'='*70}\n")
            f.write(f"Best Epochs per Group\n")
            f.write(f"{'='*70}\n")
            for result in group_results:
                auc = result['metrics'].get('auc', 'N/A')
                auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else auc
                f.write(f"Classes {result['label_head']} to {result['label_tail']}: Epoch {result['best_epoch']} (AUC: {auc_str})\n")
            f.write("\n")
            
            # Class definitions
            f.write(f"{'='*70}\n")
            f.write(f"Class Definitions\n")
            f.write(f"{'='*70}\n")
            f.write(f"HEAD classes: {self.HEAD_CLASSES}\n")
            f.write(f"MID classes: {self.MID_CLASSES}\n")
            f.write(f"TAIL classes: {self.TAIL_CLASSES}\n\n")
            
            # Per-class metrics
            f.write(f"{'='*70}\n")
            f.write(f"Per-Class Metrics\n")
            f.write(f"{'='*70}\n")
            f.write(per_class_df.to_string(index=False))
            f.write("\n\n")
            
            # Tier summary
            f.write(f"{'='*70}\n")
            f.write(f"Tier Summary\n")
            f.write(f"{'='*70}\n")
            f.write(tier_df.to_string(index=False))
            f.write("\n\n")
        
        print(f"✅ Detailed report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate group experiment results')
    parser.add_argument('--model_name', type=str, default='MedViT_tiny',
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='chestmnist',
                        help='Dataset name')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment (e.g., exp1, exp2)')
    parser.add_argument('--group_configs', nargs='+', required=True,
                        help='List of "head,tail" pairs. E.g., "3,3" "2,0" "5,8"')
    parser.add_argument('--sample', type=float, default=0.1,
                        help='Sample fraction used in training')
    parser.add_argument('--loss_function', type=str, default='default',
                        help='Loss function directory name')
    
    args = parser.parse_args()
    
    # Parse group configs
    group_configs = []
    for config in args.group_configs:
        head, tail = map(int, config.split(','))
        group_configs.append((head, tail))
    
    aggregator = GroupResultAggregator(
        model_name=args.model_name,
        dataset=args.dataset,
        experiment_name=args.experiment_name,
        sample=args.sample,
        loss_function=args.loss_function
    )
    
    aggregator.aggregate_results(group_configs)


if __name__ == '__main__':
    main()
