"""
Merge Group Evaluations and Analyze Best Configurations
Combines the best configuration from each group's core labels to create
a single comprehensive result with all 14 classes, then analyzes head/mid/tail performance.

Usage:
    python merge_group_expanding_algo_evaluations.py --eval_dir results/grouping_expanding_algo_evaluation
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
from analyze_single_result import SingleResultAnalyzer


class GroupEvaluationMerger:
    """Merge and analyze group evaluation results"""
    
    # Group core labels definition
    GROUP_CORE_LABELS = {
        0: [3, 2, 0],                          # Group 0 core
        1: [5, 4, 7, 8, 12, 1, 10, 9],        # Group 1 core
        2: [11, 6],                            # Group 2 core
        3: [13]                                # Group 3 core
    }
    
    # Class tier definitions
    HEAD_CLASSES = [3, 2, 0]
    MID_CLASSES = [5, 4, 7, 8]
    TAIL_CLASSES = [12, 1, 10, 9, 11, 6, 13]
    
    def __init__(self, eval_dir: str, model_name: str = 'MedViT_tiny', 
                 dataset: str = 'chestmnist', loss_function: str = 'default'):
        self.eval_dir = Path(eval_dir)
        self.model_name = model_name
        self.dataset = dataset
        self.loss_function = loss_function
        self.results_dir = Path('./results')
        
        if not self.eval_dir.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    
    def load_group_evaluations(self) -> dict:
        """
        Load all four group evaluation CSV files
        
        Returns:
            dict: {group_id: DataFrame}
        """
        group_dfs = {}
        
        for group_id in range(4):
            csv_file = self.eval_dir / f'group{group_id}_evaluation.csv'
            
            if not csv_file.exists():
                print(f"⚠️  Warning: {csv_file} not found, skipping Group {group_id}")
                continue
            
            df = pd.read_csv(csv_file)
            group_dfs[group_id] = df
            print(f"✅ Loaded Group {group_id}: {len(df)} configurations")
        
        return group_dfs
    
    def find_best_config_per_group(self, group_dfs: dict) -> dict:
        """
        Find the best configuration for each group based on core_mAP
        
        Args:
            group_dfs: Dictionary of group DataFrames
            
        Returns:
            dict: {group_id: best_config_row (as dict)}
        """
        best_configs = {}
        
        for group_id, df in group_dfs.items():
            if len(df) == 0:
                continue
            
            # Sort by core_mean_auc and get the best
            df_sorted = df.sort_values('core_mean_auc', ascending=False)
            best_row = df_sorted.iloc[0].to_dict()
            
            best_configs[group_id] = best_row
            
            print(f"  Group {group_id}: {best_row['config_name']} "
                  f"(core_mAP={best_row['core_mAP']:.4f}, core_AUC={best_row['core_mean_auc']:.4f})")
        
        return best_configs
    
    def merge_best_configs_metrics(self, best_configs: dict) -> dict:
        """
        Merge metrics from all groups' best configurations into a single result
        containing all 14 classes
        
        Args:
            best_configs: Dictionary of best configurations per group
            
        Returns:
            dict: Merged metrics with AUC_class_X and AP_class_X for all classes
        """
        print(f"\n{'='*70}")
        print(f"🔄 Merging Best Configurations from Each Group")
        print(f"{'='*70}\n")
        
        # Initialize merged result with NaN for all classes
        merged_metrics = {
            'epoch': 'mixed',  # Different groups may use different epochs
            'method': 'best_per_group_merged'
        }
        
        # Add placeholders for all 14 classes
        for class_id in range(14):
            merged_metrics[f'AUC_class_{class_id}'] = np.nan
            merged_metrics[f'AP_class_{class_id}'] = np.nan
        
        # Fill in metrics from each group's best configuration
        for group_id, config in best_configs.items():
            core_labels = self.GROUP_CORE_LABELS[group_id]
            
            print(f"Group {group_id} - Core labels {core_labels}:")
            
            for class_id in core_labels:
                # Extract AUC and AP for this class
                auc_key = f'AUC_class_{class_id}'
                ap_key = f'AP_class_{class_id}'
                
                if auc_key in config:
                    merged_metrics[auc_key] = config[auc_key]
                    print(f"  Class {class_id}: AUC={config[auc_key]:.4f}, AP={config[ap_key]:.4f}")
                else:
                    print(f"  ⚠️  Class {class_id}: Metrics not found in config")
                
                if ap_key in config:
                    merged_metrics[ap_key] = config[ap_key]
        
        print(f"\n{'='*70}\n")
        
        return merged_metrics
    
    def save_merged_csv(self, merged_metrics: dict, output_path: Path):
        """
        Save merged metrics to CSV in a format compatible with analyze_single_result.py
        
        Args:
            merged_metrics: Dictionary of merged metrics
            output_path: Path to save CSV file
        """
        # Create a DataFrame with a single row
        df = pd.DataFrame([merged_metrics])
        
        # Reorder columns to match expected format
        cols_order = ['epoch', 'method']
        
        # Add AUC columns in order
        for class_id in range(14):
            cols_order.append(f'AUC_class_{class_id}')
        
        # Add AP columns in order
        for class_id in range(14):
            cols_order.append(f'AP_class_{class_id}')
        
        # Add any remaining columns
        remaining_cols = [col for col in df.columns if col not in cols_order]
        cols_order.extend(remaining_cols)
        
        df = df[cols_order]
        
        # Save to CSV
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"✅ Merged metrics saved to: {output_path}")
        
        return df
    
    def analyze_merged_result(self, merged_csv_path: Path):
        """
        Analyze the merged result using SingleResultAnalyzer
        
        Args:
            merged_csv_path: Path to the merged CSV file
        """
        print(f"\n{'='*70}")
        print(f"📊 Analyzing Merged Result (Head/Mid/Tail Performance)")
        print(f"{'='*70}\n")
        
        try:
            # Create a custom analyzer that reads the merged CSV
            df = pd.read_csv(merged_csv_path)
            
            if len(df) == 0:
                print("❌ Empty CSV file")
                return
            
            # Extract metrics manually since we only have one row
            best_row = df.iloc[0].to_dict()
            
            # Extract per-class metrics
            all_classes = sorted(self.HEAD_CLASSES + self.MID_CLASSES + self.TAIL_CLASSES)
            
            per_class_data = {
                'class_id': [],
                'AP': [],
                'AUC': [],
                'tier': []
            }
            
            for class_id in all_classes:
                # Determine tier
                if class_id in self.HEAD_CLASSES:
                    tier = 'head'
                elif class_id in self.MID_CLASSES:
                    tier = 'mid'
                else:
                    tier = 'tail'
                
                # Extract AP and AUC
                ap_key = f'AP_class_{class_id}'
                auc_key = f'AUC_class_{class_id}'
                
                ap_value = best_row.get(ap_key, np.nan)
                auc_value = best_row.get(auc_key, np.nan)
                
                per_class_data['class_id'].append(class_id)
                per_class_data['AP'].append(ap_value)
                per_class_data['AUC'].append(auc_value)
                per_class_data['tier'].append(tier)
            
            per_class_df = pd.DataFrame(per_class_data)
            
            # Calculate tier metrics
            tier_data = []
            
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
            
            # Calculate total
            tier_data.append({
                'tier': 'total',
                'num_classes': len(per_class_df),
                'mAP': per_class_df['AP'].mean(),
                'AUC': per_class_df['AUC'].mean(),
                'class_ids': 'all'
            })
            
            tier_df = pd.DataFrame(tier_data)
            
            # Print results
            print(f"{'='*70}")
            print(f"📈 Per-Class Metrics (Merged Best Configurations)")
            print(f"{'='*70}")
            print(per_class_df.to_string(index=False))
            print(f"{'='*70}\n")
            
            print(f"{'='*70}")
            print(f"📈 Tier Summary (Merged Best Configurations)")
            print(f"{'='*70}")
            print(tier_df.to_string(index=False))
            print(f"{'='*70}\n")
            
            # Save analysis results
            output_dir = merged_csv_path.parent
            
            per_class_file = output_dir / 'merged_best_per_class.csv'
            per_class_df.to_csv(per_class_file, index=False, float_format='%.4f')
            print(f"✅ Per-class metrics saved to: {per_class_file}")
            
            tier_file = output_dir / 'merged_best_tier_summary.csv'
            tier_df.to_csv(tier_file, index=False, float_format='%.4f')
            print(f"✅ Tier summary saved to: {tier_file}")
            
            return {
                'per_class': per_class_df,
                'tier_summary': tier_df
            }
            
        except Exception as e:
            print(f"❌ Error analyzing merged result: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_comprehensive_report(self, best_configs: dict, merged_metrics: dict, 
                                   analysis_result: dict):
        """Save a comprehensive text report"""
        output_dir = self.eval_dir
        report_file = output_dir / 'merged_best_comprehensive_report.txt'
        
        with open(report_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"Merged Best Configurations Report\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Loss Function: {self.loss_function}\n")
            f.write(f"Method: Best configuration per group, merged by core labels\n\n")
            
            # Best configuration per group
            f.write(f"{'='*70}\n")
            f.write(f"Best Configuration per Group (by core_mAP)\n")
            f.write(f"{'='*70}\n")
            for group_id, config in best_configs.items():
                f.write(f"\nGroup {group_id} - Core labels: {self.GROUP_CORE_LABELS[group_id]}\n")
                f.write(f"  Config: {config['config_name']}\n")
                f.write(f"  Label Range: {config['label_head']} to {config['label_tail']}\n")
                f.write(f"  Num Labels Trained: {config['num_labels_trained']}\n")
                f.write(f"  Core mAP: {config['core_mAP']:.4f}\n")
                f.write(f"  Core AUC: {config['core_mean_auc']:.4f}\n")
                f.write(f"  Epoch: {config['epoch']}\n")
            f.write("\n")
            
            # Class definitions
            f.write(f"{'='*70}\n")
            f.write(f"Class Tier Definitions\n")
            f.write(f"{'='*70}\n")
            f.write(f"HEAD classes: {self.HEAD_CLASSES}\n")
            f.write(f"MID classes: {self.MID_CLASSES}\n")
            f.write(f"TAIL classes: {self.TAIL_CLASSES}\n\n")
            
            # Analysis results
            if analysis_result:
                f.write(f"{'='*70}\n")
                f.write(f"Per-Class Metrics (Merged)\n")
                f.write(f"{'='*70}\n")
                f.write(analysis_result['per_class'].to_string(index=False))
                f.write("\n\n")
                
                f.write(f"{'='*70}\n")
                f.write(f"Tier Summary (Merged)\n")
                f.write(f"{'='*70}\n")
                f.write(analysis_result['tier_summary'].to_string(index=False))
                f.write("\n\n")
        
        print(f"✅ Comprehensive report saved to: {report_file}")
    
    def run(self):
        """Main execution function"""
        print(f"\n{'='*70}")
        print(f"🔄 Merging Best Group Configurations")
        print(f"{'='*70}")
        print(f"Evaluation directory: {self.eval_dir}")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"{'='*70}\n")
        
        # Load all group evaluations
        print("📂 Loading group evaluation files...")
        group_dfs = self.load_group_evaluations()
        
        if not group_dfs:
            print("\n❌ No group evaluation files found!")
            return
        
        print(f"\n✅ Loaded {len(group_dfs)} group evaluations\n")
        
        # Find best configuration per group
        print("🏆 Finding best configuration per group (by core_mAP)...")
        best_configs = self.find_best_config_per_group(group_dfs)
        
        # Merge metrics from best configurations
        merged_metrics = self.merge_best_configs_metrics(best_configs)
        
        # Save merged CSV
        merged_csv_path = self.eval_dir / 'merged_best_configs.csv'
        self.save_merged_csv(merged_metrics, merged_csv_path)
        
        # Analyze merged result
        analysis_result = self.analyze_merged_result(merged_csv_path)
        
        # Save comprehensive report
        if analysis_result:
            self.save_comprehensive_report(best_configs, merged_metrics, analysis_result)
        
        print(f"\n{'='*70}")
        print(f"✅ Analysis Complete!")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Merge best configurations from each group and analyze head/mid/tail performance'
    )
    parser.add_argument('--eval_dir', type=str, 
                        default='results/grouping_expanding_algo_evaluation',
                        help='Directory containing group evaluation CSV files')
    parser.add_argument('--model_name', type=str, default='MedViT_tiny',
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='chestmnist',
                        help='Dataset name')
    parser.add_argument('--loss_function', type=str, default='default',
                        help='Loss function directory name')
    
    args = parser.parse_args()
    
    merger = GroupEvaluationMerger(
        eval_dir=args.eval_dir,
        model_name=args.model_name,
        dataset=args.dataset,
        loss_function=args.loss_function
    )
    
    merger.run()


if __name__ == '__main__':
    main()
