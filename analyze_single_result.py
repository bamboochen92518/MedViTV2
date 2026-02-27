"""
Analyze Single Result File
Analyze head, middle, tail metrics from a single model_metrics.csv file

Usage:
    python analyze_single_result.py --csv_path ./results/CBLoss/chestmnist/MedViT_tiny/sample_10pct/model_metrics.csv
    python analyze_single_result.py --csv_path ./results/default/chestmnist/MedViT_tiny/class_3_to_7_sample_10pct/model_metrics.csv --best_metric auc
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


class SingleResultAnalyzer:
    """Analyze a single CSV file for head/mid/tail metrics"""
    
    # Fixed class definitions for ChestMNIST
    HEAD_CLASSES = [3, 2, 0]
    MID_CLASSES = [5, 4, 7, 8]
    TAIL_CLASSES = [12, 1, 10, 9, 11, 6, 13]
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
    def find_best_epoch(self, metric: str = 'auc') -> dict:
        """
        Find the best epoch based on specified metric
        
        Args:
            metric: Metric to use for finding best epoch ('auc', 'mAP', 'val_accuracy')
            
        Returns:
            Dictionary containing best epoch info
        """
        if metric not in self.df.columns:
            print(f"⚠️  Warning: Metric '{metric}' not found in CSV. Available: {list(self.df.columns)}")
            metric = 'auc' if 'auc' in self.df.columns else 'val_accuracy'
            print(f"   Using '{metric}' instead.")
        
        best_idx = self.df[metric].idxmax()
        best_row = self.df.loc[best_idx]
        
        return {
            'epoch': int(best_row['epoch']) if 'epoch' in best_row else best_idx + 1,
            'metrics': best_row.to_dict()
        }
    
    def extract_per_class_metrics(self, best_row: dict) -> pd.DataFrame:
        """
        Extract per-class AP and AUC from the best epoch
        
        Args:
            best_row: Dictionary containing metrics from best epoch
            
        Returns:
            DataFrame with per-class metrics
        """
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
            
            # Extract AP
            ap_key = f'AP_class_{class_id}'
            ap_value = best_row['metrics'].get(ap_key, np.nan)
            
            # Extract AUC
            auc_key = f'AUC_class_{class_id}'
            auc_value = best_row['metrics'].get(auc_key, np.nan)
            
            per_class_data['class_id'].append(class_id)
            per_class_data['AP'].append(ap_value)
            per_class_data['AUC'].append(auc_value)
            per_class_data['tier'].append(tier)
        
        return pd.DataFrame(per_class_data)
    
    def calculate_tier_metrics(self, per_class_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregated metrics for head, mid, tail, and total
        
        Args:
            per_class_df: DataFrame with per-class metrics
            
        Returns:
            DataFrame with tier-level metrics
        """
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
        
        return pd.DataFrame(tier_data)
    
    def analyze(self, best_metric: str = 'auc', save_output: bool = True):
        """
        Main analysis function
        
        Args:
            best_metric: Metric to use for finding best epoch
            save_output: Whether to save output files
        """
        print(f"\n{'='*70}")
        print(f"📊 Analyzing Result File")
        print(f"{'='*70}")
        print(f"CSV file: {self.csv_path}")
        print(f"Best metric: {best_metric}")
        print(f"{'='*70}\n")
        
        # Find best epoch
        print(f"🔍 Finding best epoch based on '{best_metric}'...")
        best_row = self.find_best_epoch(best_metric)
        print(f"   ✅ Best epoch: {best_row['epoch']}")
        print(f"   Overall AUC: {best_row['metrics'].get('auc', 'N/A'):.4f}")
        print(f"   Overall mAP: {best_row['metrics'].get('mAP', 'N/A'):.4f}")
        print()
        
        # Extract per-class metrics
        print("📋 Extracting per-class metrics...")
        per_class_df = self.extract_per_class_metrics(best_row)
        
        # Calculate tier metrics
        print("📊 Calculating tier-level metrics...\n")
        tier_df = self.calculate_tier_metrics(per_class_df)
        
        # Print results
        print(f"\n{'='*70}")
        print(f"📈 Per-Class Metrics (Best Epoch: {best_row['epoch']})")
        print(f"{'='*70}")
        print(per_class_df.to_string(index=False))
        print(f"{'='*70}\n")
        
        print(f"\n{'='*70}")
        print(f"📈 Tier Summary (Best Epoch: {best_row['epoch']})")
        print(f"{'='*70}")
        print(tier_df.to_string(index=False))
        print(f"{'='*70}\n")
        
        # Save outputs
        if save_output:
            output_dir = self.csv_path.parent / 'analysis'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save per-class results
            per_class_file = output_dir / 'per_class_metrics.csv'
            per_class_df.to_csv(per_class_file, index=False, float_format='%.4f')
            print(f"✅ Per-class metrics saved to: {per_class_file}")
            
            # Save tier results
            tier_file = output_dir / 'tier_summary.csv'
            tier_df.to_csv(tier_file, index=False, float_format='%.4f')
            print(f"✅ Tier summary saved to: {tier_file}")
            
            # Save detailed report
            report_file = output_dir / 'analysis_report.txt'
            self._save_report(best_row, per_class_df, tier_df, report_file)
            print(f"✅ Report saved to: {report_file}")
        
        return {
            'best_epoch': best_row,
            'per_class': per_class_df,
            'tier_summary': tier_df
        }
    
    def _save_report(self, best_row: dict, per_class_df: pd.DataFrame, 
                     tier_df: pd.DataFrame, report_file: Path):
        """Save detailed text report"""
        with open(report_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"Single Result Analysis Report\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"CSV File: {self.csv_path}\n")
            f.write(f"Best Epoch: {best_row['epoch']}\n")
            f.write(f"Overall AUC: {best_row['metrics'].get('auc', 'N/A'):.4f}\n")
            f.write(f"Overall mAP: {best_row['metrics'].get('mAP', 'N/A'):.4f}\n\n")
            
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


def main():
    parser = argparse.ArgumentParser(description='Analyze single result file for head/mid/tail metrics')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the model_metrics.csv file')
    parser.add_argument('--best_metric', type=str, default='auc',
                        choices=['auc', 'mAP', 'val_accuracy'],
                        help='Metric to use for finding best epoch')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save output files')
    
    args = parser.parse_args()
    
    analyzer = SingleResultAnalyzer(args.csv_path)
    analyzer.analyze(best_metric=args.best_metric, save_output=not args.no_save)


if __name__ == '__main__':
    main()
