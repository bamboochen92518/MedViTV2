"""
Long-Tail Dataset Splitter
Split dataset classes into head, middle, and tail groups based on sample counts

This module provides utilities to analyze and split long-tail datasets into three tiers:
- Head: Classes with >= 7500 samples (high frequency)
- Middle: Classes with 2500 < samples < 7500
- Tail: Classes with <= 2500 samples (low frequency)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional
import os
from collections import Counter


class LongTailSplitter:
    """
    Split dataset classes into head, middle, and tail groups based on frequency
    
    Fixed thresholds:
    - Head: >= 7500 samples
    - Tail: <= 2500 samples
    - Middle: Everything in between
    """
    
    def __init__(self, dataset, dataset_name: str = 'dataset'):
        """
        Initialize the splitter with a dataset
        
        Args:
            dataset: PyTorch dataset or similar object with labels
            dataset_name: Name for display and saving purposes
        """
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.class_counts = None
        self.sorted_classes = None
        self.is_multilabel = False
        
        # Analyze dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze the dataset to get class distribution"""
        print(f"\n{'='*60}")
        print(f"Analyzing Dataset: {self.dataset_name}")
        print(f"{'='*60}")
        
        # Check if multi-label
        _, first_label = self.dataset[0]
        if isinstance(first_label, (torch.Tensor, np.ndarray)):
            if hasattr(first_label, 'shape') and len(first_label.shape) > 0:
                if first_label.shape[0] > 1 or (hasattr(first_label, 'size') and first_label.size > 1):
                    self.is_multilabel = True
        
        print(f"  Dataset type: {'Multi-label' if self.is_multilabel else 'Single-label'}")
        print(f"  Total samples: {len(self.dataset)}")
        
        # Count class frequencies
        if self.is_multilabel:
            # Multi-label: count how many samples have each class
            _, first_label = self.dataset[0]
            if isinstance(first_label, torch.Tensor):
                num_classes = first_label.shape[0]
            else:
                num_classes = len(first_label)
            
            class_counts = np.zeros(num_classes, dtype=int)
            
            for idx in range(len(self.dataset)):
                _, label = self.dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.numpy()
                elif not isinstance(label, np.ndarray):
                    label = np.array(label)
                label = label.flatten()
                class_counts += (label > 0).astype(int)
            
            self.class_counts = {i: int(count) for i, count in enumerate(class_counts)}
        else:
            # Single-label: count occurrences of each class
            labels = []
            for idx in range(len(self.dataset)):
                _, label = self.dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item() if label.numel() == 1 else label.numpy()[0]
                elif isinstance(label, np.ndarray):
                    label = int(label.flatten()[0])
                else:
                    label = int(label)
                labels.append(label)
            
            self.class_counts = dict(Counter(labels))
        
        # Sort classes by frequency (descending)
        self.sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"  Number of classes: {len(self.class_counts)}")
        print(f"\n  Class distribution (sorted by frequency):")
        for class_id, count in self.sorted_classes:
            print(f"    Class {class_id}: {count} samples")
        print(f"{'='*60}\n")
    
    def split(self) -> Dict[str, List[int]]:
        """
        Split into head (>=7500), middle (2500-7500), tail (<=2500)
        
        Returns:
            Dictionary with 'head', 'middle', 'tail' keys containing class lists
        """
        head = []
        middle = []
        tail = []
        
        for class_id, count in self.sorted_classes:
            if count >= 7500:
                head.append(class_id)
            elif count <= 2500:
                tail.append(class_id)
            else:
                middle.append(class_id)
        
        result = {
            'head': head,
            'middle': middle,
            'tail': tail
        }
        
        self._print_split_summary(result)
        return result
    
    def _print_split_summary(self, split_result: Dict[str, List[int]]):
        """Print summary of the split"""
        print(f"\n{'='*60}")
        print(f"Split Result: head>=7500, tail<=2500")
        print(f"{'='*60}")
        
        for tier in ['head', 'middle', 'tail']:
            classes = split_result[tier]
            if not classes:
                print(f"\n  {tier.upper()}: (empty)")
                continue
            
            counts = [self.class_counts[c] for c in classes]
            total_samples = sum(counts)
            
            print(f"\n  {tier.upper()}: {len(classes)} classes, {total_samples} samples")
            print(f"    Classes: {classes}")
            print(f"    Sample counts: {counts}")
            print(f"    Range: [{min(counts)}, {max(counts)}]")
            print(f"    Mean: {np.mean(counts):.1f}, Median: {np.median(counts):.1f}")
        
        print(f"{'='*60}\n")
    
    def visualize_split(self, 
                       split_result: Dict[str, List[int]],
                       save_path: Optional[str] = None):
        """
        Visualize the split with a bar chart
        
        Args:
            split_result: Result from the split method
            save_path: Path to save the figure (if None, generates default path)
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Prepare data
        all_classes = []
        all_counts = []
        colors = []
        
        color_map = {
            'head': '#2ecc71',  # Green
            'middle': '#f39c12',  # Orange
            'tail': '#e74c3c'  # Red
        }
        
        for tier in ['head', 'middle', 'tail']:
            classes = split_result[tier]
            for class_id in classes:
                all_classes.append(class_id)
                all_counts.append(self.class_counts[class_id])
                colors.append(color_map[tier])
        
        # Create bar chart
        x = np.arange(len(all_classes))
        bars = ax.bar(x, all_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Class ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title(f'Long-Tail Distribution: {self.dataset_name}\n(Head≥7500, Tail≤2500)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['head'], label=f"Head (≥7500): {len(split_result['head'])} classes"),
            Patch(facecolor=color_map['middle'], label=f"Middle (2500-7500): {len(split_result['middle'])} classes"),
            Patch(facecolor=color_map['tail'], label=f"Tail (≤2500): {len(split_result['tail'])} classes")
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            os.makedirs('./results/longtail_splits', exist_ok=True)
            save_path = f'./results/longtail_splits/{self.dataset_name}_longtail_split.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
        plt.close()
    
    def get_imbalance_ratio(self) -> float:
        """Calculate the imbalance ratio (max_samples / min_samples)"""
        counts = [count for _, count in self.sorted_classes]
        return max(counts) / min(counts)
    
    def export_split(self, 
                    split_result: Dict[str, List[int]],
                    output_path: Optional[str] = None) -> str:
        """
        Export split result to a file
        
        Args:
            split_result: Result from the split method
            output_path: Path to save the file (if None, generates default path)
            
        Returns:
            Path where the file was saved
        """
        if output_path is None:
            os.makedirs('./results/longtail_splits', exist_ok=True)
            output_path = f'./results/longtail_splits/{self.dataset_name}_split.txt'
        
        with open(output_path, 'w') as f:
            f.write(f"Long-Tail Split for {self.dataset_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Thresholds: Head >= 7500, Tail <= 2500\n")
            f.write(f"{'='*60}\n\n")
            
            for tier in ['head', 'middle', 'tail']:
                classes = split_result[tier]
                f.write(f"{tier.upper()}:\n")
                f.write(f"  Classes: {classes}\n")
                if classes:
                    counts = [self.class_counts[c] for c in classes]
                    f.write(f"  Sample counts: {counts}\n")
                    f.write(f"  Total samples: {sum(counts)}\n")
                f.write("\n")
            
            # Add imbalance ratio
            ir = self.get_imbalance_ratio()
            f.write(f"Overall Imbalance Ratio: {ir:.2f}\n")
        
        print(f"✅ Split exported to: {output_path}")
        return output_path


def analyze_medmnist_longtail(dataset_name: str = 'chestmnist'):
    """
    Analyze and split a MedMNIST dataset into head/middle/tail
    
    Fixed thresholds: Head >= 7500, Tail <= 2500
    
    Args:
        dataset_name: Name of the MedMNIST dataset (e.g., 'chestmnist')
        
    Example:
        >>> result, splitter = analyze_medmnist_longtail('chestmnist')
    """
    import medmnist
    from medmnist import INFO
    
    # Load dataset
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True, as_rgb=True,
                             root='./data', size=224, mmap_mode='r')
    
    # Create splitter
    splitter = LongTailSplitter(train_dataset, dataset_name)
    
    # Perform split
    result = splitter.split()
    
    # Visualize and export
    splitter.visualize_split(result)
    splitter.export_split(result)
    
    return result, splitter


if __name__ == '__main__':
    """
    Example usage: Analyze ChestMNIST dataset
    """
    print("\n" + "="*60)
    print("Long-Tail Dataset Splitter")
    print("Thresholds: Head >= 7500, Tail <= 2500")
    print("="*60 + "\n")
    
    result, splitter = analyze_medmnist_longtail('chestmnist')
    
    print("\n✅ Analysis completed!")
    print(f"\nImbalance Ratio: {splitter.get_imbalance_ratio():.2f}")
