import os
import json
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import medmnist
from medmnist import INFO, Evaluator

import requests
from zipfile import ZipFile
import pandas as pd
import shutil

# Import sampler utilities from sampler.py
from sampler import get_class_index_dict

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)

root_dir='data'
if not os.path.exists(root_dir):
            os.makedirs(root_dir)


def get_label_groups(dataset_name):
    """
    Define label grouping strategy for each dataset
    
    Args:
        dataset_name (str): Dataset name
    
    Returns:
        list: Group list in format [[group0_labels], [group1_labels], [group2_labels], ...]
              Returns None if the dataset does not support grouping
    
    Example:
        >>> get_label_groups('chestmnist')
        [[3, 2, 0], [5, 4, 7, 8, 12, 1, 10, 9], [11, 6], [13]]
    """
    groups = {
        'chestmnist': [
            [3, 2, 0],                          # Group 0: 3 classes (high frequency)
            [5, 4, 7, 8, 12, 1, 10, 9],         # Group 1: 8 classes (medium frequency)
            [11, 6],                            # Group 2: 2 classes (low frequency)
            [13]                                # Group 3: 1 class (very low frequency)
        ],
        # Add group definitions for other datasets here
        # 'pathmnist': [...],
        # 'dermamnist': [...],
    }
    return groups.get(dataset_name, None)


def get_sorted_label_order(dataset_name):
    """
    Get the sorted label order for a dataset (by frequency, descending)
    
    Args:
        dataset_name (str): Dataset name
        
    Returns:
        list: Sorted list of label indices
        
    Example:
        >>> get_sorted_label_order('chestmnist')
        [3, 2, 0, 5, 4, 7, 8, 12, 1, 10, 9, 11, 6, 13]
    """
    # ChestMNIST sorted by frequency (from DP algorithm)
    if dataset_name == 'chestmnist':
        return [3, 2, 0, 5, 4, 7, 8, 12, 1, 10, 9, 11, 6, 13]
    
    return None


def get_label_range(dataset_name, head_class, tail_class):
    """
    Get labels within a specified range based on sorted order
    
    Args:
        dataset_name (str): Dataset name
        head_class (int): Starting class (inclusive)
        tail_class (int): Ending class (inclusive)
        
    Returns:
        list: List of class labels in the range
        
    Example:
        >>> get_label_range('chestmnist', 2, 10)
        [2, 0, 5, 4, 7, 8, 12, 1, 10]
        >>> get_label_range('chestmnist', 3, 5)
        [3, 2, 0, 5]
    """
    sorted_order = get_sorted_label_order(dataset_name)
    
    if sorted_order is None:
        raise ValueError(f"No sorted order defined for dataset {dataset_name}")
    
    # Find indices of head and tail in sorted order
    try:
        head_idx = sorted_order.index(head_class)
        tail_idx = sorted_order.index(tail_class)
    except ValueError as e:
        raise ValueError(f"Class not found in sorted order: {e}")
    
    # Ensure head comes before tail
    if head_idx > tail_idx:
        raise ValueError(f"head_class ({head_class}) must come before tail_class ({tail_class}) in sorted order")
    
    # Extract range (inclusive)
    label_range = sorted_order[head_idx:tail_idx + 1]
    
    return label_range


class SampledDataset(Dataset):
    """
    Wraps a dataset to use only a fraction of samples
    
    Args:
        base_dataset: Original dataset
        sample_ratio: Fraction of data to use (0 < sample_ratio <= 1.0)
        seed: Random seed for reproducibility (default: 42)
        dataset_name: Name of the dataset (for saving plots)
        plot_distribution: Whether to plot distribution comparison (default: True)
    """
    
    def __init__(self, base_dataset, sample_ratio, seed=42, dataset_name='dataset', plot_distribution=True):
        self.base_dataset = base_dataset
        self.sample_ratio = sample_ratio
        self.dataset_name = dataset_name
        self.plot_distribution = plot_distribution
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Calculate number of samples to use
        total_samples = len(base_dataset)
        num_samples = int(total_samples * sample_ratio)
        
        # Randomly select indices
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)
        self.sampled_indices = all_indices[:num_samples].tolist()
        
        print(f"  📊 Sampling {sample_ratio*100:.1f}% of dataset:")
        print(f"     Total samples: {total_samples}")
        print(f"     Sampled: {num_samples} samples")
        print(f"     Random seed: {seed}")
        
        # Plot original and sampled distribution if enabled
        if self.plot_distribution:
            self._plot_distribution_comparison()
        else:
            print(f"  ℹ️  Distribution plotting disabled (use --plot_distribution True to enable)")

    def _print_cooccurrence_table(self, cooccurrence_matrix, num_classes):
        """
        Print co-occurrence matrix in a readable table format.
        """
        print("     ", end="")
        for j in range(num_classes):
            print(f"Label_{j:2d} ", end="")
        print()
        for i in range(num_classes):
            print(f"Label_{i:2d} ", end="")
            for j in range(num_classes):
                print(f"{cooccurrence_matrix[i, j]:8d} ", end="")
            print()

    def _plot_distribution_comparison(self):
        """
        Plot comparison of original and sampled dataset distributions.
        Handles both single-label and multi-label datasets.
        """
        print("  📊 Generating distribution comparison plot...")
        
        # Collect labels from original and sampled datasets
        original_labels = []
        sampled_labels = []
        
        # Check if this is multi-label by examining first sample
        _, first_label = self.base_dataset[0]
        is_multilabel = False
        
        if isinstance(first_label, (torch.Tensor, np.ndarray)):
            if hasattr(first_label, 'shape') and len(first_label.shape) > 0:
                if first_label.shape[0] > 1 or (hasattr(first_label, 'size') and first_label.size > 1):
                    is_multilabel = True
        
        if is_multilabel:
            # Multi-label case: count presence of each class
            print("     Detected multi-label dataset")
            
            # Get number of classes from first label
            if isinstance(first_label, torch.Tensor):
                num_classes = first_label.shape[0]
            else:
                num_classes = len(first_label)
            
            original_class_counts = np.zeros(num_classes)
            sampled_class_counts = np.zeros(num_classes)
            
            # ✨ Initialize co-occurrence matrices
            original_cooccurrence = np.zeros((num_classes, num_classes), dtype=int)
            sampled_cooccurrence = np.zeros((num_classes, num_classes), dtype=int)
            
            # Count original distribution
            for idx in range(len(self.base_dataset)):
                _, label = self.base_dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.numpy()
                elif not isinstance(label, np.ndarray):
                    label = np.array(label)
                label = label.flatten()
                original_class_counts += (label > 0).astype(int)
                
                # ✨ Count co-occurrences for original dataset
                active_labels = np.where(label > 0)[0]
                for i in active_labels:
                    for j in active_labels:
                        original_cooccurrence[i, j] += 1
            
            # Count sampled distribution
            for idx in self.sampled_indices:
                _, label = self.base_dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.numpy()
                elif not isinstance(label, np.ndarray):
                    label = np.array(label)
                label = label.flatten()
                sampled_class_counts += (label > 0).astype(int)
                
                # ✨ Count co-occurrences for sampled dataset
                active_labels = np.where(label > 0)[0]
                for i in active_labels:
                    for j in active_labels:
                        sampled_cooccurrence[i, j] += 1
            
            # ✨ Print co-occurrence matrices
            print("\n  📊 Label Co-occurrence Analysis:")
            print("  " + "="*70)
            print("  Original Dataset - Samples with both Label i AND Label j:")
            self._print_cooccurrence_table(original_cooccurrence, num_classes)
            
            print("\n  Sampled Dataset - Samples with both Label i AND Label j:")
            self._print_cooccurrence_table(sampled_cooccurrence, num_classes)
            print("  " + "="*70 + "\n")
            
            # ✨ Save co-occurrence matrices as CSV
            os.makedirs('./results/sampling_plots', exist_ok=True)
            original_csv_path = f'./results/sampling_plots/{self.dataset_name}_sample{int(self.sample_ratio*100)}_original_cooccurrence.csv'
            sampled_csv_path = f'./results/sampling_plots/{self.dataset_name}_sample{int(self.sample_ratio*100)}_sampled_cooccurrence.csv'
            
            # Create DataFrame with proper labels
            df_original = pd.DataFrame(original_cooccurrence, 
                                      index=[f'Label_{i}' for i in range(num_classes)],
                                      columns=[f'Label_{i}' for i in range(num_classes)])
            df_sampled = pd.DataFrame(sampled_cooccurrence,
                                     index=[f'Label_{i}' for i in range(num_classes)],
                                     columns=[f'Label_{i}' for i in range(num_classes)])
            
            df_original.to_csv(original_csv_path)
            df_sampled.to_csv(sampled_csv_path)
            print(f"  ✅ Co-occurrence matrices saved:")
            print(f"     Original: {original_csv_path}")
            print(f"     Sampled: {sampled_csv_path}")
            
            # ✨ Visualize co-occurrence matrices as heatmaps
            fig_cooccur, (ax_orig, ax_samp) = plt.subplots(1, 2, figsize=(18, 7))
            
            # Original co-occurrence heatmap
            im1 = ax_orig.imshow(original_cooccurrence, cmap='YlOrRd', aspect='auto')
            ax_orig.set_xlabel('Label j', fontsize=12)
            ax_orig.set_ylabel('Label i', fontsize=12)
            ax_orig.set_title('Original Dataset: Co-occurrence Matrix\n(samples with both label i AND j)', 
                             fontsize=14, fontweight='bold')
            ax_orig.set_xticks(range(num_classes))
            ax_orig.set_yticks(range(num_classes))
            
            # Add text annotations
            for i in range(num_classes):
                for j in range(num_classes):
                    text = ax_orig.text(j, i, original_cooccurrence[i, j],
                                       ha="center", va="center", color="black" if original_cooccurrence[i, j] < original_cooccurrence.max()/2 else "white",
                                       fontsize=8)
            
            plt.colorbar(im1, ax=ax_orig, label='Number of Samples')
            
            # Sampled co-occurrence heatmap
            im2 = ax_samp.imshow(sampled_cooccurrence, cmap='YlOrRd', aspect='auto')
            ax_samp.set_xlabel('Label j', fontsize=12)
            ax_samp.set_ylabel('Label i', fontsize=12)
            ax_samp.set_title(f'Sampled Dataset ({self.sample_ratio*100:.1f}%): Co-occurrence Matrix\n(samples with both label i AND j)', 
                             fontsize=14, fontweight='bold')
            ax_samp.set_xticks(range(num_classes))
            ax_samp.set_yticks(range(num_classes))
            
            # Add text annotations
            for i in range(num_classes):
                for j in range(num_classes):
                    text = ax_samp.text(j, i, sampled_cooccurrence[i, j],
                                       ha="center", va="center", color="black" if sampled_cooccurrence[i, j] < sampled_cooccurrence.max()/2 else "white",
                                       fontsize=8)
            
            plt.colorbar(im2, ax=ax_samp, label='Number of Samples')
            
            plt.tight_layout()
            cooccur_plot_path = f'./results/sampling_plots/{self.dataset_name}_sample{int(self.sample_ratio*100)}_cooccurrence.png'
            plt.savefig(cooccur_plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Co-occurrence heatmap saved to: {cooccur_plot_path}\n")
            plt.close()
            
            # Create bar plot (existing code)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            x = np.arange(num_classes)
            width = 0.35
            
            # Absolute counts
            ax1.bar(x - width/2, original_class_counts, width, label='Original', alpha=0.8, color='steelblue')
            ax1.bar(x + width/2, sampled_class_counts, width, label=f'Sampled ({self.sample_ratio*100:.1f}%)', alpha=0.8, color='coral')
            ax1.set_xlabel('Class Label', fontsize=12)
            ax1.set_ylabel('Number of Samples', fontsize=12)
            ax1.set_title('Absolute Distribution Comparison (Multi-Label)', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.legend(fontsize=11)
            ax1.grid(axis='y', alpha=0.3)
            
            # Percentage comparison
            original_pct = (original_class_counts / original_class_counts.sum()) * 100
            sampled_pct = (sampled_class_counts / sampled_class_counts.sum()) * 100
            
            ax2.bar(x - width/2, original_pct, width, label='Original', alpha=0.8, color='steelblue')
            ax2.bar(x + width/2, sampled_pct, width, label=f'Sampled ({self.sample_ratio*100:.1f}%)', alpha=0.8, color='coral')
            ax2.set_xlabel('Class Label', fontsize=12)
            ax2.set_ylabel('Percentage (%)', fontsize=12)
            ax2.set_title('Relative Distribution Comparison (Multi-Label)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.legend(fontsize=11)
            ax2.grid(axis='y', alpha=0.3)
            
        else:
            # Single-label case
            print("     Detected single-label dataset")
            
            # Collect all labels
            for idx in range(len(self.base_dataset)):
                _, label = self.base_dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item() if label.numel() == 1 else label.numpy()[0]
                elif isinstance(label, np.ndarray):
                    label = int(label.flatten()[0])
                else:
                    label = int(label)
                original_labels.append(label)
            
            for idx in self.sampled_indices:
                _, label = self.base_dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item() if label.numel() == 1 else label.numpy()[0]
                elif isinstance(label, np.ndarray):
                    label = int(label.flatten()[0])
                else:
                    label = int(label)
                sampled_labels.append(label)
            
            # Get unique classes
            unique_classes = sorted(set(original_labels))
            num_classes = len(unique_classes)
            
            # Count occurrences
            original_counts = [original_labels.count(c) for c in unique_classes]
            sampled_counts = [sampled_labels.count(c) for c in unique_classes]
            
            # Create bar plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            x = np.arange(num_classes)
            width = 0.35
            
            # Absolute counts
            ax1.bar(x - width/2, original_counts, width, label='Original', alpha=0.8, color='steelblue')
            ax1.bar(x + width/2, sampled_counts, width, label=f'Sampled ({self.sample_ratio*100:.1f}%)', alpha=0.8, color='coral')
            ax1.set_xlabel('Class Label', fontsize=12)
            ax1.set_ylabel('Number of Samples', fontsize=12)
            ax1.set_title('Absolute Distribution Comparison (Single-Label)', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(unique_classes)
            ax1.legend(fontsize=11)
            ax1.grid(axis='y', alpha=0.3)
            
            # Percentage comparison
            original_pct = [(c / sum(original_counts)) * 100 for c in original_counts]
            sampled_pct = [(c / sum(sampled_counts)) * 100 for c in sampled_counts]
            
            ax2.bar(x - width/2, original_pct, width, label='Original', alpha=0.8, color='steelblue')
            ax2.bar(x + width/2, sampled_pct, width, label=f'Sampled ({self.sample_ratio*100:.1f}%)', alpha=0.8, color='coral')
            ax2.set_xlabel('Class Label', fontsize=12)
            ax2.set_ylabel('Percentage (%)', fontsize=12)
            ax2.set_title('Relative Distribution Comparison (Single-Label)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(unique_classes)
            ax2.legend(fontsize=11)
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('./results/sampling_plots', exist_ok=True)
        save_path = f'./results/sampling_plots/{self.dataset_name}_sample{int(self.sample_ratio*100)}_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Distribution plot saved to: {save_path}")
        plt.close()
    
    def __len__(self):
        return len(self.sampled_indices)
    
    def __getitem__(self, idx):
        real_idx = self.sampled_indices[idx]
        return self.base_dataset[real_idx]
    
    def get_index_dic(self, list=True, get_labels=True):
        """Support for ClassAwareSampler - delegate to base dataset then filter indices"""
        return get_class_index_dict(self)


class ClassBalancedSampledDataset(Dataset):
    """
    Sample a subset with probability biased by inverse class frequency to reduce imbalance.

    For each sample i with active label set A_i, the per-sample inverse-frequency weight is
        inv_i = mean_{l in A_i}(1 / n_l)        (n_l = positive count of label l)
    where labels are counted within the (possibly grouped) base dataset's local label space.

    The final sampling weight is
        w_i = (inv_i / inv_max) ** alpha_balance
    so that:
        alpha_balance = 0  -> uniform random sampling (no rebalancing)
        alpha_balance = 1  -> classic inverse-frequency rebalancing
        alpha_balance > 1  -> stronger rebalancing toward rare-label samples

    Samples with no active labels receive the minimum non-zero weight.

    This isolates the IMBALANCE axis: label set is unchanged, only sampling distribution changes.
    """

    def __init__(self, base_dataset, sample_ratio, alpha_balance=1.0, seed=42,
                 dataset_name='dataset', save_dir=None):
        self.base_dataset = base_dataset
        self.sample_ratio = sample_ratio
        self.alpha_balance = float(alpha_balance)
        self.dataset_name = dataset_name
        self.save_dir = save_dir

        np.random.seed(seed)
        torch.manual_seed(seed)

        total_samples = len(base_dataset)
        num_samples = int(total_samples * sample_ratio)

        # Probe label dimension.
        _, first_label = self.base_dataset[0]
        if isinstance(first_label, torch.Tensor):
            label_dim = int(first_label.numel())
        else:
            label_dim = int(np.array(first_label).size)

        # Collect all labels once.
        all_labels = np.zeros((total_samples, label_dim), dtype=np.int8)
        for idx in range(total_samples):
            _, label = self.base_dataset[idx]
            if isinstance(label, torch.Tensor):
                arr = label.cpu().numpy().flatten()
            else:
                arr = np.array(label).flatten()
            all_labels[idx] = arr.astype(np.int8)

        class_counts = all_labels.sum(axis=0).astype(np.float64)
        # Avoid div-by-zero: labels never present get a large count placeholder so they don't dominate.
        safe_counts = np.where(class_counts > 0, class_counts, total_samples)
        inv_freq = 1.0 / safe_counts

        inv_per_sample = np.zeros(total_samples, dtype=np.float64)
        for i in range(total_samples):
            active = np.where(all_labels[i] > 0)[0]
            if active.size > 0:
                inv_per_sample[i] = inv_freq[active].mean()

        # Samples with no active labels: give them the minimum positive weight.
        positive_mask = inv_per_sample > 0
        if positive_mask.any():
            inv_per_sample[~positive_mask] = inv_per_sample[positive_mask].min()
        else:
            inv_per_sample[:] = 1.0

        # Normalize then power.
        inv_max = inv_per_sample.max()
        normalized = inv_per_sample / inv_max if inv_max > 0 else inv_per_sample

        if self.alpha_balance == 0.0:
            weights = np.ones(total_samples, dtype=np.float64)
        else:
            weights = np.power(normalized, self.alpha_balance)

        probs = weights / weights.sum()
        sampled_idx = np.random.choice(total_samples, size=num_samples, replace=False, p=probs)
        self.sampled_indices = sampled_idx.tolist()

        sampled_label_counts = all_labels[self.sampled_indices].sum(axis=0)

        print(f"  📊 Class-balanced sampling enabled:")
        print(f"     Total samples: {total_samples}")
        print(f"     Sampled: {num_samples} ({sample_ratio*100:.1f}%)")
        print(f"     Alpha_balance: {self.alpha_balance}")
        print(f"     Original per-class counts: {class_counts.astype(int).tolist()}")
        print(f"     Sampled  per-class counts: {sampled_label_counts.astype(int).tolist()}")

        self._save_sampled_cooccurrence()

    def _save_sampled_cooccurrence(self):
        if not self.save_dir:
            return

        # Resolve original label space (if base dataset is GroupedDataset, expand to original IDs).
        if hasattr(self.base_dataset, 'group_labels'):
            group_labels = list(self.base_dataset.group_labels)
            label_space = max(group_labels) + 1
        else:
            _, first_label = self.base_dataset[0]
            if isinstance(first_label, torch.Tensor):
                label_space = int(first_label.numel())
            else:
                label_space = int(np.array(first_label).size)
            group_labels = list(range(label_space))

        co = np.zeros((label_space, label_space), dtype=np.int64)
        for idx in self.sampled_indices:
            _, label = self.base_dataset[idx]
            if isinstance(label, torch.Tensor):
                arr = label.cpu().numpy().flatten()
            else:
                arr = np.array(label).flatten()
            active_local = [i for i, v in enumerate(arr) if v > 0 and i < len(group_labels)]
            active_orig = [group_labels[i] for i in active_local]
            for i in active_orig:
                for j in active_orig:
                    co[i, j] += 1

        os.makedirs(self.save_dir, exist_ok=True)
        cols = [f'Label_{i}' for i in range(label_space)]
        df = pd.DataFrame(co, index=cols, columns=cols)
        save_path = os.path.join(self.save_dir, 'sampled_cooccurrence.csv')
        df.to_csv(save_path)
        print(f"  ✅ Saved per-run sampled co-occurrence: {save_path}")

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        real_idx = self.sampled_indices[idx]
        return self.base_dataset[real_idx]

    def get_index_dic(self, list=True, get_labels=True):
        return get_class_index_dict(self)


class GroupedDataset(Dataset):
    """
    Wraps the original dataset to keep only samples from a specific group and remap labels
    
    Functionality:
    1. Filter samples belonging to the specified group
    2. For multi-label datasets: Keep multi-label format with remapped indices
    3. For single-label datasets: Remap original labels to [0, num_classes_in_group-1] range
    
    Example:
        Original dataset has 14 classes (0-13)
        Group 1 contains classes [3, 4, 5, 6, 7, 8]
        
        For multi-label:
        - Original label: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        - Filtered label: [1, 0, 1, 0, 0, 0] (only indices 3,4,5,6,7,8)
        
        For single-label:
        - Original label: 5
        - Remapped label: 2 (5 is the 3rd element in [3,4,5,6,7,8])
    
    Args:
        base_dataset: Original dataset
        group_labels: List of original labels in this group, e.g., [3, 4, 5, 6, 7, 8]
    """
    
    def __init__(self, base_dataset, group_labels):
        self.base_dataset = base_dataset
        self.group_labels = group_labels
        self.num_classes_in_group = len(group_labels)
        
        # Build label mapping dictionary for single-label case
        # Example: {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}
        self.label_mapping = {old_label: new_label 
                             for new_label, old_label in enumerate(group_labels)}
        
        print(f"  Building GroupedDataset for classes {group_labels}...")
        
        # Detect if this is a multi-label dataset by checking first sample
        _, first_label = base_dataset[0]
        self.is_multilabel = False
        
        if isinstance(first_label, (torch.Tensor, np.ndarray)):
            if hasattr(first_label, 'shape') and len(first_label.shape) > 0:
                if first_label.shape[0] > 1 or (hasattr(first_label, 'size') and first_label.size > 1):
                    self.is_multilabel = True
                    print(f"  ✓ Detected multi-label dataset")
        
        # For multi-label datasets, keep ALL samples (including [0,0,0,...])
        # For single-label datasets, filter by label
        if self.is_multilabel:
            # Keep all samples - just extract the relevant labels from each sample
            self.filtered_indices = list(range(len(base_dataset)))
            print(f"  ✓ Keeping all {len(self.filtered_indices)} samples (including samples with all zeros)")
        else:
            # For single-label: filter samples belonging to this group
            self.filtered_indices = []
            for idx in range(len(base_dataset)):
                _, label = base_dataset[idx]
                
                # Handle different label formats
                if isinstance(label, torch.Tensor):
                    if label.numel() == 1:
                        label = label.item()
                    else:
                        label = torch.argmax(label).item() if label.dim() > 0 else label.item()
                elif hasattr(label, 'item'):
                    if label.size == 1:
                        label = label.item()
                    else:
                        label = int(label.flatten()[0])
                elif hasattr(label, '__len__') and not isinstance(label, str):
                    label = int(label[0]) if len(label) > 0 else 0
                else:
                    label = int(label)
                
                if label in group_labels:
                    self.filtered_indices.append(idx)
            
            print(f"  ✓ Group contains {len(self.filtered_indices)} samples "
                  f"from original {len(base_dataset)} samples")
            print(f"  ✓ Label mapping: {self.label_mapping}")
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # Get original sample
        real_idx = self.filtered_indices[idx]
        image, old_label = self.base_dataset[real_idx]
        
        if self.is_multilabel:
            # For multi-label: extract only the labels for this group
            if isinstance(old_label, torch.Tensor):
                old_label_array = old_label.numpy()
            else:
                old_label_array = np.array(old_label)
            
            old_label_flat = old_label_array.flatten()
            
            # Create new label vector with only group classes
            new_label = np.zeros(self.num_classes_in_group, dtype=np.float32)
            for new_idx, old_idx in enumerate(self.group_labels):
                if old_idx < len(old_label_flat):
                    new_label[new_idx] = old_label_flat[old_idx]
            
            return image, torch.from_numpy(new_label)
        else:
            # For single-label: remap the label
            if isinstance(old_label, torch.Tensor):
                if old_label.numel() == 1:
                    old_label = old_label.item()
                else:
                    old_label = torch.argmax(old_label).item() if old_label.dim() > 0 else old_label.item()
            elif hasattr(old_label, 'item'):
                if old_label.size == 1:
                    old_label = old_label.item()
                else:
                    old_label = int(old_label.flatten()[0])
            elif hasattr(old_label, '__len__') and not isinstance(old_label, str):
                old_label = int(old_label[0]) if len(old_label) > 0 else 0
            else:
                old_label = int(old_label)
            
            new_label = self.label_mapping[old_label]
            
            return image, new_label
    
    def get_index_dic(self, list=True, get_labels=True):
        """Support for ClassAwareSampler"""
        return get_class_index_dict(self)


class PADatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'zr7vgbcyr2-1.zip')
        self.dataset_extracted_dir = self.root_dir
        self.source_images_dirs = [os.path.join(self.root_dir, 'images', f'imgs_part_{i}') for i in range(1, 4)]
        self.organized_images_dir = os.path.join(self.root_dir, 'PAD-Dataset')
        self.metadata_file_path = os.path.join(self.root_dir, 'metadata.csv')
        
    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
    def extract_dataset(self):
        if not os.path.exists(os.path.join(self.root_dir, 'images')):
            print("Extracting main dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Main extraction complete.")
        
    def extract_inner_datasets(self):
        for i, source_images_dir in enumerate(self.source_images_dirs, start=1):
            inner_zip_path = os.path.join(self.root_dir, f'images/imgs_part_{i}.zip')
            if not os.path.exists(source_images_dir):
                print(f"Extracting {inner_zip_path}...")
                with ZipFile(inner_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(source_images_dir))
                print(f"Extraction of {inner_zip_path} complete.")
        
    def organize_images(self):
        if os.path.exists(self.organized_images_dir):
            print("Images are already organized.")
            return
        
        if not os.path.exists(self.metadata_file_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_file_path}")
        
        metadata = pd.read_csv(self.metadata_file_path)
        
        os.makedirs(self.organized_images_dir, exist_ok=True)
        
        diagnostic_labels = metadata['diagnostic'].unique()
        
        for label in diagnostic_labels:
            os.makedirs(os.path.join(self.organized_images_dir, label), exist_ok=True)
        
        for _, row in metadata.iterrows():
            img_id = row['img_id']
            diagnostic = row['diagnostic']
            
            for source_dir in self.source_images_dirs:
                source_path = os.path.join(source_dir, img_id)
                if os.path.exists(source_path):
                    destination_path = os.path.join(self.organized_images_dir, diagnostic, img_id)
                    shutil.move(source_path, destination_path)
                    break
        
        print("Images moved successfully.")
        
    def get_dataset(self):
        if os.path.exists(self.organized_images_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.organized_images_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            self.extract_inner_datasets()
            self.organize_images()
            return self.organized_images_dir



class FetalDatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'FETAL_PLANES_ZENODO.zip')
        self.dataset_extracted_dir = self.root_dir
        self.organized_images_dir = os.path.join(self.root_dir, 'Fetal-Dataset')
        self.excel_file_path = os.path.join(self.root_dir, 'FETAL_PLANES_DB_data.xlsx')
        self.source_images_dir = os.path.join(self.root_dir, 'Images')
        
    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
    def extract_dataset(self):
        if not os.path.exists(self.excel_file_path) or not os.path.exists(self.source_images_dir):
            print("Extracting dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Extraction complete.")
        
    def organize_images(self):
        if os.path.exists(self.organized_images_dir):
            print("Images are already organized.")
            return
        
        if not os.path.exists(self.excel_file_path):
            raise FileNotFoundError(f"Excel file not found at {self.excel_file_path}")
        
        df = pd.read_excel(self.excel_file_path)
        
        os.makedirs(self.organized_images_dir, exist_ok=True)
        
        plane_labels = df['Plane'].unique()
        
        for label in plane_labels:
            os.makedirs(os.path.join(self.organized_images_dir, str(label)), exist_ok=True)
        
        for _, row in df.iterrows():
            img_id = row['Image_name']
            plane = row['Plane']
            source_path = os.path.join(self.source_images_dir, f'{img_id}.png')
            destination_path = os.path.join(self.organized_images_dir, str(plane), f'{img_id}.png')
            
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
        
        print("Images moved successfully.")
        
    def get_dataset(self):
        if os.path.exists(self.organized_images_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.organized_images_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            self.organize_images()
            return self.organized_images_dir


class ISICDatasetManager:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.train_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip'
        self.test_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip'
        self.train_gt_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip'
        self.test_gt_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_GroundTruth.zip'
        self.train_path = os.path.join(self.base_dir, 'ISIC2018_Train')
        self.test_path = os.path.join(self.base_dir, 'ISIC2018_Test')

        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)

    def download_and_extract(self, url, extract_to):
        local_filename = os.path.join(self.base_dir, url.split('/')[-1])
        if not os.path.exists(local_filename):
            print(f"Downloading {url}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
        print(f"Extracting {local_filename}...")
        with ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")

    def organize_by_labels(self, metadata_path, image_dir, output_base_dir):
        metadata = pd.read_csv(metadata_path)
        labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        for label in labels:
            os.makedirs(os.path.join(output_base_dir, label), exist_ok=True)

        def move_image(row):
            image_name = f"{row['image']}.jpg"
            source_path = os.path.join(image_dir, image_name)
            for label in labels:
                if row[label] == 1.0:
                    target_path = os.path.join(output_base_dir, label, image_name)
                    shutil.move(source_path, target_path)
                    break
        metadata.apply(move_image, axis=1)

    def setup_dataset(self):
        
        # Organize training and test images
        train_categorized = os.path.join(self.train_path, 'Categorized')
        test_categorized = os.path.join(self.test_path, 'Categorized')
        
        if os.path.exists(train_categorized):
            print("Dataset already exists. Returning the root directory.")
            return train_categorized, test_categorized
        else:
            # Download and extract training and test datasets
            self.download_and_extract(self.train_url, self.train_path)
            self.download_and_extract(self.test_url, self.test_path)
            self.download_and_extract(self.train_gt_url, self.train_path)
            self.download_and_extract(self.test_gt_url, self.test_path)

            
            self.organize_by_labels(
                os.path.join(self.train_path, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'),
                os.path.join(self.train_path, 'ISIC2018_Task3_Training_Input'),
                train_categorized
            )
            self.organize_by_labels(
                os.path.join(self.test_path, 'ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'),
                os.path.join(self.test_path, 'ISIC2018_Task3_Test_Input'),
                test_categorized
            )

            return train_categorized, test_categorized


class CPNDatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/dvntn9yhd2-1.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'dvntn9yhd2-1.zip')
        self.dataset_extracted_dir = os.path.join(self.root_dir, 'dvntn9yhd2-1')
        self.organized_images_dir = os.path.join(self.root_dir, 'CPN-Dataset')
        
    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
    def extract_dataset(self):
        if not os.path.exists(self.dataset_extracted_dir):
            print("Extracting main dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Main extraction complete.")
        
    def extract_inner_dataset(self):
        inner_zip_path = os.path.join(self.dataset_extracted_dir, 'Covid19-Pneumonia-Normal Chest X-Ray Images Dataset.zip')
        if not os.path.exists(self.organized_images_dir):
            os.makedirs(self.organized_images_dir)
            print("Extracting inner dataset...")
            with ZipFile(inner_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.organized_images_dir)
            print("Inner extraction complete.")
        
    def get_dataset(self):
        if os.path.exists(self.organized_images_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.organized_images_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            self.extract_inner_dataset()
            return self.organized_images_dir


class KvasirDatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://datasets.simula.no/downloads/kvasir/kvasir-dataset.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'kvasir-dataset.zip')
        self.dataset_dir = os.path.join(self.root_dir, 'kvasir-dataset')
        
    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
    def extract_dataset(self):
        if not os.path.exists(self.dataset_dir):
            print("Extracting dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Extraction complete.")
        
    def get_dataset(self):
        if os.path.exists(self.dataset_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.dataset_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            return self.dataset_dir



        
        
def build_dataset(args):
    train_transform, test_transform = build_transform(args)
    
    if args.dataset == 'Kvasir':
        # Define the sizes for the splits
        train_size = 2408
        val_size = 392
        test_size = 1200
        nb_classes = 8
        downloader = KvasirDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")  
    elif args.dataset == 'CPN':
        # Define the sizes for the splits
        train_size = 3140
        val_size = 521
        test_size = 1567
        nb_classes = 3
        downloader = CPNDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'Fetal':
        # Define the sizes for the splits
        train_size = 7446
        val_size = 1237
        test_size = 3717
        nb_classes = 6
        downloader = FetalDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'PAD':
        # Define the sizes of each split
        train_size = 1384
        val_size = 227
        test_size = 687
        nb_classes = 6
        downloader = PADatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'ISIC2018':
        nb_classes = 7
        manager = ISICDatasetManager()
        train_path, test_path = manager.setup_dataset()
        print(f"Dataset is available at: {train_path}")
        train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform) 
        test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform) 
        return train_dataset, test_dataset, nb_classes
    elif args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        n_channels = info['n_channels']
        nb_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        print("Number of channels: ", n_channels)
        print("Number of classes: ", nb_classes)
        train_dataset = DataClass(split='train', transform=train_transform, download=True, as_rgb=True, root='./data', size=224, mmap_mode='r')
        test_dataset = DataClass(split='test', transform=test_transform, download=True, as_rgb=True, root='./data', size=224, mmap_mode='r')
        
        # ✨ Handle label range training
        if hasattr(args, 'label_head') and args.label_head is not None and hasattr(args, 'label_tail') and args.label_tail is not None:
            # Get label range based on sorted order
            from datasets import get_label_range
            group_labels = get_label_range(args.dataset, args.label_head, args.label_tail)
            
            print(f"\n{'='*60}")
            print(f"🎯 Label Range Training Mode Enabled")
            print(f"{'='*60}")
            print(f"  Sorted order: {get_sorted_label_order(args.dataset)}")
            print(f"  Range: {args.label_head} to {args.label_tail}")
            print(f"  Selected classes: {group_labels}")
            
            # Wrap dataset with GroupedDataset
            train_dataset = GroupedDataset(train_dataset, group_labels)
            test_dataset = GroupedDataset(test_dataset, group_labels)
            
            # Update number of classes
            nb_classes = len(group_labels)
            print(f"  Final number of classes: {nb_classes}")
            print(f"{'='*60}\n")
        # ✨ Handle group training
        elif hasattr(args, 'group') and args.group is not None:
            label_groups = get_label_groups(args.dataset)
            
            if label_groups is None:
                raise ValueError(f"Dataset {args.dataset} does not support group training. "
                               f"Please add group definition in get_label_groups() function.")
            
            if args.group < 0 or args.group >= len(label_groups):
                raise ValueError(f"Group {args.group} is out of range. "
                               f"Valid groups: 0-{len(label_groups)-1}")
            
            print(f"\n{'='*60}")
            print(f"🎯 Group Training Mode Enabled")
            print(f"{'='*60}")
            print(f"  Training Group {args.group} of {len(label_groups)}")
            
            group_labels = label_groups[args.group]
            print(f"  Original classes in this group: {group_labels}")
            
            # Wrap dataset with GroupedDataset
            train_dataset = GroupedDataset(train_dataset, group_labels)
            test_dataset = GroupedDataset(test_dataset, group_labels)
            
            # Update number of classes
            nb_classes = len(group_labels)
            print(f"  Final number of classes: {nb_classes}")
            print(f"{'='*60}\n")
        
        # ✨ Handle sampling (apply after grouping if both are specified)
        if hasattr(args, 'sample') and args.sample is not None:
            print(f"\n{'='*60}")
            print(f"📊 Dataset Sampling Enabled")
            print(f"{'='*60}")

            if hasattr(args, 'class_balanced_sampling') and args.class_balanced_sampling:
                train_dataset = ClassBalancedSampledDataset(
                    train_dataset,
                    args.sample,
                    alpha_balance=args.class_balanced_alpha if hasattr(args, 'class_balanced_alpha') else 1.0,
                    seed=42,
                    dataset_name=args.dataset,
                    save_dir=args.result_save_dir if hasattr(args, 'result_save_dir') else None
                )
            else:
                # Get plot_distribution parameter from args
                plot_dist = args.plot_distribution if hasattr(args, 'plot_distribution') else True

                train_dataset = SampledDataset(
                    train_dataset,
                    args.sample,
                    seed=42,
                    dataset_name=args.dataset,
                    plot_distribution=plot_dist
                )
            # Note: We don't sample test dataset to keep full evaluation
            print(f"  Note: Test dataset is NOT sampled (full evaluation)")
            print(f"{'='*60}\n")
        
        return train_dataset, test_dataset, nb_classes
    else:
        raise NotImplementedError()
    
    full_dataset = datasets.ImageFolder(root=data_dir)  # Load without transform
    # Verify the total number of images matches the sum of the splits
    assert train_size + val_size + test_size == len(full_dataset), "The sum of the splits must equal the total number of images"

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Apply the transformations
    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_dataset.indices)
    val_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=test_transform), val_dataset.indices)
    test_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=test_transform), test_dataset.indices)

    print("Number of the class = %d" % nb_classes)

    return train_dataset, test_dataset, nb_classes


def build_transform(args):
    t_train = []
    # this should always dispatch to transforms_imagenet_train
    t_train.append(transforms.RandomResizedCrop(224))
    t_train.append(transforms.AugMix(alpha= 0.4))
    #t_train.append(transforms.Lambda(lambda image: image.convert('RGB')))
    t_train.append(transforms.RandomHorizontalFlip(p=0.4))
    t_train.append(transforms.ToTensor())
    t_train.append(transforms.Normalize(mean=[.5], std=[.5]))
        

    t_test = []
    t_test.append(transforms.Resize((224, 224)))
    #t_test.append(transforms.Lambda(lambda image: image.convert('RGB')))
    t_test.append(transforms.ToTensor())
    t_test.append(transforms.Normalize(mean=[.5], std=[.5]))
    return transforms.Compose(t_train), transforms.Compose(t_test)