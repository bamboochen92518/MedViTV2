"""
Class-Aware Sampler for Imbalanced Multi-Label Datasets

This module provides sampling strategies for handling class imbalance in 
multi-label classification tasks, particularly useful for medical imaging datasets.

Classes:
    RandomCycleIter: Infinite iterator with random shuffling
    ClassAwareSampler: Balanced sampler for imbalanced datasets

Functions:
    class_aware_sample_generator: Generator for class-aware sampling
    get_class_index_dict: Build class-to-samples mapping

Author: MedViT Team
Date: 2026
"""

import random
import torch
import numpy as np
from torch.utils.data.sampler import Sampler


class RandomCycleIter:
    """
    Randomly iterate through a list infinitely with shuffling when exhausted.
    
    This iterator cycles through data indefinitely, reshuffling when it reaches the end.
    Used for minority class oversampling in ClassAwareSampler.
    
    Args:
        data (list): List of items to iterate over
        seed (int): Random seed for reproducibility
    
    Example:
        >>> iter = RandomCycleIter([1, 2, 3])
        >>> [next(iter) for _ in range(10)]
        [2, 3, 1, 1, 3, 2, 3, 1, 2, ...]  # Shuffles after every 3 items
    """
    
    def __init__(self, data, seed=42):
        self.data = list(data)
        self.length = len(self.data)
        self.index = 0
        self.seed = seed
        random.seed(seed)
        random.shuffle(self.data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= self.length:
            self.index = 0
            random.shuffle(self.data)
        
        item = self.data[self.index]
        self.index += 1
        return item


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    """
    Generate samples with class-aware strategy for balanced sampling.
    
    This generator ensures balanced class representation by:
    1. Randomly selecting a class
    2. Drawing multiple samples from that class
    3. Moving to next class after exhausting samples
    
    Args:
        cls_iter: Iterator over class indices
        data_iter_list: List of iterators, one per class
        n: Total number of samples to generate
        num_samples_cls: Number of samples to draw from each selected class
    
    Yields:
        int: Sample index from the dataset
    
    Example:
        If num_samples_cls=3, the generator will:
        - Select class 5 → yield samples [100, 101, 102]
        - Select class 2 → yield samples [50, 51, 52]
        - Select class 7 → yield samples [200, 201, 202]
        - ...
    """
    i = 0
    j = 0
    while i < n:
        if j >= num_samples_cls:
            j = 0

        if j == 0:
            # Select a new class and draw num_samples_cls samples from it
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            # Continue yielding from the previously selected class
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):
    """
    Class-Aware Sampler for handling imbalanced multi-label datasets.
    
    This sampler ensures balanced class representation during training by:
    1. Treating each class equally regardless of sample count
    2. Oversampling minority classes through cycling
    3. Grouping samples from the same class together (controlled by num_samples_cls)
    
    Particularly useful for:
    - Multi-label classification (e.g., ChestMNIST where one image can have multiple diseases)
    - Highly imbalanced datasets (some classes have 10x more samples than others)
    
    Args:
        data_source: Dataset object (must implement get_index_dic method)
        num_samples_cls (int): Number of consecutive samples from each class before switching.
                               Higher values = more class cohesion per batch
                               Recommended: 2-4 for batch_size 16-32
        reduce (int): Factor to reduce total samples per epoch.
                      Higher values = faster epoch, fewer repeated samples
                      Default: 4 means ~25% of full balanced sampling
    
    Example:
        >>> sampler = ClassAwareSampler(dataset, num_samples_cls=3, reduce=4)
        >>> loader = DataLoader(dataset, batch_size=24, sampler=sampler)
        # Each batch will have ~8 different classes (24 / 3 = 8)
        # Each class contributes 3 consecutive samples
        
    Typical Usage Scenarios:
        - ChestMNIST: Class 3 (Effusion) has 8964 samples, Class 13 (Hernia) has 227 samples
          Without sampler: Model sees Class 3 ~40x more often than Class 13
          With sampler: Both classes are seen equally during training
    """
    
    def __init__(self, data_source, num_samples_cls=3, reduce=4):
        random.seed(0)
        torch.manual_seed(0)
        
        # Get class information from dataset
        self.cls_data_list, self.gt_labels = data_source.get_index_dic(list=True, get_labels=True)
        num_classes = len(self.cls_data_list)
        
        print(f'\n{"=" * 60}')
        print(f"🎯 Class-Aware Sampler Activated")
        print(f'{"=" * 60}')
        print(f"  Number of classes: {num_classes}")
        print(f"  Samples per class before switch: {num_samples_cls}")
        print(f"  Reduce factor: {reduce}")
        
        # Print class distribution
        class_counts = [len(x) for x in self.cls_data_list]
        print(f"  Class sample counts:")
        for i, count in enumerate(class_counts):
            print(f"    Class {i}: {count} samples")
        
        self.epoch = 0
        self.num_classes = num_classes
        self.num_samples_cls = num_samples_cls
        
        # Create iterators
        self.class_iter = RandomCycleIter(range(num_classes))
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list]
        
        # Calculate total samples per epoch
        # Formula: (max class size) * (number of classes) / reduce
        max_class_size = max(class_counts)
        self.num_samples = int(max_class_size * num_classes / reduce)
        
        print(f"  Max class size: {max_class_size}")
        print(f"  Samples per epoch: {self.num_samples}")
        print(f"  Original dataset size: {sum(class_counts)}")
        print(f"  Sampling ratio: {self.num_samples / sum(class_counts):.2f}x")
        print(f'{"=" * 60}\n')

    def __iter__(self):
        return class_aware_sample_generator(
            self.class_iter, 
            self.data_iter_list,
            self.num_samples, 
            self.num_samples_cls
        )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """Set the epoch counter (useful for distributed training)"""
        self.epoch = epoch


def get_class_index_dict(dataset):
    """
    Build class-to-samples mapping for multi-label datasets.
    
    This function analyzes a multi-label dataset and creates:
    1. A dictionary mapping each class to sample indices containing that class
    2. A list of all multi-label annotations
    
    Args:
        dataset: PyTorch dataset with multi-label annotations
                 Each label should be a binary vector (e.g., [0, 1, 0, 1, 0])
    
    Returns:
        cls_data_list (list of lists): cls_data_list[i] contains indices of samples with class i
        gt_labels (list of arrays): All ground truth labels
    
    Example:
        >>> cls_data_list, gt_labels = get_class_index_dict(dataset)
        >>> print(f"Class 0 appears in samples: {cls_data_list[0]}")
        [5, 12, 23, 45, ...]  # Indices of samples containing class 0
        
    Notes:
        - For multi-label datasets: A sample can belong to multiple classes
        - For single-label datasets: Converts to one-hot encoding internally
        - Handles both torch.Tensor and numpy.ndarray label formats
    """
    print("  🔍 Building class index dictionary...")
    
    # Get first sample to determine format
    _, first_label = dataset[0]
    
    # Determine number of classes
    if isinstance(first_label, torch.Tensor):
        num_classes = first_label.shape[0] if len(first_label.shape) > 0 else 1
    elif isinstance(first_label, np.ndarray):
        num_classes = first_label.shape[0] if len(first_label.shape) > 0 else 1
    else:
        # Single-label case: sample a subset to determine number of classes
        num_classes = len(set([int(dataset[i][1]) for i in range(min(1000, len(dataset)))]))
    
    # Initialize
    cls_data_list = [[] for _ in range(num_classes)]
    gt_labels = []
    
    # Iterate through dataset
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        
        # Convert to numpy array
        if isinstance(label, torch.Tensor):
            label_array = label.numpy()
        elif isinstance(label, np.ndarray):
            label_array = label
        else:
            # Single-label: convert to one-hot
            label_array = np.zeros(num_classes)
            label_array[int(label)] = 1
        
        label_flat = label_array.flatten()
        gt_labels.append(label_flat)
        
        # Add this sample to all classes it belongs to
        for class_idx in range(num_classes):
            if label_flat[class_idx] > 0:  # Class is present
                cls_data_list[class_idx].append(idx)
    
    print(f"  ✓ Index dictionary built successfully")
    print(f"  ✓ Total samples: {len(gt_labels)}")
    
    return cls_data_list, gt_labels
