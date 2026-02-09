"""
Label Grouper Module
Implements Dynamic Programming based label grouping algorithm for imbalanced datasets
"""

import statistics
from typing import Dict, List
from abc import ABC, abstractmethod


class LabelGrouper(ABC):
    """Abstract base class for label grouping algorithms"""
    
    @abstractmethod
    def group_label(self, label_count: Dict[str, int]) -> List[List[str]]:
        """
        Group labels based on their counts
        
        Args:
            label_count: Dictionary mapping label names to their counts
            
        Returns:
            List of label groups, where each group is a list of label names
        """
        pass
    
    def calculate_CVIR(self, irlbl_list: List[float]) -> float:
        """
        Calculate Coefficient of Variation for Imbalance Ratio
        
        Args:
            irlbl_list: List of imbalance ratios
            
        Returns:
            Coefficient of Variation (std / mean)
        """
        if len(irlbl_list) <= 1:
            return 0.0
        
        mean_val = statistics.mean(irlbl_list)
        if mean_val == 0:
            return 0.0
        
        std_val = statistics.stdev(irlbl_list)
        return std_val / mean_val


class DPLabelGrouper(LabelGrouper):
    """
    Dynamic Programming based Label Grouper
    
    Groups labels by minimizing the number of groups while maintaining:
    - Mean Imbalance Ratio (MeanIRLbl) <= threshold
    - Coefficient of Variation (CVIR) <= threshold
    """
    
    def __init__(self, mean_ir_threshold: float = 2.5, cvir_threshold: float = 0.4):
        """
        Initialize the grouper with thresholds
        
        Args:
            mean_ir_threshold: Maximum allowed mean imbalance ratio (default: 2.5)
            cvir_threshold: Maximum allowed coefficient of variation (default: 0.4)
        """
        self.mean_ir_threshold = mean_ir_threshold
        self.cvir_threshold = cvir_threshold
        self.cache = {}
    
    def group_label(self, label_count: Dict[str, int]) -> List[List[str]]:
        """
        Group labels using Dynamic Programming algorithm
        
        Args:
            label_count: Dictionary mapping label names to their counts
            
        Returns:
            List of label groups (optimal grouping)
        """
        # Sort labels by count in descending order
        sorted_dict = dict(sorted(label_count.items(), key=lambda item: item[1], reverse=True))
        
        # Generate valid combinations
        valid_start_index_dict = self.generate_valid_combination(sorted_dict)
        
        # Apply DP to find optimal grouping
        least_num_group_start_index = self.dp(valid_start_index_dict)
        print("============== DP Complete ================")
        
        # Convert indices to label names
        label_name_list = list(sorted_dict.keys())
        best_group_list = []
        
        least_num_group_start_index.append(len(label_name_list))
        for i in range(len(least_num_group_start_index) - 1):
            group = label_name_list[least_num_group_start_index[i]:least_num_group_start_index[i + 1]]
            best_group_list.append(group)
        
        print(f"Optimal grouping: {best_group_list}")
        return best_group_list
    
    def generate_valid_combination(self, sorted_dict: Dict[str, int]) -> Dict[int, List[int]]:
        """
        Generate valid group combinations based on MeanIRLbl and CVIR constraints
        
        Args:
            sorted_dict: Sorted dictionary of label counts (descending order)
            
        Returns:
            Dictionary mapping end index to list of valid start indices
        """
        label_count_tuple_list = [(label, count) for label, count in sorted_dict.items()]
        label_count_num = len(label_count_tuple_list)
        
        # valid_start_index_list[i] contains all valid start indices for groups ending at i
        valid_start_index_list = [[] for _ in range(label_count_num)]
        
        for i in range(label_count_num):
            _, max_count = label_count_tuple_list[i]
            irlbl_list = []
            
            for j in range(i, label_count_num):
                _, count = label_count_tuple_list[j]
                # Calculate imbalance ratio
                irlbl_list.append(max_count / count)
                mean_IRLbl = statistics.mean(irlbl_list)
                
                if len(irlbl_list) > 1:
                    cvir = self.calculate_CVIR(irlbl_list)
                else:
                    cvir = 0
                
                # Check if this group satisfies constraints
                if mean_IRLbl <= self.mean_ir_threshold and cvir <= self.cvir_threshold:
                    valid_start_index_list[j].append(i)
        
        return valid_start_index_list
    
    def dp(self, valid_start_index_list: List[List[int]]) -> List[int]:
        """
        Dynamic Programming to find minimum number of groups
        
        Each cache[i] stores the best list of start indices for labels[:i+1]
        
        Args:
            valid_start_index_list: List of valid start indices for each position
            
        Returns:
            List of start indices representing optimal grouping
        """
        self.cache = {(-1): []}
        
        for i in range(len(valid_start_index_list)):
            min_len = float('inf')
            best_solution = []
            
            for valid_start_index in valid_start_index_list[i]:
                temp_len = len(self.cache[valid_start_index - 1]) + 1
                if temp_len <= min_len:
                    min_len = temp_len
                    best_solution = self.cache[valid_start_index - 1] + [valid_start_index]
            
            self.cache[i] = best_solution
        
        return self.cache[len(valid_start_index_list) - 1]


def compute_chestmnist_groups(data_root: str = './data') -> List[List[int]]:
    """
    Compute label groups for ChestMNIST dataset using DP algorithm
    
    Args:
        data_root: Root directory containing the dataset
        
    Returns:
        List of label groups (as indices)
    """
    import medmnist
    from medmnist import INFO
    
    # Load ChestMNIST dataset
    info = INFO['chestmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load training data to get label distribution
    train_dataset = DataClass(split='train', download=True, as_rgb=True, 
                             root=data_root, size=224, mmap_mode='r')
    
    # Count labels
    label_count = {}
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        # Handle multi-label case
        if hasattr(label, 'flatten'):
            label_array = label.flatten()
            for idx, val in enumerate(label_array):
                if val > 0:
                    if idx not in label_count:
                        label_count[idx] = 0
                    label_count[idx] += 1
        else:
            label_idx = int(label)
            if label_idx not in label_count:
                label_count[label_idx] = 0
            label_count[label_idx] += 1
    
    print("\n" + "="*60)
    print("ChestMNIST Label Distribution:")
    print("="*60)
    # Sort by sample count (descending order) instead of by class index
    sorted_labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
    for label_idx, count in sorted_labels:
        print(f"  Class {label_idx}: {count} samples")
    print("="*60 + "\n")
    
    # Convert label indices to strings for the grouper
    label_count_str = {str(k): v for k, v in label_count.items()}
    
    # Apply DP grouping algorithm
    grouper = DPLabelGrouper(mean_ir_threshold=2.5, cvir_threshold=0.4)
    groups_str = grouper.group_label(label_count_str)
    
    # Convert back to integers
    groups_int = [[int(label) for label in group] for group in groups_str]
    
    return groups_int


if __name__ == '__main__':
    """
    Test the label grouper on ChestMNIST dataset
    """
    print("\n" + "="*60)
    print("Testing DPLabelGrouper on ChestMNIST Dataset")
    print("="*60 + "\n")
    
    groups = compute_chestmnist_groups()
    
    print("\n" + "="*60)
    print("Final Grouping Result:")
    print("="*60)
    for i, group in enumerate(groups):
        print(f"  Group {i}: Classes {group} ({len(group)} classes)")
    print("="*60 + "\n")
