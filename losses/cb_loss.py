"""
Class-Balanced Loss

A loss function that re-weights samples based on the effective number of samples per class.
Designed to handle class imbalance by giving more weight to rare classes.

Reference:
    Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    Paper: https://arxiv.org/abs/1901.05555
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CBLoss(nn.Module):
    """
    Class-Balanced Loss (Re-implementation Version).
    
    This version applies CB weights independently to each label before summing them up.
    This means that each label within a sample gets its own weight based on class frequency.
    
    This loss re-weights the contribution of each class based on the effective number of samples.
    The effective number is calculated as: (1 - β^n) / (1 - β), where n is the number of samples
    and β is a hyperparameter (typically 0.999 or 0.9999).
    
    Suitable for:
        - Highly imbalanced datasets (e.g., long-tail distribution)
        - Both multi-label and single-label classification
        - Medical imaging datasets with rare diseases
    
    Args:
        dataset: Training dataset (must support len() and indexing)
        num_classes (int): Number of classes
        beta (float): Hyperparameter for effective number calculation. Default: 0.9999
            - β → 1: More emphasis on balancing (suitable for extreme imbalance)
            - β → 0: Less emphasis on balancing (closer to uniform weighting)
        loss_type (str): Base loss function to use. Options: 'sigmoid' | 'softmax'. Default: 'sigmoid'
        reduction (str): Specifies reduction. Options: 'none' | 'mean' | 'sum'. Default: 'mean'
    
    Shape:
        - Input: (N, C) where N is batch size and C is number of classes
        - Target: 
            - Multi-label: (N, C) binary labels
            - Single-label: (N,) class indices
        - Output: scalar if reduction='mean' (default)
    
    Example:
        >>> # For multi-label (ChestMNIST)
        >>> loss_fn = CBLoss(dataset=train_dataset, num_classes=14, beta=0.9999)
        >>> outputs = torch.randn(32, 14)
        >>> targets = torch.randint(0, 2, (32, 14)).float()
        >>> loss = loss_fn(outputs, targets)
        
        >>> # For single-label (PathMNIST)
        >>> loss_fn = CBLoss(dataset=train_dataset, num_classes=9, beta=0.999, loss_type='softmax')
        >>> outputs = torch.randn(32, 9)
        >>> targets = torch.randint(0, 9, (32,))
        >>> loss = loss_fn(outputs, targets)
    """
    
    def __init__(self, dataset, num_classes, beta=0.9999, loss_type='sigmoid', reduction='mean'):
        super(CBLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.loss_type = loss_type
        self.reduction = reduction
        
        # Calculate class weights from dataset
        self.class_weights = self._calculate_class_weights(dataset, num_classes, beta)
        
        print(f"  📊 Class-Balanced Loss (Re-implementation) initialized:")
        print(f"     Beta: {beta}")
        print(f"     Loss type: {loss_type}")
        print(f"     Class weights (top 5): {self.class_weights[:5].cpu().numpy()}")
    
    def _calculate_class_weights(self, dataset, num_classes, beta):
        """
        Calculate class weights based on effective number of samples.
        
        Args:
            dataset: Training dataset
            num_classes (int): Number of classes
            beta (float): Hyperparameter for effective number
        
        Returns:
            Tensor: Class weights of shape (num_classes,)
        """
        print(f"  🔍 Calculating class weights from dataset...")
        
        # Count samples per class
        samples_per_class = np.zeros(num_classes)
        
        # Check if this is multi-label or single-label
        _, first_label = dataset[0]
        is_multilabel = False
        
        if isinstance(first_label, (torch.Tensor, np.ndarray)):
            if hasattr(first_label, 'shape') and len(first_label.shape) > 0:
                if first_label.shape[0] > 1 or (hasattr(first_label, 'size') and first_label.size > 1):
                    is_multilabel = True
        
        if is_multilabel:
            # Multi-label: count positive samples for each class
            print(f"     Detected multi-label dataset")
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.numpy()
                elif not isinstance(label, np.ndarray):
                    label = np.array(label)
                label = label.flatten()
                samples_per_class += (label > 0).astype(int)
        else:
            # Single-label: count samples for each class
            print(f"     Detected single-label dataset")
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item() if label.numel() == 1 else label.numpy()[0]
                elif isinstance(label, np.ndarray):
                    label = int(label.flatten()[0])
                else:
                    label = int(label)
                samples_per_class[label] += 1
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        
        # Avoid division by zero for classes with no samples
        effective_num = np.where(samples_per_class == 0, 1.0, effective_num)
        
        # Calculate weights
        weights = (1.0 - beta) / effective_num
        
        # Normalize weights (optional, makes interpretation easier)
        weights = weights / weights.sum() * num_classes
        
        # Print statistics
        print(f"     Samples per class (min/max/mean): {samples_per_class.min():.0f} / {samples_per_class.max():.0f} / {samples_per_class.mean():.0f}")
        print(f"     Weight ratio (max/min): {weights.max() / weights.min():.2f}x")
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Raw logits from model
                - For sigmoid: (N, C)
                - For softmax: (N, C)
            targets (Tensor): Ground truth labels
                - Multi-label: (N, C) binary labels
                - Single-label: (N,) class indices
        
        Returns:
            Tensor: Computed loss
        """
        device = inputs.device
        weights = self.class_weights.to(device)
        
        if self.loss_type == 'sigmoid':
            # Multi-label classification
            # BCE loss with class weights
            targets = targets.float()
            
            # Calculate BCE loss per sample per class
            # BCE = -[y*log(p) + (1-y)*log(1-p)]
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
            
            # Apply class weights independently to each label
            # Weight each class differently based on its frequency
            weighted_loss = bce_loss * weights.unsqueeze(0)
            
            if self.reduction == 'mean':
                return weighted_loss.mean()
            elif self.reduction == 'sum':
                return weighted_loss.sum()
            else:
                return weighted_loss
        
        elif self.loss_type == 'softmax':
            # Single-label classification
            # Cross entropy loss with class weights
            targets = targets.long()
            
            # Calculate CE loss
            ce_loss = F.cross_entropy(
                inputs, targets, weight=weights, reduction=self.reduction
            )
            
            return ce_loss
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Use 'sigmoid' or 'softmax'.")


class CBLossOriginal(nn.Module):
    """
    Class-Balanced Loss (Original Official Implementation).
    
    In this version, for each sample, the class weights corresponding to the positive labels
    are averaged to create a single scalar weight. This scalar weight is then used to rescale
    the entire loss of that sample. This means all labels within a single sample are weighted
    equally, but the weight varies across samples.
    
    This loss re-weights the contribution of each class based on the effective number of samples.
    The effective number is calculated as: (1 - β^n) / (1 - β), where n is the number of samples
    and β is a hyperparameter (typically 0.999 or 0.9999).
    
    Suitable for:
        - Highly imbalanced datasets (e.g., long-tail distribution)
        - Multi-label classification where sample-level weighting is preferred
        - Medical imaging datasets with rare diseases
    
    Args:
        dataset: Training dataset (must support len() and indexing)
        num_classes (int): Number of classes
        beta (float): Hyperparameter for effective number calculation. Default: 0.9999
            - β → 1: More emphasis on balancing (suitable for extreme imbalance)
            - β → 0: Less emphasis on balancing (closer to uniform weighting)
        loss_type (str): Base loss function to use. Options: 'sigmoid' | 'softmax'. Default: 'sigmoid'
        reduction (str): Specifies reduction. Options: 'none' | 'mean' | 'sum'. Default: 'mean'
    
    Shape:
        - Input: (N, C) where N is batch size and C is number of classes
        - Target: 
            - Multi-label: (N, C) binary labels
            - Single-label: (N,) class indices
        - Output: scalar if reduction='mean' (default)
    
    Example:
        >>> # For multi-label (ChestMNIST)
        >>> loss_fn = CBLossOriginal(dataset=train_dataset, num_classes=14, beta=0.9999)
        >>> outputs = torch.randn(32, 14)
        >>> targets = torch.randint(0, 2, (32, 14)).float()
        >>> loss = loss_fn(outputs, targets)
    """
    
    def __init__(self, dataset, num_classes, beta=0.9999, loss_type='sigmoid', reduction='mean'):
        super(CBLossOriginal, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.loss_type = loss_type
        self.reduction = reduction
        
        # Calculate class weights from dataset
        self.class_weights = self._calculate_class_weights(dataset, num_classes, beta)
        
        print(f"  📊 Class-Balanced Loss (Original Official) initialized:")
        print(f"     Beta: {beta}")
        print(f"     Loss type: {loss_type}")
        print(f"     Class weights (top 5): {self.class_weights[:5].cpu().numpy()}")
        print(f"     ⚠️  Using sample-level averaging of weights")
    
    def _calculate_class_weights(self, dataset, num_classes, beta):
        """
        Calculate class weights based on effective number of samples.
        
        Args:
            dataset: Training dataset
            num_classes (int): Number of classes
            beta (float): Hyperparameter for effective number
        
        Returns:
            Tensor: Class weights of shape (num_classes,)
        """
        print(f"  🔍 Calculating class weights from dataset...")
        
        # Count samples per class
        samples_per_class = np.zeros(num_classes)
        
        # Check if this is multi-label or single-label
        _, first_label = dataset[0]
        is_multilabel = False
        
        if isinstance(first_label, (torch.Tensor, np.ndarray)):
            if hasattr(first_label, 'shape') and len(first_label.shape) > 0:
                if first_label.shape[0] > 1 or (hasattr(first_label, 'size') and first_label.size > 1):
                    is_multilabel = True
        
        if is_multilabel:
            # Multi-label: count positive samples for each class
            print(f"     Detected multi-label dataset")
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.numpy()
                elif not isinstance(label, np.ndarray):
                    label = np.array(label)
                label = label.flatten()
                samples_per_class += (label > 0).astype(int)
        else:
            # Single-label: count samples for each class
            print(f"     Detected single-label dataset")
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item() if label.numel() == 1 else label.numpy()[0]
                elif isinstance(label, np.ndarray):
                    label = int(label.flatten()[0])
                else:
                    label = int(label)
                samples_per_class[label] += 1
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        
        # Avoid division by zero for classes with no samples
        effective_num = np.where(samples_per_class == 0, 1.0, effective_num)
        
        # Calculate weights
        weights = (1.0 - beta) / effective_num
        
        # Normalize weights (optional, makes interpretation easier)
        weights = weights / weights.sum() * num_classes
        
        # Print statistics
        print(f"     Samples per class (min/max/mean): {samples_per_class.min():.0f} / {samples_per_class.max():.0f} / {samples_per_class.mean():.0f}")
        print(f"     Weight ratio (max/min): {weights.max() / weights.min():.2f}x")
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Raw logits from model
                - For sigmoid: (N, C)
                - For softmax: (N, C)
            targets (Tensor): Ground truth labels
                - Multi-label: (N, C) binary labels
                - Single-label: (N,) class indices
        
        Returns:
            Tensor: Computed loss
        """
        device = inputs.device
        weights = self.class_weights.to(device)
        
        if self.loss_type == 'sigmoid':
            # Multi-label classification
            # Original implementation: average weights of positive labels per sample
            targets = targets.float()
            
            # Calculate BCE loss per sample per class (no reduction)
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )  # Shape: (N, C)
            
            # For each sample, get the weights of positive labels and average them
            # This creates a scalar weight per sample
            positive_mask = targets > 0  # Shape: (N, C)
            
            # Expand weights to match batch: (1, C) -> (N, C)
            weights_expanded = weights.unsqueeze(0).expand_as(targets)
            
            # Get weights only for positive labels
            weights_for_positives = weights_expanded * positive_mask  # Shape: (N, C)
            
            # Count number of positive labels per sample
            num_positives = positive_mask.sum(dim=1, keepdim=True)  # Shape: (N, 1)
            num_positives = torch.clamp(num_positives, min=1)  # Avoid division by zero
            
            # Average weight per sample (sum of positive weights / number of positives)
            avg_weight_per_sample = weights_for_positives.sum(dim=1, keepdim=True) / num_positives  # Shape: (N, 1)
            
            # Apply the averaged scalar weight to the entire sample's loss
            weighted_loss = bce_loss * avg_weight_per_sample  # Shape: (N, C)
            
            if self.reduction == 'mean':
                return weighted_loss.mean()
            elif self.reduction == 'sum':
                return weighted_loss.sum()
            else:
                return weighted_loss
        
        elif self.loss_type == 'softmax':
            # Single-label classification
            # For single-label, both versions behave the same
            targets = targets.long()
            
            # Calculate CE loss
            ce_loss = F.cross_entropy(
                inputs, targets, weight=weights, reduction=self.reduction
            )
            
            return ce_loss
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Use 'sigmoid' or 'softmax'.")
