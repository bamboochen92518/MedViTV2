"""
Distribution-Balanced Loss (ResampleLoss)

Combines re-weighting and re-sampling strategies to handle long-tailed distribution.
Specifically designed for datasets where class frequencies follow a long-tail distribution.

Reference:
    Wu et al. "Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets" (ECCV 2020)
    GitHub: https://github.com/wutong16/DistributionBalancedLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DBFocalLoss(nn.Module):
    """
    Distribution-Balanced Focal Loss (ResampleLoss).
    
    This loss combines three key components:
    1. Re-weighting: Adjusts loss weights based on class frequency
    2. Re-sampling: Modulates gradient flow to simulate re-sampling
    3. Focal Loss: Focuses on hard examples
    
    The loss is particularly effective for long-tailed multi-label classification,
    where some classes have orders of magnitude more samples than others.
    
    Suitable for:
        - Long-tailed multi-label datasets
        - Medical imaging with rare diseases
        - Imbalanced ChestMNIST, COCO, etc.
    
    Args:
        dataset: Training dataset (to calculate class frequencies)
        num_classes (int): Number of classes
        use_sigmoid (bool): Use sigmoid activation. Default: True
        alpha (float): Re-sampling strength. Default: 0.1
            Controls the degree of re-sampling: 0 = no re-sampling, 1 = full re-sampling
        beta (float): Re-sampling temperature. Default: 10.0
            Controls the smoothness of the re-sampling map
        gamma (float): Focal loss focusing parameter. Default: 2.0
        focal_alpha (float): Focal loss balance parameter. Default: 2.0
        neg_scale (float): Negative logit regularization scale. Default: 2.0
            Applies regularization to reduce false positives
        init_bias (float): Initial bias for logit adjustment. Default: 0.05
        reduction (str): Reduction method. Default: 'mean'
    
    Shape:
        - Input: (N, C) where N is batch size and C is number of classes
        - Target: (N, C) binary labels
        - Output: scalar if reduction='mean'
    
    Example:
        >>> # For ChestMNIST with long-tail distribution
        >>> loss_fn = DBFocalLoss(
        ...     dataset=train_dataset,
        ...     num_classes=14,
        ...     alpha=0.1,
        ...     beta=10.0,
        ...     gamma=2.0
        ... )
        >>> outputs = torch.randn(32, 14)
        >>> targets = torch.randint(0, 2, (32, 14)).float()
        >>> loss = loss_fn(outputs, targets)
    """
    
    def __init__(self, dataset, num_classes, use_sigmoid=True, alpha=0.1, beta=10.0,
                 gamma=2.0, focal_alpha=2.0, neg_scale=2.0, init_bias=0.05, 
                 reduction='mean'):
        super(DBFocalLoss, self).__init__()
        
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha  # Re-sampling strength
        self.beta = beta    # Temperature
        self.gamma = gamma  # Focal gamma
        self.focal_alpha = focal_alpha
        self.neg_scale = neg_scale
        self.init_bias = init_bias
        self.reduction = reduction
        
        # Calculate class frequencies and re-balancing weights
        self.class_freq, self.neg_class_freq = self._get_class_freq(dataset, num_classes)
        self.resample_weights = self._get_resample_weights()
        
        print(f"  🎯 Distribution-Balanced Focal Loss initialized:")
        print(f"     Alpha (re-sampling): {alpha}")
        print(f"     Beta (temperature): {beta}")
        print(f"     Gamma (focal): {gamma}")
        print(f"     Negative scale: {neg_scale}")
        print(f"     Class freq (top 5): {self.class_freq[:5].cpu().numpy()}")
    
    def _get_class_freq(self, dataset, num_classes):
        """
        Calculate positive and negative class frequencies.
        
        Returns:
            class_freq: Positive sample frequency for each class (N, C)
            neg_class_freq: Negative sample frequency for each class (N, C)
        """
        print(f"  🔍 Calculating class frequencies from dataset...")
        
        pos_freq = np.zeros(num_classes)
        neg_freq = np.zeros(num_classes)
        
        # Count positive and negative samples
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            elif not isinstance(label, np.ndarray):
                label = np.array(label)
            label = label.flatten()
            
            pos_freq += (label > 0).astype(int)
            neg_freq += (label == 0).astype(int)
        
        # Convert to frequency (probability)
        total = len(dataset)
        pos_freq = pos_freq / total
        neg_freq = neg_freq / total
        
        print(f"     Positive freq (min/max/mean): {pos_freq.min():.4f} / {pos_freq.max():.4f} / {pos_freq.mean():.4f}")
        print(f"     Negative freq (min/max/mean): {neg_freq.min():.4f} / {neg_freq.max():.4f} / {neg_freq.mean():.4f}")
        
        return torch.tensor(pos_freq, dtype=torch.float32), torch.tensor(neg_freq, dtype=torch.float32)
    
    def _get_resample_weights(self):
        """
        Calculate re-sampling weights using the map function.
        
        The map function smoothly transitions from head to tail classes:
            map(f) = α * (1 - exp(-β * f^γ))
        
        where f is the class frequency.
        """
        # Re-sampling map function
        # For head classes (high freq): weight ≈ 0 (no re-sampling)
        # For tail classes (low freq): weight ≈ α (full re-sampling)
        resample_weight = self.alpha * (1 - torch.exp(-self.beta * self.class_freq ** 0.2))
        
        return resample_weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Raw logits (N, C)
            targets (Tensor): Binary labels (N, C)
        
        Returns:
            Tensor: Computed loss
        """
        targets = targets.float()
        device = inputs.device
        
        # Move frequency tensors to device
        class_freq = self.class_freq.to(device)
        neg_class_freq = self.neg_class_freq.to(device)
        resample_weights = self.resample_weights.to(device)
        
        # Apply logit adjustment for negative samples (regularization)
        # This reduces false positives by scaling down negative logits
        neg_scale = torch.ones_like(inputs) * self.neg_scale
        neg_scale = neg_scale * (1 - targets)  # Only apply to negatives
        inputs = inputs - self.init_bias * neg_scale
        
        # Get probabilities
        pred = torch.sigmoid(inputs)
        
        # Basic BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # === Focal Loss Component ===
        # p_t: probability for the target class
        pred_t = pred * targets + (1 - pred) * (1 - targets)
        focal_weight = (1 - pred_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        # === Re-weighting Branch ===
        # Balance positive and negative samples based on frequency
        # For tail classes: increase weight to compensate for low frequency
        pos_weight = 1.0 / (class_freq + 1e-8)  # Inverse frequency weighting
        neg_weight = 1.0 / (neg_class_freq + 1e-8)
        
        # Normalize weights
        pos_weight = pos_weight / pos_weight.max()
        neg_weight = neg_weight / neg_weight.max()
        
        # Apply re-weighting
        reweight = targets * pos_weight + (1 - targets) * neg_weight
        focal_loss = focal_loss * reweight
        
        # === Re-sampling Branch ===
        # Modulate gradient flow to simulate re-sampling
        # For tail classes: increase gradient (as if we sampled more)
        resample_weight_expanded = resample_weights.unsqueeze(0).expand_as(targets)
        
        # Apply re-sampling weight only to positive samples of tail classes
        resample_modulator = 1.0 + resample_weight_expanded * targets
        focal_loss = focal_loss * resample_modulator
        
        # Apply focal alpha balance
        focal_loss = self.focal_alpha * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ResampleLoss(DBFocalLoss):
    """
    Alias for DBFocalLoss to match the original paper's naming.
    
    This is the same as DBFocalLoss, provided for compatibility with
    the original implementation.
    """
    pass
