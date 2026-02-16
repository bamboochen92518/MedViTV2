"""
Asymmetric Loss for Multi-Label Classification

ASL addresses the issue of positive-negative imbalance in multi-label classification
by applying different focusing parameters to positive and negative samples.

Reference:
    Ridnik et al. "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    Paper: https://arxiv.org/abs/2009.14119
"""

import torch
import torch.nn as nn


class ASLoss(nn.Module):
    """
    Asymmetric Loss (Optimized Version).
    
    This loss applies asymmetric focusing to positive and negative samples,
    addressing the imbalance in multi-label classification where negatives
    significantly outnumber positives.
    
    Key features:
        - Asymmetric focusing: Different γ for positive (γ_pos) and negative (γ_neg) samples
        - Probability shifting: Clips negative probabilities to reduce easy negatives' contribution
        - Memory optimized: Uses inplace operations
    
    Suitable for:
        - Multi-label classification (primary use case)
        - Datasets with severe positive-negative imbalance (e.g., medical imaging)
        - ChestMNIST, where each image may have 0-3 diseases out of 14 classes
    
    Args:
        gamma_neg (float): Focusing parameter for negative samples. Default: 4
            Higher values down-weight easy negatives more aggressively
        gamma_pos (float): Focusing parameter for positive samples. Default: 1
            Usually lower than gamma_neg since positives are already rare
        clip (float): Probability margin for hard negative samples. Default: 0.05
            Shifts negative probabilities: p_neg' = min(p_neg + clip, 1)
        eps (float): Small constant for numerical stability. Default: 1e-8
        reduction (str): Reduction method. Options: 'mean' | 'sum'. Default: 'mean'
        disable_torch_grad_focal_loss (bool): Disable gradient for focal weight calculation.
            Default: True (for efficiency)
    
    Shape:
        - Input: (N, C) where N is batch size and C is number of classes
        - Target: (N, C) binary labels (0 or 1)
        - Output: scalar if reduction='mean' or 'sum'
    
    Example:
        >>> # For ChestMNIST (14 classes, multi-label)
        >>> loss_fn = ASLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        >>> outputs = torch.randn(32, 14)  # logits
        >>> targets = torch.randint(0, 2, (32, 14)).float()
        >>> loss = loss_fn(outputs, targets)
        
        >>> # Adjust for more aggressive negative focusing
        >>> loss_fn = ASLoss(gamma_neg=6, gamma_pos=0, clip=0.1)
    """
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
                 reduction='mean', disable_torch_grad_focal_loss=True):
        super(ASLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        
        # Pre-allocate tensors for memory efficiency (will be reused)
        self.targets = None
        self.anti_targets = None
        self.xs_pos = None
        self.xs_neg = None
        self.asymmetric_w = None
        self.loss = None
        
        print(f"  🎯 Asymmetric Loss initialized:")
        print(f"     Gamma_neg: {gamma_neg} (for negative samples)")
        print(f"     Gamma_pos: {gamma_pos} (for positive samples)")
        print(f"     Clip: {clip} (probability margin)")
        print(f"     Reduction: {reduction}")
    
    def forward(self, x, y):
        """
        Args:
            x (Tensor): Raw logits from model (N, C)
            y (Tensor): Binary target labels (N, C)
        
        Returns:
            Tensor: Computed loss (scalar)
        """
        self.targets = y
        self.anti_targets = 1 - y
        
        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        
        # Asymmetric Clipping
        # Shift negative class probabilities to reduce contribution of easy negatives
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)
        
        # Basic Cross Entropy calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        
        # Asymmetric Focusing
        # Apply different focusing parameters to positive and negative samples
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets
            )
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            self.loss *= self.asymmetric_w
        
        # Apply reduction
        if self.reduction == 'mean':
            return -self.loss.mean()
        elif self.reduction == 'sum':
            return -self.loss.sum()
        else:
            return -self.loss


class ASLSingleLabel(nn.Module):
    """
    Asymmetric Loss for Single-Label Classification.
    
    Adapted version of ASL for single-label (multi-class) problems.
    Applies asymmetric focusing to the softmax predictions.
    
    Args:
        gamma_pos (float): Focusing parameter for positive (target) class. Default: 0
        gamma_neg (float): Focusing parameter for negative (non-target) classes. Default: 4
        eps (float): Label smoothing parameter. Default: 0.1
        reduction (str): Reduction method. Options: 'mean' | 'sum'. Default: 'mean'
    
    Shape:
        - Input: (N, C) where N is batch size and C is number of classes
        - Target: (N,) class indices
        - Output: scalar if reduction='mean' or 'sum'
    
    Example:
        >>> # For PathMNIST (9 classes, single-label)
        >>> loss_fn = ASLSingleLabel(gamma_pos=0, gamma_neg=4, eps=0.1)
        >>> outputs = torch.randn(32, 9)  # logits
        >>> targets = torch.randint(0, 9, (32,))
        >>> loss = loss_fn(outputs, targets)
    """
    
    def __init__(self, gamma_pos=0, gamma_neg=4, eps=0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()
        
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        
        print(f"  🎯 Asymmetric Loss (Single-Label) initialized:")
        print(f"     Gamma_pos: {gamma_pos}")
        print(f"     Gamma_neg: {gamma_neg}")
        print(f"     Label smoothing: {eps}")
    
    def forward(self, inputs, target):
        """
        Args:
            inputs (Tensor): Raw logits from model (N, C)
            target (Tensor): Class indices (N,)
        
        Returns:
            Tensor: Computed loss (scalar)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        
        # Convert target indices to one-hot encoding
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )
        
        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets
        )
        log_preds = log_preds * asymmetric_w
        
        # Label smoothing
        if self.eps > 0:
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )
        
        # Loss calculation
        loss = -self.targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        
        return loss
