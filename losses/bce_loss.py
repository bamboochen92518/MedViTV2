"""
Binary Cross Entropy Loss with Logits

Standard BCE loss for multi-label classification.
This is a wrapper around PyTorch's BCEWithLogitsLoss for consistency.
"""

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with Logits.
    
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss.
    
    Suitable for:
        - Multi-label classification
        - Binary classification
    
    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Default: 'mean'
        pos_weight (Tensor, optional): Weight of positive examples. Must be a vector with length equal to the number of classes.
    
    Shape:
        - Input: (N, C) where N is batch size and C is number of classes
        - Target: (N, C) same shape as input
        - Output: scalar if reduction='mean' (default)
    
    Example:
        >>> loss_fn = BCELoss()
        >>> outputs = torch.randn(32, 14)  # batch_size=32, num_classes=14
        >>> targets = torch.randint(0, 2, (32, 14)).float()
        >>> loss = loss_fn(outputs, targets)
    """
    
    def __init__(self, reduction='mean', pos_weight=None):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Raw logits from model (before sigmoid)
            targets (Tensor): Binary labels (0 or 1)
        
        Returns:
            Tensor: Computed loss
        """
        return self.loss_fn(inputs, targets)
