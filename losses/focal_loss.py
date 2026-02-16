"""
Focal Loss for Addressing Class Imbalance

Focal Loss down-weights the contribution of easy examples and focuses training
on hard negatives, which is particularly useful for imbalanced datasets.

Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    Paper: https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss.
    
    Focal Loss applies a modulating term to the cross entropy loss to focus learning
    on hard examples. The modulating factor (1 - p_t)^γ reduces the loss contribution
    from easy examples (where p_t is high) and extends the range in which an example
    receives low loss.
    
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
        - p_t is the model's estimated probability for the target class
        - α_t is the weighting factor (balances positive/negative examples)
        - γ is the focusing parameter (typically 2.0)
    
    Suitable for:
        - Highly imbalanced datasets
        - Both multi-label and single-label classification
        - When easy examples dominate the training
    
    Args:
        alpha (float): Weighting factor for class balance. Default: 0.25
            Higher alpha gives more weight to the minority class
        gamma (float): Focusing parameter. Default: 2.0
            Higher gamma puts more focus on hard examples
            - γ = 0: equivalent to standard cross entropy
            - γ = 2: typical choice (from paper)
            - γ = 5: very aggressive focusing
        reduction (str): Reduction method. Options: 'none' | 'mean' | 'sum'. Default: 'mean'
        use_sigmoid (bool): Use sigmoid (for multi-label) or softmax (for single-label).
            Default: True
    
    Shape:
        - Input: (N, C) where N is batch size and C is number of classes
        - Target:
            - If use_sigmoid=True: (N, C) binary labels for multi-label
            - If use_sigmoid=False: (N,) class indices for single-label
        - Output: scalar if reduction='mean' or 'sum'
    
    Example:
        >>> # For multi-label (ChestMNIST)
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0, use_sigmoid=True)
        >>> outputs = torch.randn(32, 14)
        >>> targets = torch.randint(0, 2, (32, 14)).float()
        >>> loss = loss_fn(outputs, targets)
        
        >>> # For single-label (PathMNIST)
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0, use_sigmoid=False)
        >>> outputs = torch.randn(32, 9)
        >>> targets = torch.randint(0, 9, (32,))
        >>> loss = loss_fn(outputs, targets)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', use_sigmoid=True):
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        
        print(f"  🎯 Focal Loss initialized:")
        print(f"     Alpha: {alpha} (class balance weight)")
        print(f"     Gamma: {gamma} (focusing parameter)")
        print(f"     Mode: {'Sigmoid (multi-label)' if use_sigmoid else 'Softmax (single-label)'}")
        print(f"     Reduction: {reduction}")
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Raw logits from model (N, C)
            targets (Tensor): 
                - If use_sigmoid=True: Binary labels (N, C)
                - If use_sigmoid=False: Class indices (N,)
        
        Returns:
            Tensor: Computed loss
        """
        if self.use_sigmoid:
            return self._focal_loss_sigmoid(inputs, targets)
        else:
            return self._focal_loss_softmax(inputs, targets)
    
    def _focal_loss_sigmoid(self, inputs, targets):
        """
        Focal loss with sigmoid (for multi-label classification).
        
        Args:
            inputs (Tensor): Logits (N, C)
            targets (Tensor): Binary labels (N, C)
        
        Returns:
            Tensor: Focal loss
        """
        targets = targets.float()
        
        # Get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Calculate alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _focal_loss_softmax(self, inputs, targets):
        """
        Focal loss with softmax (for single-label classification).
        
        Args:
            inputs (Tensor): Logits (N, C)
            targets (Tensor): Class indices (N,)
        
        Returns:
            Tensor: Focal loss
        """
        targets = targets.long()
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # p_t: probability of the target class
        p_t = (p * targets_one_hot).sum(dim=1)
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss with alpha weighting
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
