"""
Loss Functions for Imbalanced Medical Image Classification

This module provides various loss functions designed for handling class imbalance
in medical image classification tasks, particularly for multi-label scenarios.

Available Loss Functions:
    - BCELoss: Binary Cross Entropy Loss
    - CBLoss: Class-Balanced Loss
    - ASLoss: Asymmetric Loss (for multi-label)
    - FocalLoss: Focal Loss (for hard examples)
    - DBFocalLoss: Distribution-Balanced Focal Loss (for long-tail)

Usage:
    from losses import get_loss_function
    
    loss_fn = get_loss_function(
        loss_name='Focal',
        num_classes=14,
        dataset=train_dataset
    )
"""

from .bce_loss import BCELoss
from .cb_loss import CBLoss
from .asl_loss import ASLoss, ASLSingleLabel
from .focal_loss import FocalLoss
from .dbfocal_loss import DBFocalLoss, ResampleLoss


__all__ = [
    'BCELoss',
    'CBLoss',
    'ASLoss',
    'ASLSingleLabel',
    'FocalLoss',
    'DBFocalLoss',
    'ResampleLoss',
    'get_loss_function'
]


def get_loss_function(loss_name, num_classes, dataset=None, task='multi-label, binary-class', **kwargs):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_name (str): Name of the loss function
            Options: 'BCE', 'CBLoss', 'ASL', 'Focal', 'DBFocal', 'default'
        num_classes (int): Number of classes
        dataset: Training dataset (needed for CBLoss and DBFocal to calculate class weights)
        task (str): Task type ('multi-label, binary-class' or 'multi-class')
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        Loss function instance
    
    Example:
        >>> loss_fn = get_loss_function('Focal', num_classes=14, dataset=train_dataset)
        >>> loss = loss_fn(outputs, targets)
    """
    loss_name = loss_name.upper() if loss_name else 'DEFAULT'
    
    if loss_name == 'DEFAULT':
        # Use default PyTorch losses based on task
        import torch.nn as nn
        if task == 'multi-label, binary-class':
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()
    
    elif loss_name == 'BCE':
        return BCELoss()
    
    elif loss_name == 'CBLOSS' or loss_name == 'CB':
        if dataset is None:
            raise ValueError("CBLoss requires dataset to calculate class weights. "
                           "Please pass dataset parameter.")
        return CBLoss(dataset=dataset, num_classes=num_classes, **kwargs)
    
    elif loss_name == 'ASL':
        if task == 'multi-label, binary-class':
            return ASLoss(**kwargs)
        else:
            return ASLSingleLabel(**kwargs)
    
    elif loss_name == 'FOCAL':
        # Auto-detect use_sigmoid based on task
        use_sigmoid = (task == 'multi-label, binary-class')
        return FocalLoss(use_sigmoid=use_sigmoid, **kwargs)
    
    elif loss_name == 'DBFOCAL' or loss_name == 'DB' or loss_name == 'RESAMPLE':
        if dataset is None:
            raise ValueError("DBFocalLoss requires dataset to calculate class frequencies. "
                           "Please pass dataset parameter.")
        return DBFocalLoss(dataset=dataset, num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available options: 'default', 'BCE', 'CBLoss', 'ASL', 'Focal', 'DBFocal'")
