"""
LDACE + CCL Losses for Multi-Label Classification.

This module implements:
    - LDACELoss: pairwise dependency-aware BCE-style loss
    - CCLLoss: calibration loss via canonical Expected Calibration Error (ECE)
    - LDACECCLLoss: weighted sum of LDACE and CCL

Designed for multi-label tasks (e.g., ChestMNIST).
"""

import torch
import torch.nn as nn


def get_wmat(size, diag_weight=1.0, off_diag_weight=0.5):
    """Create class-pair weight matrix used by LDACE."""
    matrix = torch.full((size, size), off_diag_weight)
    indices = torch.arange(size)
    matrix[indices, indices] = diag_weight
    return matrix


def canonical_ece(labels, predictions, num_bins=15):
    """
    Canonical ECE for multi-label predictions.

    Args:
        labels (Tensor): shape (C, N) with binary labels.
        predictions (Tensor): shape (C, N) with confidence in [0, 1].
        num_bins (int): number of confidence bins.

    Returns:
        Tensor: scalar ECE averaged over classes.
    """
    if not isinstance(num_bins, int):
        if torch.is_tensor(num_bins):
            num_bins = int(num_bins.item())
        else:
            num_bins = int(num_bins)
    num_bins = max(1, num_bins)

    labels = labels.float()
    predictions = predictions.float().clamp(0.0, 1.0)

    c, n = predictions.shape
    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=predictions.device)
    total_ece = predictions.new_tensor(0.0)

    for cls_idx in range(c):
        cls_pred = predictions[cls_idx]
        cls_lbl = labels[cls_idx]
        cls_ece = predictions.new_tensor(0.0)

        for b in range(num_bins):
            left = bin_edges[b]
            right = bin_edges[b + 1]
            if b == num_bins - 1:
                in_bin = (cls_pred >= left) & (cls_pred <= right)
            else:
                in_bin = (cls_pred >= left) & (cls_pred < right)

            prop = in_bin.float().mean()
            if prop.item() > 0:
                acc_in_bin = cls_lbl[in_bin].mean()
                conf_in_bin = cls_pred[in_bin].mean()
                cls_ece = cls_ece + torch.abs(acc_in_bin - conf_in_bin) * prop

        total_ece = total_ece + cls_ece

    return total_ece / max(c, 1)


class LDACELoss(nn.Module):
    """
    LDACE loss operating on pairwise class interactions.

    Args:
        num_classes (int): number of labels/classes.
        diag_weight (float): pair weight on diagonal (same class pair).
        off_diag_weight (float): pair weight on off-diagonal (cross-class pair).
        eps (float): numerical stability epsilon.
    """

    def __init__(self, num_classes, diag_weight=1.0, off_diag_weight=0.5, eps=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.deno = (num_classes * (num_classes + 1)) / 2
        self.wm = get_wmat(num_classes, diag_weight=diag_weight, off_diag_weight=off_diag_weight)
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (Tensor): logits with shape (N, C, F) or (N, C)
            y_true (Tensor): binary labels with shape (N, C)
        """
        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(-1)

        y_pred = y_pred.sigmoid().clamp(min=self.eps, max=1 - self.eps)
        y_pred = y_pred @ y_pred.transpose(1, 2)

        y_true = y_true.float()
        y_true = y_true.unsqueeze(2) @ y_true.unsqueeze(1)

        wgts = self.wm.unsqueeze(0).expand(y_pred.shape[0], -1, -1).to(y_pred.device)
        loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        loss = loss * wgts

        return torch.sum(loss) / (loss.shape[0] * self.deno)


class CCLLoss(nn.Module):
    """
    CCL calibration term based on canonical ECE.

    Args:
        num_bins (int): number of bins for ECE.
        eps (float): numerical stability epsilon.
    """

    def __init__(self, num_bins=15, eps=1e-7):
        super().__init__()
        self.num_bins = num_bins
        self.eps = eps

    def forward(self, y_pred, labels):
        """
        Args:
            y_pred (Tensor): logits with shape (N, C, F) or (N, C)
            labels (Tensor): binary labels with shape (N, C)
        """
        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(-1)

        y_pred = y_pred.sigmoid().clamp(min=self.eps, max=1 - self.eps)
        y_pred = y_pred @ y_pred.transpose(1, 2)
        indices = torch.arange(y_pred.shape[-1], device=y_pred.device)
        y_pred_diag = y_pred[:, indices, indices]

        return canonical_ece(
            labels=labels.float().transpose(1, 0),
            predictions=y_pred_diag.transpose(1, 0),
            num_bins=self.num_bins,
        )


class LDACECCLLoss(nn.Module):
    """
    Combined LDACE + CCL loss.

    total_loss = lambda_ldace * LDACE + lambda_ccl * CCL
    """

    def __init__(
        self,
        num_classes,
        num_bins=15,
        lambda_ldace=1.0,
        lambda_ccl=1.0,
        diag_weight=1.0,
        off_diag_weight=0.5,
        eps=1e-7,
    ):
        super().__init__()
        self.lambda_ldace = lambda_ldace
        self.lambda_ccl = lambda_ccl
        self.ldace = LDACELoss(
            num_classes=num_classes,
            diag_weight=diag_weight,
            off_diag_weight=off_diag_weight,
            eps=eps,
        )
        self.ccl = CCLLoss(num_bins=num_bins, eps=eps)

    def forward(self, y_pred, y_true):
        ldace_loss = self.ldace(y_pred, y_true)
        ccl_loss = self.ccl(y_pred, y_true)
        return self.lambda_ldace * ldace_loss + self.lambda_ccl * ccl_loss