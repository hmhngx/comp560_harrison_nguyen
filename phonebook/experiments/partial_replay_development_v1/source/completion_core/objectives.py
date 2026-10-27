"""Explicit full-batch supervised-token weighting."""
import torch
from torch.nn import functional as F


def weighted_token_cross_entropy(logits, targets, row_weights):
    """Fixed row weights applied to *token sums*, normalized by weighted tokens.

    Integer weights reproduce full-batch literal row duplication. Book-balanced
    objectives instead require weights alpha/T_A and (1-alpha)/T_B. Neither
    interpretation implies equivalence to sequential minibatch optimizer steps.
    """
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError("Expected logits [rows, time, classes] and matching targets [rows, time]")
    if not logits.is_floating_point() or targets.dtype != torch.long:
        raise ValueError("Expected floating logits and int64 targets")
    if not isinstance(row_weights, torch.Tensor) or row_weights.ndim != 1 \
            or row_weights.shape[0] != targets.shape[0]:
        raise ValueError("row_weights must be a tensor with one weight per row")
    if row_weights.dtype == torch.bool or row_weights.is_complex():
        raise ValueError("row_weights must be real numeric weights, excluding booleans")
    if targets.device != logits.device or row_weights.device != logits.device:
        raise ValueError("logits, targets and row_weights must share a device")
    original_weights = row_weights.to(dtype=torch.float64)
    weights = row_weights.to(dtype=logits.dtype)
    if not torch.isfinite(original_weights).all() or (original_weights < 0).any() \
            or not torch.isfinite(weights).all() or ((original_weights > 0) & (weights == 0)).any():
        raise ValueError("row_weights must be finite and nonnegative in the logits dtype")
    token_counts = (targets != -100).sum(dim=1)
    denominator = (weights * token_counts).sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("Weighted supervised-token total must be finite and positive")
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             ignore_index=-100, reduction="none").reshape_as(targets)
    return (losses * (weights / denominator)[:, None]).sum()

