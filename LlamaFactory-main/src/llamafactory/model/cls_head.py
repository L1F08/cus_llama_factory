# Copyright 2025 Custom (crash detection extension). Apache-2.0 License.
"""Binary classification head + Focal Loss for VLM crash detection.

Replaces token-level CE loss with a small MLP head on the LLM's last hidden
state, avoiding materialization of the full [B, T, vocab_size] logits tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassificationHead(nn.Module):
    """Classification head on LLM hidden state. Outputs [logit_safe, logit_high_risk]."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.dropout(hidden)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-class focal loss for binary classification.

    Args:
        logits: [B, 2] - index 0 = safe, index 1 = high_risk
        labels: [B] - 0 (safe) or 1 (high_risk)
        alpha: weight for positive class (high_risk). > 0.5 prefers recall.
        gamma: focusing parameter.
    """
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    alpha_weight = torch.where(
        labels == 1,
        torch.full_like(pt, alpha),
        torch.full_like(pt, 1.0 - alpha),
    )
    focal_term = (1.0 - pt).clamp(min=1e-6) ** gamma
    loss = -alpha_weight * focal_term * log_pt

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def attach_cls_head(
    model: nn.Module, hidden_size: int, dropout: float = 0.1, dtype=torch.bfloat16
) -> BinaryClassificationHead:
    """Attach a cls_head to the (un-wrapped) base model. Returns the head."""
    head = BinaryClassificationHead(hidden_size, dropout=dropout).to(dtype=dtype)
    model.cls_head = head
    for p in head.parameters():
        p.requires_grad_(True)
    return head
