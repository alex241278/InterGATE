"""
Losses and evaluation metrics.
"""

from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight


# ────────────────────────────────────────────────────────────
# Focal loss
# ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


# ────────────────────────────────────────────────────────────
# AUC helpers
# ────────────────────────────────────────────────────────────
def safe_auc_macro_micro_ovr(y_true: np.ndarray, proba: np.ndarray):
    C = proba.shape[1]
    aucs, y_stack, s_stack = [], [], []

    for c in range(C):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, proba[:, c]))
        y_stack.append(yt)
        s_stack.append(proba[:, c])

    auc_macro = float(np.mean(aucs)) if aucs else float("nan")

    if y_stack:
        y_flat = np.concatenate(y_stack)
        s_flat = np.concatenate(s_stack)
        auc_micro = roc_auc_score(y_flat, s_flat) if len(np.unique(y_flat)) > 1 else float("nan")
    else:
        auc_micro = float("nan")

    return auc_macro, auc_micro


# ────────────────────────────────────────────────────────────
# Full metric computation
# ────────────────────────────────────────────────────────────
def compute_metrics_full(y_true: np.ndarray, proba: np.ndarray, label_map: Dict) -> Dict:
    """Return dict of acc, bal_acc, macro_f1, weighted_f1, per-class F1, AUCs."""
    C = proba.shape[1]
    pred = proba.argmax(axis=1)

    m = {}
    m["acc"] = accuracy_score(y_true, pred)
    m["bal_acc"] = balanced_accuracy_score(y_true, pred)
    m["macro_f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
    m["weighted_f1"] = f1_score(y_true, pred, average="weighted", zero_division=0)

    f1_per = f1_score(y_true, pred, average=None, labels=np.arange(C), zero_division=0)

    if isinstance(list(label_map.keys())[0], str):
        id_to_name = {v: k for k, v in label_map.items()}
    else:
        id_to_name = dict(label_map)

    for cid in range(C):
        name = id_to_name.get(cid, str(cid))
        m[f"f1_{name}"] = float(f1_per[cid])

    auc_macro, auc_micro = safe_auc_macro_micro_ovr(y_true, proba)
    m["auc_macro_ovr"] = auc_macro
    m["auc_micro_ovr"] = auc_micro
    return m


# ────────────────────────────────────────────────────────────
# Class weights
# ────────────────────────────────────────────────────────────
def compute_class_weights_balanced(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    w = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=y_train)
    return torch.tensor(w, dtype=torch.float32)
