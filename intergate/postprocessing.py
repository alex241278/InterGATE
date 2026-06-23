"""
Post-processing: OVR thresholds, gene importance (Grad×Input),
post-hoc bundle loading, classification reports.
"""

from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt


# ── OVR thresholds ───────────────────────────────


def ovr_thresholds_from_val(proba_val, y_val, n_classes, grid=None):
    """
    Calcula umbrales OVR óptimos por clase sobre validación.

    proba_val: (n, C)
    y_val: (n,)
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)

    thr = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        y_bin = (y_val == c).astype(int)
        best_t, best_f1 = 0.5, -1
        for t in grid:
            pred_bin = (proba_val[:, c] >= t).astype(int)
            f1 = f1_score(y_bin, pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr


def predict_with_ovr_thresholds(proba, thr):
    """
    Predice usando umbrales OVR por clase.

    proba: (n, C), thr: (C,)
    """
    n, C = proba.shape
    pred = np.zeros(n, dtype=int)
    for i in range(n):
        ok = np.where(proba[i] >= thr)[0]
        if ok.size == 0:
            pred[i] = int(np.argmax(proba[i]))
        elif ok.size == 1:
            pred[i] = int(ok[0])
        else:
            pred[i] = int(ok[np.argmax(proba[i, ok])])
    return pred


def plot_confusion_matrix(y_true, y_pred, class_names, title="Matriz de Confusión"):
    """Genera y muestra la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    n_classes = len(class_names)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm)
    plt.colorbar()
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

    return cm_df


# ── Gene importance (Grad×Input) ────────────────


def gene_importance_grad_x_input(
    model,
    x_gene: np.ndarray,
    x_graph: np.ndarray,
    target_class: Optional[int],
    device: torch.device,
    gene_mask: Optional[np.ndarray] = None,
    zero_outside: bool = True,
) -> np.ndarray:
    model.eval()

    x_gene_use = x_gene.astype(np.float32).copy()
    if gene_mask is not None and zero_outside:
        x_gene_use[~gene_mask] = 0.0

    xg = torch.from_numpy(x_gene_use).to(device=device).unsqueeze(0)  # (1,G)
    xh = torch.from_numpy(x_graph.astype(np.float32)).to(device=device).unsqueeze(0)  # (1,F)
    xg.requires_grad_(True)

    out = model(xg, None, xh)
    logits = out[0] if isinstance(out, (tuple, list)) else out

    c = int(torch.argmax(logits, dim=1).item()) if target_class is None else int(target_class)
    z = logits[0, c]

    model.zero_grad(set_to_none=True)
    if xg.grad is not None:
        xg.grad.zero_()

    z.backward()

    grad = xg.grad.detach().cpu().numpy()[0]  # (G,)
    imp = np.abs(x_gene_use * grad)           # (G,)

    if gene_mask is not None:
        imp[~gene_mask] = 0.0

    return imp


def aggregate_gene_importance(
    model,
    Xg: np.ndarray,
    Xh: np.ndarray,
    y_true: np.ndarray,
    class_id: int,
    device: torch.device,
    max_samples: int = 200,
    gene_mask: Optional[np.ndarray] = None,
    zero_outside: bool = True,
) -> np.ndarray:
    idx = np.where(y_true == class_id)[0]
    if idx.size == 0:
        return np.zeros((Xg.shape[1],), dtype=np.float32)
    if idx.size > max_samples:
        idx = np.random.choice(idx, size=max_samples, replace=False)

    imps = []
    for k in tqdm(idx, desc=f"grad×input class={class_id}"):
        imps.append(
            gene_importance_grad_x_input(
                model,
                Xg[k], Xh[k],
                target_class=class_id,
                device=device,
                gene_mask=gene_mask,
                zero_outside=zero_outside,
            )
        )
    return np.mean(imps, axis=0).astype(np.float32)
