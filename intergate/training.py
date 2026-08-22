"""
Training loops: predict, train_one_epoch, train_graph_learning,
pretrain_masked_gene_model, finetune_pruned.
"""

import math
import os
import time
from collections import deque
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .losses import compute_metrics_full, FocalLoss
from .utils import cleanup_memory
from .config import CFG


def reported_class_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    label_map: dict,
    class_name: str,
) -> Dict[str, Any]:
    """Métricas one-vs-rest para una única clase mostrada en el log."""
    class_name = str(class_name)
    if class_name in label_map:
        class_id = int(label_map[class_name])
    else:
        matches = [
            int(class_id)
            for class_id, name in label_map.items()
            if str(name) == class_name
        ]
        if len(matches) != 1:
            raise KeyError(
                f"TRAIN_REPORT_LABEL={class_name!r} no aparece una sola vez "
                f"en label_map={label_map}."
            )
        class_id = matches[0]

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if proba.ndim != 2 or not 0 <= class_id < proba.shape[1]:
        raise ValueError(
            f"Probabilidades incompatibles con {class_name}={class_id}: "
            f"shape={proba.shape}."
        )

    pred = proba.argmax(axis=1)
    y_pos = y_true == class_id
    pred_pos = pred == class_id
    tp = int(np.sum(y_pos & pred_pos))
    fp = int(np.sum(~y_pos & pred_pos))
    fn = int(np.sum(y_pos & ~pred_pos))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) else 0.0
    )
    auc = (
        float(roc_auc_score(y_pos.astype(np.int8), proba[:, class_id]))
        if np.unique(y_pos).size == 2 else float("nan")
    )
    return {
        "class_id": int(class_id),
        "class_name": class_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "support": int(y_pos.sum()),
    }


def _format_reported_class(report: Dict[str, Any]) -> str:
    """Formatea el resumen compacto de la clase objetivo."""
    return (
        f"P={report['precision']:.3f} "
        f"R={report['recall']:.3f} "
        f"F1={report['f1']:.3f} "
        f"AUC={report['auc']:.3f} "
        f"n={int(report['support'])}"
    )


def _mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _connectivity_options(model):
    """Resuelve la conectividad desde el brazo de ablación, no desde CFG.

    Los modelos antiguos que no guardaban estos atributos conservan el
    comportamiento previo mediante el fallback a la configuración global.
    """
    core = getattr(model, "gnn", model)
    enabled = bool(getattr(
        core, "add_connectivity_penalty", CFG.ADD_CONNECTIVITY_PENALTY
    ))
    min_deg = float(getattr(
        core, "connectivity_min_deg", CFG.CONNECTIVITY_MIN_DEG
    ))
    use_abs = bool(getattr(
        core, "connectivity_use_abs", CFG.CONNECTIVITY_USE_ABS
    ))
    conn_lambda = float(CFG.CONNECTIVITY_LAMBDA) if enabled else 0.0
    return conn_lambda, min_deg, use_abs


# ── Prediction ────────────────────────────────────

@torch.no_grad()
def predict_proba_xh_mode(model, adj, loader, device, xh_mode="orig"):
    """Predice probabilidades con control explícito del modo de Xh ('orig'|'zero')."""
    model.eval()

    # device real del modelo (robusto ante parámetros sueltos en CPU)
    try:
        dev = model.inp_lin1.weight.device
    except Exception:
        try:
            dev = model.edge_logit.device
        except Exception:
            dev = device

    all_proba, all_y = [], []

    for Xg, Xh, yb, _ in loader:
        Xg = torch.as_tensor(Xg) if not torch.is_tensor(Xg) else Xg
        Xh = torch.as_tensor(Xh) if not torch.is_tensor(Xh) else Xh
        yb = torch.as_tensor(yb) if not torch.is_tensor(yb) else yb

        Xg = Xg.to(device=dev, dtype=torch.float32, non_blocking=True)
        Xh = Xh.to(device=dev, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device=dev, dtype=torch.long, non_blocking=True)

        if xh_mode == "zero":
            Xh = torch.zeros_like(Xh)

        logits4, _, _ = model(Xg, adj, Xh)
        proba = torch.softmax(logits4, dim=1)

        all_proba.append(proba.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())

    return np.concatenate(all_proba, axis=0), np.concatenate(all_y, axis=0)



# ── Train one epoch ───────────────────────────────

def _cross_entropy_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Optional[nn.Module],
) -> torch.Tensor:
    """Versión sin reducción compatible con CrossEntropyLoss."""
    if loss_fn is None:
        return F.cross_entropy(logits, target, reduction="none")
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return F.cross_entropy(
            logits,
            target,
            weight=loss_fn.weight,
            ignore_index=loss_fn.ignore_index,
            reduction="none",
            label_smoothing=float(getattr(loss_fn, "label_smoothing", 0.0)),
        )
    raise TypeError(
        "Los pesos por subtipo requieren CrossEntropyLoss (o loss_fn=None), "
        f"no {type(loss_fn).__name__}."
    )


def _batch_sample_weights(
    sample_weights_full,
    sample_ids,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if sample_weights_full is None:
        return None
    ids = (
        sample_ids.detach().cpu().numpy()
        if torch.is_tensor(sample_ids)
        else np.asarray(sample_ids)
    )
    if torch.is_tensor(sample_weights_full):
        out = sample_weights_full.detach().cpu()[ids]
    else:
        out = np.asarray(sample_weights_full, dtype=np.float32)[ids]
    return torch.as_tensor(
        out, dtype=torch.float32, device=device
    ).reshape(-1)


def train_one_epoch(
    model: nn.Module,
    adj: torch.Tensor,
    loader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    mixup_prob: float = 0.5,
    accum_steps: int = 1,
    xh_mode: str = "orig",
    edge_l1_per_edge: float = 0.0,
    sparsity_target: float = 0.05,
    gate_tau: float = 1.0,
    ao_ids: Optional[tuple[int, int]] = None,   # (idA, idO)
    aux_lambda: float = 0.5,
    conn_lambda: float = 0.0,
    conn_min_deg: float = 0.05,
    conn_use_abs: bool = True,
    sample_weights_full=None,
    structural_regularizers: bool = True,
) -> float:
    model.train()
    model.gate_tau = float(gate_tau)
    # Device real del modelo (robusto si algún parámetro quedó en CPU)
    try:
        device = model.inp_lin1.weight.device
    except Exception:
        try:
            device = model.edge_logit.device
        except Exception:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                pass
    total = 0.0
    n = 0
    opt.zero_grad(set_to_none=True)

    for step, (Xg, Xh, yb, sample_ids) in enumerate(loader, start=1):
        Xg = torch.as_tensor(Xg) if not torch.is_tensor(Xg) else Xg
        Xg = Xg.to(device=device, dtype=torch.float32, non_blocking=True)
        Xh = torch.as_tensor(Xh) if not torch.is_tensor(Xh) else Xh
        Xh = Xh.to(device=device, dtype=torch.float32, non_blocking=True)
        yb = torch.as_tensor(yb) if not torch.is_tensor(yb) else yb
        yb = yb.to(device=device, dtype=torch.long, non_blocking=True)
        batch_weights = _batch_sample_weights(
            sample_weights_full, sample_ids, device
        )

        if xh_mode == "zero":
            Xh = torch.zeros_like(Xh)

        do_mix = use_mixup and (np.random.rand() < mixup_prob) and (Xg.shape[0] >= 2)

        if do_mix:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(Xg.size(0), device=device)
            Xg_mix = lam * Xg + (1 - lam) * Xg[idx]
            Xh_mix = lam * Xh + (1 - lam) * Xh[idx]
            y_a, y_b = yb, yb[idx]

            logits4, _, _ = model(Xg_mix, adj, Xh_mix)

            if batch_weights is not None:
                w_a = batch_weights
                w_b = batch_weights[idx]
                loss_a = _cross_entropy_per_sample(logits4, y_a, loss_fn)
                loss_b = _cross_entropy_per_sample(logits4, y_b, loss_fn)
                # Los pesos están normalizados a media 1 en TRAIN. Promediar
                # w*loss (en lugar de renormalizar cada microbatch por sum(w))
                # preserva la ponderación global incluso con microbatches
                # pequeños y acumulación de gradiente.
                loss = (
                    lam * w_a * loss_a
                    + (1.0 - lam) * w_b * loss_b
                ).mean()
            elif loss_fn is None:
                loss = (
                    lam * F.cross_entropy(logits4, y_a)
                    + (1 - lam) * F.cross_entropy(logits4, y_b)
                )
            else:
                loss = lam * loss_fn(logits4, y_a) + (1 - lam) * loss_fn(logits4, y_b)

        else:
            logits4, _, logits_ao = model(Xg, adj, Xh)
            if batch_weights is not None:
                loss_vec = _cross_entropy_per_sample(logits4, yb, loss_fn)
                loss_main = (loss_vec * batch_weights).mean()
            else:
                loss_main = (
                    F.cross_entropy(logits4, yb)
                    if loss_fn is None else loss_fn(logits4, yb)
                )

            # ----- AUX A vs O -----
            loss_aux = 0.0
            if ao_ids is not None:
                idA, idO = ao_ids
                mask = (yb == idA) | (yb == idO)
                if mask.any():
                    y_ao = (yb[mask] == idO).long()  # 0=A, 1=O
                    loss_aux = F.cross_entropy(logits_ao[mask], y_ao)

            #loss = loss_main + float(aux_lambda) * loss_aux
            loss = loss_main


        if structural_regularizers:
            # La densidad objetivo es independiente de la L1 y permanece activa
            # durante todo Phase 1, incluso cuando EDGE_L1_PER_EDGE=0.
            sparsity_mode = str(getattr(CFG, "SPARSITY_MODE", "target")).lower()
            if sparsity_mode == "target":
                loss = loss + float(CFG.SPARSITY_LAMBDA) * model.sparsity_penalty(
                    target=float(sparsity_target),
                    entropy_beta=float(CFG.SPARSITY_BINARIZE),
                )
            elif sparsity_mode == "l1" and edge_l1_per_edge > 0:
                loss = loss + float(edge_l1_per_edge) * model.edge_l1_sum()
            else:
                raise ValueError(f"SPARSITY_MODE no reconocido: {sparsity_mode!r}")

            if conn_lambda and float(conn_lambda) > 0:
                loss = loss + float(conn_lambda) * model.connectivity_penalty(
                    min_deg=float(conn_min_deg), use_abs=bool(conn_use_abs)
                )

        loss = loss / max(1, accum_steps)
        loss.backward()

        if step % max(1, accum_steps) == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        bs = Xg.shape[0]
        total += float(loss.item()) * bs * max(1, accum_steps)
        n += bs

    if (step % max(1, accum_steps)) != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)

    return total / max(1, n)



# ── Train graph learning (Phase 1) ───────────────

def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def keep_schedule(epoch, start_ep=1, end_ep=None, start_keep=1.0, end_keep=None):
    """Schedule exponencial de keep_ratio (hard-pruning durante el entrenamiento)."""
    if end_ep is None:
        from .config import InterGATEConfig
        end_ep = InterGATEConfig.STAB_EPOCHS - 5
    if end_keep is None:
        from .config import InterGATEConfig
        end_keep = InterGATEConfig.STAB_KEEP_FINAL
    if epoch < start_ep:
        return float(start_keep)
    if epoch >= end_ep:
        return float(end_keep)
    t = (epoch - start_ep) / (end_ep - start_ep)
    return float(np.exp(np.log(start_keep) + t*(np.log(end_keep)-np.log(start_keep))))


def train_graph_learning(
    model: nn.Module,
    adj: Optional[torch.Tensor] = None,
    dl_tr=None,
    dl_va=None,
    device: Optional[torch.device] = None,
    label_map: Optional[dict] = None,
    loss_fn: Optional[nn.Module] = None,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    patience: Optional[int] = None,
    accum_steps: int = 1,
    warmup: int = 1,
    edge_l1_after: float = 4e-5,
    edge_l1_per_edge: Optional[float] = None,
    aux_lambda: float = 0.5,
    keep_min: Optional[float] = None,
    class_weights=None,
    class_weights_t: Optional[torch.Tensor] = None,
    sample_weights_full=None,
    **kwargs,
):
    """
    Entrena el modelo en modo *graph-learning* y selecciona por macro-F1.

    En V6, ``CFG.GRAPH_LEARNING_XH_MODE`` debe ser ``zero``: X_h no entra en
    los gates ni en el clasificador durante Phase 1.

    Compatibilidad:
      - API antigua (este notebook): train_graph_learning(model, adj, dl_tr, dl_va, device, label_map, ...)
      - API nueva (ablación): train_graph_learning(model, adj=None, dl_tr=..., dl_va=..., epochs=..., class_weights=..., edge_l1_per_edge=..., keep_min=...)
    """
    # ---- defaults robustos ----
    if patience is None:
        from .config import InterGATEConfig
        patience = InterGATEConfig.PATIENCE

    if device is None:
        device = CFG.DEVICE

    if label_map is None:
        if "label_map" in globals():
            label_map = globals()["label_map"]
        else:
            raise ValueError("label_map no está definido. Pásalo como argumento o define label_map globalmente.")

    # ---- class weights / loss_fn ----
    if loss_fn is None:
        if class_weights_t is None and class_weights is not None:
            class_weights_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights_t is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
        else:
            loss_fn = nn.CrossEntropyLoss()

    # ---- edge L1 coef ----
    if edge_l1_per_edge is None:
        edge_l1_per_edge = float(edge_l1_after)

    # ---- hard-keep schedule ----
    end_keep = float(keep_min) if keep_min is not None else 0.001

    # Test loader opcional: solo para depuración, nunca para selección del modelo
    dl_te = kwargs.get("dl_te", None)

    graph_xh_mode = str(getattr(CFG, "GRAPH_LEARNING_XH_MODE", "zero")).lower()
    if graph_xh_mode != "zero":
        raise ValueError(
            "InterGATE V6 exige GRAPH_LEARNING_XH_MODE='zero'; "
            f"recibido {graph_xh_mode!r}."
        )
    forward_xh_mode = "zero"
    core = getattr(model, "gnn", model)
    core.use_xgraph_in_head = False
    conn_lambda_model, conn_min_deg_model, conn_use_abs_model = (
        _connectivity_options(model)
    )

    print(
        "[GraphLearn config] "
        f"X_h={graph_xh_mode} | "
        f"X_h_in_head={bool(getattr(core, 'use_xgraph_in_head', True))} | "
        f"connectivity={conn_lambda_model > 0.0} "
        f"(lambda={conn_lambda_model:g})",
        flush=True,
    )

    opt = torch.optim.AdamW(
        per_label_param_groups(model, float(lr)),
        weight_decay=weight_decay,
    )

    best_state = None
    best_score = -float("inf")
    pat = 0
    epoch_times = deque(maxlen=20)
    t_start = time.time()

    # ---- obtener ids de A y O desde label_map (robusto) ----
    # OJO: en este notebook se usan clases LumA/LumB; mantenemos robustez.
    try:
        if isinstance(label_map, dict) and "LumA" in label_map and "LumB" in label_map:
            idA, idO = label_map["LumA"], label_map["LumB"]
        else:
            inv = {v: k for k, v in label_map.items()}
            idA, idO = inv["LumA"], inv["LumB"]
        ao_ids = (int(idA), int(idO))
    except Exception:
        ao_ids = None  # si no aplica, desactiva aux

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        report_label = str(CFG.TRAIN_REPORT_LABEL)

        # ====== PODA DURA (durante entrenamiento) ======
        keep_train = keep_schedule(epoch, end_keep=end_keep)
        if hasattr(model, "set_hard_keep_ratio"):
            model.set_hard_keep_ratio(keep_train)
            try:
                hard_info = model.hard_prune_info()
                if bool(getattr(CFG, "TRAIN_LOG_FULL_PRUNING", False)):
                    print("[TRAIN hard]", hard_info, flush=True)
                else:
                    class_info = next(
                        (
                            item for item in hard_info.get("per_class", [])
                            if item.get("class_name") == report_label
                        ),
                        hard_info,
                    )
                    print(
                        f"[TRAIN hard][{report_label}] epoch={epoch:03d} "
                        f"keep={keep_train:.6f} "
                        f"kept={class_info.get('kept', '?')}/"
                        f"{class_info.get('total', '?')}",
                        flush=True,
                    )
            except Exception:
                pass

        # schedule tau
        gate_tau = 0.1
        for last_epoch, tau_value in getattr(CFG, "GATE_TAU_SCHEDULE", ((10, 0.5), (20, 0.25), (10**9, 0.1))):
            if epoch <= int(last_epoch):
                gate_tau = float(tau_value)
                break
        if hasattr(model, "gate_tau"):
            model.gate_tau = gate_tau

        # schedule lambda L1
        if epoch <= warmup:
            edge_l1 = 0.0
        else:
            edge_l1 = float(edge_l1_per_edge)

        loss = train_one_epoch(
            model, adj, dl_tr, opt, device,
            loss_fn=loss_fn,
            use_mixup=False,
            accum_steps=accum_steps,
            xh_mode=forward_xh_mode,
            edge_l1_per_edge=edge_l1,
            # La densidad objetivo sigue al schedule de poda dura: la presión
            # blanda y la dura piden lo mismo en cada época.
            sparsity_target=float(keep_train),
            gate_tau=gate_tau,
            ao_ids=ao_ids,
            aux_lambda=float(aux_lambda),
            conn_lambda=conn_lambda_model,
            conn_min_deg=conn_min_deg_model,
            conn_use_abs=conn_use_abs_model,
            sample_weights_full=sample_weights_full,
            structural_regularizers=True,
        )

        # ── Instrumentación del gate (PASO 2) ────────────────────────
        # Sin esto el colapso solo se ve al final, cuando ya se ha exportado
        # un grafo sin sentido. Coste: un forward sin gradiente por época.
        if bool(getattr(CFG, "LOG_GATE_STATS", False)):
            with torch.no_grad():
                _core = getattr(model, "gnn", model)
                _cid = 0 if getattr(_core, "per_label_graphs", False) else None
                _g = _core.edge_gate(None, class_id=_cid).flatten().float()
                _el = _core.edge_logit.detach().flatten().float()
                print(f"    [gate-base] ep={epoch:3d} tau={gate_tau:.3f} l1={edge_l1:.1e} "
                      f"| med={_g.median():.3e} max={_g.max():.3e} "
                      f">0.5={100*(_g > 0.5).float().mean():5.1f}% "
                      f"| edge_logit mean={_el.mean():+.4f} sd={_el.std():.3e}")

        # Validación con el mismo modo de X_h que el entrenamiento.
        proba_va0, y_va0 = predict_proba_xh_mode(
            model, adj, dl_va, device, xh_mode=forward_xh_mode
        )
        m_va0 = compute_metrics_full(y_va0, proba_va0, label_map)
        report_va0 = reported_class_metrics(
            y_va0, proba_va0, label_map, report_label
        )
        select_metric = str(getattr(CFG, "SELECT_METRIC", "macro_f1"))
        if select_metric not in m_va0:
            raise KeyError(
                f"CFG.SELECT_METRIC={select_metric!r} no existe. "
                f"Disponibles: {sorted(m_va0)}"
            )
        crit = m_va0[select_metric]

        report_te0 = None
        if dl_te is not None:
            proba_te0, y_te0 = predict_proba_xh_mode(
                model, adj, dl_te, device, xh_mode=forward_xh_mode
            )
            report_te0 = reported_class_metrics(
                y_te0, proba_te0, label_map, report_label
            )
        
        # El checkpoint exportable se elige únicamente cuando la cuota dura ya
        # ha alcanzado kappa. De otro modo una validación alta en la fase densa
        # podría seleccionar logits que nunca se entrenaron bajo el TopK final.
        at_target_budget = bool(
            float(keep_train) <= float(end_keep) * (1.0 + 1e-12)
        )
        if not at_target_budget:
            pat = 0
        elif crit == crit and crit > best_score:
            best_score = crit
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            pat = 0
        else:
            pat += 1

        dt = time.time() - t0
        epoch_times.append(dt)
        avg_dt = sum(epoch_times) / len(epoch_times)
        elapsed = time.time() - t_start
        eta_stop = avg_dt * max(patience - pat, 0)
        eta_max = avg_dt * (epochs - epoch)

        te_msg = (
            f" | TEST {_format_reported_class(report_te0)}"
            if report_te0 is not None else ""
        )
        print(
            f"[GraphLearn] epoch={epoch:03d} train_loss={loss:.4f} | "
            f"VAL[{report_label}] {_format_reported_class(report_va0)}"
            f"{te_msg} | "
            f"target_budget={at_target_budget} pat={pat} | "
            f"ETA_stop~{fmt_time(eta_stop)} ETA_max~{fmt_time(eta_max)} elapsed={fmt_time(elapsed)}"
        )

        if at_target_budget and pat >= patience:
            print("Early stopping (graph learning).")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_state, best_score



# ── Pretraining (masked gene modeling) ────────────

# ============================================================
# 9.x) (Opcional) Pretraining self-supervised + Stability selection
# ============================================================

def pretrain_masked_gene_model(
    model: nn.Module,
    dl_tr,
    device: torch.device,
    mask_prob: float = 0.15,
    epochs: int = 20,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    use_xgraph: bool = True,
    freeze_gates: bool = True,
):
    """Masked gene modeling: enmascara entradas x_gene y reconstruye por nodo (MSE en posiciones enmascaradas)."""
    model.train()
    if freeze_gates and hasattr(model, "edge_logit"):
        model.edge_logit.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        tot = 0.0
        n = 0
        for xg, xh, yb, idx in dl_tr:
            xg = torch.as_tensor(xg, device=device, dtype=torch.float32)
            xh = torch.as_tensor(xh, device=device, dtype=torch.float32)
            # máscara Bernoulli por nodo
            mask = (torch.rand_like(xg) < float(mask_prob))
            xg_in = xg.clone()
            xg_in[mask] = 0.0

            pred = model.forward_reconstruct(xg_in, x_graph=(xh if (use_xgraph and xh.numel() > 0) else None))  # (B,N)
            loss = F.mse_loss(pred[mask], xg[mask])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss.item()) * xg.shape[0]
            n += xg.shape[0]

        print(f"[PRETRAIN] ep={ep:03d} mse_masked={tot/max(1,n):.6f}")

    if freeze_gates and hasattr(model, "edge_logit"):
        model.edge_logit.requires_grad_(True)

    return model


def stability_selection_edges(
    build_model_fn,
    seeds: List[int],
    keep_ratio: float,
    epochs: int,
    device: torch.device,
):
    """Entrena múltiples seeds y devuelve frecuencia de selección de aristas (en el universo actual)."""
    from collections import Counter
    counter = Counter()
    total_edges = None

    for s in seeds:
        set_seed(int(s))
        model_s = build_model_fn().to(device)
        with torch.no_grad():
            if hasattr(model_s, "edge_logit"):
                model_s.edge_logit.data.fill_(-2.5)
        # entreno graph-learning corto (xh=0)
        class_w = compute_class_weights(y[train_idx], len(label_map))
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
        _m, best_state, best_score = train_graph_learning(
            model_s, adj, dl_tr, dl_va, device, label_map,
            loss_fn=loss_fn,
            lr=CFG.LR,
            weight_decay=CFG.WEIGHT_DECAY,
            epochs=int(epochs),
            patience=max(10, int(epochs//6)),
            accum_steps=CFG.ACCUM_STEPS,
            warmup=1,
            edge_l1_after=5e-4,
            aux_lambda=0.2
        )
        if best_state is not None:
            model_s.load_state_dict(best_state)

        ei_p, ew_p, et_p, thr, kept, E = export_pruned_graph(model_s, keep_ratio, keep_self_loops=False)
        total_edges = E
        # codifica arista por (src,tgt,type)
        key = torch.stack([ei_p[0], ei_p[1], et_p], dim=0).cpu().numpy()
        for k in key.T:
            counter[tuple(map(int, k))] += 1

        print(f"[STAB] seed={s} kept={kept}/{E} thr_logit={thr:.4f}")

    # frecuencia
    import pandas as pd
    rows = []
    for (u,v,t), c in counter.items():
        rows.append({"src": u, "tgt": v, "type": t, "count": c, "freq": c/len(seeds)})
    df = pd.DataFrame(rows).sort_values(["freq","count"], ascending=False)
    return df, total_edges


# ── Finetune pruned (Phase 2) ────────────────────

def finetune_pruned(
    model_pruned, adj, dl_tr, dl_va, device, label_map, loss_fn,
    lr, weight_decay, epochs_A=5, epochs_B=80, accum_steps=1,
    use_mixup_B=True, mixup_alpha=0.2, mixup_prob=0.5,
    best_metric: str = "macro_f1",
    patience_B: int = 25,
    min_delta: float = 1e-4,
    save_best_path: Optional[str] = None,
    lr_tab_mult: float = 2.0,
    lr_fusion_mult: float = 2.0,
    sample_weights_full=None,
    xh_mode_B: str = "orig",
):
    """Fine-tune the fixed consensus under the prespecified two-stage protocol.

    Phase 2A runs the GNN alone: the tabular/fusion branch is not called,
    ``X_h`` is zero, and mixup is off. Phase 2B calls the complete hybrid model
    with the configured ``X_h`` mode and optional mixup. Structural masks,
    logits, class deltas and typed calibration are frozen in both stages; the
    fixed message-passing path bypasses structural gates entirely.
    """

    model_pruned = model_pruned.to(device)
    core_model = getattr(model_pruned, "gnn", model_pruned)
    if not hasattr(core_model, "assert_fixed_graph_phase2"):
        raise TypeError("El modelo de Phase 2 no expone el contrato de grafo fijo.")
    core_model.assert_fixed_graph_phase2()

    phase2_a_mode = str(getattr(CFG, "PHASE2_A_MODE", "graph_only")).lower()
    if phase2_a_mode != "graph_only":
        raise ValueError(
            "El protocolo principal exige PHASE2_A_MODE='graph_only', no "
            f"{phase2_a_mode!r}."
        )
    if str(xh_mode_B).lower() != "orig":
        raise ValueError(
            "El protocolo principal exige X_h como covariable en Phase 2B "
            f"(xh_mode_B='orig'), no {xh_mode_B!r}."
        )

    is_hybrid = bool(getattr(model_pruned, "is_hybrid", False)) and hasattr(
        model_pruned, "gnn"
    )
    if bool(getattr(CFG, "USE_HYBRID_MODEL", True)) and not is_hybrid:
        raise RuntimeError(
            "El protocolo principal requiere el modelo híbrido en Phase 2B."
        )
    phase2a_model = model_pruned.gnn if is_hybrid else model_pruned

    named = list(model_pruned.named_parameters())
    gnn_params, tab_params, fus_params = [], [], []
    for n, p in named:
        if not p.requires_grad:
            continue
        if n.startswith("gnn."):
            gnn_params.append(p)
        elif n.startswith("tab") or ".tab" in n:
            tab_params.append(p)
        elif n.startswith("fusion") or ".fusion" in n:
            fus_params.append(p)
        else:
            gnn_params.append(p)

    param_groups = []
    if len(gnn_params) > 0:
        param_groups.append({"params": gnn_params, "lr": float(lr), "weight_decay": float(weight_decay)})
    if len(tab_params) > 0:
        param_groups.append({"params": tab_params, "lr": float(lr) * float(lr_tab_mult), "weight_decay": float(weight_decay)})
    if len(fus_params) > 0:
        param_groups.append({"params": fus_params, "lr": float(lr) * float(lr_fusion_mult), "weight_decay": float(weight_decay)})

    if len(param_groups) == 0:
        raise RuntimeError("No hay parámetros entrenables en model_pruned.")

    opt = torch.optim.AdamW(param_groups)
    conn_lambda_model, conn_min_deg_model, conn_use_abs_model = (
        _connectivity_options(model_pruned)
    )

    # La cabeza auxiliar LumA/LumB sólo se activa si ambas clases están
    # presentes en el endpoint multiclase elegido.
    if "LumA" in label_map and "LumB" in label_map:
        ao_ids = (int(label_map["LumA"]), int(label_map["LumB"]))
    else:
        ao_ids = None

    print(
        "[Phase2 contract] A=graph_only,X_h=0,mixup=off | "
        "B=hybrid,X_h=orig,mixup=config | fixed_graph=yes,gates=bypassed"
    )

    # ---- Phase 2A: GNN real, sin ejecutar tabular/fusión ----
    report_label = str(CFG.TRAIN_REPORT_LABEL)
    for ep in range(1, int(epochs_A) + 1):
        loss = train_one_epoch(
            phase2a_model, adj, dl_tr, opt, device,
            loss_fn=loss_fn,
            use_mixup=False,
            accum_steps=accum_steps,
            xh_mode="zero",
            edge_l1_per_edge=0.0,
            gate_tau=1.0,
            ao_ids=ao_ids,
            aux_lambda=0.5,
            conn_lambda=conn_lambda_model,
            conn_min_deg=conn_min_deg_model,
            conn_use_abs=conn_use_abs_model,
            sample_weights_full=sample_weights_full,
            structural_regularizers=False,
        )
        proba_va0, y_va0 = predict_proba_xh_mode(
            phase2a_model, adj, dl_va, device, xh_mode="zero"
        )
        report_va0 = reported_class_metrics(
            y_va0, proba_va0, label_map, report_label
        )
        print(
            f"[FT-A] epoch={ep:03d} train_loss={loss:.4f} | "
            f"VAL[{report_label}] {_format_reported_class(report_va0)}"
        )

    # ---- Phase 2B: híbrido completo + X_h + best checkpoint ----
    best_val = -1e18
    best_ep = -1
    best_state = None
    bad = 0

    for ep in range(1, int(epochs_B) + 1):
        loss = train_one_epoch(
            model_pruned, adj, dl_tr, opt, device,
            loss_fn=loss_fn,
            use_mixup=bool(use_mixup_B),
            mixup_alpha=float(mixup_alpha),
            mixup_prob=float(mixup_prob),
            accum_steps=accum_steps,
            xh_mode=str(xh_mode_B),
            edge_l1_per_edge=0.0,
            gate_tau=1.0,
            ao_ids=ao_ids,
            aux_lambda=0.5,
            conn_lambda=conn_lambda_model,
            conn_min_deg=conn_min_deg_model,
            conn_use_abs=conn_use_abs_model,
            sample_weights_full=sample_weights_full,
            structural_regularizers=False,
        )

        proba_va, y_va = predict_proba_xh_mode(
            model_pruned, adj, dl_va, device, xh_mode=str(xh_mode_B)
        )
        m_va = compute_metrics_full(y_va, proba_va, label_map)
        report_va = reported_class_metrics(
            y_va, proba_va, label_map, report_label
        )

        val = float(m_va.get(best_metric, float("nan")))
        if (not math.isfinite(val)):
            bad += 1
        elif val > (best_val + float(min_delta)):
            best_val = val
            best_ep = ep
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model_pruned.state_dict().items()}

            if save_best_path is not None:
                _mkdir(os.path.dirname(save_best_path))
                torch.save(
                    {
                        "epoch": int(ep),
                        "best_metric": str(best_metric),
                        "best_val": float(best_val),
                        "state_dict": best_state,
                    },
                    save_best_path
                )
        else:
            bad += 1

        print(
            f"[FT-B] epoch={ep:03d} train_loss={loss:.4f} | "
            f"VAL[{report_label}] {_format_reported_class(report_va)} | "
            f"bad={bad}"
        )

        if int(patience_B) > 0 and bad >= int(patience_B):
            print(
                f"[FT-B] Early stop: sin mejora en {patience_B} épocas."
            )
            break

    if best_state is not None:
        model_pruned.load_state_dict(best_state, strict=False)
        print(f"[FT] Checkpoint restaurado: epoch={best_ep}.")
    else:
        print("[FT] WARNING: no se guardó ningún best checkpoint válido.")

    return model_pruned


# ════════════════════════════════════════════════════════════════════════
# Grafo por etiqueta: penalizaciones y grupo de LR
# ════════════════════════════════════════════════════════════════════════

def per_label_penalty(model, cfg=None) -> "torch.Tensor":
    """L1 del delta + termino de diversidad, ya ponderados.

    Devuelve un tensor 0-dim. Si ``per_label_graphs`` esta apagado devuelve
    exactamente 0, asi que sumarlo a la perdida es inerte.
    """
    from .config import CFG as _C
    cfg = cfg or _C
    core = getattr(model, "gnn", model)
    if not getattr(core, "per_label_graphs", False):
        return torch.zeros((), device=next(model.parameters()).device)
    return (float(cfg.PER_LABEL_DELTA_L1) * core.per_label_delta_l1()
            + float(cfg.PER_LABEL_DELTA_DIVERSITY_LAMBDA) * core.per_label_diversity())


def per_label_param_groups(model, base_lr: float, cfg=None):
    """Separa ``edge_logit_delta`` en su propio grupo con LR multiplicado.

    Devuelve una lista de param groups lista para el optimizador. Si la feature
    esta apagada devuelve un unico grupo con todos los parametros, es decir, lo
    mismo que se hacia antes.
    """
    from .config import CFG as _C
    cfg = cfg or _C
    core = getattr(model, "gnn", model)
    delta = getattr(core, "edge_logit_delta", None)
    if delta is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": base_lr}]
    delta_ids = {id(delta)}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in delta_ids]
    return [
        {"params": rest, "lr": base_lr},
        {"params": [delta], "lr": base_lr * float(cfg.PER_LABEL_DELTA_LR_MULT)},
    ]
