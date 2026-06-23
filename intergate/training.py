"""
Training loops: predict, train_one_epoch, train_graph_learning,
pretrain_masked_gene_model, finetune_pruned.
"""

import math
import os
import time
from collections import deque
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import compute_metrics_full, FocalLoss
from .utils import cleanup_memory
from .config import CFG


def _mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


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
    gate_tau: float = 1.0,
    ao_ids: Optional[tuple[int, int]] = None,   # (idA, idO)
    aux_lambda: float = 0.5,
    conn_lambda: float = 0.0,
    conn_min_deg: float = 0.05,
    conn_use_abs: bool = True,
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

    for step, (Xg, Xh, yb, _) in enumerate(loader, start=1):
        Xg = torch.as_tensor(Xg) if not torch.is_tensor(Xg) else Xg
        Xg = Xg.to(device=device, dtype=torch.float32, non_blocking=True)
        Xh = torch.as_tensor(Xh) if not torch.is_tensor(Xh) else Xh
        Xh = Xh.to(device=device, dtype=torch.float32, non_blocking=True)
        yb = torch.as_tensor(yb) if not torch.is_tensor(yb) else yb
        yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

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

            if loss_fn is None:
                loss = lam * F.cross_entropy(logits4, y_a) + (1 - lam) * F.cross_entropy(logits4, y_b)
            else:
                loss = lam * loss_fn(logits4, y_a) + (1 - lam) * loss_fn(logits4, y_b)

        else:
            logits4, _, logits_ao = model(Xg, adj, Xh)
            loss_main = F.cross_entropy(logits4, yb) if loss_fn is None else loss_fn(logits4, yb)

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


        if edge_l1_per_edge > 0:
            loss = loss + float(edge_l1_per_edge) * model.edge_l1_sum()

        if conn_lambda and float(conn_lambda) > 0:
            loss = loss + float(conn_lambda) * model.connectivity_penalty(min_deg=float(conn_min_deg), use_abs=bool(conn_use_abs))

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
    **kwargs,
):
    """
    Entrena el modelo en modo *graph-learning* (Xh=0) y selecciona por macro-F1 en validación.

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

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        # ====== PODA DURA (durante entrenamiento) ======
        keep_train = keep_schedule(epoch, end_keep=end_keep)
        if hasattr(model, "set_hard_keep_ratio"):
            model.set_hard_keep_ratio(keep_train)
            try:
                print("[TRAIN hard]", model.hard_prune_info())
            except Exception:
                pass

        # schedule tau
        if epoch <= 10:
            gate_tau = 0.5
        elif epoch <= 20:
            gate_tau = 0.25
        else:
            gate_tau = 0.1
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
            xh_mode="zero",
            edge_l1_per_edge=edge_l1,
            gate_tau=gate_tau,
            ao_ids=ao_ids,
            aux_lambda=float(aux_lambda),
            conn_lambda=(CFG.CONNECTIVITY_LAMBDA if CFG.ADD_CONNECTIVITY_PENALTY else 0.0),
            conn_min_deg=CFG.CONNECTIVITY_MIN_DEG,
            conn_use_abs=CFG.CONNECTIVITY_USE_ABS,
        )

        # VAL con Xh=0 (criterio de selección)
        proba_va0, y_va0 = predict_proba_xh_mode(model, adj, dl_va, device, xh_mode="zero")
        m_va0 = compute_metrics_full(y_va0, proba_va0, label_map)
        crit = m_va0.get("macro_f1", float("nan"))

        crit0 = float("nan")
        if dl_te is not None:
            proba_te0, y_te0 = predict_proba_xh_mode(model, adj, dl_te, device, xh_mode="zero")
            m_te0 = compute_metrics_full(y_te0, proba_te0, label_map)
            crit0 = m_te0.get("macro_f1", float("nan"))
        
        # early stopping
        if crit == crit and crit > best_score:
            best_score = crit
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1

        dt = time.time() - t0
        epoch_times.append(dt)
        avg_dt = sum(epoch_times) / len(epoch_times)
        elapsed = time.time() - t_start
        eta_stop = avg_dt * max(patience - pat, 0)
        eta_max = avg_dt * (epochs - epoch)

        te_msg = f" | F1te={crit0:.3f}" if dl_te is not None else ""
        print(
            f"[GraphLearn] epoch={epoch:03d} loss={loss:.4f} | "
            f"VAL(Xh=0) auc_macro={m_va0.get('auc_macro_ovr', float('nan')):.3f} "
            f"acc={m_va0.get('acc', float('nan')):.3f} | "
            f"F1={m_va0.get('macro_f1', float('nan')):.3f}"
            f"{te_msg} | "
            f"best={best_score:.3f} pat={pat} | "
            f"ETA_stop~{fmt_time(eta_stop)} ETA_max~{fmt_time(eta_max)} elapsed={fmt_time(elapsed)}"
        )

        if pat >= patience:
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
):
    """Fine-tune en 2 fases sobre el grafo podado/fijo con best checkpoint real en Phase B."""

    model_pruned = model_pruned.to(device)

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

    # ids LumA / LumB (robusto)
    if "LumA" in label_map and "LumB" in label_map:
        idA, idO = label_map["LumA"], label_map["LumB"]
    else:
        inv = {v: k for k, v in label_map.items()}
        idA, idO = inv.get("LumA", 0), inv.get("LumB", 1)

    ao_ids = (int(idA), int(idO))

    # ---- Fase A: Xh=0 ----
    for ep in range(1, int(epochs_A) + 1):
        loss = train_one_epoch(
            model_pruned, adj, dl_tr, opt, device,
            loss_fn=loss_fn,
            use_mixup=False,
            accum_steps=accum_steps,
            xh_mode="zero",
            edge_l1_per_edge=0.0,
            gate_tau=1.0,
            ao_ids=ao_ids,
            aux_lambda=0.5,
            conn_lambda=(CFG.CONNECTIVITY_LAMBDA if CFG.ADD_CONNECTIVITY_PENALTY else 0.0),
            conn_min_deg=CFG.CONNECTIVITY_MIN_DEG,
            conn_use_abs=CFG.CONNECTIVITY_USE_ABS,
        )
        proba_va0, y_va0 = predict_proba_xh_mode(model_pruned, adj, dl_va, device, xh_mode="zero")
        m_va0 = compute_metrics_full(y_va0, proba_va0, label_map)
        print(f"[FT-A ep{ep:03d}] loss={loss:.4f} | VAL Xh=0 macroF1={m_va0.get('macro_f1', float('nan')):.3f} auc={m_va0.get('auc_macro_ovr', float('nan')):.3f} acc={m_va0.get('acc', float('nan')):.3f}")

    # ---- Fase B: Xh=orig + best checkpoint ----
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
            xh_mode="orig",
            edge_l1_per_edge=0.0,
            gate_tau=1.0,
            ao_ids=ao_ids,
            aux_lambda=0.5,
            conn_lambda=(CFG.CONNECTIVITY_LAMBDA if CFG.ADD_CONNECTIVITY_PENALTY else 0.0),
            conn_min_deg=CFG.CONNECTIVITY_MIN_DEG,
            conn_use_abs=CFG.CONNECTIVITY_USE_ABS,
        )

        proba_va, y_va = predict_proba_xh_mode(model_pruned, adj, dl_va, device, xh_mode="orig")
        m_va = compute_metrics_full(y_va, proba_va, label_map)

        val = float(m_va.get(best_metric, float("nan")))
        print(
            f"[FT-B ep{ep:03d}] loss={loss:.4f} | "
            f"VAL orig macroF1={m_va.get('macro_f1', float('nan')):.3f} "
            f"auc={m_va.get('auc_macro_ovr', float('nan')):.3f} "
            f"acc={m_va.get('acc', float('nan')):.3f} | "
            f"best({best_metric})={best_val:.3f}@{best_ep} bad={bad}"
        )

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

        if int(patience_B) > 0 and bad >= int(patience_B):
            print(f"[FT-B] Early stop: sin mejora en {patience_B} épocas. Best={best_val:.4f} @ ep{best_ep}.")
            break

    if best_state is not None:
        model_pruned.load_state_dict(best_state, strict=False)
        print(f"[FT] Restored BEST checkpoint: {best_metric}={best_val:.4f} @ ep{best_ep}")
    else:
        print("[FT] WARNING: no se guardó ningún best checkpoint válido.")

    return model_pruned
