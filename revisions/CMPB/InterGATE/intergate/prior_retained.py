"""
prior_retained.py
=================

Comparación entre el **grafo prior** (backbone candidato) y el **subgrafo
retenido** por un experto de clase (p. ej. Basal-like).

Contenido
---------
1. Utilidades de arista (frames prior / retenido en el mismo espacio de índices).
2. `PriorRetainedContext`: contenedor que fija UN layout compartido y produce
   las tres figuras comparables:
      - `figure_prior`     : prior coloreado por tipo de arista.
      - `figure_retained`  : subgrafo retenido coloreado por tipo.
      - `figure_overlay`   : prior en gris + retenido superpuesto en color.
      - `figure_triptych`  : las tres en un único panel A/B/C.
3. Análisis cuantitativos prior→retenido:
      - `retention_by_edge_type`  (Fisher + odds ratio por tipo)
      - `degree_bias`             (¿el gate solo elige hubs?)
      - `null_model_topology`     (topología frente a subgrafos aleatorios)
      - `class_specificity`       (comparación entre dos clases elegidas)
      - `gate_vs_prior`           (¿el gate copia el peso prior?)
      - `centrality_table`        (hubs del subgrafo retenido)

El módulo no importa torch de forma obligatoria: acepta tensores por *duck
typing* (`.detach().cpu().numpy()`) y arrays de numpy indistintamente.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

__all__ = [
    "ETYPE_LABELS",
    "PALETTES",
    "edges_to_frame",
    "load_prior_frame",
    "load_retained_frame",
    "PriorRetainedContext",
    "retention_by_edge_type",
    "degree_bias",
    "null_model_topology",
    "class_specificity",
    "gate_vs_prior",
    "centrality_table",
]


# ══════════════════════════════════════════════════════════════════════
# 0) Convenciones de tipo de arista
# ══════════════════════════════════════════════════════════════════════
# El backbone de InterGATE codifica:
#   0 = HuRI            → interacción física (PPI)
#   1 = OmniPath (+)    → activación
#   2 = OmniPath (−)    → inhibición
#   3 = STRING          → asociación funcional (CFG.STRING_EDGE_TYPE)

ETYPE_LABELS: Dict[int, str] = {
    0: "Physical (HuRI PPI)",
    1: "Activation (OmniPath)",
    2: "Inhibition (OmniPath)",
    3: "Functional (STRING)",
}

ETYPE_LABELS_ES: Dict[int, str] = {
    0: "Física (HuRI PPI)",
    1: "Activación (OmniPath)",
    2: "Inhibición (OmniPath)",
    3: "Funcional (STRING)",
}

#: Paletas disponibles.
#:  - ``"repo"``  : idéntica a ``intergate.visualization._ETYPE_COLOR`` (por
#:                  defecto, para que las figuras nuevas casen con las del
#:                  notebook 3 y con las del manuscrito).
#:  - ``"rby"``   : rojo=física, azul=activación, verde=inhibición.
PALETTES: Dict[str, Dict[int, str]] = {
    "repo": {0: "#4393c3", 1: "#1a9850", 2: "#d73027", 3: "#9e9e9e"},
    "rby": {0: "#d73027", 1: "#4393c3", 2: "#1a9850", 3: "#9e9e9e"},
}

_GREY = "#c8ccd0"


def _np(x):
    """Tensor de torch, lista o array → np.ndarray (sin importar torch)."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _pal(palette) -> Dict[int, str]:
    if isinstance(palette, dict):
        return palette
    return PALETTES[str(palette)]


# ══════════════════════════════════════════════════════════════════════
# 1) Frames de aristas
# ══════════════════════════════════════════════════════════════════════

def edges_to_frame(
    edge_index,
    edge_type=None,
    edge_weight=None,
    n_nodes: Optional[int] = None,
) -> pd.DataFrame:
    """(2,E) → DataFrame con claves dirigida (`dkey`) y no dirigida (`ukey`).

    Las claves permiten intersectar prior y retenido sin depender de nombres
    de gen ni del orden de las aristas.
    """
    ei = _np(edge_index).astype(np.int64)
    if ei.ndim != 2:
        raise ValueError(f"edge_index debe ser 2D, recibido {ei.shape}")
    if ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.T
    if ei.shape[0] != 2:
        raise ValueError(f"edge_index debe ser (2,E), recibido {ei.shape}")

    u = ei[0]
    v = ei[1]
    if n_nodes is None:
        n_nodes = int(max(u.max(), v.max())) + 1 if u.size else 1
    n_nodes = int(n_nodes)

    lo = np.minimum(u, v).astype(np.int64)
    hi = np.maximum(u, v).astype(np.int64)

    df = pd.DataFrame(
        {
            "u": u.astype(np.int64),
            "v": v.astype(np.int64),
            "dkey": u.astype(np.int64) * n_nodes + v.astype(np.int64),
            "ukey": lo * n_nodes + hi,
        }
    )
    et = _np(edge_type)
    df["edge_type"] = (
        np.asarray(et).reshape(-1).astype(np.int64)
        if et is not None
        else np.zeros(len(df), dtype=np.int64)
    )
    ew = _np(edge_weight)
    if ew is not None:
        df["prior_weight"] = np.asarray(ew).reshape(-1).astype(float)
    df.attrs["n_nodes"] = n_nodes
    return df


def load_prior_frame(
    edge_index,
    edge_type=None,
    edge_weight=None,
    genes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Frame del backbone completo (espacio de índices de `genes`)."""
    n_nodes = len(genes) if genes is not None else None
    df = edges_to_frame(edge_index, edge_type, edge_weight, n_nodes=n_nodes)
    df.attrs["genes"] = list(genes) if genes is not None else None
    return df


def load_retained_frame(
    bundle_dir: str,
    class_name: str = "Basal-like",
    *,
    n_nodes: Optional[int] = None,
    genes_full: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Lee `edges_{class}.csv` del bundle (índices FULL, con métricas del gate).

    Si el CSV no existe se cae a `graph_{class}.pt` (requiere torch) y, en
    último término, a `graph_bundle.pt`.
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(class_name))
    csv_path = os.path.join(bundle_dir, f"edges_{safe}.csv")

    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        u = raw["source_idx_full"].to_numpy(np.int64)
        v = raw["target_idx_full"].to_numpy(np.int64)
        if n_nodes is None:
            n_nodes = len(genes_full) if genes_full is not None else int(max(u.max(), v.max())) + 1
        n_nodes = int(n_nodes)
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        df = raw.copy()
        df["u"] = u
        df["v"] = v
        df["dkey"] = u * n_nodes + v
        df["ukey"] = lo * n_nodes + hi
        df.attrs["n_nodes"] = n_nodes
        df.attrs["source"] = csv_path
        return df

    import torch  # solo se necesita en el camino de respaldo

    pt_path = os.path.join(bundle_dir, f"graph_{safe}.pt")
    if os.path.exists(pt_path):
        g = torch.load(pt_path, map_location="cpu", weights_only=False)
    else:
        gb = torch.load(
            os.path.join(bundle_dir, "graph_bundle.pt"),
            map_location="cpu",
            weights_only=False,
        )
        wanted = str(class_name).casefold()
        matches = [
            cg
            for cg in gb.get("class_graphs_full", {}).values()
            if str(cg.get("class_name", "")).casefold() == wanted
        ]
        if not matches:
            raise KeyError(f"No encuentro el grafo de clase {class_name!r} en {bundle_dir}")
        g = matches[0]
        if genes_full is None:
            genes_full = gb.get("genes_full", None)

    df = edges_to_frame(
        g["edge_index"],
        g.get("edge_type", None),
        g.get("edge_weight", None),
        n_nodes=len(genes_full) if genes_full is not None else n_nodes,
    )
    for extra in ("learned_logit", "learned_gate", "selection_frequency"):
        if g.get(extra, None) is not None:
            df[extra] = _np(g[extra]).reshape(-1).astype(float)
    df.attrs["source"] = pt_path if os.path.exists(pt_path) else "graph_bundle.pt"
    return df


def _attach_gene_names(df: pd.DataFrame, genes: Sequence[str]) -> pd.DataFrame:
    genes_arr = np.asarray([str(g) for g in genes], dtype=object)
    out = df.copy()
    if "source_gene" not in out.columns:
        out["source_gene"] = genes_arr[out["u"].to_numpy()]
    if "target_gene" not in out.columns:
        out["target_gene"] = genes_arr[out["v"].to_numpy()]
    return out


# ══════════════════════════════════════════════════════════════════════
# 2) Contexto de figuras (layout compartido)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PriorRetainedContext:
    """Prior + retenido restringidos a un mismo conjunto de nodos y layout.

    Parameters
    ----------
    prior : DataFrame de `load_prior_frame`.
    retained : DataFrame de `load_retained_frame`.
    genes : lista de genes (espacio FULL, post `apply_connected_only`).
    scope : "induced" | "hop1" | "full"
        - ``induced`` (recomendado): prior inducido sobre los nodos retenidos.
          Es la comparación justa y la que permite superponer.
        - ``hop1``: nodos retenidos + sus vecinos prior a 1 salto.
        - ``full``: backbone completo (hairball; solo para una panorámica).
    palette : "repo" | "rby" | dict {etype: color}.
    """

    prior: pd.DataFrame
    retained: pd.DataFrame
    genes: List[str]
    scope: str = "induced"
    palette: str = "repo"
    max_nodes: int = 6000

    nodes: np.ndarray = field(init=False, default=None)
    prior_sub: pd.DataFrame = field(init=False, default=None)
    pos: Dict[int, np.ndarray] = field(init=False, default=None)
    _layout_info: dict = field(init=False, default_factory=dict)

    # ── construcción ──────────────────────────────────────────────
    def __post_init__(self):
        self.genes = [str(g) for g in self.genes]
        n = len(self.genes)
        retained_nodes = np.unique(
            np.concatenate([self.retained["u"].to_numpy(), self.retained["v"].to_numpy()])
        )

        if self.scope == "induced":
            nodes = retained_nodes
        elif self.scope == "hop1":
            mask = self.prior["u"].isin(retained_nodes) | self.prior["v"].isin(retained_nodes)
            nb = self.prior.loc[mask, ["u", "v"]].to_numpy().reshape(-1)
            nodes = np.unique(np.concatenate([retained_nodes, nb]))
        elif self.scope == "full":
            nodes = np.arange(n, dtype=np.int64)
        else:
            raise ValueError("scope debe ser 'induced', 'hop1' o 'full'")

        if len(nodes) > self.max_nodes:
            raise ValueError(
                f"scope={self.scope!r} da {len(nodes)} nodos > max_nodes={self.max_nodes}. "
                "Sube max_nodes si de verdad quieres dibujarlo (será un hairball)."
            )

        self.nodes = np.asarray(nodes, dtype=np.int64)
        keep = np.zeros(n, dtype=bool)
        keep[self.nodes] = True
        m = keep[self.prior["u"].to_numpy()] & keep[self.prior["v"].to_numpy()]
        self.prior_sub = self.prior.loc[m].reset_index(drop=True)

    # ── layout ────────────────────────────────────────────────────
    def compute_layout(
        self,
        method: str = "auto",
        seed: int = 42,
        iterations: int = 200,
        k: Optional[float] = None,
    ) -> Dict[int, np.ndarray]:
        """Un único layout, calculado sobre el prior inducido (superconjunto).

        Al compartirlo, los tres paneles son directamente comparables.
        """
        G = nx.Graph()
        G.add_nodes_from(self.nodes.tolist())
        G.add_edges_from(
            zip(self.prior_sub["u"].to_numpy().tolist(), self.prior_sub["v"].to_numpy().tolist())
        )
        G.remove_edges_from(nx.selfloop_edges(G))

        N = G.number_of_nodes()
        if method == "auto":
            method = "kamada" if N <= 700 else "spring"

        if method in ("kamada", "kamada_kawai", "kk"):
            pos = nx.kamada_kawai_layout(G)
        elif method in ("circular",):
            pos = nx.circular_layout(G)
        else:
            if k is None:
                k = 2.2 / np.sqrt(max(N, 2))
            pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations)

        self.pos = {int(a): np.asarray(b, dtype=float) for a, b in pos.items()}
        self._layout_info = {"method": method, "n_nodes": N, "seed": seed}
        return self.pos

    def _ensure_layout(self):
        if self.pos is None:
            self.compute_layout()

    # ── primitivas de dibujo ──────────────────────────────────────
    def _segments(self, df: pd.DataFrame) -> np.ndarray:
        p = self.pos
        u = df["u"].to_numpy()
        v = df["v"].to_numpy()
        ok = np.array([(a in p) and (b in p) and (a != b) for a, b in zip(u, v)])
        u, v = u[ok], v[ok]
        segs = np.empty((len(u), 2, 2), dtype=float)
        for i, (a, b) in enumerate(zip(u, v)):
            segs[i, 0] = p[int(a)]
            segs[i, 1] = p[int(b)]
        return segs, ok

    def _node_sizes(self, df: pd.DataFrame, base=14.0, scale=180.0) -> np.ndarray:
        deg = np.zeros(len(self.genes), dtype=float)
        idx, cnt = np.unique(
            np.concatenate([df["u"].to_numpy(), df["v"].to_numpy()]), return_counts=True
        )
        deg[idx] = cnt
        d = deg[self.nodes]
        if d.max() > d.min():
            d = (d - d.min()) / (d.max() - d.min())
        else:
            d = np.zeros_like(d)
        return base + scale * d

    def _draw_edges(
        self,
        ax,
        df: pd.DataFrame,
        *,
        palette,
        color: Optional[str] = None,
        lw: float = 0.6,
        alpha: float = 0.55,
        zorder: int = 1,
        arrows: bool = False,
        arrowsize: int = 9,
    ):
        if len(df) == 0:
            return
        pal = _pal(palette)
        if arrows:
            for _, row in df.iterrows():
                a, b = int(row["u"]), int(row["v"])
                if a == b or a not in self.pos or b not in self.pos:
                    continue
                c = color or pal.get(int(row.get("edge_type", 0)), "#888888")
                ax.annotate(
                    "",
                    xy=self.pos[b],
                    xytext=self.pos[a],
                    zorder=zorder,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=c,
                        lw=lw,
                        alpha=alpha,
                        shrinkA=3,
                        shrinkB=5,
                        mutation_scale=arrowsize,
                        connectionstyle="arc3,rad=0.08",
                    ),
                )
            return

        segs, ok = self._segments(df)
        if len(segs) == 0:
            return
        if color is not None:
            colors = color
        else:
            et = df["edge_type"].to_numpy()[ok]
            colors = [pal.get(int(t), "#888888") for t in et]
        ax.add_collection(
            LineCollection(segs, colors=colors, linewidths=lw, alpha=alpha, zorder=zorder)
        )

    def _draw_nodes(self, ax, sizes, color="#2b2b2b", edgecolor="white", lw=0.25,
                    alpha=0.95, zorder=3):
        xy = np.array([self.pos[int(n)] for n in self.nodes])
        ax.scatter(
            xy[:, 0], xy[:, 1], s=sizes, c=color, edgecolors=edgecolor,
            linewidths=lw, alpha=alpha, zorder=zorder,
        )
        return xy

    def _draw_labels(self, ax, df, top_labels=25, force_genes=None, fontsize=8,
                     color="#111111", zorder=6):
        deg = np.zeros(len(self.genes), dtype=float)
        idx, cnt = np.unique(
            np.concatenate([df["u"].to_numpy(), df["v"].to_numpy()]), return_counts=True
        )
        deg[idx] = cnt
        order = self.nodes[np.argsort(-deg[self.nodes])][: int(top_labels)]
        show = {int(i) for i in order}
        if force_genes:
            g2i = {g: i for i, g in enumerate(self.genes)}
            for g in force_genes:
                if g in g2i and g2i[g] in set(self.nodes.tolist()):
                    show.add(int(g2i[g]))
        for i in show:
            x, y = self.pos[int(i)]
            ax.text(
                x, y, self.genes[int(i)], fontsize=fontsize, fontweight="bold",
                color=color, ha="center", va="center", zorder=zorder,
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=0.6),
            )

    def _finish(self, ax, title, stats=None):
        ax.set_axis_off()
        ax.autoscale_view()
        ax.margins(0.04)
        if title:
            ax.set_title(title, fontsize=13, pad=10)
        if stats:
            ax.text(
                0.01, 0.01, stats, transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="#d0d0d0"),
            )

    def _etype_legend(self, ax, df, palette, loc="lower right", extra=None):
        pal = _pal(palette)
        present = sorted({int(t) for t in df["edge_type"].unique()})
        counts = df["edge_type"].value_counts().to_dict()
        handles = [
            Line2D([0], [0], color=pal.get(t, "#888888"), lw=3,
                   label=f"{ETYPE_LABELS.get(t, f'type {t}')} (n={counts.get(t, 0)})")
            for t in present
        ]
        if extra:
            handles = list(extra) + handles
        ax.legend(handles=handles, loc=loc, fontsize=8, framealpha=0.9)

    # ── figuras ───────────────────────────────────────────────────
    def figure_prior(
        self,
        *,
        title: Optional[str] = None,
        figsize=(11, 9),
        lw: float = 0.35,
        alpha: float = 0.45,
        top_labels: int = 25,
        force_genes: Optional[List[str]] = None,
        node_color: str = "#3a3a3a",
        save_path: Optional[str] = None,
        dpi: int = 300,
        ax=None,
    ):
        """1.1 — Prior coloreado por tipo de arista."""
        self._ensure_layout()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        df = self.prior_sub
        self._draw_edges(ax, df, palette=self.palette, lw=lw, alpha=alpha, zorder=1)
        self._draw_nodes(ax, self._node_sizes(df), color=node_color)
        self._draw_labels(ax, df, top_labels=top_labels, force_genes=force_genes)
        if title is None:
            title = f"Prior graph ({self.scope}) — coloured by edge type"
        self._finish(
            ax, title,
            stats=f"nodes={len(self.nodes)} | prior edges={len(df)}",
        )
        self._etype_legend(ax, df, self.palette)
        if save_path:
            (fig or ax.figure).savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[SAVE] {save_path}")
        return fig, ax

    def figure_retained(
        self,
        *,
        title: Optional[str] = None,
        figsize=(11, 9),
        lw: float = 1.2,
        alpha: float = 0.85,
        arrows: Optional[bool] = None,
        arrow_max: int = 900,
        top_labels: int = 25,
        force_genes: Optional[List[str]] = None,
        node_color: str = "#1f1f1f",
        save_path: Optional[str] = None,
        dpi: int = 300,
        ax=None,
    ):
        """1.2 — Subgrafo retenido, mismo código de color que el notebook."""
        self._ensure_layout()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        df = self.retained
        if arrows is None:
            arrows = len(df) <= arrow_max
        self._draw_edges(
            ax, df, palette=self.palette, lw=lw, alpha=alpha, zorder=2, arrows=arrows
        )
        self._draw_nodes(ax, self._node_sizes(df), color=node_color)
        self._draw_labels(ax, df, top_labels=top_labels, force_genes=force_genes)
        if title is None:
            title = "Retained subgraph — coloured by edge type"
        self._finish(
            ax, title,
            stats=f"nodes={len(self.nodes)} | retained edges={len(df)}",
        )
        self._etype_legend(ax, df, self.palette)
        if save_path:
            (fig or ax.figure).savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[SAVE] {save_path}")
        return fig, ax

    def figure_overlay(
        self,
        *,
        title: Optional[str] = None,
        figsize=(11, 9),
        prior_lw: float = 0.3,
        prior_alpha: float = 0.35,
        prior_color: str = _GREY,
        ret_lw: float = 1.4,
        ret_alpha: float = 0.95,
        arrows: Optional[bool] = None,
        arrow_max: int = 900,
        top_labels: int = 25,
        force_genes: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        dpi: int = 300,
        ax=None,
    ):
        """1.3 — Prior completo en gris + retenido superpuesto en color."""
        self._ensure_layout()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ret_keys = set(self.retained["dkey"].tolist())
        bg = self.prior_sub.loc[~self.prior_sub["dkey"].isin(ret_keys)]

        self._draw_edges(ax, bg, palette=self.palette, color=prior_color,
                         lw=prior_lw, alpha=prior_alpha, zorder=1)

        # nodos: gris los que solo están en el prior, oscuros los retenidos
        ret_nodes = set(
            np.unique(
                np.concatenate([self.retained["u"].to_numpy(), self.retained["v"].to_numpy()])
            ).tolist()
        )
        sizes = self._node_sizes(self.prior_sub, base=8.0, scale=60.0)
        is_ret = np.array([int(n) in ret_nodes for n in self.nodes])
        xy = np.array([self.pos[int(n)] for n in self.nodes])
        ax.scatter(xy[~is_ret, 0], xy[~is_ret, 1], s=sizes[~is_ret], c="#dfe3e6",
                   edgecolors="none", zorder=2)
        ax.scatter(xy[is_ret, 0], xy[is_ret, 1], s=self._node_sizes(self.retained)[is_ret],
                   c="#1f1f1f", edgecolors="white", linewidths=0.3, zorder=4)

        if arrows is None:
            arrows = len(self.retained) <= arrow_max
        self._draw_edges(ax, self.retained, palette=self.palette, lw=ret_lw,
                         alpha=ret_alpha, zorder=5, arrows=arrows)
        self._draw_labels(ax, self.retained, top_labels=top_labels,
                          force_genes=force_genes, zorder=7)

        if title is None:
            title = "Retained subgraph over the prior (grey)"
        kept_pct = 100.0 * len(self.retained) / max(len(self.prior_sub), 1)
        self._finish(
            ax, title,
            stats=(
                f"prior edges={len(self.prior_sub)} | retained={len(self.retained)} "
                f"({kept_pct:.2f}% of the induced prior)"
            ),
        )
        self._etype_legend(
            ax, self.retained, self.palette,
            extra=[Line2D([0], [0], color=prior_color, lw=3, label="Prior (not retained)")],
        )
        if save_path:
            (fig or ax.figure).savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[SAVE] {save_path}")
        return fig, ax

    def figure_triptych(
        self,
        *,
        figsize=(24, 8.5),
        titles=("A  Prior (candidate backbone)", "B  Retained subgraph",
                "C  Retained over prior"),
        save_path: Optional[str] = None,
        dpi: int = 300,
        **kw,
    ):
        """Las tres vistas en un único panel (figura de manuscrito).

        Los kwargs se filtran por panel, así que se puede pasar cualquier
        argumento de las tres funciones sin que ninguna reviente.
        """
        import inspect

        self._ensure_layout()
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        def _keep(fn):
            names = set(inspect.signature(fn).parameters)
            return {k: v for k, v in kw.items() if k in names}

        self.figure_prior(ax=axes[0], title=titles[0], **_keep(self.figure_prior))
        self.figure_retained(ax=axes[1], title=titles[1], **_keep(self.figure_retained))
        self.figure_overlay(ax=axes[2], title=titles[2], **_keep(self.figure_overlay))
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[SAVE] {save_path}")
        return fig, axes

    # ── grafos networkx listos para métricas ──────────────────────
    def nx_retained(self, directed=True) -> nx.Graph:
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(self.genes[i] for i in self.nodes)
        for _, r in self.retained.iterrows():
            a, b = int(r["u"]), int(r["v"])
            if a == b:
                continue
            G.add_edge(self.genes[a], self.genes[b], edge_type=int(r.get("edge_type", 0)))
        return G


# ══════════════════════════════════════════════════════════════════════
# 3) Análisis prior → retenido
# ══════════════════════════════════════════════════════════════════════

def retention_by_edge_type(
    prior: pd.DataFrame,
    retained: pd.DataFrame,
    labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """¿Qué tipos de arista sobreviven al gate más de lo esperado?

    Test de Fisher 2×2 por tipo (tipo t vs resto) × (retenida vs no).
    """
    from scipy.stats import fisher_exact

    labels = labels or ETYPE_LABELS
    ret_keys = set(retained["dkey"].tolist())
    is_ret = prior["dkey"].isin(ret_keys).to_numpy()
    et = prior["edge_type"].to_numpy()

    n_total = len(prior)
    n_ret = int(is_ret.sum())
    rows = []
    for t in sorted(set(et.tolist())):
        m = et == t
        a = int((m & is_ret).sum())          # tipo t retenida
        b = int(m.sum()) - a                 # tipo t no retenida
        c = n_ret - a                        # otro tipo retenida
        d = (n_total - int(m.sum())) - c     # otro tipo no retenida
        try:
            orr, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        except Exception:
            orr, p = float("nan"), float("nan")
        rows.append(
            {
                "edge_type": int(t),
                "label": labels.get(int(t), f"type {t}"),
                "n_prior": int(m.sum()),
                "prior_pct": 100.0 * m.sum() / max(n_total, 1),
                "n_retained": a,
                "retained_pct": 100.0 * a / max(n_ret, 1),
                "retention_rate_pct": 100.0 * a / max(int(m.sum()), 1),
                "odds_ratio": float(orr),
                "log2_enrichment": float(
                    np.log2(((a / max(n_ret, 1)) + 1e-12) / ((m.sum() / max(n_total, 1)) + 1e-12))
                ),
                "p_fisher": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        from scipy.stats import false_discovery_control  # scipy >= 1.11

        try:
            out["q_fdr"] = false_discovery_control(out["p_fisher"].fillna(1.0).to_numpy())
        except Exception:
            out["q_fdr"] = np.nan
    return out


def plot_retention_by_edge_type(df: pd.DataFrame, palette="repo", figsize=(11, 4.2),
                                save_path=None, dpi=300):
    """Barras: composición prior vs retenido + tasa de retención por tipo."""
    pal = _pal(palette)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(df))
    w = 0.38
    axes[0].bar(x - w / 2, df["prior_pct"], w, label="Prior", color="#b8bfc6")
    axes[0].bar(
        x + w / 2, df["retained_pct"], w, label="Retained",
        color=[pal.get(int(t), "#888") for t in df["edge_type"]],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["label"], rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("% of edges")
    axes[0].set_title("Edge-type composition")
    axes[0].legend(fontsize=8)

    axes[1].bar(
        x, df["retention_rate_pct"],
        color=[pal.get(int(t), "#888") for t in df["edge_type"]],
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["label"], rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("% of prior edges retained")
    axes[1].set_title("Retention rate by type")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    return fig, axes


def degree_bias(prior: pd.DataFrame, retained: pd.DataFrame, n_nodes: int,
                n_perm: int = 500, seed: int = 0) -> dict:
    """¿El gate se limita a elegir hubs del prior?

    Compara el grado prior medio de los extremos de las aristas retenidas con
    el de aristas del prior muestreadas al azar (mismo número).
    """
    from scipy.stats import mannwhitneyu

    deg = np.bincount(
        np.concatenate([prior["u"].to_numpy(), prior["v"].to_numpy()]), minlength=n_nodes
    ).astype(float)

    ret_nodes = np.unique(
        np.concatenate([retained["u"].to_numpy(), retained["v"].to_numpy()])
    )
    all_nodes = np.where(deg > 0)[0]

    u_stat, p_nodes = mannwhitneyu(deg[ret_nodes], deg[all_nodes], alternative="two-sided")

    ret_edge_deg = 0.5 * (deg[retained["u"].to_numpy()] + deg[retained["v"].to_numpy()])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    idx_all = np.arange(len(prior))
    for i in range(n_perm):
        s = rng.choice(idx_all, size=len(retained), replace=False)
        null[i] = np.mean(
            0.5 * (deg[prior["u"].to_numpy()[s]] + deg[prior["v"].to_numpy()[s]])
        )
    obs = float(ret_edge_deg.mean())
    p_edges = float((np.sum(null >= obs) + 1) / (n_perm + 1))

    return {
        "median_prior_degree_retained_nodes": float(np.median(deg[ret_nodes])),
        "median_prior_degree_all_nodes": float(np.median(deg[all_nodes])),
        "p_mannwhitney_nodes": float(p_nodes),
        "mean_edge_endpoint_degree_observed": obs,
        "mean_edge_endpoint_degree_null": float(null.mean()),
        "null_sd": float(null.std()),
        "z": float((obs - null.mean()) / (null.std() + 1e-12)),
        "p_empirical_edges": p_edges,
        "_degrees": deg,
        "_retained_nodes": ret_nodes,
        "_null": null,
        "_observed": obs,
    }


def _topology_metrics(u: np.ndarray, v: np.ndarray) -> dict:
    G = nx.Graph()
    G.add_edges_from(zip(u.tolist(), v.tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_components": 0, "lcc_frac": 0.0,
                "transitivity": 0.0, "modularity": 0.0, "mean_degree": 0.0}
    comps = list(nx.connected_components(G))
    lcc = max(len(c) for c in comps)
    try:
        parts = nx.community.louvain_communities(G, seed=0)
        modu = nx.community.modularity(G, parts)
    except Exception:
        modu = float("nan")
    return {
        "n_nodes": G.number_of_nodes(),
        "n_components": len(comps),
        "lcc_frac": lcc / G.number_of_nodes(),
        "transitivity": nx.transitivity(G),
        "modularity": float(modu),
        "mean_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
    }


def null_model_topology(
    prior: pd.DataFrame,
    retained: pd.DataFrame,
    n_perm: int = 200,
    seed: int = 0,
    match_edge_types: bool = True,
) -> pd.DataFrame:
    """Topología del subgrafo retenido frente a subgrafos aleatorios del prior.

    `match_edge_types=True` muestrea el mismo número de aristas *por tipo*, de
    modo que la comparación no queda confundida por la composición de tipos.
    """
    rng = np.random.default_rng(seed)
    obs = _topology_metrics(retained["u"].to_numpy(), retained["v"].to_numpy())

    pu, pv, pt = prior["u"].to_numpy(), prior["v"].to_numpy(), prior["edge_type"].to_numpy()
    if match_edge_types:
        by_type = {t: np.where(pt == t)[0] for t in np.unique(pt)}
        want = retained["edge_type"].value_counts().to_dict()
    else:
        by_type = {-1: np.arange(len(prior))}
        want = {-1: len(retained)}

    rows = []
    for _ in range(int(n_perm)):
        sel = []
        for t, k in want.items():
            pool = by_type.get(t, np.array([], dtype=int))
            k = int(min(k, len(pool)))
            if k:
                sel.append(rng.choice(pool, size=k, replace=False))
        if not sel:
            continue
        sel = np.concatenate(sel)
        rows.append(_topology_metrics(pu[sel], pv[sel]))

    null = pd.DataFrame(rows)
    out = []
    for m in ["n_nodes", "n_components", "lcc_frac", "transitivity", "modularity", "mean_degree"]:
        vals = null[m].to_numpy(dtype=float)
        o = float(obs[m])
        p_hi = (np.sum(vals >= o) + 1) / (len(vals) + 1)
        p_lo = (np.sum(vals <= o) + 1) / (len(vals) + 1)
        out.append(
            {
                "metric": m,
                "observed": o,
                "null_mean": float(np.nanmean(vals)),
                "null_sd": float(np.nanstd(vals)),
                "z": float((o - np.nanmean(vals)) / (np.nanstd(vals) + 1e-12)),
                "p_two_sided": float(min(1.0, 2 * min(p_hi, p_lo))),
            }
        )
    res = pd.DataFrame(out)
    res.attrs["null_draws"] = null
    return res


def class_specificity(
    bundle_dir: str,
    classes: Sequence[str] = ("LumA", "LumB"),
    genes_full: Optional[Sequence[str]] = None,
) -> dict:
    """Solapamiento de aristas/genes entre los dos expertos de clase."""
    frames = {
        c: load_retained_frame(bundle_dir, c, genes_full=genes_full) for c in classes
    }
    a, b = classes[0], classes[1]
    ea = set(frames[a]["ukey"].tolist())
    eb = set(frames[b]["ukey"].tolist())
    na = set(np.unique(np.concatenate([frames[a]["u"], frames[a]["v"]])).tolist())
    nb = set(np.unique(np.concatenate([frames[b]["u"], frames[b]["v"]])).tolist())

    def _jac(x, y):
        return len(x & y) / max(len(x | y), 1)

    out = {
        "frames": frames,
        "edges": {a: len(ea), b: len(eb), "shared": len(ea & eb),
                  f"{a}_only": len(ea - eb), f"{b}_only": len(eb - ea),
                  "jaccard": _jac(ea, eb)},
        "nodes": {a: len(na), b: len(nb), "shared": len(na & nb),
                  f"{a}_only": len(na - nb), f"{b}_only": len(nb - na),
                  "jaccard": _jac(na, nb)},
    }
    if genes_full is not None:
        g = np.asarray([str(x) for x in genes_full], dtype=object)
        out["genes_exclusive"] = {
            a: sorted(g[list(na - nb)].tolist()),
            b: sorted(g[list(nb - na)].tolist()),
        }
    return out


def gate_vs_prior(retained: pd.DataFrame) -> pd.DataFrame:
    """¿El gate aprendido reordena el prior o solo lo copia?

    Correlación de Spearman entre |peso prior| y logit/gate aprendido, global
    y por tipo de arista.
    """
    from scipy.stats import spearmanr

    if "learned_logit" not in retained.columns and "learned_gate" not in retained.columns:
        raise KeyError(
            "El frame no trae learned_logit/learned_gate. Usa edges_{clase}.csv "
            "de un bundle reciente."
        )
    score_col = "learned_logit" if "learned_logit" in retained.columns else "learned_gate"
    w = retained.get("prior_weight")
    if w is None:
        raise KeyError("Falta prior_weight en el frame retenido.")

    rows = []
    groups = [("all", retained)] + [
        (ETYPE_LABELS.get(int(t), f"type {t}"), retained[retained["edge_type"] == t])
        for t in sorted(retained["edge_type"].unique())
    ]
    for name, sub in groups:
        if len(sub) < 5 or sub[score_col].isna().all():
            continue
        r, p = spearmanr(np.abs(sub["prior_weight"]), sub[score_col], nan_policy="omit")
        rows.append({"group": name, "n": len(sub), "spearman_rho": float(r),
                     "p_value": float(p), "score": score_col})
    return pd.DataFrame(rows)


def centrality_table(
    G: nx.Graph,
    top: int = 40,
    betweenness_k: Optional[int] = 300,
) -> pd.DataFrame:
    """Hubs del subgrafo retenido (grado, betweenness aprox., PageRank)."""
    U = G.to_undirected() if G.is_directed() else G
    deg = dict(U.degree())
    k = None if U.number_of_nodes() <= (betweenness_k or 0) else betweenness_k
    btw = nx.betweenness_centrality(U, k=k, seed=0) if U.number_of_nodes() > 2 else {}
    try:
        pr = nx.pagerank(U)
    except Exception:
        pr = {n: np.nan for n in U.nodes()}
    df = pd.DataFrame(
        {
            "gene": list(U.nodes()),
            "degree": [deg.get(n, 0) for n in U.nodes()],
            "betweenness": [btw.get(n, np.nan) for n in U.nodes()],
            "pagerank": [pr.get(n, np.nan) for n in U.nodes()],
        }
    )
    if G.is_directed():
        df["in_degree"] = [G.in_degree(n) for n in df["gene"]]
        df["out_degree"] = [G.out_degree(n) for n in df["gene"]]
    return df.sort_values("degree", ascending=False).head(int(top)).reset_index(drop=True)
