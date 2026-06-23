"""
Visualization: pruned subgraph plotting, components grid,
graph colored by axes, dotplot, barplot.
"""

import math
import os
import random
import re
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from .axes import (
    AXES, AXIS_COLORS, AXIS_PRIORITY,
    map_genes_to_axes, _blend_hex, _darken_hex, _lighten_hex,
)

# Edge-type palette compatible with viz.py
# 0=HuRI (PPI), 1=OmniPath activation, 2=OmniPath inhibition
_ETYPE_COLOR = {
    0: "#4393c3",
    1: "#1a9850",
    2: "#d73027",
}
_ETYPE_LABEL = {
    0: "HuRI (PPI)",
    1: "OmniPath activation",
    2: "OmniPath inhibition",
}

def clean_pathway_label(label: str) -> str:
    s = re.sub(r"\s*\((?:GO|R-HSA|hsa|WP|KEGG)[^)]*\)\s*$", "", str(label))
    s = re.sub(r"\s*Homo sapiens\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()

# ── Pruned subgraph plot ───────────────────────


def plot_pruned_subgraph_all_components(edge_index_p, edge_weight_p, genes_kegg=None,
                                        label_topk=25, seed=7, title="Pruned subgraph (ALL components)"):
    ei = edge_index_p.detach().cpu().numpy() if torch.is_tensor(edge_index_p) else np.asarray(edge_index_p)
    ew = edge_weight_p.detach().cpu().numpy() if torch.is_tensor(edge_weight_p) else np.asarray(edge_weight_p)

    G = nx.Graph()
    # Importante: añadimos nodos explícitamente para no perder ninguno
    active_nodes = np.unique(ei)
    G.add_nodes_from([int(n) for n in active_nodes])

    for u, v, w in zip(ei[0], ei[1], ew):
        u = int(u); v = int(v)
        if u == v:
            continue
        G.add_edge(u, v, weight=float(w))

    #comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)

    # Layout por componente + desplazamiento
    pos = {}


    return G, pos, comps

# (USO) Ejemplo: descomenta y ejecuta cuando tengas edge_index_p/edge_weight_p
# G_all, pos_all, comps_all = plot_pruned_subgraph_all_components(edge_index_p,edge_weight_p)



# ── Components grid ──────────────────────────────



def plot_components_grid(
    G,
    genes,
    ncols=5,
    seed=7,
    font_size=8,
    label_topk=None,
):
    comps = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)
    n = len(comps)
    nrows = int(math.ceil(n / ncols))

    fig = plt.figure(figsize=(ncols*4, nrows*4))
    for i, comp in enumerate(comps):
        ax = fig.add_subplot(nrows, ncols, i+1)
        sub = G.subgraph(comp).copy()
        pos = nx.spring_layout(sub, seed=seed)

        nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.3, width=0.8)
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=30)

        if label_topk is None:
            label_nodes = list(sub.nodes())
        else:
            degs = dict(sub.degree())
            label_nodes = [k for k,_ in sorted(degs.items(), key=lambda x: x[1], reverse=True)[:int(label_topk)]]

        labels = {int(nid): genes[int(nid)] for nid in label_nodes}
        nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=font_size)

        ax.set_title(f"Comp {i+1} | n={len(comp)}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
# plot_components_grid(G_all, genes=genes_model)


# ── Graph colored by axes (nb3) ─────────────────


# ============================================================
# CONFIG
# ============================================================
OUT_FIG = "./figures_axes_refined/fig_axis_graph_colored_by_axes.png"

BUNDLE_PATH = "./artifacts_ablation/FULL/seed_1234/graph_bundle.pt"   # cambia si lo mueves
DRAW_DIRECTED = True                       # mejor False para limpio
SHOW_GIANT_COMPONENT = False                 # dibuja solo el componente mayor
SEED_LAYOUT = 42
RANDOM_SEED = 42

FIGSIZE = (18, 14)
DPI = 300
SAVE_FIG = True
OUT_DOTPLOT   = "./figures_axes_refined/dotplot_enrichment.png"
OUT_BARPLOT   = "./figures_axes_refined/barplot_enrichment.png"

# Etiquetas
LABEL_MODE = "topk_per_axis"                # "none" | "topk_per_axis"
TOPK_LABELS_PER_AXIS = 3
TOPK_LABELS_RESIDUAL = 8
ALWAYS_LABEL_MULTI_AXIS = True

# Nodos
BASE_NODE_SIZE = 120
DEGREE_SIZE_SCALE = 38
RESIDUAL_NODE_SCALE = 0.90
NODE_ALPHA_AXIS = 0.96
NODE_ALPHA_RESIDUAL = 0.72
NODE_EDGEWIDTH = 0.9
NODE_EDGEWIDTH_MULTI = 2.0

# Aristas
EDGE_ALPHA_INTRA = 0.28
EDGE_ALPHA_CROSS = 0.18
EDGE_ALPHA_TO_RESIDUAL = 0.10
EDGE_ALPHA_RESIDUAL = 0.06

EDGE_WIDTH_MIN = 0.6
EDGE_WIDTH_MAX = 2.3

# Layout
MODULE_RADIUS = 4.0           # separación de módulos
MODULE_JITTER = 0.45          # dispersión inicial dentro de módulo
SPRING_ITER = 350

# ============================================================
# EJES
# ============================================================
AXES = {
    "Luminal Hormonal": [
        "ESR1","PGR","GATA3","GREB1","ESR2","PHLDA1","LRIG1","RXRA"
    ],
    "Cell Cycle Mitotic": [
        "AURKA","AURKB","BUB1","CDC7","CDC23","CDCA3","CDK1","CDK2",
        "CEP55","E2F1","FOXM1","PLK1","TTK","ZWINT"
    ],
    "HER2 RTK MAPK": [
        "ERBB2","EGFR","IGF1R","MET","PDGFRB","AKT1","MAP2K1","MAP2K2",
        "MAPK1","MAPK3","MAPK14","PLCG1","SRC","LYN"
    ],
    "Basal Plasticity TNBC": [
        "KRT6A","KRT16","SOX10","VGLL1","VGLL3","EGFR","EMP1","MSLN"
    ],
    "Immune Lymphoid Signaling": [
        "CD79A","CXCL9","DEF6","GRAP2","HCK","JAK3","ITPKB","PAX5",
        "SYK","S1PR1","SOCS1","TRAF1","TRAF2","ZBP1"
    ],
    "DNA Damage p53 Checkpoint": [
        "ATM","ERCC3","FANCG","KAT5","PRKDC","TP53","MDM2","PTEN",
        "DAPK1","STK11","SIRT1"
    ],
    "Adhesion Cytoskeleton Invasion": [
        "ABI2","FLNA","ITGB1","ITGB1BP1","MMP2","PTK2","RHOA","ROCK1",
        "LASP1","NCK1","NCK2","SDCBP"
    ],
    "Androgen Apocrine": [
        "AR","MSLN"
    ],
}
AXIS_PRIORITY = [
    "Luminal Hormonal",
    "Cell Cycle Mitotic",
    "HER2 RTK MAPK",
    "Basal Plasticity TNBC",
    "Immune Lymphoid Signaling",
    "DNA Damage p53 Checkpoint",
    "Adhesion Cytoskeleton Invasion",
    "Androgen Apocrine",
]

AXIS_COLORS = {
    "Luminal Hormonal": "#d62728",               # rojo
    "Cell Cycle Mitotic": "#ff7f0e",             # naranja
    "HER2 RTK MAPK": "#1f77b4",                  # azul
    "Basal Plasticity TNBC": "#9467bd",          # morado
    "Immune Lymphoid Signaling": "#2ca02c",      # verde
    "DNA Damage p53 Checkpoint": "#8c564b",      # marrón
    "Adhesion Cytoskeleton Invasion": "#17becf", # cian
    "Androgen Apocrine": "#e377c2",              # rosa
    "Residual": "#bdbdbd",                       # gris
}

# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# ============================================================
# HELPERS
# ============================================================
def hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb01_to_hex(rgb):
    rgb = tuple(int(max(0, min(1, c)) * 255) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def blend_hex(c1, c2, w=0.5):
    r1, g1, b1 = hex_to_rgb01(c1)
    r2, g2, b2 = hex_to_rgb01(c2)
    rgb = (
        (1 - w) * r1 + w * r2,
        (1 - w) * g1 + w * g2,
        (1 - w) * b1 + w * b2,
    )
    return rgb01_to_hex(rgb)

def lighten_hex(c, amount=0.20):
    r, g, b = hex_to_rgb01(c)
    rgb = (
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )
    return rgb01_to_hex(rgb)

def darken_hex(c, amount=0.15):
    r, g, b = hex_to_rgb01(c)
    rgb = (r * (1 - amount), g * (1 - amount), b * (1 - amount))
    return rgb01_to_hex(rgb)

def safe_tensor_to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# ============================================================
# BLOQUE INTERACTIVO — solo se ejecuta como script, no al importar
# ============================================================
if __name__ == "__main__":
  bundle = torch.load(BUNDLE_PATH, map_location="cpu",weights_only=False)

  genes = bundle["genes_comp"]
  edge_index = safe_tensor_to_numpy(bundle["edge_index_comp"])
  edge_weight = safe_tensor_to_numpy(bundle.get("edge_weight_comp", None))

  if isinstance(genes, np.ndarray):
      genes = genes.tolist()
  genes = [str(g) for g in genes]

  assert edge_index.shape[0] == 2, f"edge_index_comp debe ser (2, E), recibido {edge_index.shape}"

  print(f"Nodos bundle: {len(genes)}")
  print(f"Aristas raw: {edge_index.shape[1]}")

  # ============================================================
  # MAPEO gene -> ejes
  # ============================================================
  gene_to_axes = {}
  for axis, gset in AXES.items():
      for g in gset:
          gene_to_axes.setdefault(g, []).append(axis)

  def assign_axis(gene):
      axes_here = gene_to_axes.get(gene, [])
      if not axes_here:
          return "Residual"
      for ax in AXIS_PRIORITY:
          if ax in axes_here:
              return ax
      return axes_here[0]

  node_axis = {g: assign_axis(g) for g in genes}
  multi_axis_genes = {g for g, axs in gene_to_axes.items() if len(axs) > 1 and g in genes}

  if multi_axis_genes:
      print("\nGenes multi-eje:")
      for g in sorted(multi_axis_genes):
          print(f"  {g}: {gene_to_axes[g]} -> asignado a {node_axis[g]}")

  # ============================================================
  # CONSTRUCCIÓN DEL GRAFO
  # ============================================================
  G = nx.DiGraph() if DRAW_DIRECTED else nx.Graph()
  G.add_nodes_from(genes)

  # Si hay edge_weight, usarlo; si no, 1.0
  if edge_weight is not None and edge_weight.ndim == 1 and len(edge_weight) == edge_index.shape[1]:
      ew = edge_weight.astype(float)
      # normalización robusta a [0,1]
      ew_abs = np.abs(ew)
      if ew_abs.max() > ew_abs.min():
          ew_norm = (ew_abs - ew_abs.min()) / (ew_abs.max() - ew_abs.min())
      else:
          ew_norm = np.ones_like(ew_abs)
  else:
      ew_norm = np.ones(edge_index.shape[1], dtype=float)

  for k, (u_idx, v_idx) in enumerate(edge_index.T):
      u = genes[int(u_idx)]
      v = genes[int(v_idx)]
      if u == v:
          continue
      G.add_edge(u, v, weight=float(ew_norm[k]))

  print(f"Grafo inicial: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

  # componente gigante para visual
  if SHOW_GIANT_COMPONENT:
      if DRAW_DIRECTED:
          comp_nodes = max(nx.weakly_connected_components(G), key=len)
      else:
          comp_nodes = max(nx.connected_components(G), key=len)
      G = G.subgraph(comp_nodes).copy()
      print(f"Componente gigante: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

  # ============================================================
  # LAYOUT: inicializar por módulos en círculo y luego refinar
  # ============================================================
  module_order = AXIS_PRIORITY + ["Residual"]
  n_modules = len(module_order)
  centers = {}

  for i, axis in enumerate(module_order):
      theta = 2 * math.pi * i / n_modules
      centers[axis] = np.array([MODULE_RADIUS * math.cos(theta), MODULE_RADIUS * math.sin(theta)])

  pos_init = {}
  for g in G.nodes():
      ax = node_axis[g]
      center = centers[ax]
      jitter = np.random.normal(loc=0.0, scale=MODULE_JITTER, size=2)
      pos_init[g] = center + jitter

  # peso para spring_layout
  for u, v in G.edges():
      w = G[u][v].get("weight", 1.0)
      # un poco más de "atracción" si están en el mismo módulo
      if node_axis[u] == node_axis[v] and node_axis[u] != "Residual":
          spring_w = 1.5 + 1.5 * w
      else:
          spring_w = 0.5 + 1.0 * w
      G[u][v]["spring_weight"] = spring_w

  pos = nx.spring_layout(
      G,
      pos=pos_init,
      seed=SEED_LAYOUT,
      iterations=SPRING_ITER,
      weight="spring_weight",
      k=1.15 / np.sqrt(max(1, G.number_of_nodes()))
  )

  # ============================================================
  # ESTILOS DE NODOS
  # ============================================================
  deg = dict(G.degree())

  def node_size(g):
      size = BASE_NODE_SIZE + DEGREE_SIZE_SCALE * deg[g]
      if node_axis[g] == "Residual":
          size *= RESIDUAL_NODE_SCALE
      return size

  # ============================================================
  # CLASIFICACIÓN DE ARISTAS Y COLORES
  # ============================================================
  edges_intra = {ax: [] for ax in AXIS_PRIORITY}
  edges_cross = []
  edges_to_residual = []
  edges_residual = []

  for u, v, d in G.edges(data=True):
      au = node_axis[u]
      av = node_axis[v]
      w = d.get("weight", 1.0)
      width = EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * float(w)

      if au == av:
          if au == "Residual":
              edges_residual.append((u, v, width))
          else:
              edges_intra[au].append((u, v, width))
      else:
          if au == "Residual" or av == "Residual":
              edges_to_residual.append((u, v, width))
          else:
              mixed = blend_hex(AXIS_COLORS[au], AXIS_COLORS[av], 0.5)
              mixed = darken_hex(mixed, 0.05)
              edges_cross.append((u, v, width, mixed))

  # ============================================================
  # DIBUJO
  # ============================================================
  plt.figure(figsize=FIGSIZE)

  # ----- aristas residuales puras
  if edges_residual:
      nx.draw_networkx_edges(
          G, pos,
          edgelist=[(u, v) for u, v, _ in edges_residual],
          width=[w for _, _, w in edges_residual],
          edge_color="#d9d9d9",
          alpha=EDGE_ALPHA_RESIDUAL,
          arrows=DRAW_DIRECTED,
          arrowsize=10 if DRAW_DIRECTED else 0,
          connectionstyle="arc3,rad=0.03" if DRAW_DIRECTED else "arc3"
      )

  # ----- aristas hacia residuales
  if edges_to_residual:
      nx.draw_networkx_edges(
          G, pos,
          edgelist=[(u, v) for u, v, _ in edges_to_residual],
          width=[w for _, _, w in edges_to_residual],
          edge_color="#b0b0b0",
          alpha=EDGE_ALPHA_TO_RESIDUAL,
          arrows=DRAW_DIRECTED,
          arrowsize=10 if DRAW_DIRECTED else 0,
          connectionstyle="arc3,rad=0.03" if DRAW_DIRECTED else "arc3"
      )

  # ----- aristas entre módulos
  if edges_cross:
      nx.draw_networkx_edges(
          G, pos,
          edgelist=[(u, v) for u, v, _, _ in edges_cross],
          width=[w for _, _, w, _ in edges_cross],
          edge_color=[c for _, _, _, c in edges_cross],
          alpha=EDGE_ALPHA_CROSS,
          arrows=DRAW_DIRECTED,
          arrowsize=10 if DRAW_DIRECTED else 0,
          connectionstyle="arc3,rad=0.05" if DRAW_DIRECTED else "arc3"
      )

  # ----- aristas intra-módulo (por eje, con color propio)
  for ax in AXIS_PRIORITY:
      e = edges_intra[ax]
      if not e:
          continue
      edge_col = lighten_hex(AXIS_COLORS[ax], 0.10)
      nx.draw_networkx_edges(
          G, pos,
          edgelist=[(u, v) for u, v, _ in e],
          width=[w for _, _, w in e],
          edge_color=edge_col,
          alpha=EDGE_ALPHA_INTRA,
          arrows=DRAW_DIRECTED,
          arrowsize=10 if DRAW_DIRECTED else 0,
          connectionstyle="arc3,rad=0.03" if DRAW_DIRECTED else "arc3"
      )

  # ----- nodos residuales
  residual_nodes = [g for g in G.nodes() if node_axis[g] == "Residual"]
  if residual_nodes:
      nx.draw_networkx_nodes(
          G, pos,
          nodelist=residual_nodes,
          node_color=AXIS_COLORS["Residual"],
          node_size=[node_size(g) for g in residual_nodes],
          alpha=NODE_ALPHA_RESIDUAL,
          linewidths=0.7,
          edgecolors="#8c8c8c"
      )

  # ----- nodos por eje
  for ax in AXIS_PRIORITY:
      nodelist = [g for g in G.nodes() if node_axis[g] == ax and g not in multi_axis_genes]
      if not nodelist:
          continue
      nx.draw_networkx_nodes(
          G, pos,
          nodelist=nodelist,
          node_color=AXIS_COLORS[ax],
          node_size=[node_size(g) for g in nodelist],
          alpha=NODE_ALPHA_AXIS,
          linewidths=NODE_EDGEWIDTH,
          edgecolors="black"
      )

  # ----- overlay para genes multi-eje (borde dorado)
  multi_nodes_in_graph = [g for g in G.nodes() if g in multi_axis_genes]
  if multi_nodes_in_graph:
      nx.draw_networkx_nodes(
          G, pos,
          nodelist=multi_nodes_in_graph,
          node_color=[AXIS_COLORS[node_axis[g]] for g in multi_nodes_in_graph],
          node_size=[node_size(g) * 1.10 for g in multi_nodes_in_graph],
          alpha=1.0,
          linewidths=NODE_EDGEWIDTH_MULTI,
          edgecolors="#FFD700"  # dorado
      )

  # ============================================================
  # ETIQUETAS
  # ============================================================
  labels = {}

  if LABEL_MODE == "topk_per_axis":
      # top hubs por eje
      for ax in AXIS_PRIORITY:
          genes_axis = [g for g in G.nodes() if node_axis[g] == ax]
          genes_axis = sorted(genes_axis, key=lambda x: deg[x], reverse=True)[:TOPK_LABELS_PER_AXIS]
          for g in genes_axis:
              labels[g] = g

      # top residuales
      res_top = sorted(residual_nodes, key=lambda x: deg[x], reverse=True)[:TOPK_LABELS_RESIDUAL]
      for g in res_top:
          labels[g] = g

  if ALWAYS_LABEL_MULTI_AXIS:
      for g in multi_nodes_in_graph:
          labels[g] = g

  if labels:
      nx.draw_networkx_labels(
          G, pos,
          labels=labels,
          font_size=8,
          font_weight="bold",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
      )

  # ============================================================
  # LEYENDA
  # ============================================================
  legend_nodes = []
  for ax in AXIS_PRIORITY + ["Residual"]:
      n_ax = sum(1 for g in G.nodes() if node_axis[g] == ax)
      legend_nodes.append(
          Line2D(
              [0], [0],
              marker="o",
              color="w",
              markerfacecolor=AXIS_COLORS[ax],
              markeredgecolor="black" if ax != "Residual" else "#8c8c8c",
              markersize=10,
              label=f"{ax} (n={n_ax})"
          )
      )

  legend_extra = [
      Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
             markeredgecolor="#FFD700", markeredgewidth=2.0, markersize=10,
             label="Multi-axis gene"),
      Line2D([0], [0], color="#666666", lw=2, alpha=EDGE_ALPHA_CROSS,
             label="Inter-module edge"),
      Line2D([0], [0], color="#b0b0b0", lw=2, alpha=EDGE_ALPHA_TO_RESIDUAL,
             label="Edge to residual"),
  ]

  plt.legend(
      handles=legend_nodes + legend_extra,
      loc="upper left",
      bbox_to_anchor=(1.02, 1.0),
      borderaxespad=0.0,
      frameon=False,
      fontsize=10
  )

  # ============================================================
  # TÍTULO Y SALIDA
  # ============================================================
  title = "Subgraph colored by biological axis"
  if SHOW_GIANT_COMPONENT:
      title += " (componente gigante)"
  plt.title(title, fontsize=17, pad=18)

  plt.axis("off")
  plt.tight_layout()

  if SAVE_FIG:
      plt.savefig(OUT_FIG, dpi=DPI, bbox_inches="tight")
      print(f"Figura guardada en: {os.path.abspath(OUT_FIG)}")

  plt.show()

  # ============================================================
  # RESUMEN AUXILIAR
  # ============================================================
  axis_membership = pd.DataFrame({
      "gene": list(G.nodes()),
      "axis_assigned": [node_axis[g] for g in G.nodes()],
      "degree": [deg[g] for g in G.nodes()],
      "multi_axis": [g in multi_axis_genes for g in G.nodes()],
      "all_axes": [", ".join(gene_to_axes.get(g, [])) if g in gene_to_axes else "" for g in G.nodes()],
  }).sort_values(["axis_assigned", "degree"], ascending=[True, False])

  display(axis_membership.head(20))


# ── Dotplot & Barplot ─────────────────────────────

# ============================================================
# DOTPLOT
# ============================================================
def make_dotplot(
    df,
    out_path,
    source_colors,
    *,
    label_col="label",
    source_col="source",
    gene_ratio_col="gene_ratio",
    pval_col="neg_log10_p",
    size_col="intersection_size",
    figsize=(12, None),
    dpi=180,
    clean_labels=True,
    title="Pathway Enrichment · Dotplot",
    xlabel="Gene Ratio (intersection / query)",
    colorbar_label="−log₁₀(adj. p-value)",
    size_legend_title="Genes overlap",
    source_legend_title="Source",
    cmap_colors=("#9BD7F4", "#6C8EBF", "#7B61A8", "#D55E00"),
    facecolor="white",
    axes_facecolor="white",
    grid_color="#D9D9D9",
    spine_color="#B0B0B0",
    text_color="black",
    edgecolor_points="white",
    save=True,
    fontsize_label=12,
    fontsize_tick=10,
    fontsize_ytick=10,
    fontsize_legend=10,
    fontsize_xlabel=12,
    fontsize_title=14,
    panel_label=None,
    panel_label_x=-0.02,
    panel_label_y=1.02,
    panel_label_fontsize=18,
    panel_label_bold=True,
):
    df_plot = df.copy()

    if clean_labels:
        df_plot[label_col] = df_plot[label_col].map(clean_pathway_label)

    n_rows = len(df_plot)
    fig_h = max(6, n_rows * 0.48) if figsize[1] is None else figsize[1]
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(axes_facecolor)

    cmap = LinearSegmentedColormap.from_list("enrich_clean", list(cmap_colors))
    norm = plt.Normalize(df_plot[pval_col].min(), df_plot[pval_col].max())

    s_min, s_max = 60, 500
    s_range = df_plot[size_col].max() - df_plot[size_col].min()

    dot_sizes = (
        np.full(len(df_plot), (s_min + s_max) / 2, dtype=float)
        if s_range == 0
        else (
            s_min
            + (df_plot[size_col].values - df_plot[size_col].min()) / s_range * (s_max - s_min)
        ).astype(float)
    )

    sc = ax.scatter(
        df_plot[gene_ratio_col].values,
        df_plot[label_col].values,
        c=df_plot[pval_col].values,
        s=dot_sizes,
        cmap=cmap,
        norm=norm,
        edgecolors=edgecolor_points,
        linewidths=0.5,
        alpha=0.95,
        zorder=3,
    )

    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color=grid_color, zorder=0)
    ax.set_axisbelow(True)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label(colorbar_label, color=text_color, fontsize=fontsize_label, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=text_color, fontsize=fontsize_tick)
    cbar.outline.set_edgecolor(spine_color)

    # leyenda de tamaños
    q_vals = np.percentile(df_plot[size_col], [25, 50, 75]).astype(int)
    for cnt in np.unique(q_vals):
        if s_range == 0:
            s = (s_min + s_max) / 2
        else:
            s = s_min + (cnt - df_plot[size_col].min()) / (s_range + 1e-9) * (s_max - s_min)

        ax.scatter(
            [], [], s=float(s), c="#999999",
            edgecolors="black", linewidths=0.3,
            label=f"n={cnt}", alpha=0.85
        )

    sz_leg = ax.legend(
        loc="upper right",
        fontsize=fontsize_legend,
        frameon=False,
        facecolor="white",
        edgecolor=spine_color,
        title=size_legend_title,
        title_fontsize=8,
    )
    ax.add_artist(sz_leg)

    # colorear etiquetas según fuente
    for tick, src in zip(ax.get_yticklabels(), df_plot[source_col].values):
        tick.set_color(source_colors.get(src, "#444444"))
        tick.set_fontsize(fontsize_ytick)

    # leyenda de fuentes
    source_patches = [
        mpatches.Patch(color=source_colors.get(s, "#888888"), label=s)
        for s in pd.unique(df_plot[source_col])
    ]
    src_leg = ax.legend(
        handles=source_patches,
        loc="lower right",
        fontsize=fontsize_legend,
        frameon=False,
        facecolor="white",
        edgecolor=spine_color,
        title=source_legend_title,
        title_fontsize=8,
    )

    ax.set_xlabel(xlabel, color=text_color, fontsize=fontsize_xlabel, labelpad=8)
    ax.set_title(title, color=text_color, fontsize=fontsize_title, fontweight="bold", pad=14, loc="left")
    ax.tick_params(colors=text_color, labelsize=fontsize_tick)

    for sp in ax.spines.values():
        sp.set_edgecolor(spine_color)

    plt.tight_layout()
    if panel_label:
        ax.text(panel_label_x, panel_label_y, panel_label,
                transform=ax.transAxes,
                fontsize=panel_label_fontsize,
                fontweight="bold" if panel_label_bold else "normal",
                va="bottom", ha="right")
    if save:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.close()
    print(f"✅ Dotplot → {out_path}")


# ============================================================
# BARPLOT
# ============================================================
def make_barplot(
    df,
    out_path,
    source_colors,
    *,
    fdr_cutoff=0.05,
    label_col="label",
    source_col="source",
    pval_col="neg_log10_p",
    size_col="intersection_size",
    figsize=(12, None),
    dpi=180,
    clean_labels=True,
    title="Pathway Enrichment · Barplot",
    xlabel="−log₁₀(adj. p-value)",
    source_legend_title="Source",
    show_counts=True,
    facecolor="white",
    axes_facecolor="white",
    grid_color="#D9D9D9",
    spine_color="#B0B0B0",
    text_color="black",
    fdr_line_color="#D55E00",
    save=True,
    fontsize_counts=10,
    fontsize_ytick=10,
    fontsize_xtick=10,
    fontsize_legend=10,
    fontsize_xlabel=14,
    fontsize_title=14,
    panel_label=None,
    panel_label_x=-0.02,
    panel_label_y=1.02,
    panel_label_fontsize=18,
    panel_label_bold=True,
):
    df_plot = df.copy()

    if clean_labels:
        df_plot[label_col] = df_plot[label_col].map(clean_pathway_label)

    n_rows = len(df_plot)
    fig_h = max(6, n_rows * 0.48) if figsize[1] is None else figsize[1]
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(axes_facecolor)

    bar_colors = [source_colors.get(s, "#888888") for s in df_plot[source_col].values]

    bars = ax.barh(
        df_plot[label_col].values,
        df_plot[pval_col].values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.72,
        alpha=0.9,
    )

    x_max = df_plot[pval_col].max()

    if show_counts:
        for bar, n in zip(bars, df_plot[size_col].values):
            ax.text(
                bar.get_width() + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"n={n}",
                va="center",
                ha="left",
                color="#555555",
                fontsize=fontsize_counts,
            )

    fdr_line = -np.log10(fdr_cutoff)
    ax.axvline(
        fdr_line,
        color=fdr_line_color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label=f"FDR = {fdr_cutoff}"
    )

    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color=grid_color, zorder=0)
    ax.set_axisbelow(True)

    for tick, src in zip(ax.get_yticklabels(), df_plot[source_col].values):
        tick.set_color(source_colors.get(src, "#444444"))
        tick.set_fontsize(fontsize_ytick)

    source_patches = [
        mpatches.Patch(color=source_colors.get(s, "#888888"), label=s)
        for s in pd.unique(df_plot[source_col])
    ]
    leg = ax.legend(
        handles=source_patches,
        loc="lower right",
        fontsize=fontsize_legend,
        frameon=False,
        facecolor="white",
        edgecolor=spine_color,
        title=source_legend_title,
        title_fontsize=8,
    )

    ax.set_xlabel(xlabel, color=text_color, fontsize=fontsize_xlabel, labelpad=8)
    ax.set_title(title, color=text_color, fontsize=fontsize_title, fontweight="bold", pad=14, loc="left")
    ax.tick_params(colors=text_color, labelsize=fontsize_xtick)

    for sp in ax.spines.values():
        sp.set_edgecolor(spine_color)

    plt.tight_layout()
    if panel_label:
        ax.text(panel_label_x, panel_label_y, panel_label,
                transform=ax.transAxes,
                fontsize=panel_label_fontsize,
                fontweight="bold" if panel_label_bold else "normal",
                va="bottom", ha="right")
    if save:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.close()
    print(f"✅ Barplot → {out_path}")





# ── draw_global_graph_from_bundle ──────────────

# ============================================================
# GRAFO DIRIGIDO "BONITO" Y CONFIGURABLE (NetworkX + matplotlib)
#   - Flechas dirigidas
#   - Color/tamaño nodos parametrizable
#   - Color/grosor aristas parametrizable
#   - Labels top-N (para evitar saturación)
# ============================================================

from typing import Optional, Dict, Any

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    mn, mx = np.nanmin(v), np.nanmax(v)
    if np.isclose(mx - mn, 0.0):
        return np.zeros_like(v)
    return (v - mn) / (mx - mn + eps)

def _safe_rank_top(values, topk):
    values = np.asarray(values)
    if values.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(-values)  # desc
    return order[:min(int(topk), len(order))]

def _pick_first(d: Dict[str, Any], keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

# ------------------------------------------------------------
# Extraer GRAFO GLOBAL desde tu bundle (único grafo)
# ------------------------------------------------------------
def extract_global_graph_from_bundle(bundle):
    """
    Devuelve:
      edge_index: (2, E) en espacio compacto/global
      edge_weight: (E,) o None
      genes: lista de genes (long N)
      edge_type / edge_sign: opcional (si existe)
    """
    g = bundle["graph"]

    genes = list(g.get("genes_comp", []))
    if len(genes) == 0:
        # fallback raro
        genes = list(g.get("genes", []))
    if len(genes) == 0:
        raise ValueError("No encuentro genes en bundle['graph']['genes_comp'] ni ['genes'].")

    edge_index = _pick_first(g, [
        "edge_index_compact", "edge_index_comp", "edge_index", "edge_index_pruned"
    ])
    edge_weight = _pick_first(g, [
        "edge_weight_compact", "edge_weight", "edge_attr", "edge_score"
    ])

    # si existe info de tipo/signo de arista, la cogemos (opcional)
    # Buscar también en el nivel superior del bundle (edge_type_compact, etc.)
    edge_type = _pick_first(g, [
        "edge_type_compact", "edge_type_comp", "edge_type",
        "edge_types", "edge_channel", "edge_type_global"
    ])
    if edge_type is None and isinstance(bundle, dict):
        edge_type = _pick_first(bundle, [
            "edge_type_compact", "edge_type_comp", "edge_type",
            "edge_types", "edge_channel", "edge_type_global"
        ])
    edge_sign = _pick_first(g, [
        "edge_sign_compact", "edge_sign_comp", "edge_sign",
        "signed_edge", "edge_polarity"
    ])
    if edge_sign is None and isinstance(bundle, dict):
        edge_sign = _pick_first(bundle, [
            "edge_sign_compact", "edge_sign_comp", "edge_sign",
            "signed_edge", "edge_polarity"
        ])

    if edge_index is None:
        raise ValueError("No encuentro edge_index en bundle['graph'].")

    edge_index = _to_np(edge_index).astype(np.int64)
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T

    if edge_weight is not None:
        edge_weight = _to_np(edge_weight).astype(float).reshape(-1)

    if edge_type is not None:
        edge_type = _to_np(edge_type).reshape(-1)

    if edge_sign is not None:
        edge_sign = _to_np(edge_sign).reshape(-1)

    return edge_index, edge_weight, genes, edge_type, edge_sign

# ------------------------------------------------------------
# Construir DiGraph dirigido
# ------------------------------------------------------------
def build_directed_nx_graph(
    edge_index,
    genes,
    edge_weight=None,
    edge_type=None,
    edge_sign=None,
    remove_self_loops=True,
    aggregate_parallel="sum_abs",   # "sum_abs" | "last"
):
    """
    edge_index: (2,E), índices [0..N-1]
    genes: lista de genes (N)
    edge_weight: (E,) opcional
    edge_type: (E,) opcional (0=HuRI, 1=OP+, 2=OP-)
    edge_sign: (E,) opcional (ej: -1/+1)
    """
    edge_index = _to_np(edge_index).astype(np.int64)
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T

    N = len(genes)
    G = nx.DiGraph()

    for i, g in enumerate(genes):
        G.add_node(i, gene=str(g))

    E = edge_index.shape[1]
    if edge_weight is None:
        edge_weight = np.ones(E, dtype=float)
    else:
        edge_weight = _to_np(edge_weight).astype(float).reshape(-1)

    if edge_type is not None:
        edge_type = _to_np(edge_type).reshape(-1)

    if edge_sign is not None:
        edge_sign = _to_np(edge_sign).reshape(-1)

    for e in range(E):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])

        if remove_self_loops and u == v:
            continue
        if not (0 <= u < N and 0 <= v < N):
            continue

        w = float(edge_weight[e]) if e < len(edge_weight) else 1.0
        et = None
        if edge_type is not None and e < len(edge_type):
            try:
                et = int(edge_type[e])
            except Exception:
                et = None

        s = None
        if edge_sign is not None and e < len(edge_sign):
            try:
                s = float(edge_sign[e])
            except Exception:
                s = None

        if G.has_edge(u, v):
            if aggregate_parallel == "sum_abs":
                G[u][v]["weight"] = float(G[u][v].get("weight", 0.0)) + abs(w)
                if et is not None and "etype" not in G[u][v]:
                    G[u][v]["etype"] = et
                if s is not None:
                    # si hay conflicto de signo, lo dejamos a 0 (desconocido/mixto)
                    prev_s = G[u][v].get("sign", None)
                    if prev_s is None:
                        G[u][v]["sign"] = s
                    elif np.sign(prev_s) != np.sign(s):
                        G[u][v]["sign"] = 0.0
            else:
                G[u][v]["weight"] = abs(w)
                if et is not None:
                    G[u][v]["etype"] = et
                if s is not None:
                    G[u][v]["sign"] = s
        else:
            G.add_edge(u, v, weight=abs(w))
            if et is not None:
                G[u][v]["etype"] = et
            if s is not None:
                G[u][v]["sign"] = s

    return G

# ------------------------------------------------------------
# Dibujo dirigido configurable
# ------------------------------------------------------------
def draw_directed_gene_graph(
    G: nx.DiGraph,
    *,
    # --- nodo (métrica) ---
    node_score: Optional[np.ndarray] = None,      # len N, ej importancia Grad×Input
    node_score_name: str = "Node score",
    node_color_mode: str = "score",               # "score" | "constant" | "degree"
    node_color_constant: str = "skyblue",
    node_cmap = plt.cm.viridis,
    node_vmin: Optional[float] = None,
    node_vmax: Optional[float] = None,

    node_size_mode: str = "score",                # "score" | "constant" | "degree"
    node_size_constant: float = 200.0,
    node_size_base: float = 80.0,
    node_size_scale: float = 900.0,

    node_alpha: float = 0.95,
    node_edgecolor: str = "black",
    node_linewidth: float = 0.4,

    # --- labels ---
    top_labels: int = 30,                         # cuántos nombres mostrar
    label_fontsize: int = 9,
    label_fontweight: str = "bold",
    label_fontcolor: str = "black",
    label_bbox_alpha: float = 0.65,
    force_label_genes: Optional[List[str]] = None,  # genes que siempre se etiquetan

    # --- highlight ---                             # Añadimos estellitas
    highlight_genes: Optional[List[str]] = None,
    highlight_marker: str = "*",
    highlight_size: float = 350.0,
    highlight_color: str = "#FFD700",   # dorado
    highlight_edgecolor: str = "white",
    highlight_linewidth: float = 1.2,
    highlight_zorder: int = 10,
    highlight_offset_frac: float = 0.025,

    # --- aristas (métrica) ---
    edge_color_mode: str = "etype",               # "etype" | "weight" | "constant" | "sign"
    edge_color_constant: str = "gray",
    edge_cmap = plt.cm.plasma,
    edge_vmin: Optional[float] = None,
    edge_vmax: Optional[float] = None,

    edge_width_mode: str = "weight",              # "weight" | "constant"
    edge_width_constant: float = 1.0,
    edge_width_min: float = 1.0,
    edge_width_max: float = 4.5,

    edge_alpha: float = 0.55,
    edge_pos_color: str = "tab:red",              # si edge_color_mode="sign"
    edge_neg_color: str = "tab:blue",
    edge_zero_color: str = "gray",

    # --- flechas dirigidas ---
    arrows: bool = True,
    arrowstyle: str = "-|>",
    arrowsize: int = 18,
    connectionstyle: str = "arc3,rad=0.12",       # como viz.py
    min_source_margin: int = 6,
    min_target_margin: int = 6,

    # --- layout ---
    layout: str = "spring",                       # "spring" | "kamada" | "circular" | "shell" | "fr"
    seed: int = 42,
    k: Optional[float] = None,
    iterations: int = 300,

    # --- figura ---
    title: Optional[str] = None,
    figsize=(14, 12),
    facecolor: str = "white",
    show_colorbar_nodes: bool = True,
    show_colorbar_edges: bool = False,
    annotate_stats: bool = True,

    # --- output ---
    save_path: Optional[str] = None,
    dpi: int = 220,
):
    if G.number_of_nodes() == 0:
        print("Grafo vacío.")
        return None, None

    nodes = list(G.nodes())
    N = len(nodes)
    edges = list(G.edges())
    M = len(edges)

    # -------------------------
    # Scores de nodos
    # -------------------------
    if node_score is None:
        deg = np.array([G.out_degree(n) + G.in_degree(n) for n in nodes], dtype=float)
        node_score_arr = deg
        if node_score_name == "Node score":
            node_score_name = "In+Out degree"
    else:
        node_score_arr = _to_np(node_score).astype(float).reshape(-1)
        if len(node_score_arr) != N:
            raise ValueError(f"node_score len={len(node_score_arr)} != N={N}")

    node_score_norm = _normalize(node_score_arr)

    # Pre-compute axis mapping if any mode needs it
    _node_axis = _multi_axis = _gene_names_ax = None
    if node_color_mode == "axis" or edge_color_mode == "axis":
        _gene_names_ax = [str(G.nodes[n].get("gene", n)) for n in nodes]
        _node_axis, _gene_to_axes, _multi_axis = map_genes_to_axes(_gene_names_ax)

    # Color nodos
    if node_color_mode == "constant":
        node_colors = node_color_constant
        node_mappable = None
    elif node_color_mode == "degree":
        deg = np.array([G.out_degree(n) + G.in_degree(n) for n in nodes], dtype=float)
        vals = deg
        node_colors = vals
        node_mappable = vals
        if node_score_name == "Node score":
            node_score_name = "Degree"
    elif node_color_mode == "axis":
        node_colors = [AXIS_COLORS.get(_node_axis[g], "#bdbdbd") for g in _gene_names_ax]
        node_mappable = None
    else:  # "score"
        vals = node_score_arr
        node_colors = vals
        node_mappable = vals

    # Tamaño nodos
    if node_size_mode == "constant":
        node_sizes = np.full(N, float(node_size_constant))
    elif node_size_mode == "degree":
        deg = np.array([G.out_degree(n) + G.in_degree(n) for n in nodes], dtype=float)
        dnorm = _normalize(deg)
        node_sizes = node_size_base + node_size_scale * (0.15 + 0.85 * dnorm)
    else:  # "score"
        node_sizes = node_size_base + node_size_scale * (0.15 + 0.85 * node_score_norm)

    # -------------------------
    # Métricas de aristas
    # -------------------------
    edge_weights = np.array([float(G[u][v].get("weight", 1.0)) for u, v in edges], dtype=float) if M > 0 else np.array([])
    edge_weights_norm = _normalize(edge_weights) if M > 0 else np.array([])

    edge_signs = None
    if M > 0:
        edge_signs = np.array([G[u][v].get("sign", np.nan) for u, v in edges], dtype=float)

    e_types = None
    if M > 0:
        e_types = np.array([G[u][v].get("etype", -1) for u, v in edges], dtype=int)

    # Color aristas
    if edge_color_mode == "constant":
        edge_colors = edge_color_constant
        edge_mappable = None
    elif edge_color_mode == "etype":
        if e_types is not None and np.any(e_types >= 0):
            edge_colors = [_ETYPE_COLOR.get(int(et), "#888888") for et in e_types]
        else:
            edge_colors = edge_color_constant
        edge_mappable = None
    elif edge_color_mode == "sign":
        # rojo = positivo, azul = negativo, gris = 0/NaN
        edge_colors = []
        for s in edge_signs:
            if np.isnan(s) or s == 0:
                edge_colors.append(edge_zero_color)
            elif s > 0:
                edge_colors.append(edge_pos_color)
            else:
                edge_colors.append(edge_neg_color)
        edge_mappable = None
    elif edge_color_mode == "axis":
        _n2g = {n: str(G.nodes[n].get("gene", n)) for n in nodes}
        edge_colors = []
        for u, v in edges:
            au = _node_axis.get(_n2g[u], "Residual")
            av = _node_axis.get(_n2g[v], "Residual")
            if au == av:
                if au == "Residual":
                    edge_colors.append("#888888")
                else:
                    edge_colors.append(_lighten_hex(AXIS_COLORS[au], 0.10))
            elif au == "Residual" or av == "Residual":
                non_res = au if av == "Residual" else av
                edge_colors.append(_lighten_hex(AXIS_COLORS.get(non_res, "#888888"), 0.30))
            else:
                edge_colors.append(_darken_hex(_blend_hex(AXIS_COLORS[au], AXIS_COLORS[av], 0.5), 0.05))
        edge_mappable = None
    else:  # "weight"
        edge_colors = edge_weights if M > 0 else "gray"
        edge_mappable = edge_weights if M > 0 else None

    # Grosor aristas
    if edge_width_mode == "constant":
        edge_widths = np.full(M, float(edge_width_constant)) if M > 0 else []
    else:
        edge_widths = edge_width_min + (edge_width_max - edge_width_min) * edge_weights_norm if M > 0 else []

    # -------------------------
    # Layout
    # -------------------------
    if layout.lower() in ["kamada", "kamada_kawai", "kk"]:
        pos = nx.kamada_kawai_layout(G, weight="weight")
    elif layout.lower() in ["circular", "circle"]:
        pos = nx.circular_layout(G)
    elif layout.lower() in ["shell"]:
        pos = nx.shell_layout(G)
    elif layout.lower() in ["fr", "fruchterman"]:
        pos = nx.fruchterman_reingold_layout(G, seed=seed)
    else:  # spring
        if k is None:
            k = 1.6 / np.sqrt(max(N, 2))
        pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations, weight="weight")

    # -------------------------
    # Figura
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(facecolor)

    # Aristas (estilo viz.py: agrupadas por tipo cuando existe etype)
    edge_artist = None
    use_viz_etype_style = (
        arrows and isinstance(G, nx.DiGraph) and
        e_types is not None and np.any(e_types >= 0) and
        edge_color_mode == "etype"
    )

    if use_viz_etype_style:
        types_present = sorted(set(int(et) for et in e_types if int(et) >= 0))
        for et in types_present:
            mask = e_types == et
            elist = [edges[i] for i in range(len(edges)) if mask[i]]
            widths = np.asarray(edge_widths)[mask]
            color = _ETYPE_COLOR.get(et, "#888888")
            edge_artist = nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=elist,
                width=widths,
                alpha=edge_alpha,
                edge_color=color,
                arrows=True,
                arrowsize=arrowsize,
                arrowstyle="-|>",
                node_size=node_sizes,
                connectionstyle=connectionstyle,
                min_source_margin=min_source_margin,
                min_target_margin=min_target_margin,
            )
    else:
        edge_artist = nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=edge_alpha,
            arrows=arrows,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            connectionstyle=connectionstyle,
            min_source_margin=min_source_margin,
            min_target_margin=min_target_margin,
            node_size=node_sizes,   # ayuda a que la flecha no se meta dentro del nodo
        )

    # Nodos
    nodes_artist = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=nodes,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=node_cmap if node_color_mode not in ("constant", "axis") else None,
        vmin=node_vmin,
        vmax=node_vmax,
        alpha=node_alpha,
        linewidths=node_linewidth,
        edgecolors=node_edgecolor,
    )

    # Multi-axis golden border overlay
    if node_color_mode == "axis" and _multi_axis:
        _n2g_m = {n: str(G.nodes[n].get("gene", n)) for n in nodes}
        _multi_nodes = [n for n in nodes if _n2g_m[n] in _multi_axis]
        if _multi_nodes:
            _mi = [nodes.index(n) for n in _multi_nodes]
            nx.draw_networkx_nodes(
                G, pos, ax=ax, nodelist=_multi_nodes,
                node_size=[float(node_sizes[i]) * 1.10 for i in _mi],
                node_color=[AXIS_COLORS.get(_node_axis[_n2g_m[n]], "#bdbdbd") for n in _multi_nodes],
                alpha=1.0, linewidths=2.0, edgecolors="#FFD700",
            )

    # Labels top-N
    # criterio por score si existe, si no degree
    label_metric = node_score_arr
    idx_top = _safe_rank_top(label_metric, top_labels)
    labels = {}
    for i in idx_top:
        n = nodes[i]
        labels[n] = str(G.nodes[n].get("gene", n))

    # Añadir genes forzados aunque no estén en top-N
    if force_label_genes:
        gene_to_node_lbl = {G.nodes[n].get("gene", n): n for n in nodes}
        for g in force_label_genes:
            if g in gene_to_node_lbl and gene_to_node_lbl[g] not in labels:
                labels[gene_to_node_lbl[g]] = g

    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=label_fontsize,
        font_weight=label_fontweight,
        font_color=label_fontcolor,
    )

    # ── Estrellas en genes destacados ──
    if highlight_genes:
        gene_to_node = {G.nodes[n].get("gene", n): n for n in nodes}
        hl_nodes = [gene_to_node[g] for g in highlight_genes if g in gene_to_node]
        if hl_nodes:
            hl_xy = np.array([pos[n] for n in hl_nodes])
            # offset vertical proporcional al rango del layout
            y_range = hl_xy[:, 1].ptp() if len(pos) < 2 else np.ptp([p[1] for p in pos.values()])
            offset_y = y_range * highlight_offset_frac          # ← ajusta este factor a gusto
            ax.scatter(
                hl_xy[:, 0]+0.03, hl_xy[:, 1] + offset_y,
                marker=highlight_marker,
                s=highlight_size,
                facecolors=highlight_color,
                edgecolors=highlight_edgecolor,
                linewidths=highlight_linewidth,
                zorder=highlight_zorder,
            )


    # Título / texto
    if title is None:
        title = f"Grafo dirigido (N={N}, E={M})"
    ax.set_title(title, fontsize=14, pad=12)
    ax.axis("off")

    if annotate_stats:
        txt = (
            f"Nodos={N} | Aristas={M}\n"
            f"Labels mostrados={len(labels)}\n"
            f"NodeColor={node_color_mode} | NodeSize={node_size_mode}\n"
            f"EdgeColor={edge_color_mode} | EdgeWidth={edge_width_mode}"
        )
        ax.text(
            0.01, 0.01, txt,
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="lightgray")
        )

    # Leyenda de tipos de arista como en viz.py
    if use_viz_etype_style:
        types_in_graph = sorted(set(int(et) for et in e_types if int(et) >= 0))
        patches = [
            mpatches.Patch(color=_ETYPE_COLOR[t], label=_ETYPE_LABEL.get(t, f"type {t}"))
            for t in types_in_graph if t in _ETYPE_COLOR
        ]
        if patches:
            ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.8)

    # Leyenda de ejes biológicos
    if node_color_mode == "axis" or edge_color_mode == "axis":
        _present_axes = []
        for axn in AXIS_PRIORITY + ["Residual"]:
            _cnt = sum(1 for g in _gene_names_ax if _node_axis.get(g, "Residual") == axn)
            if _cnt > 0:
                _present_axes.append((axn, _cnt))
        _handles = [
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=AXIS_COLORS.get(axn, "#bdbdbd"),
                   markeredgecolor="black" if axn != "Residual" else "#8c8c8c",
                   markersize=9, label=f"{axn} (n={cnt})")
            for axn, cnt in _present_axes
        ]
        if _multi_axis:
            _handles.append(Line2D([0],[0], marker="o", color="w",
                markerfacecolor="white", markeredgecolor="#FFD700",
                markeredgewidth=2.0, markersize=9, label="Multi-axis gene"))
        ax.legend(handles=_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0, frameon=True, fontsize=9, framealpha=0.85)

    # Colorbar nodos
    if show_colorbar_nodes and (node_color_mode != "constant") and (node_mappable is not None):
        cbar_n = plt.colorbar(nodes_artist, ax=ax, fraction=0.035, pad=0.01)
        cbar_n.set_label(node_score_name)

    # Colorbar aristas (solo si edge_color_mode="weight")
    if show_colorbar_edges and (edge_color_mode == "weight") and (edge_mappable is not None) and (M > 0):
        # Creamos mappable manual para colorbar de aristas
        sm = plt.cm.ScalarMappable(cmap=edge_cmap)
        vmin = np.nanmin(edge_weights) if edge_vmin is None else edge_vmin
        vmax = np.nanmax(edge_weights) if edge_vmax is None else edge_vmax
        sm.set_clim(vmin, vmax)
        cbar_e = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.06)
        cbar_e.set_label("Edge weight")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

    plt.show()
    return fig, ax

# ------------------------------------------------------------
# Wrapper rápido desde bundle (grafo único)
# ------------------------------------------------------------
def draw_global_graph_from_bundle(
    bundle,
    node_score_global=None,      # opcional: importancia por gen en espacio genes_comp
    title="Grafo global dirigido",
    **kwargs
):
    edge_index, edge_weight, genes, edge_type, edge_sign = extract_global_graph_from_bundle(bundle)

    # si no hay edge_sign pero sí edge_type y es binario/±1, lo usamos como signo
    sign = edge_sign
    if sign is None and edge_type is not None:
        et = _to_np(edge_type)
        # Heurística: si solo hay 2-3 valores y parecen signo
        vals = np.unique(et[~pd.isna(et)] if "pd" in globals() else et)
        if len(vals) <= 3:
            sign = et

    G = build_directed_nx_graph(
        edge_index=edge_index,
        genes=genes,
        edge_weight=edge_weight,
        edge_type=edge_type,
        edge_sign=sign,
        remove_self_loops=True,
        aggregate_parallel="sum_abs",
    )

    return draw_directed_gene_graph(
        G,
        node_score=node_score_global,
        title=title,
        **kwargs
    )


# ------------------------------------------------------------
# Helper: inspeccionar conexiones de un nodo/gen
# ------------------------------------------------------------

def _etype_to_label(et):
    try:
        et = int(et)
    except Exception:
        return "unknown"
    return {
        0: "HuRI PPI",
        1: "activation",
        2: "inhibition",
    }.get(et, f"type_{et}")


def _sign_to_label(sign, etype=None):
    # prioriza etype si está disponible
    try:
        if etype is not None:
            et = int(etype)
            if et == 1:
                return "activation"
            if et == 2:
                return "inhibition"
            if et == 0:
                return "PPI"
    except Exception:
        pass

    if sign is None:
        return "unknown"
    try:
        s = float(sign)
    except Exception:
        return "unknown"
    if s > 0:
        return "activation"
    if s < 0:
        return "inhibition"
    return "unknown"


def get_node_connections_from_bundle(bundle, node_gene: str, sort_by: str = "abs_weight", include_self_loops: bool = False):
    """
    Devuelve un DataFrame con todas las aristas incidentes a `node_gene`,
    separando si son entrantes o salientes.

    Columnas:
      source, target, queried_gene, neighbor, direction, edge_type,
      interaction, sign_label, weight
    """
    edge_index, edge_weight, genes, edge_type, edge_sign = extract_global_graph_from_bundle(bundle)

    genes = [str(g) for g in genes]
    if node_gene not in genes:
        raise ValueError(f"'{node_gene}' no está en genes_comp. Ejemplos: {genes[:10]}")

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    q = gene_to_idx[node_gene]

    E = edge_index.shape[1]
    if edge_weight is None:
        edge_weight = np.ones(E, dtype=float)
    else:
        edge_weight = _to_np(edge_weight).astype(float).reshape(-1)

    if edge_type is not None:
        edge_type = _to_np(edge_type).reshape(-1)
    if edge_sign is not None:
        edge_sign = _to_np(edge_sign).reshape(-1)

    rows = []
    for e in range(E):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])
        if (not include_self_loops) and u == v:
            continue
        if u != q and v != q:
            continue

        et = None
        if edge_type is not None and e < len(edge_type):
            try:
                et = int(edge_type[e])
            except Exception:
                et = None

        sg = None
        if edge_sign is not None and e < len(edge_sign):
            try:
                sg = float(edge_sign[e])
            except Exception:
                sg = None

        direction = "outgoing" if u == q else "incoming"
        neighbor_idx = v if u == q else u

        rows.append({
            "source": genes[u],
            "target": genes[v],
            "queried_gene": node_gene,
            "neighbor": genes[neighbor_idx],
            "direction": direction,
            "edge_type": et,
            "interaction": _etype_to_label(et),
            "sign_label": _sign_to_label(sg, etype=et),
            "weight": float(edge_weight[e]) if e < len(edge_weight) else 1.0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if sort_by == "abs_weight":
        df = df.assign(abs_weight=df["weight"].abs()).sort_values(
            ["direction", "abs_weight", "neighbor"], ascending=[True, False, True]
        ).drop(columns=["abs_weight"])
    elif sort_by in df.columns:
        df = df.sort_values(["direction", sort_by, "neighbor"], ascending=[True, False, True])
    else:
        df = df.sort_values(["direction", "neighbor"])

    df = df.reset_index(drop=True)
    return df


def print_node_connections_from_bundle(bundle, node_gene: str, max_rows: Optional[int] = None):
    """Imprime de forma compacta las conexiones entrantes/salientes de un gen."""
    df = get_node_connections_from_bundle(bundle, node_gene=node_gene)
    if df.empty:
        print(f"[node] {node_gene}: sin conexiones incidentes en el bundle.")
        return df

    print(f"\n[node] {node_gene} | conexiones={len(df)}")
    for direction in ["incoming", "outgoing"]:
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        print(f"\n  {direction.upper()} ({len(sub)}):")
        sub_it = sub if max_rows is None else sub.head(int(max_rows))
        for _, r in sub_it.iterrows():
            print(
                f"    {r['source']} -> {r['target']} | neighbor={r['neighbor']} | "
                f"type={r['interaction']} | sign={r['sign_label']} | weight={r['weight']:.4f}"
            )
    return df
