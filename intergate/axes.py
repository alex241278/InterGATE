"""
Biological axes: definitions, score computation, OVR analysis, and plotting.
"""

import math
import re
import copy
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

def _signed_formatter(x, _):
    """Format colorbar ticks with explicit +/− sign (Unicode minus for symmetry)."""
    if x > 0:
        return f"+{x:.1f}"
    elif x < 0:
        return f"\u2212{abs(x):.1f}"
    return f"{x:.1f}"


# ── Axes definitions & palettes ────────────────────

import re

# =========================
# 1) Definir ejes biológicos
# =========================
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

# =========================
# 2) Configuración global de figuras
# =========================
SUBTYPE_PALETTE = {
    "Normal": "#6C8EBF",
    "LumA":   "#4CAF50",
    "LumB":   "#E69F00",
    "HER2":   "#D55E00",
    "TNBC":   "#7B61A8",
}

PLOT_DEFAULTS = {
    "figure_size": (10, 6),   # mismo tamaño para TODAS las figuras
    "dpi": 150,
    "bbox_inches": "tight",
    "class_order": ["Normal", "LumA", "LumB", "HER2", "TNBC"],
    "class_palette": copy.deepcopy(SUBTYPE_PALETTE),
    "boxplot_palette": copy.deepcopy(SUBTYPE_PALETTE),
    "embedding_palette": copy.deepcopy(SUBTYPE_PALETTE),
    "mean_heatmap_cmap": "coolwarm",
    "corr_cmap": "coolwarm",
    "one_vs_rest_cmap": "coolwarm",
    "default_save_dir": "./figures_axes_refined",
    # ── Font sizes (used as defaults in all plot functions) ──
    "title_fontsize": 14,
    "axis_label_fontsize": 12,
    "tick_fontsize": 11,
    "annot_fontsize": 9,
    "legend_fontsize": 11,
    "colorbar_label_fontsize": 11,
    "colorbar_tick_fontsize": 10,
    "scatter_size": 18,
}





# ── Helpers, score computation, plotting ────────────


# =========================
# 3) Helpers
# =========================
def _to_samples_x_genes(expr_df: pd.DataFrame, meta_df: pd.DataFrame, sample_col="sample"):
    """
    Devuelve expresión en formato samples x genes.
    Detecta si expr viene genes x samples o samples x genes.
    """
    if sample_col not in meta_df.columns:
        raise ValueError(f"meta_df no contiene la columna '{sample_col}'")

    meta_samples = set(meta_df[sample_col].astype(str))

    idx_overlap = len(set(expr_df.index.astype(str)) & meta_samples)
    col_overlap = len(set(expr_df.columns.astype(str)) & meta_samples)

    if idx_overlap >= col_overlap and idx_overlap > 0:
        X = expr_df.copy()
        X.index = X.index.astype(str)
        X.columns = X.columns.astype(str)
        orientation = "samples_x_genes"
    elif col_overlap > 0:
        X = expr_df.T.copy()
        X.index = X.index.astype(str)
        X.columns = X.columns.astype(str)
        orientation = "genes_x_samples -> transposed"
    else:
        raise ValueError("No se pudieron alinear muestras entre expr y meta (ni por índice ni por columnas).")

    X = X.apply(pd.to_numeric, errors="coerce")
    return X, orientation


def _dedup_gene_columns(X: pd.DataFrame, how="mean"):
    """
    Si hay genes duplicados como columnas, agregarlos.
    """
    if not X.columns.duplicated().any():
        return X
    if how == "mean":
        return X.T.groupby(level=0).mean().T
    elif how == "median":
        return X.T.groupby(level=0).median().T
    elif how == "max":
        return X.T.groupby(level=0).max().T
    else:
        return X.loc[:, ~X.columns.duplicated(keep="first")]


def _zscore_by_gene(X: pd.DataFrame):
    """
    Z-score por gen (columna) across samples.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0, np.nan)
    Z = (X - mu) / sd
    return Z


def prettify_axis_name(name: str) -> str:
    s = str(name).replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def prettify_axis_list(names):
    return [prettify_axis_name(x) for x in names]


def _get_present_class_order(df: pd.DataFrame, label_col="label", class_order=None):
    if class_order is None:
        class_order = PLOT_DEFAULTS["class_order"]
    classes_present = set(df[label_col].astype(str))
    return [c for c in class_order if c in classes_present]


def _resolve_palette(class_order, palette=None, fallback_cmap="Set2"):
    if palette is None:
        palette = PLOT_DEFAULTS["class_palette"]
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        return {cl: cmap(i / max(1, len(class_order) - 1)) for i, cl in enumerate(class_order)}
    if isinstance(palette, dict):
        return {cl: palette.get(cl, plt.get_cmap(fallback_cmap)(i / max(1, len(class_order) - 1)))
                for i, cl in enumerate(class_order)}
    palette = list(palette)
    return {cl: palette[i % len(palette)] for i, cl in enumerate(class_order)}


def _finalize_figure(fig, save_dir=None, save_name=None, dpi=None, bbox_inches=None):
    dpi = dpi or PLOT_DEFAULTS["dpi"]
    bbox_inches = bbox_inches or PLOT_DEFAULTS["bbox_inches"]
    if save_dir is not None and save_name is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / save_name
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to: {out_path.resolve()}")


def _bw(bold_flag: bool) -> str:
    """Return matplotlib fontweight string."""
    return "bold" if bold_flag else "normal"


def _xtick_offset(align: str) -> float:
    """Return tick position offset: -0.5 (start of cell), 0 (center), +0.5 (end)."""
    if align == "start":
        return -0.5
    elif align == "end":
        return 0.5
    return 0.0


def _brighten_color(color, factor: float = 1.0):
    """Brighten (factor>1) or darken (factor<1) a color. Returns RGBA tuple."""
    import colorsys
    r, g, b, a = to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, max(0.0, l * factor))
    s = min(1.0, max(0.0, s * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2, a)


def _brighten_palette(palette_map: dict, factor: float = 1.0) -> dict:
    """Apply brightness factor to all colors in a palette dict."""
    if factor == 1.0:
        return palette_map
    return {k: _brighten_color(v, factor) for k, v in palette_map.items()}


def _brighten_cmap(cmap, factor: float = 1.0):
    """Return a brightened version of a matplotlib colormap."""
    if factor == 1.0:
        if isinstance(cmap, str):
            return plt.get_cmap(cmap)
        return cmap
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    colors = [_brighten_color(base(i / 255), factor) for i in range(256)]
    return LinearSegmentedColormap.from_list(f"{base.name}_bright", colors, N=256)


def _add_panel_label(fig, label: Optional[str], fontsize: int = 18,
                     bold: bool = True, x: float = 0.01, y: float = 0.98):
    """Add a panel label (A, B, C …) at a custom position in figure coords."""
    if label:
        fig.text(
            x, y, label,
            fontsize=fontsize, fontweight=_bw(bold),
            va="top", ha="left",
            transform=fig.transFigure,
        )


# =========================
# 4) Calcular scores por eje
# =========================
def compute_axis_scores(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    axes: dict,
    sample_col="sample",
    label_col="label",
    min_genes_per_axis=2,
    restrict_labels=("LumA","LumB","HER2","TNBC","Normal","normal")
):
    meta = meta_df.copy()
    meta[sample_col] = meta[sample_col].astype(str)

    if label_col in meta.columns and restrict_labels is not None:
        meta = meta[meta[label_col].astype(str).isin(set(map(str, restrict_labels)))].copy()

    X, orientation = _to_samples_x_genes(expr_df, meta, sample_col=sample_col)

    common = [s for s in meta[sample_col] if s in X.index]
    meta = meta[meta[sample_col].isin(common)].copy()
    meta = meta.drop_duplicates(subset=[sample_col], keep="first")
    meta = meta.set_index(sample_col, drop=False).loc[common].reset_index(drop=True)
    X = X.loc[common].copy()

    X = _dedup_gene_columns(X, how="mean")
    Xz = _zscore_by_gene(X)

    axis_scores = pd.DataFrame(index=Xz.index)
    axis_info_rows = []

    col_upper = pd.Index([str(c).upper() for c in Xz.columns])
    upper_to_real = {}
    for real, up in zip(Xz.columns, col_upper):
        if up not in upper_to_real:
            upper_to_real[up] = real

    for axis_name, genes in axes.items():
        genes_u = [g.upper() for g in genes]
        present_u = [g for g in genes_u if g in upper_to_real]
        missing_u = [g for g in genes_u if g not in upper_to_real]
        present_real = [upper_to_real[g] for g in present_u]

        if len(present_real) >= min_genes_per_axis:
            axis_scores[axis_name] = Xz[present_real].mean(axis=1, skipna=True)
        else:
            axis_scores[axis_name] = np.nan

        axis_info_rows.append({
            "Axis": axis_name,
            "Axis_pretty": prettify_axis_name(axis_name),
            "n_axis_genes": len(genes),
            "n_present": len(present_real),
            "n_missing": len(missing_u),
            "present_genes": ", ".join(map(str, present_real)),
            "missing_genes": ", ".join(missing_u),
            "coverage": len(present_real) / max(1, len(genes)),
        })

    axis_info = pd.DataFrame(axis_info_rows).sort_values(["n_present","coverage"], ascending=False).reset_index(drop=True)

    meta_scores = meta.copy()
    meta_scores = meta_scores.set_index(sample_col, drop=False)
    meta_scores = meta_scores.join(axis_scores, how="left")
    meta_scores = meta_scores.reset_index(drop=True)

    print(f"Orientación detectada: {orientation}")
    print(f"Muestras usadas: {meta_scores.shape[0]}")
    print(f"Genes en expresión (tras dedup): {Xz.shape[1]}")

    return meta_scores, axis_info, Xz


# =========================
# 5) Visualizaciones
# =========================
def plot_axis_boxplots(
    meta_scores: pd.DataFrame,
    label_col="label",
    axes_order=None,
    figure_size=None,
    palette=None,
    color_brightness: float = 1.0,
    save_dir=None,
    save_prefix="boxplot",
    save_format: str = "svg",
    dpi=None,
    # ── Font sizes (separate x / y) ──
    title_fontsize=None,
    xlabel_fontsize=None,
    ylabel_fontsize=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    # ── Bold flags ──
    bold_title=False,
    bold_xlabel=False,
    bold_ylabel=False,
    bold_xticks=False,
    bold_yticks=False,
    # ── Panel label (A, B, …) ──
    panel_label: Optional[str] = None,
    panel_label_fontsize: int = 18,
    panel_label_bold: bool = True,
    panel_label_x: float = 0.01,
    panel_label_y: float = 0.98,
):
    """
    Un gráfico por eje (matplotlib, sin seaborn), con títulos en inglés
    y cajas coloreadas.
    """
    if axes_order is None:
        axes_order = [c for c in meta_scores.columns if c in AXES]
    tmp = meta_scores.copy()
    tmp[label_col] = tmp[label_col].astype(str).replace({"normal": "Normal"})
    class_order = _get_present_class_order(tmp, label_col=label_col)
    palette_map = _resolve_palette(class_order, palette or PLOT_DEFAULTS["boxplot_palette"])
    palette_map = _brighten_palette(palette_map, color_brightness)
    figure_size = figure_size or PLOT_DEFAULTS["figure_size"]

    _tfs = title_fontsize or PLOT_DEFAULTS["title_fontsize"]
    _xlfs = xlabel_fontsize or PLOT_DEFAULTS.get("axis_label_fontsize", 12)
    _ylfs = ylabel_fontsize or PLOT_DEFAULTS.get("axis_label_fontsize", 12)
    _xtkfs = xtick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _ytkfs = ytick_fontsize or PLOT_DEFAULTS["tick_fontsize"]

    for ax_name in axes_order:
        fig, ax = plt.subplots(figsize=figure_size)
        data = [tmp.loc[tmp[label_col] == cl, ax_name].dropna().values for cl in class_order]
        bp = ax.boxplot(
            data,
            tick_labels=class_order,
            showfliers=False,
            patch_artist=True,
            widths=0.65,
        )
        for patch, cl in zip(bp["boxes"], class_order):
            patch.set_facecolor(palette_map[cl])
            patch.set_alpha(0.85)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)
        for element in ["whiskers", "caps", "medians"]:
            for artist in bp[element]:
                artist.set_color("black")
                artist.set_linewidth(1.0)
        ax.set_xlabel("Subtype", fontsize=_xlfs, fontweight=_bw(bold_xlabel))
        ax.set_ylabel("Axis score (z-mean)", fontsize=_ylfs, fontweight=_bw(bold_ylabel))
        ax.set_title(
            f"{prettify_axis_name(ax_name)} across subtypes",
            fontsize=_tfs, fontweight=_bw(bold_title),
        )
        ax.tick_params(axis="x", rotation=0, labelsize=_xtkfs)
        ax.tick_params(axis="y", labelsize=_ytkfs)
        if bold_xticks:
            for lbl in ax.get_xticklabels():
                lbl.set_fontweight("bold")
        if bold_yticks:
            for lbl in ax.get_yticklabels():
                lbl.set_fontweight("bold")
        _add_panel_label(fig, panel_label, panel_label_fontsize, panel_label_bold, panel_label_x, panel_label_y)
        fig.tight_layout()
        safe_ax_name = ax_name.replace(" ", "_").lower()
        _finalize_figure(
            fig,
            save_dir=save_dir,
            save_name=f"{save_prefix}_{safe_ax_name}.{save_format}" if save_dir is not None else None,
            dpi=dpi,
        )
        plt.show()

def plot_axis_heatmap_by_subtype(
    meta_scores: pd.DataFrame,
    label_col="label",
    axes_order=None,
    figure_size=None,
    cmap=None,
    color_brightness: float = 1.0,
    vmin: float = -1.0,
    vmax: float = 1.0,
    save_dir=None,
    save_name="mean_axis_scores_by_subtype",
    save_format: str = "svg",
    dpi=None,
    title="Mean axis scores by subtype",
    # ── Font sizes (separate x / y) ──
    title_fontsize=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    annot_fontsize=None,
    colorbar_label_fontsize=None,
    colorbar_tick_fontsize=None,
    # ── Bold flags ──
    bold_title=False,
    bold_xticks=False,
    bold_yticks=False,
    bold_annot=False,
    bold_colorbar_label=False,
    # ── Colorbar label position ──
    colorbar_label_rotation: Optional[float] = None,
    colorbar_label_pad: Optional[float] = None,
    # ── X-tick alignment relative to cell: "center", "start", "end" ──
    xtick_align: str = "center",
    # ── Y-axis order (list of subtype names) ──
    y_order: Optional[List[str]] = None,
    # ── Panel label (A, B, …) ──
    panel_label: Optional[str] = None,
    panel_label_fontsize: int = 18,
    panel_label_bold: bool = True,
    panel_label_x: float = 0.01,
    panel_label_y: float = 0.98,
):
    if axes_order is None:
        axes_order = [c for c in meta_scores.columns if c in AXES]

    tmp = meta_scores.copy()
    tmp[label_col] = tmp[label_col].astype(str).replace({"normal":"Normal"})

    if y_order is not None:
        class_order = [c for c in y_order if c in set(tmp[label_col].astype(str))]
    else:
        class_order = _get_present_class_order(tmp, label_col=label_col)

    mat = tmp.groupby(label_col)[axes_order].mean(numeric_only=True)
    mat = mat.reindex(class_order)

    figure_size = figure_size or PLOT_DEFAULTS["figure_size"]
    cmap = cmap or PLOT_DEFAULTS["mean_heatmap_cmap"]
    cmap = _brighten_cmap(cmap, color_brightness)

    _xtkfs = xtick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _ytkfs = ytick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _afs = annot_fontsize or PLOT_DEFAULTS["annot_fontsize"]
    _tfs = title_fontsize or PLOT_DEFAULTS["title_fontsize"]

    fig, ax = plt.subplots(figsize=figure_size)
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    _xoff = _xtick_offset(xtick_align)
    ax.set_xticks([i + _xoff for i in range(len(axes_order))])
    ax.set_xticklabels(prettify_axis_list(axes_order), rotation=35, ha="right",
                       fontsize=_xtkfs, fontweight=_bw(bold_xticks))
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=_ytkfs, fontweight=_bw(bold_yticks))
    if title:
        ax.set_title(title, fontsize=_tfs, fontweight=_bw(bold_title))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=_afs, fontweight=_bw(bold_annot))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    _cbar_kw = dict(
        fontsize=colorbar_label_fontsize or PLOT_DEFAULTS["colorbar_label_fontsize"],
        fontweight=_bw(bold_colorbar_label),
    )
    if colorbar_label_rotation is not None:
        _cbar_kw["rotation"] = colorbar_label_rotation
    if colorbar_label_pad is not None:
        _cbar_kw["labelpad"] = colorbar_label_pad
    cbar.set_label("Mean axis score", **_cbar_kw)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize or PLOT_DEFAULTS["colorbar_tick_fontsize"])
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_signed_formatter))

    _add_panel_label(fig, panel_label, panel_label_fontsize, panel_label_bold, panel_label_x, panel_label_y)
    fig.tight_layout()
    _finalize_figure(fig, save_dir=save_dir, save_name=f"{save_name}.{save_format}" if save_dir is not None else None, dpi=dpi)
    plt.show()

    return mat


def plot_axis_correlation(
    meta_scores: pd.DataFrame,
    axes_order=None,
    figure_size=None,
    cmap=None,
    color_brightness: float = 1.0,
    save_dir=None,
    save_name="correlacion_spearman_ejes",
    save_format: str = "svg",
    dpi=None,
    title="Spearman correlation between axes",
    # ── Font sizes (separate x / y) ──
    title_fontsize=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    annot_fontsize=None,
    colorbar_label_fontsize=None,
    colorbar_tick_fontsize=None,
    # ── Bold flags ──
    bold_title=False,
    bold_xticks=False,
    bold_yticks=False,
    bold_annot=False,
    bold_colorbar_label=False,
    # ── Colorbar label position ──
    colorbar_label_rotation: Optional[float] = None,
    colorbar_label_pad: Optional[float] = None,
    # ── X-tick alignment relative to cell: "center", "start", "end" ──
    xtick_align: str = "center",
    # ── Panel label ──
    panel_label: Optional[str] = None,
    panel_label_fontsize: int = 18,
    panel_label_bold: bool = True,
    panel_label_x: float = 0.01,
    panel_label_y: float = 0.98,
):
    if axes_order is None:
        axes_order = [c for c in meta_scores.columns if c in AXES]

    corr = meta_scores[axes_order].corr(method="spearman")
    figure_size = figure_size or PLOT_DEFAULTS["figure_size"]
    cmap = cmap or PLOT_DEFAULTS["corr_cmap"]
    cmap = _brighten_cmap(cmap, color_brightness)

    _xtkfs = xtick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _ytkfs = ytick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _afs = annot_fontsize or PLOT_DEFAULTS["annot_fontsize"]
    _tfs = title_fontsize or PLOT_DEFAULTS["title_fontsize"]

    fig, ax = plt.subplots(figsize=figure_size)
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    _xoff = _xtick_offset(xtick_align)
    ax.set_xticks([i + _xoff for i in range(len(axes_order))])
    ax.set_xticklabels(prettify_axis_list(axes_order), rotation=35, ha="right",
                       fontsize=_xtkfs, fontweight=_bw(bold_xticks))
    ax.set_yticks(range(len(axes_order)))
    ax.set_yticklabels(prettify_axis_list(axes_order),
                       fontsize=_ytkfs, fontweight=_bw(bold_yticks))
    if title:
        ax.set_title(title, fontsize=_tfs, fontweight=_bw(bold_title))

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=_afs, fontweight=_bw(bold_annot))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    _cbar_kw = dict(
        fontsize=colorbar_label_fontsize or PLOT_DEFAULTS["colorbar_label_fontsize"],
        fontweight=_bw(bold_colorbar_label),
    )
    if colorbar_label_rotation is not None:
        _cbar_kw["rotation"] = colorbar_label_rotation
    if colorbar_label_pad is not None:
        _cbar_kw["labelpad"] = colorbar_label_pad
    cbar.set_label("ρ", **_cbar_kw)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize or PLOT_DEFAULTS["colorbar_tick_fontsize"])
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_signed_formatter))

    _add_panel_label(fig, panel_label, panel_label_fontsize, panel_label_bold, panel_label_x, panel_label_y)
    fig.tight_layout()
    _finalize_figure(fig, save_dir=save_dir, save_name=f"{save_name}.{save_format}" if save_dir is not None else None, dpi=dpi)
    plt.show()

    return corr


def plot_axis_embedding(
    meta_scores: pd.DataFrame,
    label_col="label",
    axes_order=None,
    figure_size=None,
    palette=None,
    color_brightness: float = 1.0,
    save_dir=None,
    pca_save_name="pca_axes_space",
    umap_save_name="umap_axes_space",
    save_format: str = "svg",
    dpi=None,
    pca_title="PCA in axis space",
    umap_title="UMAP in axis space",
    scatter_size=None,
    # ── Font sizes (separate x / y) ──
    title_fontsize=None,
    xlabel_fontsize=None,
    ylabel_fontsize=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    legend_fontsize=None,
    # ── Legend marker size ──
    legend_markersize: float = 8,
    # ── Signed tick format (+/−) on axes ──
    signed_xticks: bool = True,
    signed_yticks: bool = True,
    xtick_decimals: Optional[int] = None,
    ytick_decimals: Optional[int] = None,
    # ── Bold flags ──
    bold_title=False,
    bold_xlabel=False,
    bold_ylabel=False,
    bold_xticks=False,
    bold_yticks=False,
    bold_legend=False,
    # ── Panel labels (one per subplot) ──
    pca_panel_label: Optional[str] = None,
    umap_panel_label: Optional[str] = None,
    panel_label_fontsize: int = 18,
    panel_label_bold: bool = True,
    panel_label_x: float = 0.01,
    panel_label_y: float = 0.98,
):
    """
    PCA (siempre) + UMAP opcional si está instalado.
    Todo en inglés y con leyenda sin marco.
    """
    if axes_order is None:
        axes_order = [c for c in meta_scores.columns if c in AXES]

    tmp = meta_scores.copy()
    tmp[label_col] = tmp[label_col].astype(str).replace({"normal":"Normal"})
    X = tmp[axes_order].copy().fillna(tmp[axes_order].mean())

    class_order = _get_present_class_order(tmp, label_col=label_col)
    palette_map = _resolve_palette(class_order, palette or PLOT_DEFAULTS["embedding_palette"])
    palette_map = _brighten_palette(palette_map, color_brightness)
    figure_size = figure_size or PLOT_DEFAULTS["figure_size"]

    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc.values, full_matrices=False)
    pca2 = U[:, :2] * S[:2]

    _tfs = title_fontsize or PLOT_DEFAULTS["title_fontsize"]
    _xlfs = xlabel_fontsize or PLOT_DEFAULTS["axis_label_fontsize"]
    _ylfs = ylabel_fontsize or PLOT_DEFAULTS["axis_label_fontsize"]
    _xtkfs = xtick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _ytkfs = ytick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _lgfs = legend_fontsize or PLOT_DEFAULTS["legend_fontsize"]
    _ss = scatter_size or PLOT_DEFAULTS["scatter_size"]

    # ── Helper to style an embedding axes ──
    def _style_emb_ax(ax_obj, xl, yl, ttl):
        ax_obj.set_xlabel(xl, fontsize=_xlfs, fontweight=_bw(bold_xlabel))
        ax_obj.set_ylabel(yl, fontsize=_ylfs, fontweight=_bw(bold_ylabel))
        if ttl:
            ax_obj.set_title(ttl, fontsize=_tfs, fontweight=_bw(bold_title))
        ax_obj.tick_params(axis="x", labelsize=_xtkfs)
        ax_obj.tick_params(axis="y", labelsize=_ytkfs)
        # Signed tick format
        if signed_xticks:
            ax_obj.xaxis.set_major_formatter(ticker.FuncFormatter(_signed_formatter))
        if signed_yticks:
            ax_obj.yaxis.set_major_formatter(ticker.FuncFormatter(_signed_formatter))
        # Custom decimal precision (overrides signed formatter if set)
        if xtick_decimals is not None:
            def _xfmt(x, _pos):
                s = f"{x:.{xtick_decimals}f}"
                if signed_xticks and x > 0:
                    s = "+" + s
                elif signed_xticks and x < 0:
                    s = "\u2212" + f"{abs(x):.{xtick_decimals}f}"
                return s
            ax_obj.xaxis.set_major_formatter(ticker.FuncFormatter(_xfmt))
        if ytick_decimals is not None:
            def _yfmt(x, _pos):
                s = f"{x:.{ytick_decimals}f}"
                if signed_yticks and x > 0:
                    s = "+" + s
                elif signed_yticks and x < 0:
                    s = "\u2212" + f"{abs(x):.{ytick_decimals}f}"
                return s
            ax_obj.yaxis.set_major_formatter(ticker.FuncFormatter(_yfmt))
        if bold_xticks:
            for lbl in ax_obj.get_xticklabels():
                lbl.set_fontweight("bold")
        if bold_yticks:
            for lbl in ax_obj.get_yticklabels():
                lbl.set_fontweight("bold")
        leg = ax_obj.legend(frameon=False, fontsize=_lgfs, markerscale=legend_markersize / max(1, _ss**0.5))
        if bold_legend and leg:
            for t in leg.get_texts():
                t.set_fontweight("bold")
        # Set legend marker sizes explicitly
        for handle in leg.legend_handles:
            handle.set_sizes([legend_markersize ** 2])

    fig, ax = plt.subplots(figsize=figure_size)
    for cl in class_order:
        m = (tmp[label_col] == cl).values
        if m.sum() == 0:
            continue
        ax.scatter(pca2[m, 0], pca2[m, 1], label=cl, alpha=0.8, s=_ss,
                   color=palette_map[cl], edgecolors="none")
    _style_emb_ax(ax, "PC1", "PC2", pca_title)
    _add_panel_label(fig, pca_panel_label, panel_label_fontsize, panel_label_bold, panel_label_x, panel_label_y)
    fig.tight_layout()
    _finalize_figure(fig, save_dir=save_dir, save_name=f"{pca_save_name}.{save_format}" if save_dir is not None else None, dpi=dpi)
    plt.show()

    try:
        import umap
        emb = umap.UMAP(n_neighbors=20, min_dist=0.2, random_state=42).fit_transform(X.values)

        fig, ax = plt.subplots(figsize=figure_size)
        for cl in class_order:
            m = (tmp[label_col] == cl).values
            if m.sum() == 0:
                continue
            ax.scatter(emb[m, 0], emb[m, 1], label=cl, alpha=0.8, s=_ss,
                       color=palette_map[cl], edgecolors="none")
        _style_emb_ax(ax, "UMAP1", "UMAP2", umap_title)
        _add_panel_label(fig, umap_panel_label, panel_label_fontsize, panel_label_bold, panel_label_x, panel_label_y)
        fig.tight_layout()
        _finalize_figure(fig, save_dir=save_dir, save_name=f"{umap_save_name}.{save_format}" if save_dir is not None else None, dpi=dpi)
        plt.show()
    except Exception as e:
        print(f"[UMAP opcional no disponible] {e}")


# =========================
# 6) Estadística simple por eje (Kruskal)
# =========================
def axis_stats_by_subtype(meta_scores: pd.DataFrame, label_col="label", axes_order=None):
    try:
        from scipy.stats import kruskal
    except Exception:
        print("scipy no está disponible; salto stats.")
        return None

    if axes_order is None:
        axes_order = [c for c in meta_scores.columns if c in AXES]

    tmp = meta_scores.copy()
    tmp[label_col] = tmp[label_col].astype(str).replace({"normal":"Normal"})
    class_order = [c for c in ["Normal","LumA","LumB","HER2","TNBC"] if c in set(tmp[label_col])]

    rows = []
    for ax_name in axes_order:
        groups = [tmp.loc[tmp[label_col] == cl, ax_name].dropna().values for cl in class_order]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        rows.append({"Axis": ax_name, "Axis_pretty": prettify_axis_name(ax_name), "Kruskal_H": float(H), "p_value": float(p)})

    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    try:
        from statsmodels.stats.multitest import multipletests
        if len(out) > 0:
            out["p_fdr"] = multipletests(out["p_value"].values, method="fdr_bh")[1]
    except Exception:
        pass

    return out


# =========================
# 7) Cobertura de ejes en la firma del grafo
# =========================
def summarize_axis_overlap_with_graph(graph_genes, axes: dict):
    if isinstance(graph_genes, pd.DataFrame):
        if 'gene' not in graph_genes.columns:
            raise ValueError("Si graph_genes es DataFrame, debe tener columna 'gene'.")
        gset = set(graph_genes['gene'].astype(str).str.upper())
    elif isinstance(graph_genes, pd.Series):
        gset = set(graph_genes.astype(str).str.upper())
    else:
        gset = set(map(lambda x: str(x).upper(), graph_genes))

    rows = []
    for ax_name, genes in axes.items():
        present = [g for g in genes if str(g).upper() in gset]
        absent = [g for g in genes if str(g).upper() not in gset]
        rows.append({
            'Axis': ax_name,
            'Axis_pretty': prettify_axis_name(ax_name),
            'n_axis_genes': len(genes),
            'n_in_graph': len(present),
            'coverage_in_graph': len(present) / max(1, len(genes)),
            'present_in_graph': ', '.join(present),
            'missing_from_graph': ', '.join(absent),
        })

    out = pd.DataFrame(rows).sort_values(
        ['coverage_in_graph', 'n_in_graph', 'Axis'],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    return out



# ── One-vs-rest analysis ────────────────────────────


# ============================================================
# One-vs-rest por subtipo para scores de ejes
# - Test: Mann-Whitney U (robusto)
# - Tamaño de efecto:
#     * delta_mean (media subtipo - media resto)
#     * rank_biserial (derivado de U; rango [-1, 1])
# - Ajuste múltiple: FDR BH (global y por subtipo)
# ============================================================

def one_vs_rest_axis_analysis(
    meta_scores: pd.DataFrame,
    label_col: str = "label",
    axes_order=None,
    class_order=("Normal", "LumA", "LumB", "HER2", "TNBC"),
):
    try:
        from scipy.stats import mannwhitneyu
    except Exception as e:
        raise ImportError(f"Necesitas scipy para este análisis. Error: {e}")

    try:
        from statsmodels.stats.multitest import multipletests
        has_statsmodels = True
    except Exception:
        has_statsmodels = False

    df = meta_scores.copy()
    df[label_col] = df[label_col].astype(str).replace({"normal": "Normal"})

    if axes_order is None:
        exclude = {label_col, "sample", "batch", "sample_meta", "sample_norm", "sample_norm_expr", "sample_expr"}
        axes_order = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    present_classes = [c for c in class_order if c in set(df[label_col])]
    rows = []

    for subtype in present_classes:
        df_sub = df[df[label_col] == subtype].copy()
        df_rest = df[df[label_col] != subtype].copy()

        for ax in axes_order:
            x = pd.to_numeric(df_sub[ax], errors="coerce").dropna().values
            y = pd.to_numeric(df_rest[ax], errors="coerce").dropna().values

            if len(x) < 3 or len(y) < 3:
                continue

            U, p = mannwhitneyu(x, y, alternative="two-sided")

            n1 = len(x)
            n0 = len(y)
            auc = U / (n1 * n0)
            rank_biserial = 2 * auc - 1

            mean_sub = float(np.mean(x))
            mean_rest = float(np.mean(y))
            delta_mean = mean_sub - mean_rest

            med_sub = float(np.median(x))
            med_rest = float(np.median(y))
            delta_median = med_sub - med_rest

            s1 = np.std(x, ddof=1)
            s0 = np.std(y, ddof=1)
            sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / max(1, (n1 + n0 - 2)))
            cohen_d = (mean_sub - mean_rest) / (sp + 1e-12)

            rows.append({
                "Subtype": subtype,
                "Axis": ax,
                "Axis_pretty": prettify_axis_name(ax),
                "n_subtype": n1,
                "n_rest": n0,
                "mean_subtype": mean_sub,
                "mean_rest": mean_rest,
                "delta_mean": delta_mean,
                "median_subtype": med_sub,
                "median_rest": med_rest,
                "delta_median": delta_median,
                "U": float(U),
                "p_value": float(p),
                "rank_biserial": float(rank_biserial),
                "AUC_effect": float(auc),
                "cohen_d": float(cohen_d),
                "direction": "up_in_subtype" if delta_mean > 0 else "down_in_subtype",
            })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    if has_statsmodels:
        out["p_fdr_global"] = multipletests(out["p_value"].values, method="fdr_bh")[1]
    else:
        out["p_fdr_global"] = np.nan

    out["p_fdr_subtype"] = np.nan
    if has_statsmodels:
        for subtype in out["Subtype"].unique():
            m = out["Subtype"] == subtype
            out.loc[m, "p_fdr_subtype"] = multipletests(out.loc[m, "p_value"].values, method="fdr_bh")[1]

    out = out.sort_values(["Subtype", "p_fdr_subtype", "p_value", "Axis"]).reset_index(drop=True)

    return out


def summarize_one_vs_rest(
    ovr_df: pd.DataFrame,
    top_n: int = 3,
    sort_by: str = "delta_mean",
):
    if ovr_df is None or ovr_df.empty:
        return pd.DataFrame()

    rows = []
    for subtype in ovr_df["Subtype"].unique():
        d = ovr_df[ovr_df["Subtype"] == subtype].copy()

        d_pos = d.sort_values(sort_by, ascending=False).head(top_n)
        d_neg = d.sort_values(sort_by, ascending=True).head(top_n)

        rows.append({
            "Subtype": subtype,
            "Top_up_axes": " | ".join([f"{r.Axis_pretty} (Δ mean={r.delta_mean:+.3f}, rb={r.rank_biserial:+.3f})" for _, r in d_pos.iterrows()]),
            "Top_down_axes": " | ".join([f"{r.Axis_pretty} (Δ mean={r.delta_mean:+.3f}, rb={r.rank_biserial:+.3f})" for _, r in d_neg.iterrows()]),
        })

    return pd.DataFrame(rows)


def plot_one_vs_rest_heatmap(
    ovr_df: pd.DataFrame,
    value_col="delta_mean",
    fdr_col="p_fdr_subtype",
    figure_size=None,
    cmap=None,
    color_brightness: float = 1.0,
    vmin: float = -1.0,
    vmax: float = 1.0,
    save_dir=None,
    save_name="one_vs_rest_heatmap",
    save_format: str = "svg",
    dpi=None,
    title="One-vs-rest effect sizes by subtype",
    colorbar_label=None,
    # ── Font sizes (separate x / y) ──
    title_fontsize=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    annot_fontsize=None,
    colorbar_label_fontsize=None,
    colorbar_tick_fontsize=None,
    # ── Bold flags ──
    bold_title=False,
    bold_xticks=False,
    bold_yticks=False,
    bold_annot=False,
    bold_colorbar_label=False,
    # ── Colorbar label position ──
    colorbar_label_rotation: Optional[float] = None,
    colorbar_label_pad: Optional[float] = None,
    # ── X-tick alignment relative to cell: "center", "start", "end" ──
    xtick_align: str = "center",
    # ── Y-axis order ──
    y_order: Optional[List[str]] = None,
    # ── Panel label ──
    panel_label: Optional[str] = None,
    panel_label_fontsize: int = 18,
    panel_label_bold: bool = True,
    panel_label_x: float = 0.01,
    panel_label_y: float = 0.98,
):
    if ovr_df is None or ovr_df.empty:
        print("No hay resultados one-vs-rest.")
        return None

    if y_order is not None:
        subtypes = [c for c in y_order if c in set(ovr_df["Subtype"])]
    else:
        subtypes = list(ovr_df["Subtype"].unique())
    axes = list(ovr_df["Axis"].unique())

    mat = ovr_df.pivot(index="Subtype", columns="Axis", values=value_col).reindex(index=subtypes, columns=axes)
    pmat = ovr_df.pivot(index="Subtype", columns="Axis", values=fdr_col).reindex(index=subtypes, columns=axes)

    figure_size = figure_size or PLOT_DEFAULTS["figure_size"]
    cmap = cmap or PLOT_DEFAULTS.get("one_vs_rest_cmap", "coolwarm")
    cmap = _brighten_cmap(cmap, color_brightness)

    if colorbar_label is None:
        if value_col == "delta_mean":
            colorbar_label = "Δ mean score"
        elif value_col == "rank_biserial":
            colorbar_label = "Rank-biserial correlation"
        else:
            colorbar_label = prettify_axis_name(value_col)

    _xtkfs = xtick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _ytkfs = ytick_fontsize or PLOT_DEFAULTS["tick_fontsize"]
    _afs = annot_fontsize or PLOT_DEFAULTS["annot_fontsize"]
    _tfs = title_fontsize or PLOT_DEFAULTS["title_fontsize"]

    fig, ax = plt.subplots(figsize=figure_size)
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    _xoff = _xtick_offset(xtick_align)
    ax.set_xticks([i + _xoff for i in range(len(axes))])
    ax.set_xticklabels(prettify_axis_list(axes), rotation=35, ha="right",
                       fontsize=_xtkfs, fontweight=_bw(bold_xticks))
    ax.set_yticks(range(len(subtypes)))
    ax.set_yticklabels(subtypes, fontsize=_ytkfs, fontweight=_bw(bold_yticks))
    if title:
        ax.set_title(title, fontsize=_tfs, fontweight=_bw(bold_title))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    _cbar_kw = dict(
        fontsize=colorbar_label_fontsize or PLOT_DEFAULTS["colorbar_label_fontsize"],
        fontweight=_bw(bold_colorbar_label),
    )
    if colorbar_label_rotation is not None:
        _cbar_kw["rotation"] = colorbar_label_rotation
    if colorbar_label_pad is not None:
        _cbar_kw["labelpad"] = colorbar_label_pad
    cbar.set_label(colorbar_label, **_cbar_kw)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize or PLOT_DEFAULTS["colorbar_tick_fontsize"])
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_signed_formatter))

    for i in range(len(subtypes)):
        for j in range(len(axes)):
            p = pmat.iloc[i, j]
            if pd.isna(p):
                txt = ""
            elif p < 1e-3:
                txt = "***"
            elif p < 1e-2:
                txt = "**"
            elif p < 5e-2:
                txt = "*"
            else:
                txt = ""
            if txt:
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=_afs, fontweight=_bw(bold_annot))

    _add_panel_label(fig, panel_label, panel_label_fontsize, panel_label_bold, panel_label_x, panel_label_y)
    fig.tight_layout()
    _finalize_figure(fig, save_dir=save_dir, save_name=f"{save_name}.{save_format}" if save_dir is not None else None, dpi=dpi)
    plt.show()

    return mat, pmat


# ═════════════════════════════════════════════════════════════
#  Axis colours & palette
# ═════════════════════════════════════════════════════════════

AXIS_PRIORITY = list(AXES.keys())

AXIS_COLORS = {
    "Luminal Hormonal":              "#d62728",
    "Cell Cycle Mitotic":            "#ff7f0e",
    "HER2 RTK MAPK":                "#1f77b4",
    "Basal Plasticity TNBC":        "#9467bd",
    "Immune Lymphoid Signaling":    "#2ca02c",
    "DNA Damage p53 Checkpoint":    "#8c564b",
    "Adhesion Cytoskeleton Invasion": "#17becf",
    "Androgen Apocrine":            "#e377c2",
    "Residual":                     "#bdbdbd",
}


# ═════════════════════════════════════════════════════════════
#  Colour helpers
# ═════════════════════════════════════════════════════════════

def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb01_to_hex(rgb):
    rgb = tuple(int(max(0, min(1, c)) * 255) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _blend_hex(c1: str, c2: str, w: float = 0.5) -> str:
    r1, g1, b1 = _hex_to_rgb01(c1)
    r2, g2, b2 = _hex_to_rgb01(c2)
    return _rgb01_to_hex(((1-w)*r1 + w*r2, (1-w)*g1 + w*g2, (1-w)*b1 + w*b2))


def _lighten_hex(c: str, amount: float = 0.20) -> str:
    r, g, b = _hex_to_rgb01(c)
    return _rgb01_to_hex((r + (1-r)*amount, g + (1-g)*amount, b + (1-b)*amount))


def _darken_hex(c: str, amount: float = 0.15) -> str:
    r, g, b = _hex_to_rgb01(c)
    return _rgb01_to_hex((r*(1-amount), g*(1-amount), b*(1-amount)))


def _safe_tensor_to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ═════════════════════════════════════════════════════════════
#  Gene → axis mapping
# ═════════════════════════════════════════════════════════════

def map_genes_to_axes(
    genes: List[str],
    axes: Optional[Dict[str, List[str]]] = None,
    priority: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, List[str]], set]:
    """
    Returns
    -------
    node_axis : dict   gene → single assigned axis (or "Residual")
    gene_to_axes : dict  gene → list of all axes it belongs to
    multi_axis_genes : set  genes appearing in ≥ 2 axes
    """
    axes = axes or AXES
    priority = priority or AXIS_PRIORITY

    gene_to_axes: Dict[str, List[str]] = {}
    for axis, gset in axes.items():
        for g in gset:
            gene_to_axes.setdefault(g, []).append(axis)

    def _assign(gene):
        axs = gene_to_axes.get(gene, [])
        if not axs:
            return "Residual"
        for ax in priority:
            if ax in axs:
                return ax
        return axs[0]

    gene_set = set(genes)
    node_axis = {g: _assign(g) for g in genes}
    multi_axis_genes = {g for g, axs in gene_to_axes.items() if len(axs) > 1 and g in gene_set}

    return node_axis, gene_to_axes, multi_axis_genes


# ═════════════════════════════════════════════════════════════
#  Build NX graph from bundle arrays
# ═════════════════════════════════════════════════════════════

def _build_nx_graph_from_arrays(
    genes: List[str],
    edge_index: np.ndarray,
    edge_weight: Optional[np.ndarray],
    directed: bool = True,
    giant_component: bool = False,
) -> nx.Graph:
    """Build NetworkX graph; normalises weights to [0, 1]."""
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(genes)

    if edge_weight is not None and edge_weight.ndim == 1 and len(edge_weight) == edge_index.shape[1]:
        ew_abs = np.abs(edge_weight.astype(float))
        rng = ew_abs.max() - ew_abs.min()
        ew_norm = (ew_abs - ew_abs.min()) / rng if rng > 0 else np.ones_like(ew_abs)
    else:
        ew_norm = np.ones(edge_index.shape[1], dtype=float)

    for k, (u_idx, v_idx) in enumerate(edge_index.T):
        u, v = genes[int(u_idx)], genes[int(v_idx)]
        if u != v:
            G.add_edge(u, v, weight=float(ew_norm[k]))

    if giant_component:
        if directed:
            comp = max(nx.weakly_connected_components(G), key=len)
        else:
            comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(comp).copy()

    return G


# ═════════════════════════════════════════════════════════════
#  draw_graph_colored_by_axes  (main entry point)
# ═════════════════════════════════════════════════════════════

def draw_graph_colored_by_axes(
    genes: List[str],
    edge_index: np.ndarray,
    edge_weight: Optional[np.ndarray] = None,
    *,
    # axes config
    axes: Optional[Dict[str, List[str]]] = None,
    axis_priority: Optional[List[str]] = None,
    axis_colors: Optional[Dict[str, str]] = None,
    # layout
    directed: bool = True,
    giant_component: bool = False,
    layout: str = "axis_modules",
    module_radius: float = 4.0,
    module_jitter: float = 0.45,
    spring_iter: int = 350,
    seed: int = 42,
    # nodes
    base_node_size: float = 120,
    degree_size_scale: float = 38,
    residual_node_scale: float = 0.90,
    node_alpha_axis: float = 0.96,
    node_alpha_residual: float = 0.72,
    node_edgewidth: float = 0.9,
    node_edgewidth_multi: float = 2.0,
    # edges
    edge_alpha_intra: float = 0.28,
    edge_alpha_cross: float = 0.18,
    edge_alpha_to_residual: float = 0.10,
    edge_alpha_residual: float = 0.06,
    edge_width_min: float = 0.6,
    edge_width_max: float = 2.3,
    # labels
    label_mode: str = "topk_per_axis",
    topk_per_axis: int = 3,
    topk_residual: int = 8,
    always_label_multi: bool = True,
    label_fontsize: int = 8,
    # figure
    figsize: Tuple[float, float] = (18, 14),
    dpi: int = 300,
    title: Optional[str] = "Subgraph colored by biological axis",
    title_fontsize: int = 17,
    # save
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, nx.Graph, pd.DataFrame]:
    """
    Draw the pruned subgraph coloured by biological axis.

    Accepts raw arrays (genes, edge_index, edge_weight) so it works both
    from a live model and from a saved bundle.

    Returns (fig, G, axis_membership_df).
    """
    np.random.seed(seed)

    axes = axes or AXES
    axis_priority = axis_priority or AXIS_PRIORITY
    axis_colors = axis_colors or AXIS_COLORS

    # ── Gene → axis mapping ──
    node_axis, gene_to_axes, multi_axis_genes = map_genes_to_axes(
        genes, axes=axes, priority=axis_priority,
    )

    # ── Build graph ──
    G = _build_nx_graph_from_arrays(
        genes, edge_index, edge_weight,
        directed=directed, giant_component=giant_component,
    )

    # ── Layout ──
    module_order = axis_priority + ["Residual"]
    n_modules = len(module_order)
    centers = {}
    for i, ax in enumerate(module_order):
        theta = 2 * math.pi * i / n_modules
        centers[ax] = np.array([module_radius * math.cos(theta),
                                module_radius * math.sin(theta)])

    pos_init = {}
    for g in G.nodes():
        c = centers[node_axis[g]]
        pos_init[g] = c + np.random.normal(0.0, module_jitter, size=2)

    for u, v in G.edges():
        w = G[u][v].get("weight", 1.0)
        same = node_axis[u] == node_axis[v] and node_axis[u] != "Residual"
        G[u][v]["spring_weight"] = (1.5 + 1.5*w) if same else (0.5 + 1.0*w)

    pos = nx.spring_layout(
        G, pos=pos_init, seed=seed, iterations=spring_iter,
        weight="spring_weight",
        k=1.15 / np.sqrt(max(1, G.number_of_nodes())),
    )

    # ── Classify edges ──
    deg = dict(G.degree())
    edges_intra = {ax: [] for ax in axis_priority}
    edges_cross, edges_to_res, edges_res = [], [], []

    for u, v, d in G.edges(data=True):
        au, av = node_axis[u], node_axis[v]
        w = d.get("weight", 1.0)
        width = edge_width_min + (edge_width_max - edge_width_min) * float(w)
        if au == av:
            (edges_res if au == "Residual" else edges_intra[au]).append((u, v, width))
        elif au == "Residual" or av == "Residual":
            edges_to_res.append((u, v, width))
        else:
            mixed = _darken_hex(_blend_hex(axis_colors[au], axis_colors[av], 0.5), 0.05)
            edges_cross.append((u, v, width, mixed))

    # ── Node size helper ──
    def _ns(g):
        s = base_node_size + degree_size_scale * deg[g]
        return s * residual_node_scale if node_axis[g] == "Residual" else s

    # ── Draw ──
    fig, ax_plot = plt.subplots(figsize=figsize)

    _arrow_kw = dict(arrows=directed, arrowsize=10 if directed else 0)
    _cs = "arc3,rad=0.03" if directed else "arc3"
    _cs_cross = "arc3,rad=0.05" if directed else "arc3"

    if edges_res:
        nx.draw_networkx_edges(G, pos, ax=ax_plot,
            edgelist=[(u,v) for u,v,_ in edges_res],
            width=[w for _,_,w in edges_res],
            edge_color="#d9d9d9", alpha=edge_alpha_residual,
            connectionstyle=_cs, **_arrow_kw)

    if edges_to_res:
        nx.draw_networkx_edges(G, pos, ax=ax_plot,
            edgelist=[(u,v) for u,v,_ in edges_to_res],
            width=[w for _,_,w in edges_to_res],
            edge_color="#b0b0b0", alpha=edge_alpha_to_residual,
            connectionstyle=_cs, **_arrow_kw)

    if edges_cross:
        nx.draw_networkx_edges(G, pos, ax=ax_plot,
            edgelist=[(u,v) for u,v,_,_ in edges_cross],
            width=[w for _,_,w,_ in edges_cross],
            edge_color=[c for _,_,_,c in edges_cross],
            alpha=edge_alpha_cross, connectionstyle=_cs_cross, **_arrow_kw)

    for axn in axis_priority:
        e = edges_intra[axn]
        if not e:
            continue
        nx.draw_networkx_edges(G, pos, ax=ax_plot,
            edgelist=[(u,v) for u,v,_ in e],
            width=[w for _,_,w in e],
            edge_color=_lighten_hex(axis_colors[axn], 0.10),
            alpha=edge_alpha_intra, connectionstyle=_cs, **_arrow_kw)

    # Residual nodes
    res_nodes = [g for g in G.nodes() if node_axis[g] == "Residual"]
    if res_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax_plot, nodelist=res_nodes,
            node_color=axis_colors["Residual"],
            node_size=[_ns(g) for g in res_nodes],
            alpha=node_alpha_residual, linewidths=0.7, edgecolors="#8c8c8c")

    # Axis nodes
    for axn in axis_priority:
        nl = [g for g in G.nodes() if node_axis[g] == axn and g not in multi_axis_genes]
        if nl:
            nx.draw_networkx_nodes(G, pos, ax=ax_plot, nodelist=nl,
                node_color=axis_colors[axn],
                node_size=[_ns(g) for g in nl],
                alpha=node_alpha_axis, linewidths=node_edgewidth, edgecolors="black")

    # Multi-axis overlay (golden border)
    multi_in = [g for g in G.nodes() if g in multi_axis_genes]
    if multi_in:
        nx.draw_networkx_nodes(G, pos, ax=ax_plot, nodelist=multi_in,
            node_color=[axis_colors[node_axis[g]] for g in multi_in],
            node_size=[_ns(g)*1.10 for g in multi_in],
            alpha=1.0, linewidths=node_edgewidth_multi, edgecolors="#FFD700")

    # ── Labels ──
    labels = {}
    if label_mode == "topk_per_axis":
        for axn in axis_priority:
            top = sorted([g for g in G.nodes() if node_axis[g] == axn],
                         key=lambda x: deg[x], reverse=True)[:topk_per_axis]
            for g in top:
                labels[g] = g
        for g in sorted(res_nodes, key=lambda x: deg[x], reverse=True)[:topk_residual]:
            labels[g] = g
    if always_label_multi:
        for g in multi_in:
            labels[g] = g
    if labels:
        nx.draw_networkx_labels(G, pos, ax=ax_plot, labels=labels,
            font_size=label_fontsize, font_weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # ── Legend ──
    handles = []
    for axn in axis_priority + ["Residual"]:
        n_ax = sum(1 for g in G.nodes() if node_axis[g] == axn)
        handles.append(Line2D([0],[0], marker="o", color="w",
            markerfacecolor=axis_colors[axn],
            markeredgecolor="black" if axn != "Residual" else "#8c8c8c",
            markersize=10, label=f"{axn} (n={n_ax})"))
    handles += [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#FFD700", markeredgewidth=2.0, markersize=10,
               label="Multi-axis gene"),
        Line2D([0],[0], color="#666666", lw=2, alpha=edge_alpha_cross,
               label="Inter-module edge"),
        Line2D([0],[0], color="#b0b0b0", lw=2, alpha=edge_alpha_to_residual,
               label="Edge to residual"),
    ]
    ax_plot.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0, frameon=False, fontsize=10)

    if title:
        ax_plot.set_title(title, fontsize=title_fontsize, pad=18)
    ax_plot.axis("off")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {Path(save_path).resolve()}")

    if show:
        plt.show()

    # ── Summary DataFrame ──
    axis_membership = pd.DataFrame({
        "gene": list(G.nodes()),
        "axis_assigned": [node_axis[g] for g in G.nodes()],
        "degree": [deg[g] for g in G.nodes()],
        "multi_axis": [g in multi_axis_genes for g in G.nodes()],
        "all_axes": [", ".join(gene_to_axes.get(g, [])) for g in G.nodes()],
    }).sort_values(["axis_assigned", "degree"], ascending=[True, False])

    return fig, G, axis_membership


def draw_graph_colored_by_axes_from_bundle(
    bundle: dict,
    **kwargs,
) -> Tuple[plt.Figure, nx.Graph, pd.DataFrame]:
    """Convenience wrapper: extract arrays from a saved bundle dict and plot."""
    # Try nested "graph" dict first, then top-level
    g = bundle.get("graph", bundle)

    genes = g.get("genes_comp", bundle.get("genes_comp"))
    ei = g.get("edge_index_compact", g.get("edge_index_comp",
         bundle.get("edge_index_comp")))
    ew = g.get("edge_weight_compact", g.get("edge_weight_comp",
         bundle.get("edge_weight_comp", None)))

    genes = [str(x) for x in (genes.tolist() if hasattr(genes, "tolist") else list(genes))]
    ei = _safe_tensor_to_numpy(ei)
    ew = _safe_tensor_to_numpy(ew)

    assert ei.shape[0] == 2, f"edge_index must be (2, E), got {ei.shape}"

    return draw_graph_colored_by_axes(genes, ei, ew, **kwargs)

