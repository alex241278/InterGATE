#!/usr/bin/env python3
"""Build the five final quantitative figures for the InterGATE manuscript."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/intergate-mpl")

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


from .posthoc_biology_common import load_locked_signature_resources


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v8"
MANUSCRIPT_OUT = RESULTS / "10_manuscript_figures"
TABLES = MANUSCRIPT_OUT / "tables"
FIGURES = MANUSCRIPT_OUT / "figures"

CLASSES = ["LumA", "LumB", "HER2-enriched", "Basal-like"]
COLORS = {
    "LumA": "#4C78A8",
    "LumB": "#72B7B2",
    "HER2-enriched": "#F58518",
    "Basal-like": "#C44E52",
}
EDGE_TYPES = {0: "HuRI", 1: "OmniPath activation", 2: "OmniPath inhibition"}


def mean_sd(series: pd.Series) -> tuple[float, float]:
    return float(series.mean()), float(series.std(ddof=1))


def component_sets(edge_pairs: list[tuple[str, str]]) -> list[set[str]]:
    nodes = sorted({node for edge in edge_pairs for node in edge})
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for left, right in edge_pairs:
        union(left, right)
    groups: dict[str, set[str]] = {}
    for node in nodes:
        groups.setdefault(find(node), set()).add(node)
    return sorted(groups.values(), key=lambda x: (-len(x), sorted(x)))


def weak_topology(edges: pd.DataFrame) -> dict[str, float | int]:
    edge_pairs = list(edges[["source_gene", "target_gene"]].itertuples(index=False, name=None))
    all_nodes = {node for edge in edge_pairs for node in edge}
    comps = component_sets(edge_pairs)
    giant = max((len(x) for x in comps), default=0)
    isolated_one_edge = 0
    for component_nodes in comps:
        if sum(1 for u, v in edge_pairs if u in component_nodes and v in component_nodes) == 1:
            isolated_one_edge += 1
    degree = {node: 0 for node in all_nodes}
    for source, target in edge_pairs:
        degree[source] += 1
        degree[target] += 1
    degree_one = sum(value == 1 for value in degree.values())
    return {
        "n_nodes": len(all_nodes),
        "n_edges": len(edge_pairs),
        "n_components": len(comps),
        "giant_nodes": giant,
        "giant_fraction": giant / len(all_nodes) if all_nodes else math.nan,
        "isolated_one_edge_components": isolated_one_edge,
        "degree_one_nodes": degree_one,
    }


def save_figure(fig: plt.Figure, stem: str, *, close: bool = True) -> None:
    for ext in ("pdf", "svg"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=600)
    if close:
        plt.close(fig)


def clean_term(term: str) -> str:
    return (
        term.replace("HALLMARK_", "")
        .replace("_", " ")
        .replace("PI3K AKT MTOR", "PI3K-AKT-mTOR")
        .replace("G2M", "G2/M")
        .title()
        .replace("Pi3K-Akt-Mtor", "PI3K-AKT-mTOR")
        .replace("Myc", "MYC")
        .replace("E2F", "E2F")
        .replace("G2/M", "G2/M")
    )


def structural_tables() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_path = (
        RESULTS
        / "04_controls/locked_biology_score_w0075/biology_consensus/biology_stability_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    panel_manifest = json.loads(
        (
            RESULTS
            / "04_controls/locked_biology_score_w0075/biology_panels_locked.json"
        ).read_text()
    )

    rows: list[dict] = []
    strict_edges: dict[str, pd.DataFrame] = {}
    for cls in CLASSES:
        path = (
            RESULTS
            / f"04_controls/locked_biology_score_w0075/biology_consensus/biology_consensus_{cls}.csv"
        )
        consensus = pd.read_csv(path)
        consensus["source_gene"] = consensus.source_gene.astype(str).str.upper()
        consensus["target_gene"] = consensus.target_gene.astype(str).str.upper()
        # Recover the median learned direction from the five locked 0.075
        # seed tables instead of inserting direction counts by hand.
        differential = []
        for seed in (1234, 2025, 7, 99, 31337):
            seed_path = (
                RESULTS
                / "04_controls/locked_biology_score_w0075/seed_tables"
                / f"seed_{seed}"
                / f"biology_edges_{cls}.csv"
            )
            current = pd.read_csv(
                seed_path,
                usecols=["candidate_edge_id", "differential_logit"],
            )
            current["seed"] = seed
            differential.append(current)
        median_direction = (
            pd.concat(differential, ignore_index=True)
            .groupby("candidate_edge_id", as_index=False)
            .agg(median_differential_logit=("differential_logit", "median"))
        )
        consensus = consensus.merge(
            median_direction,
            on="candidate_edge_id",
            how="left",
            validate="one_to_one",
        )
        strict = consensus.loc[consensus["selection_count"].eq(5)].copy()
        strict_edges[cls] = strict
        strict_topology = weak_topology(strict)
        primary_genes = set(strict["source_gene"]) | set(strict["target_gene"])
        secondary_genes = set(panel_manifest["node_presence_core_5of5"][cls])
        topo = summary[cls]["topology"]
        rows.append(
            {
                "class_name": cls,
                "exact_k_edges": int(summary[cls]["k_edges"]),
                "union_edges_across_seeds": int(summary[cls]["n_union_edges"]),
                "mean_pairwise_edge_jaccard": float(summary[cls]["mean_pairwise_edge_jaccard"]),
                "strict_5of5_directed_edges": int(len(strict)),
                "primary_edge_supported_genes": int(len(primary_genes)),
                "secondary_node_presence_genes": int(len(secondary_genes)),
                "consensus_nodes": int(topo["n_nodes"]),
                "consensus_weak_components": int(topo["n_components"]),
                "consensus_giant_nodes": int(topo["giant_nodes"]),
                "consensus_giant_fraction": float(topo["giant_fraction"]),
                "consensus_isolated_one_edge_components": int(topo["dyad_components"]),
                "consensus_degree_one_nodes": int(topo["degree_one_nodes"]),
                "strict_core_nodes": int(strict_topology["n_nodes"]),
                "strict_core_weak_components": int(strict_topology["n_components"]),
                "strict_core_giant_nodes": int(strict_topology["giant_nodes"]),
                "strict_core_giant_fraction": float(strict_topology["giant_fraction"]),
                "strict_core_isolated_one_edge_components": int(
                    strict_topology["isolated_one_edge_components"]
                ),
                "strict_core_degree_one_nodes": int(strict_topology["degree_one_nodes"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "main_structural_summary.csv", index=False)
    return out, strict_edges


def basal_component_table(strict: pd.DataFrame) -> pd.DataFrame:
    edge_pairs = list(strict[["source_gene", "target_gene"]].itertuples(index=False, name=None))
    signatures, _, _, _ = load_locked_signature_resources(
        results_root=RESULTS,
        resource_dir=ROOT / "resources_biological",
    )
    signature = signatures["Basal-like"]
    ora = pd.read_csv(RESULTS / "05_biology_validation/hallmark_c2_ora_complete_family.csv.gz")
    hallmark_rows = ora.loc[
        ora["panel"].eq("strict_5of5::Basal-like")
        & ora["collection"].eq("Hallmark")
        & ora["fdr"].lt(0.05)
    ]
    hallmark_genes: dict[str, set[str]] = {}
    for row in hallmark_rows.itertuples(index=False):
        hallmark_genes[row.term] = set(str(row.overlap_genes).split(";"))

    comps = component_sets(edge_pairs)
    rows = []
    for rank, nodes in enumerate(comps, start=1):
        sub_edges = [(u, v) for u, v in edge_pairs if u in nodes and v in nodes]
        hits = sorted(nodes & signature)
        path_hits = {
            clean_term(term): len(nodes & genes)
            for term, genes in hallmark_genes.items()
            if nodes & genes
        }
        rows.append(
            {
                "component_rank": rank,
                "n_genes": len(nodes),
                "n_directed_edges": len(sub_edges),
                "basal_signature_genes_n": len(hits),
                "basal_signature_genes": ";".join(hits),
                "significant_hallmark_overlap": "; ".join(
                    f"{name} ({count})" for name, count in sorted(path_hits.items())
                ),
                "genes": ";".join(sorted(nodes)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "basal_strict_5of5_components.csv", index=False)
    return table


def biological_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    biology = RESULTS / "05_biology_validation"
    matched = pd.read_csv(biology / "matched_candidate_signature_tests.csv")
    cross = pd.read_csv(biology / "cross_class_specificity.csv")
    rewiring = pd.read_csv(
        biology / "directed_degree_type_sign_rewiring.csv"
    )
    rewiring_tests = pd.read_csv(biology / "rewiring_coherence_tests.csv")
    observed = rewiring.loc[
        rewiring.kind.eq("observed"), ["graph_class", "coherent_fraction"]
    ].rename(columns={"coherent_fraction": "observed_coherent_fraction"})
    null = (
        rewiring.loc[rewiring.kind.eq("rewired")]
        .groupby("graph_class", as_index=False)
        .agg(
            null_mean=("coherent_fraction", "mean"),
            null_sd=("coherent_fraction", "std"),
        )
    )
    rew_table = observed.merge(null, on="graph_class", validate="one_to_one")
    rew_table = rew_table.merge(
        rewiring_tests, on="graph_class", validate="one_to_one"
    )
    rew_table["fold_over_null"] = (
        rew_table.observed_coherent_fraction / rew_table.null_mean
    )
    matched.to_csv(TABLES / "matched_candidate_signature_tests_all_48.csv", index=False)
    cross.to_csv(TABLES / "cross_class_specificity_matrix.csv", index=False)
    rew_table.to_csv(TABLES / "directed_rewiring_summary.csv", index=False)

    ora = pd.read_csv(RESULTS / "05_biology_validation/hallmark_c2_ora_complete_family.csv.gz")
    basal_ora = ora.loc[
        ora["panel"].isin(["biology_graph_nodes::Basal-like", "strict_5of5::Basal-like"])
        & ora["fdr"].lt(0.05)
        & ora["meets_min_overlap"]
    ].copy()
    basal_ora.to_csv(TABLES / "basal_significant_hallmark_c2_ora.csv", index=False)
    return matched, cross, rew_table


def final_control_tables() -> dict[str, pd.DataFrame]:
    """Export the V8 tables used by main and supplementary figures."""
    review = RESULTS / "12_reviewer_controls"
    ablation = RESULTS / "09_ablation_biology_w0075"
    exports = {
        "directional_signature_locked_network_tests": pd.read_csv(
            RESULTS / "05_biology_validation/matched_candidate_signature_tests.csv"
        ),
        "external_structural_concordance": pd.read_csv(
            RESULTS / "08b_external_structural_concordance"
            / "external_concordance_matched_null_summary.csv"
        ),
        "basal_single_component_ablation_summary": pd.read_csv(
            ablation / "ablation_biology_summary.csv"
        ),
        "basal_single_component_ablation_tests": pd.read_csv(
            ablation / "ablation_matched_signature_tests.csv"
        ),
        "factorial_q_directional_signature_tests": pd.read_csv(
            review / "factorial_directional_signature_tests.csv"
        ),
        "factorial_q_paired_contrasts": pd.read_csv(
            review / "factorial_paired_contrasts.csv"
        ),
        "core_threshold_k_over_R": pd.read_csv(
            review / "core_threshold_k_over_R.csv"
        ),
        "core_strict_subset_sensitivity": pd.read_csv(
            review / "core_strict_subset_sensitivity.csv"
        ),
    }
    permutation_summary = (
        review / "label_permutation_retraining/label_permutation_summary.csv"
    )
    if permutation_summary.is_file():
        exports["label_permutation_retraining_summary"] = pd.read_csv(
            permutation_summary
        )
    for stem, table in exports.items():
        table.to_csv(TABLES / f"{stem}.csv", index=False)
    return exports


def write_consolidated_numbers(
    summary: pd.DataFrame,
    components: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    rewiring: pd.DataFrame,
) -> None:
    """Write one internally consistent numeric source for the final article."""
    payload = {
        "structural_summary": json.loads(summary.to_json(orient="records")),
        "basal_top_components": json.loads(
            components.head(10).to_json(orient="records")
        ),
        "directed_rewiring_1000": json.loads(rewiring.to_json(orient="records")),
    }
    for key, table in controls.items():
        payload[key] = json.loads(table.to_json(orient="records"))
    (TABLES / "final_results_numbers.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def figure_structural(
    summary: pd.DataFrame,
    strict_edges: dict[str, pd.DataFrame],
    components: pd.DataFrame,
    *,
    close: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)
    x = np.arange(len(CLASSES))
    colors = [COLORS[c] for c in CLASSES]

    ax = axes[0, 0]
    vals = summary.set_index("class_name").loc[CLASSES, "mean_pairwise_edge_jaccard"]
    bars = ax.bar(x, vals, color=colors, width=0.68)
    for bar, cls in zip(bars, CLASSES):
        strict_n = int(
            summary.set_index("class_name").loc[cls, "strict_5of5_directed_edges"]
        )
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{strict_n} edges", ha="center", fontsize=9)
    ax.set_xticks(x, ["LumA", "LumB", "HER2", "Basal-like"])
    ax.set_ylim(0, max(0.69, float(vals.max()) + 0.08))
    ax.set_ylabel("Mean pairwise edge Jaccard")
    ax.set_title("A  Five-seed reproducibility")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    cons = summary.set_index("class_name").loc[CLASSES, "consensus_giant_fraction"]
    core = summary.set_index("class_name").loc[CLASSES, "strict_core_giant_fraction"]
    width = 0.36
    ax.bar(x - width / 2, cons, width, color=colors, alpha=0.95, label="Exact-K consensus")
    ax.bar(x + width / 2, core, width, color=colors, alpha=0.42, hatch="//", label="Strict 5/5 edge core")
    ax.set_xticks(x, ["LumA", "LumB", "HER2", "Basal-like"])
    ax.set_ylabel("Fraction of nodes in largest component")
    ax.set_title("B  Connected structure at a common edge budget")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    top = components.head(10).copy().sort_values("component_rank", ascending=False)
    labels = [f"C{v}" for v in top["component_rank"]]
    bars = ax.barh(labels, top["n_genes"], color="#C44E52", alpha=0.88)
    for bar, edges in zip(bars, top["n_directed_edges"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{int(edges)} edges", va="center", fontsize=8.5)
    ax.set_xlabel("Genes in Basal-like strict component")
    ax.set_title("C  Largest components of the Basal-like 5/5 core")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    strict = strict_edges["Basal-like"]
    edge_counts = strict["edge_type"].map(EDGE_TYPES).value_counts().reindex(list(EDGE_TYPES.values())).fillna(0)
    direction = pd.Series({
        "Class-strengthened": int(
            (strict["median_differential_logit"] >= 0).sum()
        ),
        "Class-attenuated": int(
            (strict["median_differential_logit"] < 0).sum()
        ),
    })
    labels = list(edge_counts.index) + list(direction.index)
    values = list(edge_counts.astype(int)) + list(direction.astype(int))
    palette = ["#8C8C8C", "#55A868", "#C44E52", "#4C78A8", "#F58518"]
    bars = ax.barh(labels[::-1], values[::-1], color=palette[::-1])
    for bar in bars:
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Strict 5/5 directed edges")
    ax.set_title("D  Basal-like core composition")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Reproducibility and topology of class-specific networks", fontsize=16, fontweight="bold")
    save_figure(fig, "Figure_structural_recovery", close=close)
    return fig


def force_layout(nodes: list[str], edges: list[tuple[str, str, int]], seed: int = 42) -> dict[str, np.ndarray]:
    """Small deterministic Fruchterman--Reingold layout without extra dependencies."""
    rng = np.random.default_rng(seed)
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    pos = rng.normal(0, 0.35, size=(n, 2))
    k = 1.25 / math.sqrt(max(n, 1))
    temperature = 0.14
    for step in range(320):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta**2).sum(axis=2)) + np.eye(n)
        rep = (delta / dist[:, :, None]) * ((k * k) / np.maximum(dist, 1e-3))[:, :, None]
        force = rep.sum(axis=1)
        for source, target, _ in edges:
            i, j = index[source], index[target]
            vec = pos[i] - pos[j]
            length = max(float(np.linalg.norm(vec)), 1e-3)
            attr = (vec / length) * (length * length / k)
            force[i] -= attr
            force[j] += attr
        norms = np.linalg.norm(force, axis=1)
        pos += force / np.maximum(norms[:, None], 1e-9) * np.minimum(norms, temperature)[:, None]
        pos -= pos.mean(axis=0)
        temperature = 0.14 * (1 - (step + 1) / 330)
    span = np.ptp(pos, axis=0)
    pos = pos / np.maximum(span, 1e-9)
    return {node: pos[i] for i, node in enumerate(nodes)}


def figure_basal_core(
    strict: pd.DataFrame,
    components: pd.DataFrame,
    *,
    close: bool = True,
) -> plt.Figure:
    edge_rows = [
        (row.source_gene, row.target_gene, int(row.edge_type))
        for row in strict.itertuples(index=False)
    ]
    largest_nodes = component_sets([(u, v) for u, v, _ in edge_rows])[0]
    sub_edges = [(u, v, t) for u, v, t in edge_rows if u in largest_nodes and v in largest_nodes]
    sub_nodes = sorted(largest_nodes)

    ora = pd.read_csv(RESULTS / "05_biology_validation/hallmark_c2_ora_complete_family.csv.gz")
    hall = ora.loc[
        ora["panel"].eq("strict_5of5::Basal-like")
        & ora["collection"].eq("Hallmark")
        & ora["fdr"].lt(0.05)
        & ora["meets_min_overlap"]
    ].sort_values("fold_enrichment", ascending=True)
    hall_genes = set()
    for value in hall["overlap_genes"].dropna():
        hall_genes.update(str(value).split(";"))
    signature_genes = set(str(components.iloc[0]["basal_signature_genes"]).split(";"))

    fig = plt.figure(figsize=(14.2, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    pos = force_layout(sub_nodes, sub_edges)
    degree = {node: 0 for node in sub_nodes}
    for source, target, _ in sub_edges:
        degree[source] += 1
        degree[target] += 1
    node_colors = []
    node_sizes = []
    for node in sub_nodes:
        in_sig = node in signature_genes
        in_hall = node in hall_genes
        if in_sig and in_hall:
            node_colors.append("#C44E52")
        elif in_sig:
            node_colors.append("#F58518")
        elif in_hall:
            node_colors.append("#8064A2")
        else:
            node_colors.append("#C7C7C7")
        node_sizes.append(65 + 20 * degree[node])
    for edge_type, color in [(0, "#9E9E9E"), (1, "#55A868"), (2, "#C44E52")]:
        for source, target, etype in sub_edges:
            if etype != edge_type:
                continue
            arrow = FancyArrowPatch(
                pos[source], pos[target], arrowstyle="-|>", mutation_scale=5.5,
                linewidth=0.65, color=color, alpha=0.42,
                connectionstyle="arc3,rad=0.04", shrinkA=3.2, shrinkB=3.2,
            )
            ax.add_patch(arrow)
    coords = np.vstack([pos[node] for node in sub_nodes])
    ax.scatter(coords[:, 0], coords[:, 1], c=node_colors, s=node_sizes, linewidths=0.45, edgecolors="white", zorder=3)
    priority = {
        "EGFR", "PCNA", "VGLL1", "TEAD4", "KRT5", "KRT16", "SOX10",
        "E2F1", "TFDP1", "PLK1", "CDK1", "NOTCH1", "CEP76", "FZD7",
    }
    texts = []
    for node in sub_nodes:
        if node not in priority:
            continue
        text = ax.text(pos[node][0], pos[node][1], node, fontsize=8, fontweight="bold", ha="center", va="center", zorder=4)
        texts.append(text)
    for text in texts:
        text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])
    ax.set_title(
        f"A  Largest Basal-like strict component\n{len(sub_nodes)} genes, {len(sub_edges)} directed edges",
        loc="left",
    )
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#C44E52", markeredgecolor="white", markersize=7, label="Basal signature + Hallmark"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#F58518", markeredgecolor="white", markersize=7, label="Basal signature only"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#8064A2", markeredgecolor="white", markersize=7, label="Hallmark only"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#C7C7C7", markeredgecolor="white", markersize=7, label="Neither"),
        Line2D([0], [0], color="#9E9E9E", lw=1.2, label="HuRI"),
        Line2D([0], [0], color="#55A868", lw=1.2, label="OmniPath activation"),
        Line2D([0], [0], color="#C44E52", lw=1.2, label="OmniPath inhibition"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=7.3,
        ncol=2,
        handletextpad=0.5,
        columnspacing=0.9,
    )
    ax.axis("off")

    right = grid[0, 1].subgridspec(2, 1, height_ratios=[1.18, 1.0])
    ax2 = fig.add_subplot(right[0, 0])
    h = hall.copy()
    labels = [clean_term(t) for t in h["term"]]
    bars = ax2.barh(labels, h["fold_enrichment"], color="#8064A2")
    for bar, overlap, fdr in zip(bars, h["overlap_n"], h["fdr"]):
        ax2.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2, f"n={int(overlap)}; FDR={fdr:.3g}", va="center", fontsize=8.4)
    ax2.set_xlim(0, max(h["fold_enrichment"]) * 1.42)
    ax2.set_xlabel("Fold enrichment")
    n_strict_genes = len(set(strict.source_gene) | set(strict.target_gene))
    ax2.set_title(
        f"B  Significant Hallmarks in the {n_strict_genes}-gene 5/5 core",
        loc="left",
    )
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(right[1, 0])
    top = components.head(10).copy().sort_values("component_rank", ascending=False)
    bars = ax3.barh([f"C{i}" for i in top["component_rank"]], top["n_genes"], color="#C44E52")
    ax3.set_xlabel("Genes")
    ax3.set_title("C  Component-size profile", loc="left")
    ax3.spines[["top", "right"]].set_visible(False)
    for bar, sig in zip(bars, top["basal_signature_genes_n"]):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{int(sig)} signature genes", va="center", fontsize=8.2)
    ax3.set_xlim(0, max(top["n_genes"]) * 1.35)

    fig.suptitle("Organization of the Basal-like 5/5 edge core", fontsize=16, fontweight="bold")
    save_figure(fig, "Figure_Basal_strict_core", close=close)
    return fig


def figure_validation(
    matched: pd.DataFrame,
    cross: pd.DataFrame,
    rew: pd.DataFrame,
    *,
    close: bool = True,
) -> plt.Figure:
    fig = plt.figure(figsize=(12.4, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82])

    ax = fig.add_subplot(grid[0, 0])
    matrix = cross.pivot(index="graph_class", columns="annotation_class", values="node_fraction").loc[CLASSES, CLASSES]
    im = ax.imshow(matrix.values, cmap="magma", vmin=0, vmax=max(0.28, float(matrix.values.max())))
    for i in range(4):
        for j in range(4):
            color = "white" if matrix.iloc[i, j] < 0.15 else "black"
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", color=color, fontsize=9)
        ax.add_patch(plt.Rectangle((i - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="cyan", linewidth=1.8))
    ax.set_xticks(range(4), ["LumA", "LumB", "HER2", "Basal-like"], rotation=25, ha="right")
    ax.set_yticks(range(4), ["LumA", "LumB", "HER2", "Basal-like"])
    ax.set_xlabel("Directional annotation class")
    ax.set_ylabel("Locked network class")
    ax.set_title("A  Cross-class locked-network specificity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[0, 1])
    basal = matched.loc[(matched["graph_class"] == "Basal-like") & (matched["term"] == "Basal-like")].copy()
    order = ["at_least_one_endpoint", "both_endpoints", "unique_node_fraction"]
    basal = basal.set_index("statistic").loc[order].reset_index()
    labels = [">=1 endpoint", "Both endpoints", "Unique nodes"]
    x = np.arange(3)
    width = 0.36
    ax.bar(x - width / 2, basal["observed"], width, color="#C44E52", label="Observed")
    ax.bar(x + width / 2, basal["null_mean"], width, color="#BDBDBD", label="Matched-null mean")
    fdr_column = (
        "fdr_global_48_tests"
        if "fdr_global_48_tests" in basal.columns
        else "fdr_within_panel"
    )
    for i, row in basal.iterrows():
        ax.text(
            i,
            max(row.observed, row.null_mean) + 0.018,
            f"{row.fold_over_null:.2f}x\nFDR={row[fdr_column]:.3f}",
            ha="center",
            fontsize=8.6,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(0.54, float(basal[["observed", "null_mean"]].to_numpy().max()) * 1.28))
    ax.set_ylabel("Fraction")
    ax.set_title("B  Directional Basal-like recovery")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, :])
    rew_i = rew.set_index("graph_class").loc[CLASSES]
    x = np.arange(4)
    ax.bar(x - width / 2, rew_i["observed_coherent_fraction"], width, color=[COLORS[c] for c in CLASSES], label="Observed")
    ax.bar(x + width / 2, rew_i["null_mean"], width, yerr=rew_i["null_sd"], color="#BDBDBD", capsize=3, label="Rewired mean +/- SD")
    for i, row in enumerate(rew_i.itertuples()):
        ax.text(i, max(row.observed_coherent_fraction, row.null_mean) + 0.009, f"{row.fold_over_null:.2f}x", ha="center", fontsize=9)
    ax.set_xticks(x, ["LumA", "LumB", "HER2", "Basal-like"])
    ax.set_ylabel("Hallmark-coherent edge fraction")
    ax.set_title("C  Functional coherence beyond preserved topology")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Subtype-specific recovery and functional coherence", fontsize=16, fontweight="bold")
    save_figure(fig, "Figure_biological_validation", close=close)
    return fig


def figure_external_concordance(*, close: bool = True) -> plt.Figure:
    table = pd.read_csv(
        RESULTS / "08b_external_structural_concordance"
        / "external_concordance_matched_null_summary.csv"
    )
    primary = table.loc[table.is_primary_metric.astype(bool)].copy()
    order = ["TCGA-BRCA", "GSE21653", "GSE1456", "EXTERNAL_META"]
    primary = primary.set_index("scope").loc[order].reset_index()
    labels = ["TCGA-BRCA", "GSE21653", "GSE1456", "External meta-analysis"]
    ypos = np.arange(len(primary))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.hlines(
        ypos, primary.null_q025, primary.null_q975,
        color="#BDBDBD", lw=7, label="Matched-null 95% interval",
    )
    ax.scatter(
        primary.null_mean, ypos, color="#4D4D4D", s=48,
        zorder=3, label="Matched-null mean",
    )
    ax.scatter(
        primary.observed, ypos, color=COLORS["Basal-like"], s=82,
        zorder=4, label="Basal-like 5/5 core",
    )
    for y, row in zip(ypos, primary.itertuples(index=False)):
        ax.text(
            row.observed + 0.016, y,
            f"FDR={row.fdr_primary_four_scopes:.3f}",
            va="center", ha="left", fontsize=8.5,
        )
    ax.set_yticks(ypos, labels)
    xmin = float(primary[["null_q025", "null_mean", "observed"]].min().min())
    xmax = float(primary[["null_q975", "null_mean", "observed"]].max().max())
    margin = max(0.03, 0.12 * max(xmax - xmin, 0.1))
    ax.set_xlim(xmin - margin, xmax + 2.2 * margin)
    ax.set_xlabel("Spearman correlation: development vs external edge effects")
    ax.set_title("External concordance of Basal-like edge effects", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.03, 0.56),
        fontsize=8.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "Figure_external_structural_concordance", close=close)
    return fig


def figure_controls_sensitivity(*, close: bool = True) -> plt.Figure:
    review = RESULTS / "12_reviewer_controls"
    ablation = pd.read_csv(
        RESULTS / "09_ablation_biology_w0075" / "ablation_biology_summary.csv"
    )
    ablation_tests = pd.read_csv(
        RESULTS / "09_ablation_biology_w0075"
        / "ablation_matched_signature_tests.csv"
    )
    factorial = pd.read_csv(review / "factorial_directional_signature_tests.csv")
    threshold = pd.read_csv(review / "core_threshold_k_over_R.csv")
    subsets = pd.read_csv(review / "core_strict_subset_sensitivity.csv")

    arm_order = [
        "FULL", "NO_CONTEXT_LOSS", "NO_COHORT_QUERY_LOSS",
        "NO_EDGE_TYPE_CALIBRATION", "NO_SIGNED_WEIGHT",
    ]
    arm_labels = ["Full", "No context", "No cohort query", "No edge type", "No signed prior"]
    arm_colors = ["#2166AC", "#B2182B", "#969696", "#969696", "#969696"]
    ablation = ablation.set_index("arm").loc[arm_order]
    unique_tests = (
        ablation_tests.loc[ablation_tests.statistic.eq("unique_node_fraction")]
        .set_index("arm").loc[arm_order]
    )
    fdr_column = "fdr_global_15_tests"

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=True)
    x = np.arange(len(arm_order))

    ax = axes[0, 0]
    ax.bar(x, ablation.mean_pairwise_edge_jaccard, color=arm_colors)
    ax.set_xticks(x, arm_labels, rotation=25, ha="right")
    ax.set_ylim(0, max(0.1, float(ablation.mean_pairwise_edge_jaccard.max()) * 1.18))
    ax.set_ylabel("Mean pairwise edge Jaccard")
    ax.set_title("A  Five-seed reproducibility", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    bars = ax.bar(x, ablation.strict_edges, color=arm_colors)
    for bar, value in zip(bars, ablation.strict_edges):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(1.0, float(ablation.strict_edges.max()) * 0.025),
            f"{int(value)}",
            ha="center",
            fontsize=8.5,
        )
    ax.set_xticks(x, arm_labels, rotation=25, ha="right")
    ax.set_ylim(0, max(1.0, float(ablation.strict_edges.max()) * 1.18))
    ax.set_ylabel("Directed edges")
    ax.set_title("B  Strict 5/5 core size", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 2]
    bars = ax.bar(x, unique_tests.fold_over_null, color=arm_colors)
    for bar, row in zip(bars, unique_tests.itertuples()):
        fdr = float(getattr(row, fdr_column))
        mark = "ns" if fdr >= 0.05 else f"FDR={fdr:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, row.fold_over_null + 0.07, mark, ha="center", fontsize=7.8)
    ax.axhline(1.0, color="#4D4D4D", lw=0.9, ls="--")
    ax.set_xticks(x, arm_labels, rotation=25, ha="right")
    ax.set_ylim(0, max(1.2, float(unique_tests.fold_over_null.max()) * 1.22))
    ax.set_ylabel("Unique-node fold over matched null")
    ax.set_title("C  Directional Basal-like recovery", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    statistics = ["at_least_one_endpoint", "both_endpoints", "unique_node_fraction"]
    stat_labels = [">=1 endpoint", "Both endpoints", "Unique nodes"]
    ax = axes[1, 0]
    sx = np.arange(3)
    factorial_order = ["InterGATE_raw", "InterGATE_module", "q_raw", "q_module"]
    factorial_labels = ["InterGATE raw", "InterGATE module", "q raw", "q module"]
    factorial_colors = ["#2F6FB0", "#103D6E", "#E58B2A", "#A94F00"]
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(factorial_order))
    for arm, label, color, offset in zip(
        factorial_order, factorial_labels, factorial_colors, offsets
    ):
        values = (
            factorial.loc[factorial.arm.eq(arm)]
            .set_index("statistic").loc[statistics, "fold_over_null"]
        )
        ax.bar(sx + offset, values, width, color=color, label=label)
    ax.axhline(1.0, color="#4D4D4D", lw=0.9, ls="--")
    ax.set_xticks(sx, stat_labels, rotation=15, ha="right")
    ax.set_ylabel("Fold over matched null")
    ax.set_title("D  Learning × module factorial", loc="left")
    ax.legend(frameon=False, fontsize=7.2, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    for selector, color in (("raw", "#4C78A8"), ("module", "#C44E52")):
        block = threshold.loc[threshold.selector.eq(selector)].sort_values("k")
        ax.plot(block.k, block.n_edges, marker="o", lw=2, color=color, label=selector)
    r_value = int(threshold.R.iloc[0])
    ax.set_xticks(range(1, r_value + 1))
    ax.set_xlabel(f"Minimum selections k (R={r_value})")
    ax.set_ylabel("Edges retained at >=k/R")
    ax.set_title("E  Core threshold curve", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 2]
    positions = np.arange(2, r_value + 1)
    for offset, selector, color in (
        (-0.13, "raw", "#4C78A8"), (0.13, "module", "#C44E52")
    ):
        blocks = [
            subsets.loc[
                subsets.selector.eq(selector) & subsets.subset_R.eq(r),
                "strict_edges_RofR",
            ].to_numpy(float)
            for r in positions
        ]
        box = ax.boxplot(
            blocks,
            positions=positions + offset,
            widths=0.22,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for median in box["medians"]:
            median.set_color("white")
    ax.plot([], [], color="#4C78A8", lw=7, alpha=0.65, label="raw")
    ax.plot([], [], color="#C44E52", lw=7, alpha=0.65, label="module")
    ax.set_xticks(positions)
    ax.set_xlabel("Number of seeds in subset R")
    ax.set_ylabel("Strict R/R intersection size")
    ax.set_title("F  Seed-subset sensitivity", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Mechanism controls and parameter sensitivity", fontsize=16, fontweight="bold")
    save_figure(fig, "Figure_controls_sensitivity", close=close)
    return fig


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    summary, strict_edges = structural_tables()
    components = basal_component_table(strict_edges["Basal-like"])
    matched, cross, rew = biological_tables()
    controls = final_control_tables()
    write_consolidated_numbers(summary, components, controls, rew)
    figure_structural(summary, strict_edges, components)
    figure_basal_core(strict_edges["Basal-like"], components)
    figure_validation(matched, cross, rew)
    figure_external_concordance()
    figure_controls_sensitivity()
    print(f"Wrote five quantitative figures to {FIGURES}")


if __name__ == "__main__":
    main()
