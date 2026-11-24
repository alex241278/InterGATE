"""Evaluate the locked score panels using the existing post-lock effect estimates."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from gesubnet_comparison import PROTOCOL


def ranked(series, k):
    return set(series.sort_index().sort_values(ascending=False, kind="stable").head(k).index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "work", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(study))
    from intergate.biology import combine_biology_scores

    results = study / "results_v8"
    seeds = PROTOCOL["training_seeds"]
    prior = pd.read_csv(results / "05_biology_validation/candidate_prior_for_null.csv.gz")
    prior["gene_a"] = np.minimum(prior.source_gene.to_numpy(object), prior.target_gene.to_numpy(object))
    prior["gene_b"] = np.maximum(prior.source_gene.to_numpy(object), prior.target_gene.to_numpy(object))
    common = pd.read_csv(args.work / f"gesubnet_seed_{seeds[0]}.csv").set_index(["gene_a", "gene_b"]).index
    prior = prior.set_index(["gene_a", "gene_b"])
    prior = prior[prior.index.isin(common)].copy()
    pairs = pd.DataFrame(index=common.sort_values())
    for seed in seeds:
        folder = study / f"artifacts/intrinsic4/INTERGATE_V8/seed_{seed}"
        dense = np.load(folder / "transport_support_dense.npz")
        class_id = json.loads((folder / "meta.json").read_text())["label_map"]["Basal-like"]
        score = combine_biology_scores(dense["differential_logits"].astype(float),
                                      dense["context_score"].astype(float),
                                      dense["selection_frequency"].astype(float),
                                      context_weight=0.35, support_weight=0.075)[class_id]
        exported = pd.read_csv(results / f"04_controls/locked_biology_score_w0075/seed_tables/seed_{seed}/biology_edges_Basal-like.csv")
        np.testing.assert_allclose(score[exported.candidate_edge_id], exported.biology_score, atol=2e-6)
        prior["score"] = score[prior.candidate_edge_id]
        pairs[f"intergate_{seed}"] = prior.groupby(level=[0, 1]).score.max()
        table = pd.read_csv(args.work / f"gesubnet_seed_{seed}.csv").set_index(["gene_a", "gene_b"])
        pairs[f"gesubnet_{seed}"] = table.gesubnet_logit
    assert np.isfinite(pairs.to_numpy()).all()
    methods = ["intergate", "gesubnet"]
    for method in methods:
        pairs[method] = pairs[[f"{method}_{seed}" for seed in seeds]].median(axis=1)
    # Attach external outcomes only after scoring and export.
    effects = pd.read_csv(results / "08b_external_structural_concordance/external_candidate_differential_effects.csv.gz").set_index("candidate_edge_id")
    scopes = ["TCGA-BRCA", "GSE21653", "GSE1456", "EXTERNAL_META"]
    cols = ["development_meta_effect"] + [f"external_effect::{c}" for c in scopes]
    for col in cols:
        prior[col] = prior.candidate_edge_id.map(effects[col])
        by_pair = prior.groupby(level=[0, 1])[col]
        assert (by_pair.max() - by_pair.min()).max() < 1e-5
        pairs[col] = by_pair.mean()
    manifest = json.loads((results / "05_biology_validation/locked_signature_manifest.json").read_text())
    focal_terms = set(manifest["terms"]["Basal-like"])
    gmt = study / "resources_biological/msigdb/c2.cgp.v2026.1.Hs.symbols.gmt"
    signature = set()
    found = set()
    for line in gmt.read_text().splitlines():
        term, _, *members = line.split("\t")
        if term in focal_terms:
            signature.update(members)
            found.add(term)
    assert found == focal_terms
    signature -= set((study / "resources_biological/PAM50_symbols.txt").read_text().split())
    pairs["signature_endpoint_count"] = [int(a in signature) + int(b in signature) for a, b in pairs.index]
    rows, memberships = [], []
    for fraction in (0.01, 0.05, 0.10):
        k = max(1, round(len(pairs) * fraction))
        panels = {method: ranked(pairs[method], k) for method in methods}
        shared = len(panels["intergate"] & panels["gesubnet"])
        for method in methods:
            selected = pairs.loc[sorted(panels[method])]
            per_seed = [ranked(pairs[f"{method}_{seed}"], k) for seed in seeds]
            jaccard = [len(a & b) / len(a | b) for a, b in itertools.combinations(per_seed, 2)]
            row = {"density": fraction, "k": k, "method": method, "between_method_overlap": shared,
                   "between_method_jaccard": shared / (2 * k - shared), "within_method_mean_jaccard": np.mean(jaccard),
                   "five_seed_core_pairs": len(set.intersection(*per_seed)),
                   "at_least_one_signature_endpoint": (selected.signature_endpoint_count >= 1).sum(),
                   "two_signature_endpoints": (selected.signature_endpoint_count == 2).sum()}
            for scope in scopes:
                row[f"transport_spearman_{scope}"] = spearmanr(selected.development_meta_effect,
                                                              selected[f"external_effect::{scope}"]).statistic
            rows.append(row)
            memberships.extend({"density": fraction, "method": method, "gene_a": a, "gene_b": b}
                               for a, b in sorted(panels[method]))
    summary = {"n_pairs": len(pairs), "n_genes": len(set(pairs.index.get_level_values(0)) | set(pairs.index.get_level_values(1))),
               "n_train": 2970, "seeds":seeds, "rank_correlation_all_pairs": spearmanr(pairs.intergate, pairs.gesubnet).statistic}
    (args.output / "gesubnet_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pairs.to_csv(args.output / "gesubnet_pair_scores.csv")
    pd.DataFrame(rows).to_csv(args.output / "gesubnet_panel_metrics.csv", index=False)
    pd.DataFrame(memberships).to_csv(args.output / "gesubnet_selected_pairs.csv", index=False)


if __name__ == "__main__":
    main()
