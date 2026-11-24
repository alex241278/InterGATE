"""Reprocess trained InterGATE V8 ablations with the locked 0.075 score.

Only dense structural exports from the already completed FULL and four atomic
ablation arms are read.  No optimizer, model checkpoint or external sample is
used by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .biology import aggregate_biology_seed_tables
from .biology_validation import (
    edge_signature_statistics,
    matched_edge_signature_null,
)
from .config import CFG
from .notebook04_controls import (
    _assert_saved_arm_contract,
    _rebuild_target_biology_network,
    audit_single_component_ablations,
    build_trained_arm_specs,
)
from .reviewer_analysis import enrichment_for_panels
from .transport import TransportConfig
from .posthoc_biology_common import (
    LOCKED_CONTEXT_WEIGHT,
    LOCKED_SEEDS,
    LOCKED_SUPPORT_WEIGHT,
    LOCKED_TARGET_CLASS,
    bh_adjust,
    build_candidates,
    edge_sets_pairwise_summary,
    load_locked_signature_resources,
    output_hashes,
    safe_name,
    sha256_file,
    strict_edge_core_from_seed_tables,
    topology_from_candidate_ids,
)


def _edge_key_set(frame: pd.DataFrame) -> set[int]:
    return set(frame.candidate_edge_id.astype(int))


def _node_set(frame: pd.DataFrame) -> set[str]:
    return set(frame.source_gene.astype(str)) | set(frame.target_gene.astype(str))


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else np.nan


def _locked_full_seed_table(
    results_root: Path,
    seed: int,
    target_class: str,
) -> Path:
    return (
        results_root
        / "04_controls"
        / "locked_biology_score_w0075"
        / "seed_tables"
        / f"seed_{int(seed)}"
        / f"biology_edges_{safe_name(target_class)}.csv"
    )


def _locked_full_seed_root(results_root: Path) -> Path:
    return (
        results_root
        / "04_controls"
        / "locked_biology_score_w0075"
        / "seed_tables"
    )


def _verify_full_reconstruction(
    rebuilt: pd.DataFrame,
    reference_path: Path,
) -> None:
    if not reference_path.is_file():
        raise FileNotFoundError(str(reference_path))
    reference = pd.read_csv(reference_path)
    rebuilt_ids = _edge_key_set(rebuilt)
    reference_ids = _edge_key_set(reference)
    if rebuilt_ids != reference_ids:
        raise RuntimeError(
            "La reconstrucción FULL a 0.075 no reproduce el score ya bloqueado: "
            f"solo_rebuilt={len(rebuilt_ids-reference_ids)}, "
            f"solo_referencia={len(reference_ids-rebuilt_ids)}."
        )


def _paired_signature_tests(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    full = per_seed[per_seed.arm.eq("FULL")]
    for arm in sorted(set(per_seed.arm) - {"FULL"}):
        arm_frame = per_seed[per_seed.arm.eq(arm)]
        for statistic in sorted(per_seed.statistic.unique()):
            left = arm_frame[arm_frame.statistic.eq(statistic)].set_index("seed")[
                "value"
            ]
            right = full[full.statistic.eq(statistic)].set_index("seed")["value"]
            observed_common = set(left.index) & set(right.index)
            if observed_common != set(LOCKED_SEEDS):
                raise RuntimeError(
                    f"{arm}/{statistic}: semillas pareadas "
                    f"{sorted(observed_common)} "
                    f"!= {list(LOCKED_SEEDS)}."
                )
            common = list(LOCKED_SEEDS)
            differences = left.loc[common].to_numpy(float) - right.loc[
                common
            ].to_numpy(float)
            if np.allclose(differences, 0.0):
                p_value = 1.0
            else:
                p_value = float(stats.wilcoxon(
                    left.loc[common], right.loc[common],
                    alternative="two-sided", method="auto",
                ).pvalue)
            rows.append({
                "arm": arm,
                "statistic": statistic,
                "n_paired_seeds": len(common),
                "mean_arm": float(left.loc[common].mean()),
                "mean_full": float(right.loc[common].mean()),
                "mean_paired_difference_arm_minus_full": float(
                    differences.mean()
                ),
                "median_paired_difference_arm_minus_full": float(
                    np.median(differences)
                ),
                "wilcoxon_two_sided_p": p_value,
                "minimum_attainable_two_sided_p_with_five_nonzero_pairs": 0.0625,
            })
    output = pd.DataFrame(rows)
    output["fdr_12_paired_tests"] = bh_adjust(output.wilcoxon_two_sided_p)
    return output


def run_locked_score_ablation_biology(
    runtime: Mapping[str, Any],
    *,
    artifacts_root: str | Path,
    results_root: str | Path,
    resource_dir: str | Path,
    output_dir: str | Path,
    protocol_lock: str | Path,
    target_class: str = LOCKED_TARGET_CLASS,
    seeds: Sequence[int] = LOCKED_SEEDS,
    n_matched_draws: int = 1000,
    random_seed: int = 20260821,
    cfg=CFG,
) -> dict[str, pd.DataFrame]:
    """Rebuild, aggregate and biologically assess all five trained arms."""

    if bool(runtime.get("external_opened", True)):
        raise RuntimeError("La post-procesación de ablaciones debe usar desarrollo.")
    seeds = tuple(map(int, seeds))
    if seeds != LOCKED_SEEDS:
        raise ValueError(f"Semillas no bloqueadas: {seeds}.")
    if target_class != LOCKED_TARGET_CLASS:
        raise ValueError(f"Target no bloqueado: {target_class}.")
    if int(n_matched_draws) < 1000:
        raise ValueError("Se requieren al menos 1.000 paneles nulos por brazo.")

    artifacts_root = Path(artifacts_root)
    results_root = Path(results_root)
    resource_dir = Path(resource_dir)
    output_dir = Path(output_dir)
    protocol_lock = Path(protocol_lock)
    if not protocol_lock.is_file():
        raise FileNotFoundError(str(protocol_lock))
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_root = output_dir / "reconstructed_seed_tables"
    consensus_root = output_dir / "arm_consensus"
    strict_root = output_dir / "strict_5of5_edges"
    for directory in (reconstructed_root, consensus_root, strict_root):
        directory.mkdir(parents=True, exist_ok=True)

    base_transport = TransportConfig.from_global(cfg)
    specs = build_trained_arm_specs(base_transport)
    contract = audit_single_component_ablations(specs)
    if not contract.single_component_contract_passed.all():
        raise RuntimeError("Alguna ablación no es atómica.")
    contract_file = output_dir / "single_component_contract_rechecked.csv"
    contract.to_csv(contract_file, index=False)

    signatures, hallmark, cgp, signature_manifest = (
        load_locked_signature_resources(
            results_root=results_root,
            resource_dir=resource_dir,
        )
    )
    if target_class not in signatures:
        raise RuntimeError(f"Falta la firma bloqueada {target_class}.")
    candidates = build_candidates(runtime)
    candidates_file = output_dir / "candidate_prior_for_ablation_null.csv.gz"
    candidates.drop(columns=["null_stratum"]).to_csv(
        candidates_file, index=False, compression="gzip"
    )

    # Exact-K and the FULL strict-core size are observed V8 outputs.  Infer
    # them from the score-locked FULL tables instead of importing V7's 529/284.
    locked_full_root = _locked_full_seed_root(results_root)
    full_reference_strict, full_reference_seed_frames = (
        strict_edge_core_from_seed_tables(
            locked_full_root,
            seeds=seeds,
            class_name=target_class,
            expected_k=None,
        )
    )
    expected_exact_k = int(len(full_reference_seed_frames[0]))
    expected_full_strict_ids = set(
        full_reference_strict.candidate_edge_id.astype(int)
    )

    rebuilt_rows = []
    strict_by_arm: dict[str, pd.DataFrame] = {}
    consensus_by_arm: dict[str, pd.DataFrame] = {}
    seed_frames_by_arm: dict[str, list[pd.DataFrame]] = {}
    topology_rows = []
    per_seed_signature_rows = []
    cross_signature_rows = []
    matched_summaries = []
    matched_distributions = []

    for arm_index, (arm, spec) in enumerate(specs.items()):
        arm_artifact_root = artifacts_root / spec.artifact_name
        arm_seed_root = reconstructed_root / arm
        arm_consensus_dir = consensus_root / arm
        arm_seed_root.mkdir(parents=True, exist_ok=True)
        arm_consensus_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            run_dir = arm_artifact_root / f"seed_{seed}"
            meta_path = run_dir / "meta.json"
            dense_path = run_dir / "transport_support_dense.npz"
            if not meta_path.is_file() or not dense_path.is_file():
                raise FileNotFoundError(
                    f"Falta la corrida completada {arm}/seed={seed}: "
                    f"{meta_path} o {dense_path}."
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            _assert_saved_arm_contract(meta, arm, spec)
            table, summary, _ = _rebuild_target_biology_network(
                runtime,
                run_dir=run_dir,
                seed=seed,
                target_class=target_class,
                score_variant="LOCKED_W0075",
                context_weight=LOCKED_CONTEXT_WEIGHT,
                support_weight=LOCKED_SUPPORT_WEIGHT,
                cfg=cfg,
            )
            table = table.rename(
                columns={"transport_support": "selection_frequency"}
            )
            if len(table) != expected_exact_k:
                raise RuntimeError(
                    f"{arm}/seed={seed}: {len(table)} != {expected_exact_k}."
                )
            if arm == "FULL":
                _verify_full_reconstruction(
                    table,
                    _locked_full_seed_table(results_root, seed, target_class),
                )
            destination = arm_seed_root / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            table.to_csv(
                destination / f"biology_edges_{safe_name(target_class)}.csv",
                index=False,
            )
            rebuilt_rows.append({
                "arm": arm,
                "seed": int(seed),
                "score_context_weight": LOCKED_CONTEXT_WEIGHT,
                "score_support_weight": LOCKED_SUPPORT_WEIGHT,
                "n_edges": int(len(table)),
                **summary,
            })
            stats_seed = edge_signature_statistics(
                table, {target_class: signatures[target_class]}
            )
            stats_seed.insert(0, "seed", int(seed))
            stats_seed.insert(0, "arm", arm)
            per_seed_signature_rows.append(stats_seed)

        aggregate_biology_seed_tables(
            arm_seed_root,
            seeds=seeds,
            class_names=[target_class],
            output_dir=arm_consensus_dir,
            pool_multiplier=int(cfg.BIOLOGY_MODULE_POOL_MULTIPLIER),
            target_components=int(cfg.BIOLOGY_MODULE_TARGET_COMPONENTS),
            min_component_edges=int(cfg.BIOLOGY_MODULE_MIN_EDGES),
        )
        strict, seed_frames = strict_edge_core_from_seed_tables(
            arm_seed_root,
            seeds=seeds,
            class_name=target_class,
            expected_k=expected_exact_k,
        )
        if arm == "FULL" and set(
            strict.candidate_edge_id.astype(int)
        ) != expected_full_strict_ids:
            raise RuntimeError(
                "El núcleo FULL reconstruido no coincide con las tablas "
                "V8 bloqueadas."
            )
        strict.to_csv(
            strict_root / f"strict_5of5_{safe_name(arm)}.csv", index=False
        )
        strict_by_arm[arm] = strict
        seed_frames_by_arm[arm] = seed_frames
        consensus_path = (
            arm_consensus_dir
            / f"biology_consensus_{safe_name(target_class)}.csv"
        )
        consensus = pd.read_csv(consensus_path)
        consensus_by_arm[arm] = consensus
        pairwise = edge_sets_pairwise_summary(seed_frames)
        pairwise.insert(0, "arm", arm)
        pairwise.to_csv(
            arm_consensus_dir / "pairwise_jaccard_recomputed.csv", index=False
        )
        strict_topology = topology_from_candidate_ids(
            strict.candidate_edge_id, runtime["edge_index"]
        )
        consensus_topology = topology_from_candidate_ids(
            consensus.candidate_edge_id, runtime["edge_index"]
        )
        topology_rows.append({
            "arm": arm,
            "mean_pairwise_edge_jaccard": float(pairwise.edge_jaccard.mean()),
            "strict_edges": int(len(strict)),
            "strict_nodes": int(len(_node_set(strict))),
            **{f"strict_{key}": value for key, value in strict_topology.items()},
            **{
                f"consensus_{key}": value
                for key, value in consensus_topology.items()
            },
        })

        cross = edge_signature_statistics(strict, signatures)
        cross.insert(0, "arm", arm)
        cross_signature_rows.append(cross)
        null_summary, null_distribution = matched_edge_signature_null(
            strict,
            candidates,
            {target_class: signatures[target_class]},
            n_draws=n_matched_draws,
            seed=random_seed + arm_index * 10000,
        )
        null_summary.insert(0, "arm", arm)
        null_summary["selected_edges_excluded_from_null"] = False
        null_distribution.insert(0, "arm", arm)
        matched_summaries.append(null_summary)
        matched_distributions.append(null_distribution)

    rebuilt = pd.DataFrame(rebuilt_rows)
    topology = pd.DataFrame(topology_rows)
    per_seed_signature = pd.concat(per_seed_signature_rows, ignore_index=True)
    paired = _paired_signature_tests(per_seed_signature)
    cross_signature = pd.concat(cross_signature_rows, ignore_index=True)
    matched_summary = pd.concat(matched_summaries, ignore_index=True)
    matched_summary["fdr_global_15_tests"] = bh_adjust(
        matched_summary.empirical_p
    )
    matched_distribution = pd.concat(matched_distributions, ignore_index=True)

    full_edges = _edge_key_set(strict_by_arm["FULL"])
    full_nodes = _node_set(strict_by_arm["FULL"])
    topology["strict_edge_jaccard_vs_full"] = topology.arm.map(
        lambda arm: _jaccard(_edge_key_set(strict_by_arm[arm]), full_edges)
    )
    topology["strict_node_jaccard_vs_full"] = topology.arm.map(
        lambda arm: _jaccard(_node_set(strict_by_arm[arm]), full_nodes)
    )

    specificity_rows = []
    unique_nodes = cross_signature[
        cross_signature.statistic.eq("unique_node_fraction")
    ]
    for arm in specs:
        current = unique_nodes[unique_nodes.arm.eq(arm)].set_index("term").value
        alternatives = current.drop(index=target_class)
        specificity_rows.append({
            "arm": arm,
            "basal_like_unique_node_fraction": float(current[target_class]),
            "best_non_basal_annotation_fraction": float(alternatives.max()),
            "basal_specificity_margin": float(
                current[target_class] - alternatives.max()
            ),
            "basal_annotation_is_largest": bool(
                current[target_class] > alternatives.max()
            ),
        })
    specificity = pd.DataFrame(specificity_rows)

    panels = {
        f"strict_5of5::{arm}": _node_set(frame)
        for arm, frame in strict_by_arm.items()
    }
    ora = enrichment_for_panels(
        panels,
        universe=runtime["genes"],
        collections={"Hallmark": hallmark, "C2_CGP": cgp},
        min_overlap=2,
        return_reportable_only=False,
    )
    ora["arm"] = ora.panel.str.split("::", n=1).str[-1]
    ora["fdr_global_all_arms_same_collection"] = np.nan
    for collection, indices in ora.groupby("collection").groups.items():
        ora.loc[indices, "fdr_global_all_arms_same_collection"] = bh_adjust(
            ora.loc[indices, "p_value"]
        )
    ora["fdr_global_all_arms_both_collections"] = bh_adjust(ora.p_value)
    ora_significant = ora[
        ora.meets_min_overlap
        & ora.fdr.lt(0.05)
    ].copy()

    matched_wide = matched_summary.pivot(
        index="arm", columns="statistic",
        values=["observed", "fold_over_null", "empirical_p", "fdr_global_15_tests"],
    )
    matched_wide.columns = [
        f"basal_{metric}__{statistic}"
        for metric, statistic in matched_wide.columns
    ]
    matched_wide = matched_wide.reset_index()
    ora_counts = (
        ora_significant.groupby(["arm", "collection"])
        .size().unstack(fill_value=0)
        .add_prefix("n_significant_panel_fdr__")
        .reset_index()
    )
    summary = (
        topology.merge(specificity, on="arm", how="left", validate="one_to_one")
        .merge(matched_wide, on="arm", how="left", validate="one_to_one")
        .merge(ora_counts, on="arm", how="left", validate="one_to_one")
    )

    files = {
        "rebuild": output_dir / "locked_w0075_rebuild_audit.csv",
        "topology": output_dir / "ablation_topology_summary.csv",
        "per_seed": output_dir / "ablation_per_seed_basal_signature.csv",
        "paired": output_dir / "ablation_paired_signature_vs_full.csv",
        "cross": output_dir / "ablation_cross_signature_specificity.csv",
        "specificity": output_dir / "ablation_specificity_summary.csv",
        "matched": output_dir / "ablation_matched_signature_tests.csv",
        "matched_dist": output_dir / "ablation_matched_signature_distribution.csv.gz",
        "ora": output_dir / "ablation_hallmark_c2_ora_complete.csv.gz",
        "ora_sig": output_dir / "ablation_hallmark_c2_ora_significant.csv",
        "summary": output_dir / "ablation_biology_summary.csv",
    }
    rebuilt.to_csv(files["rebuild"], index=False)
    topology.to_csv(files["topology"], index=False)
    per_seed_signature.to_csv(files["per_seed"], index=False)
    paired.to_csv(files["paired"], index=False)
    cross_signature.to_csv(files["cross"], index=False)
    specificity.to_csv(files["specificity"], index=False)
    matched_summary.to_csv(files["matched"], index=False)
    matched_distribution.to_csv(
        files["matched_dist"], index=False, compression="gzip"
    )
    ora.to_csv(files["ora"], index=False, compression="gzip")
    ora_significant.to_csv(files["ora_sig"], index=False)
    summary.to_csv(files["summary"], index=False)

    manifest_file = output_dir / "ablation_biology_w0075_manifest.json"
    manifest = {
        "status": "complete",
        "analysis": "trained_single_component_ablation_biology_postprocessing",
        "target_class": target_class,
        "arms": list(specs),
        "seeds": list(seeds),
        "score": {
            "context_weight": LOCKED_CONTEXT_WEIGHT,
            "support_weight": LOCKED_SUPPORT_WEIGHT,
        },
        "exact_k_per_seed": expected_exact_k,
        "full_reference_strict_edges": len(expected_full_strict_ids),
        "strict_definition": "identical directed candidate edge in all five seeds",
        "signature_manifest_reused": str(
            results_root / "05_biology_validation" / "locked_signature_manifest.json"
        ),
        "msigdb_release": signature_manifest["msigdb_release"],
        "matched_null_draws_per_arm": int(n_matched_draws),
        "model_refit": False,
        "model_checkpoints_loaded": False,
        "dense_structural_exports_only": True,
        "external_inputs_read": False,
        "single_component_contract_rechecked": True,
        "protocol_lock_sha256": sha256_file(protocol_lock),
        "output_sha256": output_hashes([
            contract_file, candidates_file, *files.values(),
        ]),
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "summary": summary,
        "topology": topology,
        "specificity": specificity,
        "matched_tests": matched_summary,
        "paired_tests": paired,
        "ora_significant": ora_significant,
    }
