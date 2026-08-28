"""Static contract tying protocol, code, documentation, and graphics together."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def literal_config_fields(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fields: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "InterGATEConfig":
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                try:
                    fields[item.target.id] = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    pass
    return fields


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/protocol_v8.json").read_text(encoding="utf-8")
    )
    assert protocol["endpoint"] == [
        "LumA", "LumB", "HER2-enriched", "Basal-like"
    ]
    assert protocol["development_cohorts"] == [
        "GSE96058", "GSE25066", "GSE162228", "GSE20711"
    ]
    assert protocol["external_cohorts"] == [
        "TCGA-BRCA", "GSE21653", "GSE1456"
    ]
    assert protocol["outer_seeds"] == [1234, 2025, 7, 99, 31337]
    assert protocol["kappa"] == 0.01

    meta = protocol["meta_training"]
    assert meta["episodes"] == 60
    assert meta["microbatch_size"] == 2
    assert meta["steps_per_episode"] == 192
    assert meta["accumulation_steps"] == 120
    assert meta["samples_per_episode"] == 384
    assert meta["optimizer_steps_per_episode"] == 2
    assert meta["xh"] == "zero"

    support = protocol["support_audit"]
    assert support["steps"] == 5
    assert support["microbatch_size"] == 2
    assert support["batches_per_step"] == 64
    assert support["samples_per_step"] == 128

    biology = protocol["biology_graph"]
    assert biology["score_terms"] == {
        "absolute_centered_class_delta": 1.0,
        "signed_context_agreement": 0.35,
        "cohort_support": 0.25,
    }
    assert biology["definitive_locked_score_terms"] == {
        "absolute_centered_class_delta": 1.0,
        "signed_context_agreement": 0.35,
        "cohort_support": 0.075,
    }
    assert biology["exact_k"] is True
    assert round(52_878 * protocol["kappa"]) == 529

    phase2 = protocol["phase2"]
    assert phase2["phase2a_graph_only_epochs"] == 5
    assert phase2["phase2b_max_epochs"] == 80
    assert phase2["structural_parameters_frozen"] is True
    assert phase2["fixed_graph_bypasses_gates"] is True
    assert phase2["xh_enters_only_post_pooling"] is True
    assert phase2["node_identity_embedding"] is True
    assert phase2["expert_deep_supervision_lambda"] == 0.2
    assert phase2["tabular_full_connected_gene_universe"] is True
    assert phase2["tabular_hidden_width"] == 256
    assert phase2["tabular_branch_dropout"] == 0.3
    assert phase2["gnn_branch_dropout"] == 0.0
    assert phase2["gnn_auxiliary_loss_lambda"] == 0.5

    fields = literal_config_fields(ROOT / "intergate/config.py")
    expected_fields = {
        "MODEL_VERSION": "8.1.0",
        "TRANSPORT_PROTOCOL_VERSION": protocol["protocol_version"],
        "TRANSPORT_EPISODES": 60,
        "GRAPH_BATCH_SIZE": 2,
        "GRAPH_ACCUM_STEPS": 120,
        "TRANSPORT_STEPS_PER_EPISODE": 192,
        "TRANSPORT_SUPPORT_STEPS": 5,
        "TRANSPORT_SUPPORT_BATCHES": 64,
        "BIOLOGY_SCORE_CONTEXT_WEIGHT": 0.35,
        "BIOLOGY_SCORE_SUPPORT_WEIGHT": 0.25,
        "FT_EPOCHS_A": 5,
        "FT_EPOCHS_B": 80,
        "USE_NODE_IDENTITY": True,
        "EXPERT_DEEP_SUPERVISION_LAMBDA": 0.2,
        "TAB_FULL_GENE_UNIVERSE": True,
        "HYBRID_TAB_HIDDEN": 256,
        "BRANCH_DROPOUT_TAB": 0.3,
        "BRANCH_DROPOUT_GNN": 0.0,
        "GNN_AUX_LOSS_LAMBDA": 0.5,
    }
    for name, expected in expected_fields.items():
        assert fields.get(name) == expected, (name, fields.get(name), expected)

    aligned_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/PROTOCOL_ALIGNMENT.md",
            "docs/MANUSCRIPT_CODE_ALIGNMENT.md",
            "assets/graphical_abstract.svg",
        )
    )
    for required in (
        "299 directed edges",
        "288 genes",
        "0.607",
        "0.691",
        "82.9%",
        "2.22",
        "5.94",
        "2.31",
        "0.838",
        "0.963",
    ):
        assert required in aligned_text, required
    for stale in (
        "284 directed edges",
        "257 genes",
        "0.604",
        "0.668",
        "81.7%",
        "6.64",
        "2.47",
        "macro-F1 = 0.837",
    ):
        assert stale not in aligned_text, stale

    assert not (ROOT / "configs/protocol_v6.json").exists()
    assert not (ROOT / "notebooks/10_figuras_definitivas_v2.ipynb").exists()
    print("Protocol/manuscript/graphic alignment: OK")


if __name__ == "__main__":
    main()
