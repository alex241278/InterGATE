#!/usr/bin/env python3
"""Fail-fast checks for the InterGATE environment and prepared inputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA_IMPORTS = {
    "GEOparse": "GEOparse",
    "inmoose": "inmoose",
    "matplotlib": "matplotlib",
    "mygene": "mygene",
    "numpy": "numpy",
    "pandas": "pandas",
    "requests": "requests",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
}
ANALYSIS_IMPORTS = {
    "IPython": "ipykernel",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "numpy": "numpy",
    "pandas": "pandas",
    "requests": "requests",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "xgboost": "xgboost",
}


def ok(message: str) -> None:
    print(f"[ok] {message}")


def fail(message: str, errors: list[str]) -> None:
    print(f"[fail] {message}")
    errors.append(message)


def check_python(errors: list[str]) -> None:
    if sys.version_info[:2] != (3, 11):
        fail(
            f"Python 3.11 required; observed {sys.version.split()[0]}", errors
        )
    else:
        ok(f"Python {sys.version.split()[0]}")


def check_imports(imports: dict[str, str], errors: list[str]) -> None:
    missing = [package for module, package in imports.items() if importlib.util.find_spec(module) is None]
    if missing:
        fail("missing packages: " + ", ".join(sorted(set(missing))), errors)
    else:
        ok(f"{len(imports)} required Python modules available")
    if shutil.which("jupyter") is None:
        fail("jupyter executable not found", errors)
    else:
        ok("Jupyter executable available")


def check_protocol(errors: list[str]) -> None:
    path = ROOT / "configs/protocol_v6.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "endpoint": ["LumA", "LumB", "HER2-enriched", "Basal-like"],
        "development_cohorts": ["GSE96058", "GSE25066", "GSE162228", "GSE20711"],
        "external_cohorts": ["TCGA-BRCA", "GSE21653", "GSE1456"],
        "outer_seeds": [1234, 2025, 7, 99, 31337],
        "kappa": 0.01,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        fail(f"protocol mismatch: {observed}", errors)
        return
    graph = payload["biology_graph"]
    if graph["training_score_terms"]["cohort_support"] != 0.25:
        fail("training support weight is not 0.25", errors)
    elif graph["definitive_locked_score_terms"]["cohort_support"] != 0.075:
        fail("definitive support weight is not 0.075", errors)
    else:
        ok("protocol cohorts, seeds, kappa, and score timeline")


def _metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _expression_samples(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))[1:]


def check_prepared_data(errors: list[str]) -> None:
    folder = ROOT / "data_prepare/output_mcombat"
    specs = (
        ("train", folder / "expr_train_mcombat.csv", folder / "meta_train.csv", 3713),
        ("external", folder / "expr_test_mcombat.csv", folder / "meta_test.csv", 1143),
    )
    for label, expression, metadata, expected_n in specs:
        missing = [str(path.relative_to(ROOT)) for path in (expression, metadata) if not path.is_file()]
        if missing:
            fail(f"missing prepared {label} files: {', '.join(missing)}", errors)
            continue
        rows = _metadata(metadata)
        samples = _expression_samples(expression)
        meta_samples = [row["sample"] for row in rows]
        if len(rows) != expected_n or len(samples) != expected_n:
            fail(
                f"{label} sample count mismatch: expression={len(samples)}, metadata={len(rows)}, expected={expected_n}",
                errors,
            )
        elif samples != meta_samples:
            fail(f"{label} expression/metadata order mismatch", errors)
        elif len(set(samples)) != expected_n:
            fail(f"{label} contains duplicate sample identifiers", errors)
        else:
            ok(f"{label}: {expected_n} aligned samples")


def check_depmap(errors: list[str]) -> None:
    if os.environ.get("INTERGATE_RUN_DEPMAP", "1") == "0":
        ok("DepMap explicitly disabled; full protocol locking will not be available")
        return
    folder = Path(
        os.environ.get(
            "INTERGATE_DEPMAP_DIR", ROOT / "resources_biological/depmap"
        )
    ).expanduser().resolve()
    gene_effect = folder / "CRISPRGeneEffect.csv"
    model_candidates = (folder / "Model.csv", folder / "Models.csv")
    missing = []
    if not gene_effect.is_file():
        missing.append(str(gene_effect))
    if not any(path.is_file() for path in model_candidates):
        missing.append(f"{model_candidates[0]} or {model_candidates[1]}")
    if missing:
        fail(
            "missing licensed DepMap inputs (or set INTERGATE_RUN_DEPMAP=0 "
            "and stop at step 05): " + ", ".join(missing),
            errors,
        )
    else:
        ok(f"DepMap inputs available in {folder}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("data", "analysis", "repository"), required=True)
    args = parser.parse_args()
    errors: list[str] = []
    check_python(errors)
    check_protocol(errors)
    if args.stage == "data":
        check_imports(DATA_IMPORTS, errors)
    elif args.stage == "analysis":
        check_imports(ANALYSIS_IMPORTS, errors)
        check_prepared_data(errors)
        check_depmap(errors)
    if errors:
        print(f"\nPreflight failed with {len(errors)} issue(s).")
        return 1
    print("\nPreflight complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
