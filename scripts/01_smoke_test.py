#!/usr/bin/env python3
"""Smoke test for InterGATE before running the full notebooks.

This test is intentionally lightweight: it validates that core modules import,
that notebooks are clean, and that obvious absolute local paths are absent. It
does not require the full Zenodo data files.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULES = [
    "intergate",
    "intergate.config",
    "intergate.data",
    "intergate.graph",
    "intergate.model",
    "intergate.training",
    "intergate.ablation",
    "intergate.benchmarks",
    "intergate.backbone_blocks",
    "intergate.backbone_ablation",
    "intergate.benchmarks_gnn_baselines",
    "intergate.axes",
    "intergate.enrichment",
    "intergate.visualization",
]

ABSOLUTE_PATH_RE = re.compile(r"(/home/|/Users/|[A-Za-z]:[\\/]|Escritorio|content/drive|mnt/data)", re.IGNORECASE)


def check_imports() -> bool:
    ok = True
    print("[smoke] Import checks")
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"  [OK] {name}")
        except Exception as exc:  # pragma: no cover
            ok = False
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    return ok


def check_notebooks() -> bool:
    ok = True
    print("\n[smoke] Notebook metadata checks")
    for nb_path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        outputs = sum(len(c.get("outputs", [])) for c in nb.get("cells", []))
        exec_counts = sum(
            c.get("execution_count") is not None
            for c in nb.get("cells", [])
            if c.get("cell_type") == "code"
        )
        bad_paths = []
        for idx, cell in enumerate(nb.get("cells", []), start=1):
            src = "".join(cell.get("source", []))
            if ABSOLUTE_PATH_RE.search(src):
                bad_paths.append(idx)
        if outputs or exec_counts or bad_paths:
            ok = False
            print(
                f"  [FAIL] {nb_path.name}: outputs={outputs}, "
                f"execution_counts={exec_counts}, absolute_path_cells={bad_paths}"
            )
        else:
            print(f"  [OK] {nb_path.name}")
    return ok


def main() -> int:
    ok = check_imports()
    ok = check_notebooks() and ok
    if ok:
        print("\n[ok] Smoke test passed. Next: python scripts/00_check_setup.py")
        return 0
    print("\n[fail] Smoke test failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
