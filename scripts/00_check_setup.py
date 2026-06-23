#!/usr/bin/env python3
"""Pre-flight check for the self-contained InterGATE package."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intergate.config import CFG  # noqa: E402


EXPECTED_PIPELINE_CACHE = [
    "backbone__362de41006ca05c1__e71ae7463734.npz",
    "backbone_global__n11907__sig5f9c12d26c9c90d5__op1__huri1__score1__min0p0.npz",
    "HuRI.filtered.with_score.min0.0.impute_median.npz",
    "Xh__0575558be8d8a450__c102b7893b38__2060915c8ab9.npz",
]


def check_file(path: Path, required: bool = True) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    tag = "OK" if ok else ("MISSING" if required else "optional missing")
    print(f"[{tag:16}] {path}")
    return ok or not required


def check_dir_nonempty(path: Path, required: bool = True) -> bool:
    ok = path.exists() and path.is_dir() and any(path.iterdir())
    tag = "OK" if ok else ("MISSING" if required else "optional missing")
    print(f"[{tag:16}] {path}")
    return ok or not required


def main() -> int:
    print("InterGATE self-contained setup check")
    print(f"PROJECT_ROOT: {CFG.PROJECT_ROOT}")
    print(f"DATA_DIR:     {CFG.DATA_DIR}")
    print(f"CACHE_ROOT:   {CFG.CACHE_ROOT}")
    print(f"PIPELINE:     {CFG.PIPELINE_CACHE_DIR}")
    print(f"ARTIFACTS:    {CFG.ARTIFACTS_ROOT}")
    print("")

    ok = True
    ok &= check_file(Path(CFG.EXPR_CSV), required=True)
    ok &= check_file(Path(CFG.META_CSV), required=True)

    # These are optional because graph.py can download them, but keeping them
    # locally makes the run closer to offline/reproducible.
    check_file(Path(CFG.EXTERNAL_DATA_DIR) / "omnipath_interactions.tsv", required=False)
    check_file(Path(CFG.EXTERNAL_DATA_DIR) / "HuRI.tsv", required=False)
    check_file(Path(CFG.EXTERNAL_DATA_DIR) / "HuRI.psi", required=False)

    print("\nPaper-aligned cache artifacts:")
    for name in EXPECTED_PIPELINE_CACHE:
        ok &= check_file(Path(CFG.PIPELINE_CACHE_DIR) / name, required=True)

    print("\nPaper-aligned model artifacts:")
    ok &= check_dir_nonempty(Path(CFG.ARTIFACTS_ROOT), required=True)

    if not ok:
        print("\nFaltan datos/cache/artifacts obligatorios. Ejecuta:")
        print("  python scripts/download_zenodo_data.py --extract")
        return 1

    print("\n[ok] La estructura mínima paper-aligned está preparada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
