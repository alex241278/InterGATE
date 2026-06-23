#!/usr/bin/env python3
"""Pre-flight check for the self-contained InterGATE package."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intergate.config import CFG  # noqa: E402


def check_file(path: Path, required: bool = True) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    tag = "OK" if ok else ("MISSING" if required else "optional missing")
    print(f"[{tag:16}] {path}")
    return ok or not required


def main() -> int:
    print("InterGATE self-contained setup check")
    print(f"PROJECT_ROOT: {CFG.PROJECT_ROOT}")
    print(f"DATA_DIR:     {CFG.DATA_DIR}")
    print(f"CACHE_ROOT:   {CFG.CACHE_ROOT}")
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

    if not ok:
        print("\nFaltan datos obligatorios. Ejecuta:")
        print("  python scripts/download_zenodo_data.py --extract")
        return 1

    print("\n[ok] La estructura mínima está preparada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
