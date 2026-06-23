#!/usr/bin/env python3
"""Download and arrange InterGATE data from Zenodo.

Default DOI / record:
    https://doi.org/10.5281/zenodo.20815745

The script downloads every file attached to the Zenodo record, verifies MD5
checksums when Zenodo exposes them, optionally extracts archives, and then tries
to place the files expected by the pipeline under a self-contained layout:

    data/processed/expr_combat_corrected.csv
    data/processed/metadata_combined.csv
    data/external/omnipath_interactions.tsv    # optional; can also be downloaded by graph.py
    data/external/HuRI.tsv                     # optional; can also be downloaded by graph.py
    data/external/HuRI.psi                     # optional; can also be downloaded by graph.py
    cache/pipeline_cache/*.npz                 # paper-aligned graph/cache artifacts
    artifacts_ablation/                        # paper-aligned model/graph artifacts
    artifacts_backbone_ablation/               # optional/internal backbone-ablation artifacts
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Falta requests. Instala dependencias con: pip install -e .") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ID = "20815745"
DEFAULT_DOI = "10.5281/zenodo.20815745"

REQUIRED_PROCESSED = {
    "expr_combat_corrected.csv": [
        "expr_combat_corrected.csv",
        "expression_combat_corrected.csv",
        "*expr*combat*corrected*.csv",
        "*expression*combat*.csv",
    ],
    "metadata_combined.csv": [
        "metadata_combined.csv",
        "*metadata*combined*.csv",
        "*meta*combined*.csv",
    ],
}

OPTIONAL_EXTERNAL = {
    "omnipath_interactions.tsv": [
        "omnipath_interactions.tsv",
        "*omnipath*interactions*.tsv",
        "*omnipath*.tsv",
    ],
    "HuRI.tsv": ["HuRI.tsv", "huri.tsv", "*HuRI*.tsv", "*huri*.tsv"],
    "HuRI.psi": ["HuRI.psi", "huri.psi", "*HuRI*.psi", "*huri*.psi"],
}

# These four files are part of the paper-aligned Zenodo layout.  They are
# deliberately placed under cache/pipeline_cache/ because intergate.config.CFG
# uses PROJECT_ROOT/cache/pipeline_cache as the default pipeline cache.
EXPECTED_PIPELINE_CACHE = {
    "backbone__362de41006ca05c1__e71ae7463734.npz": [
        "backbone__362de41006ca05c1__e71ae7463734.npz",
        "*backbone__362de41006ca05c1__e71ae7463734*.npz",
    ],
    "backbone_global__n11907__sig5f9c12d26c9c90d5__op1__huri1__score1__min0p0.npz": [
        "backbone_global__n11907__sig5f9c12d26c9c90d5__op1__huri1__score1__min0p0.npz",
        "*backbone_global__n11907__sig5f9c12d26c9c90d5__op1__huri1__score1__min0p0*.npz",
    ],
    "HuRI.filtered.with_score.min0.0.impute_median.npz": [
        "HuRI.filtered.with_score.min0.0.impute_median.npz",
        "*HuRI.filtered.with_score.min0.0.impute_median*.npz",
        "*huri.filtered.with_score.min0.0.impute_median*.npz",
    ],
    "Xh__0575558be8d8a450__c102b7893b38__2060915c8ab9.npz": [
        "Xh__0575558be8d8a450__c102b7893b38__2060915c8ab9.npz",
        "*Xh__0575558be8d8a450__c102b7893b38__2060915c8ab9*.npz",
    ],
}

ARTIFACTS_DIR_CANDIDATES = {
    "artifacts_ablation",
    "artifacts-ablation",
    "artifacts ablation",
}

BACKBONE_ARTIFACTS_DIR_CANDIDATES = {
    "artifacts_backbone_ablation",
    "artifact_backbone_ablation",
    "artifacts-backbone-ablation",
    "artifact-backbone-ablation",
    "artifacts backbone ablation",
    "artifact backbone ablation",
    "artifacts_back_bone_ablation",
    "artifact_back_bone_ablation",
    "artifacts-back-bone-ablation",
    "artifact-back-bone-ablation",
    "artifacts back bone ablation",
    "artifact back bone ablation",
    "backbone_ablation",
    "backbone-ablation",
}


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def record_api_url(record_id: str) -> str:
    return f"https://zenodo.org/api/records/{record_id}"


def fetch_record(record_id: str) -> dict:
    url = record_api_url(record_id)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def file_name(entry: dict) -> str:
    return str(entry.get("key") or entry.get("filename") or entry.get("name"))


def file_download_url(entry: dict) -> str:
    links = entry.get("links", {}) or {}
    for key in ("self", "download", "content"):
        if links.get(key):
            return str(links[key])
    if entry.get("download_url"):
        return str(entry["download_url"])
    raise KeyError(f"No encuentro URL de descarga para {file_name(entry)}")


def expected_md5(entry: dict) -> Optional[str]:
    checksum = str(entry.get("checksum") or "")
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1].lower()
    if len(checksum) == 32 and all(c in "0123456789abcdefABCDEF" for c in checksum):
        return checksum.lower()
    return None


def download_file(entry: dict, raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = file_name(entry)
    out = raw_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    url = file_download_url(entry)
    want_md5 = expected_md5(entry)

    if out.exists() and out.stat().st_size > 0 and not force:
        if want_md5 is None or md5sum(out) == want_md5:
            print(f"[ok] ya existe: {out}")
            return out
        print(f"[warn] checksum distinto; se descarga de nuevo: {out.name}")

    tmp = out.with_suffix(out.suffix + ".part")
    print(f"[download] {name}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(out)

    if want_md5 is not None:
        got = md5sum(out)
        if got != want_md5:
            raise RuntimeError(f"MD5 incorrecto para {out.name}: esperado {want_md5}, obtenido {got}")
    return out


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2"))


def extract_archive(path: Path, raw_dir: Path, force: bool = False) -> Optional[Path]:
    if not is_archive(path):
        return None
    out_dir = raw_dir / (path.name.replace(".tar.gz", "").replace(".tar.bz2", "").replace(".zip", "").replace(".tgz", "").replace(".tbz2", "") + "__extracted")
    marker = out_dir / ".extracted_ok"
    if marker.exists() and not force:
        print(f"[ok] ya extraído: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {path.name} -> {out_dir}")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(out_dir)
    else:
        print(f"[warn] archivo no reconocido como comprimido: {path}")
        return None
    marker.write_text("ok\n", encoding="utf-8")
    return out_dir


def iter_candidate_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.name != ".extracted_ok" and p not in seen:
                seen.add(p)
                yield p


def find_file(roots: Iterable[Path], patterns: list[str]) -> Optional[Path]:
    files = list(iter_candidate_files(roots))
    # Exact basename, case-insensitive, first.
    lower_map = {p.name.lower(): p for p in files}
    for pat in patterns:
        if "*" not in pat and "?" not in pat:
            hit = lower_map.get(pat.lower())
            if hit:
                return hit
    # Then glob-style basename matching.
    for pat in patterns:
        for p in files:
            if fnmatch.fnmatch(p.name.lower(), pat.lower()):
                return p
    return None


def copy_if_found(roots: Iterable[Path], patterns: list[str], dest: Path) -> bool:
    hit = find_file(roots, patterns)
    if hit is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if hit.resolve() != dest.resolve():
        shutil.copy2(hit, dest)
    print(f"[layout] {dest.relative_to(PROJECT_ROOT)} <= {hit}")
    return True


def _norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def iter_candidate_dirs(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir() and p not in seen:
                seen.add(p)
                yield p


def copytree_merge(src: Path, dest: Path) -> bool:
    """Merge-copy a directory, skipping extraction marker files."""
    if not src.exists() or not src.is_dir():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    copied = False
    for item in src.iterdir():
        if item.name in {".extracted_ok", ".DS_Store"}:
            continue
        target = dest / item.name
        if item.is_dir():
            copied = copytree_merge(item, target) or copied
        elif item.is_file():
            shutil.copy2(item, target)
            copied = True
    if copied:
        print(f"[layout] {dest.relative_to(PROJECT_ROOT)} <= {src}")
    return copied


def find_named_artifact_dir(roots: Iterable[Path], candidates: set[str]) -> Optional[Path]:
    """Find an extracted artifact directory using tolerant aliases.

    Zenodo archives may contain directories such as ``artifacts_ablation/`` or
    may themselves be named ``artifacts_ablation.zip``.  After extraction, the
    latter becomes ``artifacts_ablation__extracted``.  This helper handles both
    cases and also tolerates hyphens/spaces/minor singular/plural variants.
    """
    dirs = list(iter_candidate_dirs(roots))
    aliases = {_norm_name(x) for x in candidates}

    # Prefer exact aliases, including nested cases such as
    # extracted_archive/artifacts_ablation/.
    for d in dirs:
        if _norm_name(d.name) in aliases:
            return d

    # Then accept extracted archive directories whose name starts with one of
    # the aliases, e.g. artifacts_back_bone_ablation__extracted/.
    for d in dirs:
        n = _norm_name(d.name)
        if any(n.startswith(alias) for alias in aliases) and any(d.iterdir()):
            return d

    return None


def find_artifacts_ablation_dir(roots: Iterable[Path]) -> Optional[Path]:
    return find_named_artifact_dir(roots, ARTIFACTS_DIR_CANDIDATES)


def find_backbone_artifacts_dir(roots: Iterable[Path]) -> Optional[Path]:
    return find_named_artifact_dir(roots, BACKBONE_ARTIFACTS_DIR_CANDIDATES)


def arrange_layout(
    raw_dir: Path,
    processed_dir: Path,
    external_dir: Path,
    pipeline_cache_dir: Path,
    artifacts_dir: Path,
    backbone_artifacts_dir: Path,
) -> dict:
    roots = [raw_dir] + [p for p in raw_dir.iterdir() if p.is_dir()]
    arranged = {
        "processed": {},
        "external": {},
        "pipeline_cache": {},
        "artifacts_ablation": None,
        "artifacts_backbone_ablation": None,
    }

    for dest_name, patterns in REQUIRED_PROCESSED.items():
        dest = processed_dir / dest_name
        ok = copy_if_found(roots, patterns, dest)
        arranged["processed"][dest_name] = str(dest) if ok or dest.exists() else None

    for dest_name, patterns in OPTIONAL_EXTERNAL.items():
        dest = external_dir / dest_name
        ok = copy_if_found(roots, patterns, dest)
        arranged["external"][dest_name] = str(dest) if ok or dest.exists() else None

    pipeline_cache_dir.mkdir(parents=True, exist_ok=True)
    for dest_name, patterns in EXPECTED_PIPELINE_CACHE.items():
        dest = pipeline_cache_dir / dest_name
        ok = copy_if_found(roots, patterns, dest)
        arranged["pipeline_cache"][dest_name] = str(dest) if ok or dest.exists() else None

    artifact_source = find_artifacts_ablation_dir(roots)
    if artifact_source is not None:
        copied = copytree_merge(artifact_source, artifacts_dir)
        arranged["artifacts_ablation"] = str(artifacts_dir) if copied or artifacts_dir.exists() else None
    elif artifacts_dir.exists() and any(artifacts_dir.iterdir()):
        arranged["artifacts_ablation"] = str(artifacts_dir)

    backbone_artifact_source = find_backbone_artifacts_dir(roots)
    if backbone_artifact_source is not None:
        copied = copytree_merge(backbone_artifact_source, backbone_artifacts_dir)
        arranged["artifacts_backbone_ablation"] = str(backbone_artifacts_dir) if copied or backbone_artifacts_dir.exists() else None
    elif backbone_artifacts_dir.exists() and any(backbone_artifacts_dir.iterdir()):
        arranged["artifacts_backbone_ablation"] = str(backbone_artifacts_dir)

    return arranged


def write_manifest(record: dict, downloaded: list[Path], arranged: dict, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "doi": DEFAULT_DOI,
        "record_id": str(record.get("id") or DEFAULT_RECORD_ID),
        "title": (record.get("metadata") or {}).get("title"),
        "version": (record.get("metadata") or {}).get("version"),
        "files": [
            {
                "path": str(p),
                "name": p.name,
                "size": p.stat().st_size if p.exists() else None,
                "md5": md5sum(p) if p.exists() and p.is_file() else None,
            }
            for p in downloaded
        ],
        "arranged": arranged,
    }
    (data_root / "zenodo_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = ["path\tsize\tmd5"]
    for p in sorted(downloaded):
        if p.exists() and p.is_file():
            rows.append(f"{p}\t{p.stat().st_size}\t{md5sum(p)}")
    (data_root / "checksums.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record-id", default=DEFAULT_RECORD_ID, help="Zenodo record id; default: 20815745")
    ap.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    ap.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    ap.add_argument("--external-dir", type=Path, default=PROJECT_ROOT / "data" / "external")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache" / "pipeline_cache",
                    help="Destination for paper-aligned pipeline cache .npz files")
    ap.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts_ablation",
                    help="Destination for paper-aligned ablation/model artifacts")
    ap.add_argument("--backbone-artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts_backbone_ablation",
                    help="Destination for backbone-replacement ablation artifacts")
    ap.add_argument("--extract", action="store_true", help="Extract downloaded archives")
    ap.add_argument("--force", action="store_true", help="Re-download/re-extract files even if they already exist")
    ap.add_argument("--arrange-only", action="store_true", help="Do not download; only arrange files already present under raw-dir")
    ap.add_argument("--allow-missing-cache", action="store_true",
                    help="Do not fail if the expected cache/pipeline_cache .npz files are absent")
    ap.add_argument("--allow-missing-artifacts", action="store_true",
                    help="Do not fail if artifacts_ablation/ is absent or empty")
    ap.add_argument("--allow-missing-backbone-artifacts", action="store_true",
                    help="Do not fail if artifacts_backbone_ablation/ is absent or empty")
    args = ap.parse_args(argv)

    raw_dir = args.raw_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    external_dir = args.external_dir.expanduser().resolve()
    pipeline_cache_dir = args.cache_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    backbone_artifacts_dir = args.backbone_artifacts_dir.expanduser().resolve()
    for d in [raw_dir, processed_dir, external_dir, pipeline_cache_dir, artifacts_dir, backbone_artifacts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    record = {"id": args.record_id, "metadata": {"title": None, "version": None}}

    if not args.arrange_only:
        record = fetch_record(args.record_id)
        files = record.get("files", [])
        if not files:
            raise RuntimeError(f"El registro Zenodo {args.record_id} no expone archivos descargables.")
        print(f"[record] {record.get('id')} | {(record.get('metadata') or {}).get('title')}")
        for entry in files:
            downloaded.append(download_file(entry, raw_dir, force=args.force))
    else:
        downloaded = [p for p in raw_dir.iterdir() if p.is_file()]

    if args.extract:
        for p in list(downloaded):
            extract_archive(p, raw_dir, force=args.force)

    arranged = arrange_layout(raw_dir, processed_dir, external_dir, pipeline_cache_dir, artifacts_dir, backbone_artifacts_dir)
    write_manifest(record, downloaded, arranged, PROJECT_ROOT / "data")

    required_missing = [name for name in REQUIRED_PROCESSED if not (processed_dir / name).exists()]
    cache_missing = [name for name in EXPECTED_PIPELINE_CACHE if not (pipeline_cache_dir / name).exists()]
    artifacts_missing = not (artifacts_dir.exists() and any(artifacts_dir.iterdir()))
    backbone_artifacts_missing = not (backbone_artifacts_dir.exists() and any(backbone_artifacts_dir.iterdir()))

    if required_missing:
        print("\n[ATENCIÓN] No se han encontrado todos los ficheros procesados requeridos:")
        for name in required_missing:
            print(f"  - {processed_dir / name}")
        print("\nSi los nombres en Zenodo son distintos, copia/renombra manualmente esos CSV a data/processed/.")
        return 2

    if cache_missing and not args.allow_missing_cache:
        print("\n[ATENCIÓN] Faltan ficheros obligatorios en cache/pipeline_cache/:")
        for name in cache_missing:
            print(f"  - {pipeline_cache_dir / name}")
        print("\nEl registro Zenodo debe contener estos .npz para reproducir el pipeline paper-aligned sin reconstruir cachés.")
        print("Usa --allow-missing-cache solo si quieres regenerarlos localmente.")
        return 3

    if artifacts_missing and not args.allow_missing_artifacts:
        print("\n[ATENCIÓN] No se ha encontrado una carpeta artifacts_ablation/ no vacía.")
        print(f"  - destino esperado: {artifacts_dir}")
        print("\nEl registro Zenodo debe contener la carpeta/archivo artifacts_ablation para reproducir las tablas y figuras derivadas.")
        print("Usa --allow-missing-artifacts solo si quieres regenerarla localmente.")
        return 4

    if backbone_artifacts_missing and not args.allow_missing_backbone_artifacts:
        print("\n[ATENCIÓN] No se ha encontrado una carpeta artifacts_backbone_ablation/ no vacía.")
        print(f"  - destino esperado: {backbone_artifacts_dir}")
        print("\nEl registro Zenodo debe contener la carpeta/archivo de ablación de backbones")
        print("(por ejemplo artifacts_backbone_ablation/ o artifact_back_bone_ablation/) para reproducir ST6.")
        print("Usa --allow-missing-backbone-artifacts solo si quieres regenerarla localmente.")
        return 5

    print("\n[ok] Datos preparados.")
    print(f"  processed:       {processed_dir}")
    print(f"  external:        {external_dir}")
    print(f"  pipeline cache:  {pipeline_cache_dir}")
    print(f"  artifacts:       {artifacts_dir}")
    print(f"  backbone arts.:  {backbone_artifacts_dir}")
    print("Siguiente paso: python scripts/00_check_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
