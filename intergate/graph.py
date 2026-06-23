"""
graph.py
--------
Build the backbone protein–protein interaction graph from:
  - OmniPath (directed, signed)
  - HuRI / Interactome Atlas (undirected, with optional confidence scores)

Edge types:
  0 = HuRI (physical PPI)
  1 = OmniPath stimulation (+)
  2 = OmniPath inhibition  (−)

This module mirrors the notebook cells 2 / 3.1 / 3.2 / 3.3 exactly, including:
  - manual SYMBOL→ENSG overrides  (FAM118A, TMEM30A, TMEM30B …)
  - REQUIRED_CONNECTED_GENES validation gate on every cache load
  - EXPECTED_CONNECTED_GENES optional reference check
  - per-dataset HuRI cache with mapping fingerprint
  - global backbone cache with dedup
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

from intergate.config import CFG as C


# ═══════════════════════════════════════════════════════════════════════════
#  Gene name utilities  (notebook § 2)
# ═══════════════════════════════════════════════════════════════════════════

def canonical_gene(g: str) -> str:
    """Clean a raw gene identifier to a canonical token."""
    if g is None:
        return ""
    g = str(g).strip()
    g = re.sub(r"^hsa:", "", g, flags=re.I)
    for sep in ["///", "|", ";", ","]:
        if sep in g:
            g = g.split(sep, 1)[0].strip()
    if " " in g:
        g = g.split(" ", 1)[0].strip()
    g = re.sub(r"^(ENSG\d+)\.\d+$", r"\1", g, flags=re.I)
    g = re.sub(r"^(ENST\d+)\.\d+$", r"\1", g, flags=re.I)
    return g


def canonical_gene_upper(g: str) -> str:
    return canonical_gene(g).upper()


# ═══════════════════════════════════════════════════════════════════════════
#  Expected-connected-genes reference  (notebook § 3.2)
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_gene_set(obj) -> Optional[Set[str]]:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        col = "gene" if "gene" in obj.columns else obj.columns[-1]
        vals = obj[col].tolist()
    elif isinstance(obj, pd.Series):
        vals = obj.tolist()
    elif isinstance(obj, (list, tuple, set, np.ndarray, pd.Index)):
        vals = list(obj)
    else:
        return None
    out = set()
    for x in vals:
        s = str(x).strip().upper()
        if s and s != "NAN":
            out.add(s)
    return out


def load_expected_connected_genes(
    csv_path: Optional[Path] = None,
) -> Optional[Set[str]]:
    """Load a reference set of expected connected genes from CSV (optional)."""
    csv_path = csv_path or getattr(C, "EXPECTED_CONNECTED_GENES_CSV", None)
    if csv_path is not None:
        p = Path(csv_path)
        if p.exists():
            df = pd.read_csv(p)
            s = _normalize_gene_set(df)
            if s:
                print(f"[graph] Expected-connected reference: {len(s)} genes")
                return s
    # fallback: genes_usados.csv in cwd
    p = Path("genes_usados.csv")
    if p.exists():
        df = pd.read_csv(p)
        s = _normalize_gene_set(df)
        if s:
            print(f"[graph] Expected-connected reference (cwd): {len(s)} genes")
            return s
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Cache validation helpers  (notebook § 3.2 / 3.3)
# ═══════════════════════════════════════════════════════════════════════════

def _connected_gene_set(edge_index: np.ndarray, genes_order: List[str]) -> Set[str]:
    if edge_index.shape[1] == 0:
        return set()
    deg = np.bincount(edge_index.reshape(-1), minlength=len(genes_order))
    return {str(genes_order[i]).strip().upper() for i in np.where(deg > 0)[0]}


def _validate_huri_cache(
    src: np.ndarray,
    tgt: np.ndarray,
    genes_order: List[str],
    required_connected: Optional[Set[str]] = None,
) -> Tuple[bool, dict]:
    """Validate a cached HuRI edge array."""
    if src.ndim != 1 or tgt.ndim != 1 or src.shape[0] != tgt.shape[0]:
        return False, {"reason": "shape_mismatch"}
    n = len(genes_order)
    if src.shape[0] > 0:
        if src.min() < 0 or tgt.min() < 0 or src.max() >= n or tgt.max() >= n:
            return False, {"reason": "out_of_bounds", "n_nodes": n}

    ei = np.vstack([src, tgt])
    connected = _connected_gene_set(ei, genes_order)
    missing = sorted((required_connected or set()) - connected)
    return len(missing) == 0, {"connected": len(connected), "missing_required": missing}


def _validate_backbone_cache(
    edge_index: np.ndarray,
    genes_order: List[str],
    required_connected: Optional[Set[str]] = None,
    expected_connected: Optional[Set[str]] = None,
) -> Tuple[bool, dict]:
    """Validate a cached backbone .npz (merged HuRI+OmniPath)."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return False, {"reason": "shape_mismatch"}
    n = len(genes_order)
    if edge_index.shape[1] > 0:
        if edge_index.min() < 0 or edge_index.max() >= n:
            return False, {"reason": "out_of_bounds", "n_nodes": n}

    connected = _connected_gene_set(edge_index, genes_order)
    missing_required = sorted((required_connected or set()) - connected)

    missing_expected: list = []
    extra_expected: list = []
    if expected_connected is not None:
        missing_expected = sorted(expected_connected - connected)
        extra_expected = sorted(connected - expected_connected)

    ok = (
        len(missing_required) == 0
        and len(missing_expected) == 0
        and len(extra_expected) == 0
    )
    detail = {
        "connected_genes": len(connected),
        "missing_required": missing_required,
        "n_missing_expected": len(missing_expected),
        "n_extra_expected": len(extra_expected),
        "missing_expected_head": missing_expected[:20],
        "extra_expected_head": extra_expected[:20],
    }
    return ok, detail


# ═══════════════════════════════════════════════════════════════════════════
#  SYMBOL → ENSG mapping  (notebook § 3.2 steps 1 + 2)
# ═══════════════════════════════════════════════════════════════════════════

def _mapping_cache_paths(cache_dir: Path) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / "symbol_to_ensg_cache.json",
        cache_dir / "symbol_to_ensg_unresolved.json",
    )


def _load_symbol_ensg_cache(cache_dir: Path) -> Tuple[Dict[str, str], Dict[str, bool]]:
    map_path, unres_path = _mapping_cache_paths(cache_dir)
    mapping: Dict[str, str] = {}
    unresolved: Dict[str, bool] = {}
    if map_path.exists():
        try:
            raw = json.loads(map_path.read_text(encoding="utf-8"))
            mapping = {str(k).upper(): str(v).split(".")[0].upper()
                       for k, v in raw.items() if v}
        except Exception:
            mapping = {}
    if unres_path.exists():
        try:
            raw = json.loads(unres_path.read_text(encoding="utf-8"))
            unresolved = {str(k).upper(): bool(v) for k, v in raw.items()}
        except Exception:
            unresolved = {}
    return mapping, unresolved


def _save_symbol_ensg_cache(
    cache_dir: Path,
    mapping: Dict[str, str],
    unresolved: Dict[str, bool],
) -> None:
    map_path, unres_path = _mapping_cache_paths(cache_dir)
    map_path.write_text(
        json.dumps(dict(sorted(mapping.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unres_path.write_text(
        json.dumps(dict(sorted(unresolved.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pick_ensembl_gene(row: dict) -> Optional[str]:
    ens = row.get("ensembl", None)
    if isinstance(ens, dict):
        gene = ens.get("gene")
        return str(gene).split(".")[0].upper() if gene else None
    if isinstance(ens, list):
        vals = []
        for item in ens:
            if isinstance(item, dict) and item.get("gene"):
                vals.append(str(item["gene"]).split(".")[0].upper())
        vals = sorted(set(v for v in vals if re.match(r"^ENSG\d+$", v)))
        return vals[0] if vals else None
    return None


def _build_ensg_index(
    genes: List[str],
    mapping_cache_dir: Path = C.PIPELINE_CACHE_DIR,
    manual_overrides: Optional[Dict[str, str]] = None,
    required_connected: Optional[Set[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, int], str]:
    """
    Build a stable SYMBOL→ENSG mapping with:
      1. persistent disk cache  (pipeline_cache/)
      2. multi-scope MyGene queries
      3. manual overrides       (MANUAL_SYMBOL2ENSG from config)

    Returns (symbol2ensg, ensg2idx, mapping_fingerprint).
    """
    manual_overrides = manual_overrides or getattr(C, "MANUAL_SYMBOL2ENSG", {})
    required_connected = required_connected or getattr(C, "REQUIRED_CONNECTED_GENES", set())

    cleaned = [canonical_gene_upper(g) for g in genes]
    frac_ensg = float(np.mean([bool(re.match(r"^ENSG\d+$", g)) for g in cleaned]))

    cache_map, cache_unresolved = _load_symbol_ensg_cache(mapping_cache_dir)
    symbol2ensg: Dict[str, str] = {}

    # ── Direct / cache resolution ──────────────────────────────────────────
    for g in cleaned:
        if re.match(r"^ENSG\d+$", g):
            symbol2ensg[g] = g
        elif g in cache_map:
            symbol2ensg[g] = cache_map[g]

    unresolved_symbols = [
        g for g in cleaned
        if g not in symbol2ensg and g not in cache_unresolved
    ]

    # ── MyGene multi-scope lookup ──────────────────────────────────────────
    if unresolved_symbols and frac_ensg <= 0.9:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            batch = 500
            newly_resolved: Dict[str, str] = {}
            newly_unresolved: Dict[str, bool] = {}

            for i in range(0, len(unresolved_symbols), batch):
                q = unresolved_symbols[i : i + batch]
                res = mg.querymany(
                    q,
                    scopes="symbol,alias,ensembl.gene,retired",
                    fields="ensembl.gene,symbol,alias",
                    species="human",
                    as_dataframe=False,
                    returnall=False,
                    verbose=False,
                )
                for row in res:
                    query = canonical_gene_upper(row.get("query", ""))
                    if not query:
                        continue
                    ensg = _pick_ensembl_gene(row)
                    if ensg:
                        newly_resolved[query] = ensg
                    else:
                        newly_unresolved[query] = True

            cache_map.update(newly_resolved)
            cache_unresolved.update(newly_unresolved)
            symbol2ensg.update(newly_resolved)
            _save_symbol_ensg_cache(mapping_cache_dir, cache_map, cache_unresolved)
            print(
                f"[map] SYMBOL→ENSG mapped: {len(symbol2ensg)}/{len(cleaned)} "
                f"| new={len(newly_resolved)} | unresolved={len(newly_unresolved)}"
            )
        except Exception as e:
            if not symbol2ensg:
                raise RuntimeError(
                    "Could not map SYMBOL→ENSG via MyGene and cache is empty. "
                    "Install mygene (`pip install mygene`) and ensure internet access."
                ) from e
            print(f"[map] WARNING: MyGene lookup failed, using cache/direct only ({e})")
    else:
        print(f"[map] SYMBOL→ENSG from direct/cache: {len(symbol2ensg)}/{len(cleaned)}")

    # ── Manual overrides (ALWAYS applied AFTER mygene) ─────────────────────
    for sym, ensg in manual_overrides.items():
        symbol2ensg[sym.upper()] = ensg.split(".")[0].upper()
    print(f"[map] SYMBOL→ENSG after overrides: {len(symbol2ensg)}")

    for g in sorted(required_connected):
        ensg = symbol2ensg.get(g.upper())
        print(f"[map] override check {g} → {ensg}")

    missing_req = [g for g in sorted(required_connected) if g.upper() not in symbol2ensg]
    assert not missing_req, f"Missing required ENSG mappings: {missing_req}"

    # Persist cache with overrides included
    cache_map.update({k: v for k, v in symbol2ensg.items()
                      if re.match(r"^ENSG\d+$", v)})
    try:
        _save_symbol_ensg_cache(mapping_cache_dir, cache_map, cache_unresolved)
    except Exception as e:
        print(f"[WARN] Could not save SYMBOL2ENSG cache: {e}")

    # ── Build ensg2idx over current gene universe ──────────────────────────
    ensg2idx: Dict[str, int] = {}
    for i, g in enumerate(cleaned):
        ensg = symbol2ensg.get(g)
        if ensg and ensg not in ensg2idx:
            ensg2idx[ensg] = i

    mapping_blob = json.dumps(sorted(symbol2ensg.items()), ensure_ascii=False)
    mapping_fp = hashlib.sha256(mapping_blob.encode("utf-8")).hexdigest()[:12]

    return symbol2ensg, ensg2idx, mapping_fp


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _download_if_missing(url: str, out_path: Path, chunk_mb: int = 8) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
#  OmniPath  (notebook § 3.1)
# ═══════════════════════════════════════════════════════════════════════════

def load_omnipath(
    genes: List[str],
    cache_dir: Path = C.OMNIPATH_CACHE_DIR,
    url: str = C.OMNIPATH_URL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Download (or load cached) OmniPath interactions and filter to gene universe.

    Returns (edge_index, edge_weight, edge_type).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = cache_dir / "omnipath_interactions.tsv"

    if not tsv_path.exists() or tsv_path.stat().st_size < 10_000:
        print("[OmniPath] Downloading TSV …")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        tsv_path.write_bytes(r.content)

    print(f"[OmniPath] TSV: {tsv_path} | size: {tsv_path.stat().st_size}")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"[OmniPath] columns: {df.columns.tolist()}")

    src_col = "source_genesymbol" if "source_genesymbol" in df.columns else "source"
    tgt_col = "target_genesymbol" if "target_genesymbol" in df.columns else "target"

    need = [
        src_col, tgt_col,
        "is_stimulation", "is_inhibition",
        "consensus_stimulation", "consensus_inhibition",
    ]
    df = df[need].copy()
    df[src_col] = df[src_col].astype(str).str.strip().str.upper()
    df[tgt_col] = df[tgt_col].astype(str).str.strip().str.upper()

    gene2idx = {str(g).upper(): i for i, g in enumerate(genes)}
    mask = df[src_col].isin(gene2idx) & df[tgt_col].isin(gene2idx)
    df = df.loc[mask].reset_index(drop=True)
    print(f"[OmniPath] interactions within gene universe: {len(df)}")

    src_idx = df[src_col].map(gene2idx).astype(np.int64).values
    tgt_idx = df[tgt_col].map(gene2idx).astype(np.int64).values
    edge_index = np.vstack([src_idx, tgt_idx]).astype(np.int64)

    stim = (df["consensus_stimulation"].astype(bool) | df["is_stimulation"].astype(bool)).values
    inh = (df["consensus_inhibition"].astype(bool) | df["is_inhibition"].astype(bool)).values

    edge_weight = np.ones(len(df), dtype=np.float32)
    edge_weight[inh & ~stim] = -1.0
    edge_weight[stim & ~inh] = +1.0

    # edge types: 1=OP(+), 2=OP(-)
    edge_type = np.ones(len(df), dtype=np.int64)
    edge_type[inh & ~stim] = 2

    print(f"[OmniPath] edge_index: {edge_index.shape}, "
          f"edge_weight: {edge_weight.shape}, edge_type: {edge_type.shape}")
    return edge_index, edge_weight, edge_type


# ═══════════════════════════════════════════════════════════════════════════
#  HuRI / Interactome Atlas  (notebook § 3.2)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_ensg(field: str) -> Optional[str]:
    m = re.search(r"ENSG\d{11}", str(field))
    return m.group(0) if m else None


def _extract_author_score(field: str) -> Optional[float]:
    m = re.search(r"([0-9]*\.[0-9]+|[0-9]+)", str(field))
    return float(m.group(1)) if m else None


def load_huri(
    genes: List[str],
    datasets: List[str] = C.HURI_DATASETS,
    cache_dir: Path = C.HURI_CACHE_DIR,
    use_confidence: bool = C.USE_HURI_CONFIDENCE,
    min_score: float = C.HURI_MIN_SCORE,
    default_weight: float = C.HURI_DEFAULT_WEIGHT,
    force_rebuild: bool = C.FORCE_REBUILD_HURI_CACHE,
    required_connected: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load HuRI physical PPI edges for the provided gene universe.
    Includes REQUIRED_CONNECTED_GENES validation on every cache load.

    Returns (edge_index, edge_weight, edge_type).
    """
    required_connected = required_connected or getattr(C, "REQUIRED_CONNECTED_GENES", set())

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    symbol2ensg, ensg2idx, mapping_fp = _build_ensg_index(genes)
    ensg_set = set(ensg2idx.keys())
    print(f"[HuRI] ENSG IDs in universe: {len(ensg_set)}")

    all_src, all_tgt, all_w = [], [], []

    for ds in datasets:
        ds = str(ds)
        tsv_url = f"https://interactome-atlas.org/data/{ds}.tsv"
        psi_url = f"https://interactome-atlas.org/data/{ds}.psi"
        tsv_path = cache_dir / f"{ds}.tsv"
        psi_path = cache_dir / f"{ds}.psi"

        _download_if_missing(tsv_url, tsv_path)
        use_score = use_confidence and (ds.lower() == "huri")

        # ----- WITH confidence scores -----
        if use_score:
            _download_if_missing(psi_url, psi_path)
            cache_npz = cache_dir / (
                f"{ds}.filtered.with_score.min{min_score}"
                f".impute_median.map{mapping_fp}.npz"
            )

            use_cached = False
            if cache_npz.exists() and not force_rebuild:
                z = np.load(cache_npz)
                src_tmp = z["src"].astype(np.int64)
                tgt_tmp = z["tgt"].astype(np.int64)
                w_tmp = z["w"].astype(np.float32)

                ok, detail = _validate_huri_cache(
                    src_tmp, tgt_tmp, genes,
                    required_connected=required_connected,
                )
                if ok:
                    src, tgt, w = src_tmp, tgt_tmp, w_tmp
                    use_cached = True
                    print(f"[HuRI/{ds}] cache (score) valid: {len(w)} edges")
                else:
                    print(f"[HuRI/{ds}] cache (score) IGNORED: {detail}")

            if not use_cached:
                print(f"[HuRI/{ds}] parsing PSI for author scores …")
                pair2score: dict = {}
                missing_pairs: set = set()

                with open(psi_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        cols = line.rstrip("\n").split("\t")
                        if len(cols) < 15:
                            continue
                        ea = _extract_ensg(cols[2])
                        eb = _extract_ensg(cols[3])
                        if ea is None or eb is None:
                            continue
                        ea, eb = ea.upper(), eb.upper()
                        if ea not in ensg_set or eb not in ensg_set:
                            continue
                        key = (ea, eb) if ea <= eb else (eb, ea)
                        sc = _extract_author_score(cols[14])
                        if sc is None:
                            missing_pairs.add(key)
                            continue
                        sc = float(sc)
                        if sc < float(min_score):
                            continue
                        prev = pair2score.get(key, -1.0)
                        if sc > prev:
                            pair2score[key] = sc

                median_sc = (
                    float(np.median(np.fromiter(pair2score.values(), dtype=np.float32)))
                    if pair2score else float(default_weight)
                )
                imputed = 0
                for key in missing_pairs:
                    if key in pair2score:
                        continue
                    if median_sc < float(min_score):
                        continue
                    pair2score[key] = median_sc
                    imputed += 1

                print(
                    f"[HuRI/{ds}] scores: obs={len(pair2score)-imputed} | "
                    f"missing_seen={len(missing_pairs)} | imputed={imputed} | "
                    f"median={median_sc:.4f}"
                )

                src_l, tgt_l, w_l = [], [], []
                for (ea, eb), sc in pair2score.items():
                    i, j = ensg2idx[ea], ensg2idx[eb]
                    src_l += [i, j]
                    tgt_l += [j, i]
                    w_l += [float(sc), float(sc)]

                src = np.array(src_l, dtype=np.int64)
                tgt = np.array(tgt_l, dtype=np.int64)
                w = np.array(w_l, dtype=np.float32)
                np.savez_compressed(cache_npz, src=src, tgt=tgt, w=w)
                print(f"[HuRI/{ds}] PSI→cache: {len(w)} edges (bidirectional)")

        # ----- WITHOUT confidence scores -----
        else:
            cache_npz = cache_dir / f"{ds}.filtered.no_score.map{mapping_fp}.npz"

            use_cached = False
            if cache_npz.exists() and not force_rebuild:
                z = np.load(cache_npz)
                src_tmp = z["src"].astype(np.int64)
                tgt_tmp = z["tgt"].astype(np.int64)
                w_tmp = z["w"].astype(np.float32)

                ok, detail = _validate_huri_cache(
                    src_tmp, tgt_tmp, genes,
                    required_connected=required_connected,
                )
                if ok:
                    src, tgt, w = src_tmp, tgt_tmp, w_tmp
                    use_cached = True
                    print(f"[HuRI/{ds}] cache (no_score) valid: {len(w)} edges")
                else:
                    print(f"[HuRI/{ds}] cache (no_score) IGNORED: {detail}")

            if not use_cached:
                df_h = pd.read_csv(
                    tsv_path, sep="\t", header=None, names=["ensg_a", "ensg_b"]
                )
                df_h["ensg_a"] = df_h["ensg_a"].astype(str).str.split(".").str[0].str.upper()
                df_h["ensg_b"] = df_h["ensg_b"].astype(str).str.split(".").str[0].str.upper()
                df_h = df_h[df_h["ensg_a"].isin(ensg_set) & df_h["ensg_b"].isin(ensg_set)]
                print(f"[HuRI/{ds}] interactions in universe: {len(df_h)}")

                src_l, tgt_l, w_l = [], [], []
                for ea, eb in zip(df_h["ensg_a"].values, df_h["ensg_b"].values):
                    i, j = ensg2idx[ea], ensg2idx[eb]
                    src_l += [i, j]
                    tgt_l += [j, i]
                    w_l += [float(default_weight), float(default_weight)]

                src = np.array(src_l, dtype=np.int64)
                tgt = np.array(tgt_l, dtype=np.int64)
                w = np.array(w_l, dtype=np.float32)
                np.savez_compressed(cache_npz, src=src, tgt=tgt, w=w)
                print(f"[HuRI/{ds}] TSV→cache: {len(w)} edges (bidirectional)")

        all_src.append(src)
        all_tgt.append(tgt)
        all_w.append(w)

    # ── Merge datasets ─────────────────────────────────────────────────────
    if not all_w:
        print("[HuRI] no edges loaded (empty).")
        return np.zeros((2, 0), np.int64), np.zeros(0, np.float32), np.zeros(0, np.int64)

    src_all = np.concatenate(all_src)
    tgt_all = np.concatenate(all_tgt)
    w_all = np.concatenate(all_w)

    edge_index = np.vstack([src_all, tgt_all]).astype(np.int64)
    edge_weight = w_all.astype(np.float32)
    edge_type = np.zeros(edge_weight.shape[0], dtype=np.int64)  # 0 = HuRI

    print(
        f"[HuRI/Interactome] edges={edge_index.shape[1]} | "
        f"w min/median/max={edge_weight.min():.3f}/"
        f"{np.median(edge_weight):.3f}/{edge_weight.max():.3f}"
    )

    deg = np.bincount(edge_index.reshape(-1), minlength=len(genes))
    genes_upper = [str(g).upper() for g in genes]
    connected = {genes_upper[i] for i in np.where(deg > 0)[0]}
    print(f"[HuRI only] connected genes (deg>0): {len(connected)} "
          f"({len(connected)/len(genes)*100:.1f}%)")
    for g in sorted(required_connected):
        print(f"[HuRI only check] {g}: {g.upper() in connected}")

    return edge_index, edge_weight, edge_type


# ═══════════════════════════════════════════════════════════════════════════
#  Deduplication  (notebook § 3.3)
# ═══════════════════════════════════════════════════════════════════════════

def dedup_edges(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    edge_type: np.ndarray,
    prefer_type: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduplicate by (src, tgt), keeping the edge with highest |weight|.
    On ties prefer `prefer_type` (default: HuRI=0).
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be (2,E), got {edge_index.shape}")
    E = edge_index.shape[1]
    if edge_weight.shape[0] != E or edge_type.shape[0] != E:
        raise ValueError(
            f"Dimension mismatch: E={E}, ew={edge_weight.shape}, et={edge_type.shape}"
        )
    if E == 0:
        return edge_index, edge_weight.astype(np.float32), edge_type.astype(np.int64)

    ei0 = edge_index[0].astype(np.int64, copy=False)
    ei1 = edge_index[1].astype(np.int64, copy=False)
    ew = edge_weight.astype(np.float32, copy=False)
    et = edge_type.astype(np.int64, copy=False)

    key = (ei0 << np.int64(32)) | ei1
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    src_s = ei0[order]
    tgt_s = ei1[order]
    ew_s = ew[order]
    et_s = et[order]

    keep_src, keep_tgt, keep_w, keep_t = [], [], [], []
    i = 0
    while i < E:
        j = i + 1
        bsrc, btgt, bw, bt = src_s[i], tgt_s[i], ew_s[i], et_s[i]
        while j < E and key_s[j] == key_s[i]:
            cw, ct = ew_s[j], et_s[j]
            if abs(cw) > abs(bw) or (
                abs(cw) == abs(bw) and ct == prefer_type and bt != prefer_type
            ):
                bsrc, btgt, bw, bt = src_s[j], tgt_s[j], cw, ct
            j += 1
        keep_src.append(bsrc)
        keep_tgt.append(btgt)
        keep_w.append(bw)
        keep_t.append(bt)
        i = j

    ei2 = np.vstack([np.asarray(keep_src, np.int64), np.asarray(keep_tgt, np.int64)])
    ew2 = np.asarray(keep_w, np.float32)
    et2 = np.asarray(keep_t, np.int64)
    return ei2, ew2, et2


# ═══════════════════════════════════════════════════════════════════════════
#  High-level backbone builder  (notebook § 3.3  merge + backbone cache)
# ═══════════════════════════════════════════════════════════════════════════

def build_backbone(
    genes: List[str],
    use_omnipath: bool = C.USE_OMNIPATH,
    use_huri: bool = C.USE_HURI,
    backbone_cache_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full backbone graph (HuRI + OmniPath), with:
      - per-source loading  (OmniPath, HuRI)
      - deduplication on (src, tgt)
      - global backbone cache with validation
      - REQUIRED_CONNECTED_GENES gate
      - EXPECTED_CONNECTED_GENES optional reference check

    Returns (edge_index, edge_weight, edge_type).
    """
    backbone_cache_dir = Path(
        backbone_cache_dir
        or getattr(C, "BACKBONE_CACHE_DIR", C.DATA_DIR / "backbone_cache")
    )
    backbone_cache_dir.mkdir(parents=True, exist_ok=True)
    force_rebuild = force_rebuild or getattr(C, "FORCE_REBUILD_BACKBONE_CACHE", False)

    required_connected = getattr(C, "REQUIRED_CONNECTED_GENES", set())
    expected_connected = load_expected_connected_genes()

    # ── Try backbone cache ─────────────────────────────────────────────────
    gene_sig = hashlib.md5(
        "||".join(map(str, genes)).encode("utf-8")
    ).hexdigest()[:16]
    score_flag = 1 if C.USE_HURI_CONFIDENCE else 0
    min_score_tag = str(C.HURI_MIN_SCORE).replace(".", "p")
    cache_npz = backbone_cache_dir / (
        f"backbone_global__n{len(genes)}__sig{gene_sig}"
        f"__op{int(bool(use_omnipath))}__huri{int(bool(use_huri))}"
        f"__score{score_flag}__min{min_score_tag}.npz"
    )

    if cache_npz.exists() and not force_rebuild:
        z = np.load(cache_npz)
        ei_tmp = z["edge_index"].astype(np.int64)
        ew_tmp = z["edge_weight"].astype(np.float32)
        et_tmp = z["edge_type"].astype(np.int64)

        ok, detail = _validate_backbone_cache(
            ei_tmp, genes,
            required_connected=required_connected,
            expected_connected=expected_connected,
        )
        if ok:
            print(f"[Backbone cache] loaded: {cache_npz.name}")
            _print_backbone_summary(ei_tmp, genes, required_connected, expected_connected)
            return ei_tmp, ew_tmp, et_tmp
        else:
            print(f"[Backbone cache] IGNORED: {detail}")

    # ── Build from sources ─────────────────────────────────────────────────
    ei_parts, ew_parts, et_parts = [], [], []

    if use_omnipath:
        ei, ew, et = load_omnipath(genes)
        ei_parts.append(ei)
        ew_parts.append(ew)
        et_parts.append(et)
        print(f"[Merge input] OmniPath edges={ei.shape[1]}")

    if use_huri:
        ei, ew, et = load_huri(genes, required_connected=required_connected)
        ei_parts.append(ei)
        ew_parts.append(ew)
        et_parts.append(et)
        print(f"[Merge input] HuRI edges={ei.shape[1]}")

    if not ei_parts:
        return np.zeros((2, 0), np.int64), np.zeros(0, np.float32), np.zeros(0, np.int64)

    edge_index = np.hstack(ei_parts)
    edge_weight = np.concatenate(ew_parts).astype(np.float32)
    edge_type = np.concatenate(et_parts).astype(np.int64)

    assert edge_index.shape[1] == edge_weight.shape[0] == edge_type.shape[0], (
        f"MERGE mismatch: ei={edge_index.shape}, ew={edge_weight.shape}, et={edge_type.shape}"
    )

    print(f"[graph] Before dedup: {edge_index.shape[1]} edges")
    edge_index, edge_weight, edge_type = dedup_edges(
        edge_index, edge_weight, edge_type, prefer_type=0,
    )
    print(f"[graph] After  dedup: {edge_index.shape[1]} edges")

    # ── Save backbone cache ────────────────────────────────────────────────
    np.savez_compressed(
        cache_npz,
        edge_index=edge_index.astype(np.int64),
        edge_weight=edge_weight.astype(np.float32),
        edge_type=edge_type.astype(np.int64),
    )
    print(f"[Backbone cache] saved: {cache_npz.name}")

    _print_backbone_summary(edge_index, genes, required_connected, expected_connected)
    return edge_index, edge_weight, edge_type


def _print_backbone_summary(
    edge_index: np.ndarray,
    genes: List[str],
    required_connected: Set[str],
    expected_connected: Optional[Set[str]],
) -> None:
    """Print connectivity summary (matches notebook § 3.3 final checks)."""
    deg = np.bincount(edge_index.reshape(-1), minlength=len(genes))
    connected = _connected_gene_set(edge_index, genes)
    print(
        f"[Backbone HuRI+OmniPath] connected genes (deg>0): "
        f"{len(connected)} ({len(connected)/len(genes)*100:.1f}%)"
    )

    for g in sorted(required_connected):
        print(f"[Backbone check] {g}: {g.upper() in connected}")

    if expected_connected is not None:
        missing = sorted(expected_connected - connected)
        extra = sorted(connected - expected_connected)
        print(f"[Backbone refcheck] missing={len(missing)} extra={len(extra)}")
        if missing:
            print(f"[Backbone refcheck] missing head: {missing[:20]}")
        if extra:
            print(f"[Backbone refcheck] extra head: {extra[:20]}")
