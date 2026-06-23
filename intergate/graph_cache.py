"""
graph_cache.py
--------------
Cacheo persistente del grafo backbone y de las features de regulador (X_h).

Primera ejecución  → genera, guarda a disco.
Siguientes         → carga directamente (con validación de REQUIRED_CONNECTED_GENES).

Uso en el notebook
------------------
    from graph_cache import get_or_build_backbone, get_or_build_Xh

    # 1. Grafo backbone  (reemplaza a build_backbone)
    edge_index, edge_weight, edge_type = get_or_build_backbone(
        genes, cache_dir=CACHE_DIR,
    )

    # ... filter_to_connected_genes, scale_features como siempre ...

    # 2. X_h / Xs_graph   (reemplaza a build_regulator_features)
    Xs_graph, graph_feat_names = get_or_build_Xh(
        Xs_gene, genes_conn, edge_index_conn, cache_dir=CACHE_DIR,
    )
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

import intergate.config as _cfg_module
from intergate.config import CFG as C
from intergate.graph import build_backbone, _validate_backbone_cache, load_expected_connected_genes
from intergate.data import build_regulator_features


# ── Defaults ────────────────────────────────────────────────────────────────
_DEFAULT_CACHE_DIR = Path(C.PIPELINE_CACHE_DIR)


# ── Utilidades internas ─────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _genes_fingerprint(genes: List[str]) -> str:
    """Hash determinista de la lista de genes (orden incluido)."""
    h = hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()[:16]
    return h


def _backbone_config_fingerprint() -> str:
    """Hash de los flags de config que afectan al backbone."""
    blob = json.dumps({
        "USE_HURI": C.USE_HURI,
        "USE_OMNIPATH": C.USE_OMNIPATH,
        "USE_HURI_CONFIDENCE": C.USE_HURI_CONFIDENCE,
        "HURI_MIN_SCORE": C.HURI_MIN_SCORE,
        "HURI_DEFAULT_WEIGHT": C.HURI_DEFAULT_WEIGHT,
        "HURI_DATASETS": C.HURI_DATASETS,
        # include manual overrides so a change in overrides invalidates cache
        "MANUAL_SYMBOL2ENSG": getattr(C, "MANUAL_SYMBOL2ENSG", {}),
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _xh_config_fingerprint(
    stats: Tuple[str, ...],
    min_targets: int,
    max_regulators: Optional[int],
) -> str:
    blob = json.dumps({
        "stats": list(stats),
        "min_targets": min_targets,
        "max_regulators": max_regulators,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
#  1.  BACKBONE GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def get_or_build_backbone(
    genes: List[str],
    *,
    cache_dir: Optional[Path] = None,
    force_rebuild: bool = False,
    use_omnipath: bool = C.USE_OMNIPATH,
    use_huri: bool = C.USE_HURI,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve (edge_index, edge_weight, edge_type) del backbone completo.

    - Si el fichero de caché existe Y pasa validación → lo lee.
    - Si no existe, falla validación, o force_rebuild → llama a build_backbone.

    Validación: comprueba que REQUIRED_CONNECTED_GENES estén todos presentes
    en el grafo cacheado.  Si falta alguno, se regenera automáticamente.

    Parameters
    ----------
    genes : list[str]
        Universo de genes (orden importa: define los índices de nodo).
    cache_dir : Path, optional
        Carpeta de caché.  Por defecto ``DATA_DIR/pipeline_cache``.
    force_rebuild : bool
        Si True, regenera aunque exista caché.

    Returns
    -------
    edge_index  : (2, E) int64
    edge_weight : (E,)   float32
    edge_type   : (E,)   int64
    """
    cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
    _ensure_dir(cache_dir)

    required_connected: Set[str] = getattr(C, "REQUIRED_CONNECTED_GENES", set())
    expected_connected = load_expected_connected_genes()

    gfp = _genes_fingerprint(genes)
    cfp = _backbone_config_fingerprint()
    fname = f"backbone__{gfp}__{cfp}.npz"
    cache_path = cache_dir / fname

    # ── Intentar cargar + validar ──────────────────────────────────────────
    if cache_path.exists() and not force_rebuild:
        print(f"[graph_cache] Intentando cargar backbone: {cache_path.name}")
        z = np.load(cache_path)
        edge_index  = z["edge_index"].astype(np.int64)
        edge_weight = z["edge_weight"].astype(np.float32)
        edge_type   = z["edge_type"].astype(np.int64)

        ok, detail = _validate_backbone_cache(
            edge_index, genes,
            required_connected=required_connected,
            expected_connected=expected_connected,
        )
        if ok:
            print(f"[graph_cache] Backbone válido desde caché: "
                  f"{edge_index.shape[1]} aristas, {len(genes)} genes")
            return edge_index, edge_weight, edge_type
        else:
            print(f"[graph_cache] Backbone cache IGNORADO (validación): {detail}")

    # ── Generar y guardar ──────────────────────────────────────────────────
    print(f"[graph_cache] Generando backbone (build_backbone)…")
    edge_index, edge_weight, edge_type = build_backbone(
        genes, use_omnipath=use_omnipath, use_huri=use_huri,
    )
    np.savez_compressed(
        cache_path,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
    )
    print(f"[graph_cache] Backbone guardado en: {cache_path}")
    print(f"[graph_cache]   → {edge_index.shape[1]} aristas")
    return edge_index, edge_weight, edge_type


# ═══════════════════════════════════════════════════════════════════════════
#  2.  X_h  (regulator features = Xs_graph)
# ═══════════════════════════════════════════════════════════════════════════

def get_or_build_Xh(
    Xs_gene: np.ndarray,
    genes: List[str],
    edge_index: np.ndarray,
    *,
    cache_dir: Optional[Path] = None,
    force_rebuild: bool = False,
    stats: Tuple[str, ...] = C.REG_STATS,
    min_targets: int = C.REG_MIN_GENES,
    max_regulators: Optional[int] = C.REG_MAX_REGULATORS,
    tag: str = "",
) -> Tuple[np.ndarray, List[str]]:
    """
    Devuelve (Xs_graph, graph_feat_names).

    NOTA: Xs_graph depende de Xs_gene (datos escalados), que a su vez depende
    del split.  Por eso se usa un hash de la propia matriz de entrada como
    parte de la clave de caché.  Si cambias semilla/split/escalado,
    se regenerará automáticamente.

    Parameters
    ----------
    Xs_gene : (N, G) float32
        Matriz de expresión escalada.
    genes : list[str]
        Genes conectados (tras filter_to_connected_genes).
    edge_index : (2, E) int64
        Aristas del backbone (conectadas).
    cache_dir : Path, optional
    force_rebuild : bool
    stats, min_targets, max_regulators : parámetros de build_regulator_features
    tag : str
        Etiqueta adicional para diferenciar cachés (p.ej. "seed1200").

    Returns
    -------
    Xs_graph        : (N, F) float32
    graph_feat_names: list[str]
    """
    cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
    _ensure_dir(cache_dir)

    # Fingerprint: genes + config de reguladores + hash de datos de entrada
    gfp = _genes_fingerprint(genes)
    xfp = _xh_config_fingerprint(stats, min_targets, max_regulators)
    # Hash rápido de los datos escalados (forma + muestra de valores)
    data_hash = hashlib.sha256(
        f"{Xs_gene.shape}|{Xs_gene[:3, :5].tobytes().hex()}|{Xs_gene[-3:, -5:].tobytes().hex()}".encode()
    ).hexdigest()[:12]

    tag_part = f"__{tag}" if tag else ""
    fname = f"Xh__{gfp}__{xfp}__{data_hash}{tag_part}.npz"
    cache_path = cache_dir / fname

    # ── Intentar cargar ────────────────────────────────────────────────────
    if cache_path.exists() and not force_rebuild:
        print(f"[graph_cache] Cargando Xs_graph (X_h) desde caché: {cache_path.name}")
        z = np.load(cache_path, allow_pickle=True)
        Xs_graph = z["Xs_graph"].astype(np.float32)
        graph_feat_names = z["graph_feat_names"].tolist()
        print(f"[graph_cache]   → Xs_graph shape: {Xs_graph.shape}")
        return Xs_graph, graph_feat_names

    # ── Generar y guardar ──────────────────────────────────────────────────
    print(f"[graph_cache] Generando Xs_graph (X_h) por primera vez…")
    Xs_graph, graph_feat_names = build_regulator_features(
        Xs_gene, genes, edge_index,
        stats=stats, min_targets=min_targets, max_regulators=max_regulators,
    )
    np.savez_compressed(
        cache_path,
        Xs_graph=Xs_graph,
        graph_feat_names=np.array(graph_feat_names, dtype=object),
    )
    print(f"[graph_cache] Xs_graph guardado en: {cache_path}")
    print(f"[graph_cache]   → shape: {Xs_graph.shape}")
    return Xs_graph, graph_feat_names


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Wrapper completo  (backbone + filtro + X_h de un tirón)
# ═══════════════════════════════════════════════════════════════════════════

def get_or_build_all(
    genes_full: List[str],
    Xs_gene_full: np.ndarray,
    edge_index_full: np.ndarray,
    edge_weight_full: np.ndarray,
    edge_type_full: np.ndarray,
    *,
    cache_dir: Optional[Path] = None,
    force_rebuild: bool = False,
    stats: Tuple[str, ...] = C.REG_STATS,
    min_targets: int = C.REG_MIN_GENES,
    max_regulators: Optional[int] = C.REG_MAX_REGULATORS,
    tag: str = "",
) -> Tuple[np.ndarray, List[str]]:
    """
    Solo genera/carga X_h partiendo de un backbone ya filtrado a
    genes conectados.

    Es equivalente a get_or_build_Xh pero acepta los arrays tal como
    salen de filter_to_connected_genes para mayor comodidad.
    """
    return get_or_build_Xh(
        Xs_gene=Xs_gene_full,
        genes=genes_full,
        edge_index=edge_index_full,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        stats=stats,
        min_targets=min_targets,
        max_regulators=max_regulators,
        tag=tag,
    )
