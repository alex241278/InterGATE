"""
Centralised configuration for InterGATE.

The original notebooks used paths outside the repository.  This version is
self-contained by default: data are expected under ``data/processed`` and
external resources/caches under ``data/external`` and ``cache``.  All paths can
still be overridden with environment variables.

Environment variables
---------------------
INTERGATE_DATA_DIR          Directory containing expr_combat_corrected.csv and metadata_combined.csv
INTERGATE_EXPR_CSV          Explicit expression CSV path
INTERGATE_META_CSV          Explicit metadata CSV path
INTERGATE_RAW_DATA_DIR      Directory where Zenodo files are downloaded
INTERGATE_EXTERNAL_DATA_DIR Directory for OmniPath/HuRI raw files
INTERGATE_CACHE_ROOT        Root directory for derived caches
INTERGATE_ARTIFACTS_ROOT    Directory for trained artifacts
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import os

import torch

from .utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.environ.get("INTERGATE_DATA_DIR", PROJECT_ROOT / "data" / "processed")).expanduser().resolve()
DEFAULT_RAW_DATA_DIR = Path(os.environ.get("INTERGATE_RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw")).expanduser().resolve()
DEFAULT_EXTERNAL_DATA_DIR = Path(os.environ.get("INTERGATE_EXTERNAL_DATA_DIR", PROJECT_ROOT / "data" / "external")).expanduser().resolve()
DEFAULT_CACHE_ROOT = Path(os.environ.get("INTERGATE_CACHE_ROOT", PROJECT_ROOT / "cache")).expanduser().resolve()


@dataclass
class InterGATEConfig:
    """All hyper-parameters and repository paths in a single overridable object."""

    # ── Repository / data paths ─────────────────────────────
    PROJECT_ROOT: Path = PROJECT_ROOT
    ZENODO_DOI: str = "10.5281/zenodo.19476488"
    ZENODO_RECORD_ID: str = "19476488"
    ZENODO_API_URL: str = "https://zenodo.org/api/records/19476488"

    DATA_DIR: Path = DEFAULT_DATA_DIR
    RAW_DATA_DIR: Path = DEFAULT_RAW_DATA_DIR
    EXTERNAL_DATA_DIR: Path = DEFAULT_EXTERNAL_DATA_DIR
    CACHE_ROOT: Path = DEFAULT_CACHE_ROOT

    EXPR_CSV: Optional[Path] = None
    META_CSV: Optional[Path] = None
    PIPELINE_CACHE_DIR: Optional[Path] = None
    SAMPLE_COL: str = "sample"
    COHORT_COL: str = "batch"
    LABEL_COL: str = "label"
    # ── Splits ─────────────────────────────────────────────
    SEED: int = 42
    TEST_SIZE: float = 0.20
    VAL_SIZE: float = 0.20

    # ── Scaling ────────────────────────────────────────────
    USE_QUANTILE: bool = False
    SCALE_MODE: str = "standard"  # "standard" | "minmax_-1_1" | "none"

    # ── Interactome (HuRI) ─────────────────────────────────
    USE_HURI: bool = True
    HURI_DATASETS: Tuple[str, ...] = ("HuRI",)
    HURI_CACHE_DIR: Optional[Path] = None
    USE_HURI_CONFIDENCE: bool = True
    HURI_MIN_SCORE: float = 0.0
    HURI_DEFAULT_WEIGHT: float = 1.0
    FORCE_REBUILD_HURI_CACHE: bool = True

    # ── OmniPath ───────────────────────────────────────────
    USE_OMNIPATH: bool = True
    OMNIPATH_CACHE_DIR: Optional[Path] = None
    OMNIPATH_URL: str        = (
        "https://omnipathdb.org/interactions"
        "?datasets=omnipath&genesymbols=1&directed=1&signed=1&format=tsv"
    )

    # ── Graph construction ─────────────────────────────────
    ADD_SELF_LOOPS_IN_GRAPH: bool = False
    CONNECTED_ONLY: bool = True

    # ── Regulator features (X_graph) ──────────────────────
    ADD_REGULATOR_FEATURES: bool = True
    REG_STATS: Tuple[str, ...] = ("mean", "std", "max")
    REG_MIN_GENES: int = 5
    REG_MAX_REGULATORS: Optional[int] = None

    # ── Training (Phase 1 – graph learning) ────────────────
    BATCH_SIZE: int = 20
    ACCUM_STEPS: int = 8
    PATIENCE: int = 30
    FOCAL_GAMMA: float = 2.0

    LR: float = 2e-3
    WEIGHT_DECAY: float = 1e-4

    # ── Gates / sparsity ──────────────────────────────────
    GATE_TAU_START: float = 2.0
    GATE_TAU_END: float = 0.7
    EDGE_L1_PER_EDGE: float = 1e-5

    # ── GNN backbone ──────────────────────────────────────
    HIDDEN: int = 64
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.10
    NUM_HEADS: int = 4       # GAT heads
    POOL_HEADS: int = 4      # pooling heads
    XGRAPH_DROPOUT: float = 0.00

    # ── Block / bypass ────────────────────────────────────
    KEEP_SELF_LOOPS: bool = False
    BLOCK_USE_SELF: bool = True
    BLOCK_RESIDUAL: bool = True

    # ── Loss / pruning ────────────────────────────────────
    AUX_LAMBDA: float = 0.10
    KEEP_MIN: float = 0.004

    # ── Data augmentation (Phase 2) ──────────────────────
    USE_MIXUP: bool = True
    MIXUP_ALPHA: float = 0.15
    MIXUP_P: float = 0.30

    # ── Epochs ────────────────────────────────────────────
    EPOCHS1: int = 90        # Phase 1 (graph learning)
    FT_EPOCHS_A: int = 5     # Phase 2A (Xh=0)
    FT_EPOCHS_B: int = 80    # Phase 2B (Xh=orig)

    # ── Novelty toggles ──────────────────────────────────
    EDGE_TYPE_GATING: bool = True
    SAMPLE_COND_GATING: bool = True
    SAMPLE_COND_MODE: str = "per_type"

    SIGNED_CHANNELS: bool = False
    SIGNED_CHANNELS_MODE: str = "type_only"

    ADD_CONNECTIVITY_PENALTY: bool = True
    CONNECTIVITY_LAMBDA: float = 0.002
    CONNECTIVITY_MIN_DEG: float = 0.05
    CONNECTIVITY_USE_ABS: bool = True

    DO_PRETRAIN: bool = False
    DO_STABILITY_SELECTION: bool = True

    # ── Stability selection ──────────────────────────────
    STAB_RUNS: int = 8
    STAB_EPOCHS: int = 60
    STAB_KEEP_FINAL: float = 0.004
    STAB_FREQ_THR: float = 0.5

    # ── Hybrid model (Phase 2) ──────────────────────────
    USE_HYBRID_MODEL: bool = True
    HYBRID_TAB_LAYERS: int = 2
    HYBRID_TAB_DROPOUT: float = 0.20
    HYBRID_FUSION_DROPOUT: float = 0.20
    HYBRID_BLEND_LOGITS: bool = False
    HYBRID_BLEND_INIT: float = 0.70

    # ── Artifacts ─────────────────────────────────────────
    ARTIFACTS_ROOT: str = str((PROJECT_ROOT / "artifacts_ablation").resolve())
    SAVE_ALL_CONFIGS: bool = True
    RUN_FINAL_AFTER_ABLATION: bool = True
    FINAL_SELECT_METRIC: str = "macro_f1"

    # ── Device ────────────────────────────────────────────
    DEVICE: Optional[torch.device] = None
    def __post_init__(self):
        """Resolve paths and create lightweight directories.

        The project can run fully inside the repository after executing
        ``python scripts/download_zenodo_data.py --extract``.  Per-machine
        overrides remain possible through the environment variables listed at
        the top of this file.
        """
        self.DATA_DIR = Path(os.environ.get("INTERGATE_DATA_DIR", self.DATA_DIR)).expanduser().resolve()
        self.RAW_DATA_DIR = Path(os.environ.get("INTERGATE_RAW_DATA_DIR", self.RAW_DATA_DIR)).expanduser().resolve()
        self.EXTERNAL_DATA_DIR = Path(os.environ.get("INTERGATE_EXTERNAL_DATA_DIR", self.EXTERNAL_DATA_DIR)).expanduser().resolve()
        self.CACHE_ROOT = Path(os.environ.get("INTERGATE_CACHE_ROOT", self.CACHE_ROOT)).expanduser().resolve()

        expr_env = os.environ.get("INTERGATE_EXPR_CSV")
        meta_env = os.environ.get("INTERGATE_META_CSV")
        if self.EXPR_CSV is None:
            self.EXPR_CSV = Path(expr_env).expanduser().resolve() if expr_env else self.DATA_DIR / "expr_combat_corrected.csv"
        else:
            self.EXPR_CSV = Path(self.EXPR_CSV).expanduser().resolve()
        if self.META_CSV is None:
            self.META_CSV = Path(meta_env).expanduser().resolve() if meta_env else self.DATA_DIR / "metadata_combined.csv"
        else:
            self.META_CSV = Path(self.META_CSV).expanduser().resolve()

        if self.PIPELINE_CACHE_DIR is None:
            self.PIPELINE_CACHE_DIR = self.CACHE_ROOT / "pipeline_cache"
        else:
            self.PIPELINE_CACHE_DIR = Path(self.PIPELINE_CACHE_DIR).expanduser().resolve()
        if self.HURI_CACHE_DIR is None:
            # HuRI.tsv and HuRI.psi are looked up here; derived .npz files are also cached here.
            self.HURI_CACHE_DIR = self.EXTERNAL_DATA_DIR
        else:
            self.HURI_CACHE_DIR = Path(self.HURI_CACHE_DIR).expanduser().resolve()
        if self.OMNIPATH_CACHE_DIR is None:
            # OmniPath is looked up as omnipath_interactions.tsv in this directory.
            self.OMNIPATH_CACHE_DIR = self.EXTERNAL_DATA_DIR
        else:
            self.OMNIPATH_CACHE_DIR = Path(self.OMNIPATH_CACHE_DIR).expanduser().resolve()

        self.ARTIFACTS_ROOT = str(Path(os.environ.get("INTERGATE_ARTIFACTS_ROOT", self.ARTIFACTS_ROOT)).expanduser().resolve())

        for path in [
            self.DATA_DIR, self.RAW_DATA_DIR, self.EXTERNAL_DATA_DIR,
            self.CACHE_ROOT, self.PIPELINE_CACHE_DIR,
            self.HURI_CACHE_DIR, self.OMNIPATH_CACHE_DIR,
            Path(self.ARTIFACTS_ROOT),
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

        if self.DEVICE is None:
            self.DEVICE = get_device()


# Singleton – import this in notebooks
CFG = InterGATEConfig()
