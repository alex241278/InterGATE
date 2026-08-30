"""
Centralised configuration for InterGATE.

The model reads the two immutable matrices produced by ``data_prepare``. The
external cohort block is never repartitioned inside an InterGATE notebook.

Environment variables
---------------------
INTERGATE_DATA_PREPARE_DIR  Root directory containing output_mcombat
INTERGATE_TRAIN_EXPR_CSV    Explicit M-ComBat train matrix
INTERGATE_TEST_EXPR_CSV     Explicit M-ComBat external-test matrix
INTERGATE_TRAIN_META_CSV    Explicit train metadata
INTERGATE_TEST_META_CSV     Explicit external-test metadata
INTERGATE_RAW_DATA_DIR      Directory where Zenodo files are downloaded
INTERGATE_EXTERNAL_DATA_DIR Directory for OmniPath/HuRI raw files
INTERGATE_CACHE_ROOT        Root directory for derived caches
INTERGATE_ARTIFACTS_ROOT    Directory for trained artifacts
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Optional
import hashlib
import json
import os

import torch

from .utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PREPARE_DIR = Path(os.environ.get(
    "INTERGATE_DATA_PREPARE_DIR", PROJECT_ROOT / "data_prepare"
)).expanduser().resolve()
DEFAULT_MCOMBAT_DIR = DEFAULT_DATA_PREPARE_DIR / "output_mcombat"
DEFAULT_DATA_DIR = DEFAULT_MCOMBAT_DIR
DEFAULT_RAW_DATA_DIR = Path(os.environ.get("INTERGATE_RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw")).expanduser().resolve()
DEFAULT_EXTERNAL_DATA_DIR = Path(os.environ.get("INTERGATE_EXTERNAL_DATA_DIR", PROJECT_ROOT / "data" / "external")).expanduser().resolve()
DEFAULT_CACHE_ROOT = Path(os.environ.get("INTERGATE_CACHE_ROOT", PROJECT_ROOT / "cache")).expanduser().resolve()


@dataclass
class InterGATEConfig:
    """All hyper-parameters and repository paths in a single overridable object."""

    # ── Repository / data paths ─────────────────────────────
    PROJECT_ROOT: Path = PROJECT_ROOT
    MODEL_VERSION: str = "8.1.0"
    ZENODO_DOI: str = "10.5281/zenodo.20817487"
    ZENODO_RECORD_ID: str = "20817487"
    ZENODO_API_URL: str = "https://zenodo.org/api/records/20817487"

    DATA_PREPARE_DIR: Path = DEFAULT_DATA_PREPARE_DIR
    DATA_DIR: Path = DEFAULT_DATA_DIR
    RAW_DATA_DIR: Path = DEFAULT_RAW_DATA_DIR
    EXTERNAL_DATA_DIR: Path = DEFAULT_EXTERNAL_DATA_DIR
    CACHE_ROOT: Path = DEFAULT_CACHE_ROOT

    TRAIN_EXPR_CSV: Optional[Path] = None
    TEST_EXPR_CSV: Optional[Path] = None
    TRAIN_META_CSV: Optional[Path] = None
    TEST_META_CSV: Optional[Path] = None
    SPLIT_COHORTS_CSV: Optional[Path] = None
    MCOMBAT_ARTIFACT_DIR: Optional[Path] = None
    PIPELINE_CACHE_DIR: Optional[Path] = None
    SAMPLE_COL: str = "sample"
    COHORT_COL: str = "dataset"
    LABEL_COL: str = "label"
    # ══════════════════════════════════════════════════════
    # Endpoint multiclase intrínseco de cuatro clases
    # ══════════════════════════════════════════════════════
    # Etiqueta de los controles en el metadata
    CONTROL_LABEL_NAME: str = "Normal"

    # Con INCLUDE_CONTROLS=False los controles se descartan del dataset entero
    # y el modo multiclase pasa a ser de 4 subtipos.
    INCLUDE_CONTROLS: bool = False
    MULTICLASS_LABELS_WITH_CONTROL: Tuple[str, ...] = (
        "Normal", "LumA", "LumB", "HER2-enriched", "Basal-like",
    )
    MULTICLASS_LABELS_NO_CONTROL: Tuple[str, ...] = (
        "LumA", "LumB", "HER2-enriched", "Basal-like",
    )

    # ── Filtrado de cohortes ──────────────────────────────
    # Descarta cohortes con menos de N etiquetas distintas (1 = no filtra).
    MIN_LABELS_PER_COHORT: int = 2
    # Cohortes que DEBEN contener estas clases. ("Basal-like",) junto con
    # MIN_LABELS_PER_COHORT=2 deja solo cohortes con Basal-like y al menos otra.
    REQUIRE_CLASSES: Tuple[str, ...] = ()
    # Cohortes que nunca deben caer en el filtro.
    PROTECT_COHORTS: Tuple[str, ...] = ()
    DROP_COHORTS: Tuple[str, ...] = ()

    # Partición externa declarada por data_prepare (orden preservado).
    TRAIN_COHORTS: Tuple[str, ...] = (
        "GSE96058", "GSE25066", "GSE162228", "GSE20711",
    )
    TEST_COHORTS: Tuple[str, ...] = (
        "TCGA-BRCA", "GSE21653", "GSE1456",
    )



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
    FORCE_REBUILD_HURI_CACHE: bool = False

    # ── OmniPath ───────────────────────────────────────────
    USE_OMNIPATH: bool = True
    OMNIPATH_CACHE_DIR: Optional[Path] = None
    OMNIPATH_URL: str        = (
        "https://omnipathdb.org/interactions"
        "?datasets=omnipath&genesymbols=1&directed=1&signed=1&format=tsv"
    )

    # ══════════════════════════════════════════════════════
    # STRING  +  seleccion de fuentes del backbone
    # ══════════════════════════════════════════════════════
    # "huri_omnipath" (comportamiento original)
    # "string_omnipath" | "all" (HuRI+OmniPath+STRING; default entregado)
    # "huri" | "omnipath" | "string" | "custom"
    # "custom" respeta los flags USE_* tal cual esten.
    # Se conserva la fuente del último paquete aportado por el usuario.
    GRAPH_SOURCE: str = "huri_omnipath"

    USE_STRING: bool = True
    STRING_CACHE_DIR: Optional[Path] = None
    # Ficheros descargados de STRING (https://string-db.org/cgi/download).
    # Se admiten .txt o .txt.gz.
    STRING_LINKS_FILE: Optional[Path] = (
        DEFAULT_EXTERNAL_DATA_DIR
        / "string"
        / "9606.protein.links.detailed.v12.0.txt.gz"
    )

    STRING_INFO_FILE: Optional[Path] = (
        DEFAULT_EXTERNAL_DATA_DIR
        / "string"
        / "9606.protein.info.v12.0.txt.gz"
    )

    STRING_ALIAS_FILE: Optional[Path] = (
        DEFAULT_EXTERNAL_DATA_DIR
        / "string"
        / "9606.protein.aliases.v12.0.txt.gz"
    )
    STRING_MIN_SCORE: int = 700                 # 700 = alta confianza
    STRING_SCORE_COLUMN: str = "combined_score"
    STRING_WEIGHT_SCALE: float = 1000.0         # score -> peso en [0, 1]
    STRING_MAX_WEIGHT: float = 1.0
    # Tipo de arista asignado a STRING (0=HuRI, 1=OP+, 2=OP-, 3=STRING)
    STRING_EDGE_TYPE: int = 3

    # ══════════════════════════════════════════════════════
    # Grafos predictivos por clase
    # ══════════════════════════════════════════════════════
    # Cada muestra recorre TODOS los expertos. El experto c usa
    # base_logit + edge_logit_delta[c] y aporta solo el logit de la clase c.
    # La etiqueta real nunca decide qué grafo se ejecuta (sin target leakage).
    PER_LABEL_GRAPHS: bool = True
    # V8 hace identificable la descomposición ``backbone + desviación de
    # subtipo`` centrando los deltas entre clases. No se fuerza que los grafos
    # sean disjuntos: los mecanismos realmente compartidos permanecen en el
    # backbone común.
    CENTER_CLASS_DELTAS: bool = True
    PER_LABEL_DELTA_L1: float = 1e-4
    PER_LABEL_DELTA_DIVERSITY_LAMBDA: float = 0.0
    # El delta arranca en cero y necesita moverse mas rapido que el resto.
    PER_LABEL_DELTA_LR_MULT: float = 1.75
    # Override absoluto opcional. Por defecto es None: cada experto conserva
    # la fracción kappa=KEEP_MIN del backbone candidato real. Así, al cambiar
    # HuRI/OmniPath/STRING, K cambia con el número de aristas disponibles.
    # Asignar un entero fuerza ese K solo para una ablación explícita.
    CLASS_GRAPH_TARGET_EDGES: Optional[int] = None
    # Tras estabilidad, conserva el K producido por el TopK de cada corrida
    # (derivado de kappa, o del override absoluto si se ha definido).
    CLASS_GRAPH_STABILITY_FIXED_BUDGET: bool = True
    # Metrica de validacion que decide que checkpoint se guarda en Phase 1
    # (aprendizaje del grafo). Claves disponibles en compute_metrics_full:
    # "macro_f1", "weighted_f1", "bal_acc", "acc", "auc_macro_ovr",
    # "auc_micro_ovr" y "f1_<nombre_clase>" (p. ej.
    # "f1_Basal-like").
    # Phase 2 se controla aparte con register_runtime(PHASE2_BEST_METRIC=...).
    SELECT_METRIC: str = "macro_f1"
    # Imprime media/max/%abierto del gate en cada epoca de la fase 1.
    # Diagnostico: el colapso del gate no se ve en las metricas.
    LOG_GATE_STATS: bool = False
    # El log visible informa solo de esta clase (one-vs-rest). Es independiente
    # de SELECT_METRIC: cambiar esto NO cambia que checkpoint se guarda.
    TRAIN_REPORT_LABEL: str = "Basal-like"
    # True recupera el diccionario completo de poda para depuración.
    TRAIN_LOG_FULL_PRUNING: bool = False
    # Uso de X_h durante Phase 1: V8 exige "zero".
    GRAPH_LEARNING_XH_MODE: str = "zero"
    # Uso de X_h en Phase 2B y evaluación final. Phase 2A es estrictamente
    # graph-only: ni X_h ni la rama tabular/fusión participan.
    CLASS_GRAPH_XH_MODE: str = "orig"  # "zero" | "orig"

    # ── Graph construction ─────────────────────────────────
    ADD_SELF_LOOPS_IN_GRAPH: bool = False
    CONNECTED_ONLY: bool = True

    # ── Regulator features (X_graph) ──────────────────────
    ADD_REGULATOR_FEATURES: bool = True
    REG_STATS: Tuple[str, ...] = ("mean", "std", "max")
    REG_MIN_GENES: int = 5
    REG_MAX_REGULATORS: Optional[int] = None

    # ── Training (Phase 1 – graph learning) ────────────────
    BATCH_SIZE: int = 8
    ACCUM_STEPS: int = 20
    # Fase 1 ejecuta un experto GNN por clase para cada muestra. Un microbatch
    # pequeño limita la VRAM; la acumulación mantiene el batch efectivo.
    GRAPH_BATCH_SIZE: int = 2
    GRAPH_ACCUM_STEPS: int = 120
    PATIENCE: int = 30
    FOCAL_GAMMA: float = 2.0

    LR: float = 2e-3
    WEIGHT_DECAY: float = 1e-4

    # ── Gates / sparsity ──────────────────────────────────
    # Piecewise constant gate-temperature schedule used in Phase 1.
    # Entries are (last_epoch_inclusive, tau), matching the manuscript/SF3.
    GATE_TAU_SCHEDULE: Tuple[Tuple[int, float], ...] = ((10, 0.5), (20, 0.25), (10**9, 0.1))
    EDGE_L1_PER_EDGE: float = 0.0
    SPARSITY_MODE: str = "target"
    SPARSITY_LAMBDA: float = 3000.0
    # Coeficiente beta del término de entropía binaria de R_gate.
    SPARSITY_BINARIZE: float = 0.01

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
    KEEP_MIN: float = 0.01
    KAPPA_SENSITIVITY: Tuple[float, ...] = (0.007, 0.010, 0.015, 0.020)

    # ── Data augmentation (Phase 2) ──────────────────────
    USE_MIXUP: bool = True
    MIXUP_ALPHA: float = 0.15
    MIXUP_P: float = 0.30

    # ── Epochs ────────────────────────────────────────────
    EPOCHS1: int = 90        # Phase 1 (graph learning)
    FT_EPOCHS_A: int = 5     # Phase 2A: GNN graph-only, Xh=0, sin mixup
    FT_EPOCHS_B: int = 80    # Phase 2B: híbrido + CLASS_GRAPH_XH_MODE

    # ── Novelty toggles ──────────────────────────────────
    EDGE_TYPE_GATING: bool = True
    SIGNED_CHANNELS: bool = True
    SIGNED_CHANNELS_MODE: str = "type_only"

    # El antiguo hinge intentaba dar grado a todos los nodos del interactoma y
    # favorecía muchas parejas aisladas bajo presupuestos muy pequeños. V8 no
    # lo usa en el objetivo; la coherencia se impone después mediante una
    # selección modular Exact-K auditable.
    ADD_CONNECTIVITY_PENALTY: bool = False
    CONNECTIVITY_LAMBDA: float = 0.0
    CONNECTIVITY_MIN_DEG: float = 0.05
    CONNECTIVITY_USE_ABS: bool = True

    DO_PRETRAIN: bool = False
    DO_STABILITY_SELECTION: bool = False

    # ── Reproducibilidad entre semillas ──────────────────
    # V8 ejecuta cinco semillas principales. No lanza las ocho corridas
    # internas históricas: el soporte se estima por deleción de cohorte y la
    # agregación Exact-K ocurre directamente entre estas cinco salidas.
    OUTER_RUNS: int = 5
    STABILITY_SEEDS: Tuple[int, ...] = (1234, 2025, 7, 99, 31337)
    STAB_RUNS: int = 0
    STAB_SEED_OFFSETS: Tuple[int, ...] = ()
    STAB_EPOCHS: int = 0
    STAB_KEEP_FINAL: float = 0.01
    # Campos conservados para que los módulos históricos sigan importando;
    # no gobiernan ningún notebook V8.
    STAB_FREQ_THR: float = 0.8
    STAB_STRICT_FREQ_THR: float = 1.0
    # Entre los cinco consensos Exact-K externos, la firma biológica
    # principal es el núcleo génico 5/5. El 4/5 queda como sensibilidad.
    OUTER_GENE_CORE_THR: float = 1.0
    OUTER_GENE_SENSITIVITY_THR: float = 0.8
    # Las corridas mantienen fija la matriz y la partición de desarrollo.
    # Por tanto, cuantifican reproducibilidad algorítmica multisemilla y no
    # estabilidad frente a remuestreo de pacientes o cohortes.
    STABILITY_DATA_RESAMPLING: bool = False
    STABILITY_SCOPE: str = "algorithmic_randomness_fixed_development_data"

    # ── InterGATE V8: cohort-transport structural meta-learning ─────
    # Esta es la única ruta principal de los notebooks V8. Las funciones
    # históricas se conservan como compatibilidad interna, pero no forman parte
    # de la secuencia ejecutable entregada.
    TRANSPORT_PROTOCOL_VERSION: str = (
        "intergate-v8-biology-first-v2-matched-budget"
    )
    # Un episodio retiene una cohorte de desarrollo, adapta únicamente los
    # parámetros estructurales con las otras cohortes y calcula la pérdida
    # exterior en la cohorte retenida. Con cuatro cohortes, 60 episodios
    # proporcionan 15 exposiciones exteriores por cohorte.
    TRANSPORT_EPISODES: int = 60
    TRANSPORT_INNER_LR: float = 5e-4
    TRANSPORT_FIRST_ORDER: bool = True
    TRANSPORT_META_TRAIN_WEIGHT: float = 0.25
    TRANSPORT_META_VALIDATION_WEIGHT: float = 1.0
    # Distancia soft-Jaccard entre el TopK blando del portador y el de la
    # adaptación leave-one-cohort-out. No sustituye a la cuota dura Exact-K.
    TRANSPORT_TOPOLOGY_LAMBDA: float = 0.10
    TRANSPORT_SOFT_TOPK_TAU: float = 0.10
    # La cuota alcanza kappa antes del final para que el checkpoint se elija
    # bajo el presupuesto definitivo y disponga de episodios de paciencia.
    TRANSPORT_QUOTA_END_FRACTION: float = 0.65
    TRANSPORT_PATIENCE: int = 15
    # Cada episodio ejecuta exactamente el mismo número de pasos, con muestreo
    # round-robin cohorte×clase en meta-train y clase en la cohorte retenida.
    # Esto evita que GSE20711 se recicle decenas de veces mientras GSE96058 no
    # completa ni una pasada.
    TRANSPORT_BALANCED_SAMPLING: bool = True
    # V7 ejecutaba 96 microbatches de 4 muestras: 384 draws por episodio y
    # dos actualizaciones (60 + 36 microbatches) con acumulación 60. V8.1 usa
    # microbatch 2, por lo que duplica pasos y acumulación para conservar
    # exactamente 384 draws y las mismas dos actualizaciones (120 + 72).
    TRANSPORT_STEPS_PER_EPISODE: int = 192
    TRANSPORT_CHECKPOINT_METRIC: str = "cohort_macro_f1"
    # Auditoría final: cinco pasos de influencia estructural sobre 128
    # muestras balanceadas (64 microbatches de tamaño 2) por deleción. Así se
    # conserva el soporte de V7: 32 microbatches de tamaño 4.
    TRANSPORT_SUPPORT_STEPS: int = 5
    TRANSPORT_SUPPORT_BATCHES: int = 64
    TRANSPORT_MIN_COHORT_SAMPLES: int = 2

    # ── InterGATE V8: evidencia diferencial y salida biológica ─────────
    BIOLOGY_FIRST: bool = True
    # Evidencia calculada exclusivamente en TRAIN de desarrollo. Para cada
    # arista se contrasta la coexpresión clase vs resto dentro de cada cohorte
    # y después se combinan los efectos mediante metaanálisis de efectos fijos.
    BIOLOGY_EVIDENCE_METHOD: str = "within_cohort_differential_correlation"
    BIOLOGY_MIN_SAMPLES_PER_GROUP: int = 8
    BIOLOGY_MIN_COHORTS: int = 2
    BIOLOGY_EDGE_BATCH_SIZE: int = 4096
    BIOLOGY_META_Z_CLIP: float = 6.0
    BIOLOGY_HETEROGENEITY_PENALTY: float = 0.50
    # Alineación suave entre los deltas aprendidos y la evidencia transcriptómica
    # reproducible. No utiliza firmas conocidas, PAM50, C2:CGP ni DepMap.
    BIOLOGY_CONTEXT_LAMBDA: float = 0.10
    BIOLOGY_CONTEXT_WARMUP_EPISODES: int = 8
    # Ranking biológico: contraste de delta + evidencia + soporte de cohorte.
    BIOLOGY_SCORE_CONTEXT_WEIGHT: float = 0.35
    BIOLOGY_SCORE_SUPPORT_WEIGHT: float = 0.25
    # El grafo predictivo permanece Exact-K por puntuación total. La salida
    # biológica usa el mismo K, pero selecciona un bosque modular desde un pool
    # mayor de aristas diferenciales.
    BIOLOGY_MODULE_POOL_MULTIPLIER: int = 6
    BIOLOGY_MODULE_TARGET_COMPONENTS: int = 12
    BIOLOGY_MODULE_MIN_EDGES: int = 3
    BIOLOGY_MATCHED_NULL_DRAWS: int = 1000
    BIOLOGY_REWIRING_NULL_DRAWS: int = 1000
    BIOLOGY_NONINFERIORITY_MARGIN: float = 0.02

    # ── Hybrid model (Phase 2) ──────────────────────────
    USE_HYBRID_MODEL: bool = True
    # Contrato metodológico fijo de Phase 2. No son knobs para el análisis
    # principal: assert_v8_protocol impide cambiarlos accidentalmente.
    PHASE2_A_MODE: str = "graph_only"
    PHASE2_FIXED_GRAPH_BYPASS_GATES: bool = True
    PHASE2_FREEZE_STRUCTURAL_PARAMETERS: bool = True
    HYBRID_TAB_LAYERS: int = 2
    HYBRID_TAB_DROPOUT: float = 0.20
    HYBRID_FUSION_DROPOUT: float = 0.20
    HYBRID_BLEND_LOGITS: bool = True
    HYBRID_BLEND_INIT: float = 0.70

    # ── InterGATE V8: identidad nodal y equilibrio de ramas ─────────
    V8_PROTOCOL_VERSION: str = (
        "intergate-v8-biology-first-v2-matched-budget"
    )
    V8_ARTIFACT_NAME: str = "INTERGATE_V8"
    # Debe permanecer False. El externo se abre primero en el notebook 07;
    # 08b puede volver a leerlo después, exclusivamente para la concordancia
    # estructural ya fijada y sin selección ni refit.
    ALLOW_EXTERNAL_DURING_TRAINING: bool = False
    USE_NODE_IDENTITY: bool = True
    EXPERT_DEEP_SUPERVISION_LAMBDA: float = 0.20
    TAB_FULL_GENE_UNIVERSE: bool = True
    # 0 = usar HIDDEN. >0 desacopla la anchura
    # de la rama tabular del ancho del GNN. Cambia las formas del state_dict:
    # los bundles entrenados con valores distintos NO son intercambiables ni
    # ensamblables entre si.
    HYBRID_TAB_HIDDEN: int = 256

    # Dropout de modalidad + supervisión directa para evitar que la rama
    # tabular absorba todo el gradiente de la fase 2B.
    BRANCH_DROPOUT_TAB: float = 0.30     # anula z_tab por muestra en 2B
    BRANCH_DROPOUT_GNN: float = 0.0      # simetrico, normalmente 0
    GNN_AUX_LOSS_LAMBDA: float = 0.50    # CE sobre logits4 propios del GNN

    # ── Artifacts ─────────────────────────────────────────
    # Coincide con la raíz que usan todos los notebooks definitivos. De este
    # modo, una función llamada sin ``save_root`` no escribe en un árbol
    # paralelo distinto del que luego buscan los notebooks 04--13.
    ARTIFACTS_ROOT: str = str(
        (PROJECT_ROOT / "artifacts" / "intrinsic4").resolve()
    )
    SAVE_ALL_CONFIGS: bool = True
    RUN_FINAL_AFTER_ABLATION: bool = True
    FINAL_SELECT_METRIC: str = "macro_f1"

    # ── Device ────────────────────────────────────────────
    DEVICE: Optional[torch.device] = None
    def __post_init__(self):
        """Resolve paths and create lightweight directories.

        The project can run from the repository after completing the numbered
        data-preparation notebooks. On the first model preflight, OmniPath and
        HuRI are downloaded into ``data/external`` and then cached. Per-machine
        overrides remain possible through the environment variables listed at
        the top of this file.
        """
        self.DATA_PREPARE_DIR = Path(os.environ.get(
            "INTERGATE_DATA_PREPARE_DIR", self.DATA_PREPARE_DIR
        )).expanduser().resolve()
        self.DATA_DIR = self.DATA_PREPARE_DIR / "output_mcombat"
        self.RAW_DATA_DIR = Path(os.environ.get("INTERGATE_RAW_DATA_DIR", self.RAW_DATA_DIR)).expanduser().resolve()
        self.EXTERNAL_DATA_DIR = Path(os.environ.get("INTERGATE_EXTERNAL_DATA_DIR", self.EXTERNAL_DATA_DIR)).expanduser().resolve()
        self.CACHE_ROOT = Path(os.environ.get("INTERGATE_CACHE_ROOT", self.CACHE_ROOT)).expanduser().resolve()

        path_specs = {
            "TRAIN_EXPR_CSV": ("INTERGATE_TRAIN_EXPR_CSV", "expr_train_mcombat.csv"),
            "TEST_EXPR_CSV": ("INTERGATE_TEST_EXPR_CSV", "expr_test_mcombat.csv"),
            "TRAIN_META_CSV": ("INTERGATE_TRAIN_META_CSV", "meta_train.csv"),
            "TEST_META_CSV": ("INTERGATE_TEST_META_CSV", "meta_test.csv"),
            "SPLIT_COHORTS_CSV": ("INTERGATE_SPLIT_COHORTS_CSV", "split_cohortes.csv"),
        }
        for attr, (env_name, filename) in path_specs.items():
            value = getattr(self, attr)
            env_value = os.environ.get(env_name)
            resolved = (
                Path(value).expanduser().resolve() if value is not None
                else Path(env_value).expanduser().resolve() if env_value
                else self.DATA_DIR / filename
            )
            setattr(self, attr, resolved)
        if self.MCOMBAT_ARTIFACT_DIR is None:
            self.MCOMBAT_ARTIFACT_DIR = self.DATA_DIR / "artefacto_mcombat"
        else:
            self.MCOMBAT_ARTIFACT_DIR = Path(
                self.MCOMBAT_ARTIFACT_DIR
            ).expanduser().resolve()

        if self.PIPELINE_CACHE_DIR is None:
            self.PIPELINE_CACHE_DIR = self.CACHE_ROOT / "pipeline_cache"
        else:
            self.PIPELINE_CACHE_DIR = Path(self.PIPELINE_CACHE_DIR).expanduser().resolve()
        if self.HURI_CACHE_DIR is None:
            # HuRI.tsv and HuRI.psi are looked up here; derived .npz files are also cached here.
            self.HURI_CACHE_DIR = self.EXTERNAL_DATA_DIR
        else:
            self.HURI_CACHE_DIR = Path(self.HURI_CACHE_DIR).expanduser().resolve()
        if self.STRING_CACHE_DIR is None:
            # STRING files (protein.links / info / aliases) are looked up here.
            self.STRING_CACHE_DIR = self.EXTERNAL_DATA_DIR
        else:
            self.STRING_CACHE_DIR = Path(self.STRING_CACHE_DIR).expanduser().resolve()
        for _attr in ("STRING_LINKS_FILE", "STRING_INFO_FILE", "STRING_ALIAS_FILE"):
            _v = getattr(self, _attr)
            if _v is not None:
                setattr(self, _attr, Path(_v).expanduser().resolve())
        if self.OMNIPATH_CACHE_DIR is None:
            # OmniPath is looked up as omnipath_interactions.tsv in this directory.
            self.OMNIPATH_CACHE_DIR = self.EXTERNAL_DATA_DIR
        else:
            self.OMNIPATH_CACHE_DIR = Path(self.OMNIPATH_CACHE_DIR).expanduser().resolve()

        self.ARTIFACTS_ROOT = str(Path(os.environ.get("INTERGATE_ARTIFACTS_ROOT", self.ARTIFACTS_ROOT)).expanduser().resolve())

        for path in [
            self.DATA_PREPARE_DIR, self.DATA_DIR, self.RAW_DATA_DIR,
            self.EXTERNAL_DATA_DIR,
            self.CACHE_ROOT, self.PIPELINE_CACHE_DIR,
            self.HURI_CACHE_DIR, self.OMNIPATH_CACHE_DIR, self.STRING_CACHE_DIR,
            Path(self.ARTIFACTS_ROOT),
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

        if self.DEVICE is None:
            self.DEVICE = get_device()

        self._sync_endpoint_from_data_prepare()
        self._resolve_task_and_sources()

    def _sync_endpoint_from_data_prepare(self) -> None:
        """Hace que el endpoint guardado por data_prepare gobierne el modelo."""
        manifest = self.DATA_PREPARE_DIR / "output_prepared" / "endpoint.json"
        if not manifest.exists():
            return
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        labels = tuple(map(str, payload.get("active_labels", ())))
        if labels == tuple(self.MULTICLASS_LABELS_NO_CONTROL):
            self.INCLUDE_CONTROLS = False
        elif labels == tuple(self.MULTICLASS_LABELS_WITH_CONTROL):
            self.INCLUDE_CONTROLS = True
        else:
            raise ValueError(
                "endpoint.json contiene un espacio de etiquetas no permitido: "
                f"{labels}"
            )

    # ------------------------------------------------------------------
    def _resolve_task_and_sources(self):
        """Valida el endpoint V8 y sincroniza las fuentes del grafo."""
        if int(self.GRAPH_BATCH_SIZE) <= 0 or int(self.GRAPH_ACCUM_STEPS) <= 0:
            raise ValueError(
                "GRAPH_BATCH_SIZE y GRAPH_ACCUM_STEPS deben ser positivos."
            )
        if not 0.0 < float(self.KEEP_MIN) <= 1.0:
            raise ValueError("KEEP_MIN (kappa) debe estar en (0, 1].")
        if tuple(self.KAPPA_SENSITIVITY) != (0.007, 0.010, 0.015, 0.020):
            raise ValueError(
                "El protocolo V8 fija KAPPA_SENSITIVITY=(.007,.010,.015,.020)."
            )
        if int(self.OUTER_RUNS) != 5:
            raise ValueError("El protocolo principal fija cinco semillas outer.")
        if len(self.STABILITY_SEEDS) != int(self.OUTER_RUNS):
            raise ValueError("STABILITY_SEEDS y OUTER_RUNS no coinciden.")
        if len(set(map(int, self.STABILITY_SEEDS))) != int(self.OUTER_RUNS):
            raise ValueError("Las semillas outer deben ser distintas.")
        if bool(self.DO_STABILITY_SELECTION):
            raise ValueError(
                "V8 no permite activar las ocho corridas internas históricas."
            )
        if int(self.STAB_RUNS) != 0 or tuple(self.STAB_SEED_OFFSETS):
            raise ValueError(
                "V8 fija STAB_RUNS=0 y STAB_SEED_OFFSETS=()."
            )
        if not 0.0 < float(self.STAB_FREQ_THR) <= 1.0:
            raise ValueError("STAB_FREQ_THR debe estar en (0, 1].")
        if not 0.0 < float(self.STAB_STRICT_FREQ_THR) <= 1.0:
            raise ValueError("STAB_STRICT_FREQ_THR debe estar en (0, 1].")
        if float(self.STAB_STRICT_FREQ_THR) < float(self.STAB_FREQ_THR):
            raise ValueError(
                "STAB_STRICT_FREQ_THR no puede ser menor que STAB_FREQ_THR."
            )
        if float(self.OUTER_GENE_CORE_THR) != 1.0:
            raise ValueError("La firma génica principal debe ser el núcleo 5/5.")
        if float(self.OUTER_GENE_SENSITIVITY_THR) != 0.8:
            raise ValueError("La sensibilidad génica suplementaria se fija en 4/5.")
        if bool(self.STABILITY_DATA_RESAMPLING):
            raise ValueError(
                "El protocolo principal fija los datos entre semillas; "
                "el remuestreo debe definirse como un análisis separado."
            )
        if int(self.TRANSPORT_EPISODES) < len(self.TRAIN_COHORTS):
            raise ValueError(
                "TRANSPORT_EPISODES debe cubrir al menos una vez cada cohorte "
                "de desarrollo."
            )
        if float(self.TRANSPORT_INNER_LR) <= 0:
            raise ValueError("TRANSPORT_INNER_LR debe ser positivo.")
        if float(self.TRANSPORT_META_TRAIN_WEIGHT) < 0:
            raise ValueError("TRANSPORT_META_TRAIN_WEIGHT no puede ser negativo.")
        if float(self.TRANSPORT_META_VALIDATION_WEIGHT) <= 0:
            raise ValueError(
                "TRANSPORT_META_VALIDATION_WEIGHT debe ser positivo."
            )
        if float(self.TRANSPORT_TOPOLOGY_LAMBDA) < 0:
            raise ValueError("TRANSPORT_TOPOLOGY_LAMBDA no puede ser negativo.")
        if float(self.TRANSPORT_SOFT_TOPK_TAU) <= 0:
            raise ValueError("TRANSPORT_SOFT_TOPK_TAU debe ser positivo.")
        if not 0.0 < float(self.TRANSPORT_QUOTA_END_FRACTION) < 1.0:
            raise ValueError(
                "TRANSPORT_QUOTA_END_FRACTION debe estar en (0, 1)."
            )
        if int(self.TRANSPORT_PATIENCE) <= 0:
            raise ValueError("TRANSPORT_PATIENCE debe ser positivo.")
        if int(self.TRANSPORT_SUPPORT_BATCHES) <= 0:
            raise ValueError("TRANSPORT_SUPPORT_BATCHES debe ser positivo.")
        if int(self.TRANSPORT_SUPPORT_STEPS) <= 0:
            raise ValueError("TRANSPORT_SUPPORT_STEPS debe ser positivo.")
        if int(self.TRANSPORT_STEPS_PER_EPISODE) <= 0:
            raise ValueError("TRANSPORT_STEPS_PER_EPISODE debe ser positivo.")
        phase1_draws = (
            int(self.GRAPH_BATCH_SIZE)
            * int(self.TRANSPORT_STEPS_PER_EPISODE)
        )
        phase1_updates = (
            int(self.TRANSPORT_STEPS_PER_EPISODE)
            + int(self.GRAPH_ACCUM_STEPS) - 1
        ) // int(self.GRAPH_ACCUM_STEPS)
        support_draws = (
            int(self.GRAPH_BATCH_SIZE)
            * int(self.TRANSPORT_SUPPORT_BATCHES)
        )
        if phase1_draws != 384 or phase1_updates != 2:
            raise ValueError(
                "V8.1 exige presupuesto de fase 1 pareado con V7: "
                "384 draws y 2 actualizaciones por episodio; se obtuvieron "
                f"{phase1_draws} y {phase1_updates}."
            )
        if support_draws != 128:
            raise ValueError(
                "V8.1 exige 128 muestras por paso de soporte; se obtuvieron "
                f"{support_draws}."
            )
        if str(self.TRANSPORT_CHECKPOINT_METRIC) not in {
            "cohort_macro_f1", "macro_f1"
        }:
            raise ValueError(
                "TRANSPORT_CHECKPOINT_METRIC debe ser cohort_macro_f1 o macro_f1."
            )
        if int(self.TRANSPORT_MIN_COHORT_SAMPLES) < 2:
            raise ValueError("TRANSPORT_MIN_COHORT_SAMPLES debe ser al menos 2.")
        if not bool(self.CENTER_CLASS_DELTAS):
            raise ValueError(
                "V8 exige CENTER_CLASS_DELTAS=True para identificar el "
                "backbone compartido y las desviaciones de subtipo."
            )
        if not bool(self.BIOLOGY_FIRST):
            raise ValueError("El paquete V8 exige BIOLOGY_FIRST=True.")
        if str(self.BIOLOGY_EVIDENCE_METHOD) != (
            "within_cohort_differential_correlation"
        ):
            raise ValueError("Método de evidencia biológica V8 no reconocido.")
        if int(self.BIOLOGY_MIN_SAMPLES_PER_GROUP) < 4:
            raise ValueError(
                "BIOLOGY_MIN_SAMPLES_PER_GROUP debe ser al menos 4."
            )
        if not 1 <= int(self.BIOLOGY_MIN_COHORTS) <= len(self.TRAIN_COHORTS):
            raise ValueError("BIOLOGY_MIN_COHORTS fuera del rango permitido.")
        if int(self.BIOLOGY_EDGE_BATCH_SIZE) <= 0:
            raise ValueError("BIOLOGY_EDGE_BATCH_SIZE debe ser positivo.")
        if float(self.BIOLOGY_CONTEXT_LAMBDA) < 0:
            raise ValueError("BIOLOGY_CONTEXT_LAMBDA no puede ser negativo.")
        if int(self.BIOLOGY_MODULE_POOL_MULTIPLIER) < 1:
            raise ValueError("BIOLOGY_MODULE_POOL_MULTIPLIER debe ser >=1.")
        if int(self.BIOLOGY_MODULE_TARGET_COMPONENTS) < 1:
            raise ValueError("BIOLOGY_MODULE_TARGET_COMPONENTS debe ser >=1.")
        if int(self.BIOLOGY_MODULE_MIN_EDGES) < 1:
            raise ValueError("BIOLOGY_MODULE_MIN_EDGES debe ser >=1.")
        if int(self.BIOLOGY_MATCHED_NULL_DRAWS) < 0:
            raise ValueError("BIOLOGY_MATCHED_NULL_DRAWS no puede ser negativo.")
        if int(self.BIOLOGY_REWIRING_NULL_DRAWS) < 0:
            raise ValueError("BIOLOGY_REWIRING_NULL_DRAWS no puede ser negativo.")
        if set(self.TRAIN_COHORTS) & set(self.TEST_COHORTS):
            raise ValueError("TRAIN_COHORTS y TEST_COHORTS se solapan.")
        if len(self.TRAIN_COHORTS) != 4 or len(self.TEST_COHORTS) != 3:
            raise ValueError("El protocolo externo requiere la partición fija 4/3.")
        if self.INCLUDE_CONTROLS:
            raise ValueError(
                "El protocolo intrinsic4 no admite controles ni Normal-like."
            )
        expected_labels = {"LumA", "LumB", "HER2-enriched", "Basal-like"}
        if set(self.MULTICLASS_LABELS) != expected_labels:
            raise ValueError(
                "El endpoint debe ser exactamente LumA, LumB, "
                "HER2-enriched y Basal-like."
            )
        if str(self.SPARSITY_MODE).lower() != "target":
            raise ValueError("El protocolo V8 usa exclusivamente esparsidad objetivo.")
        if float(self.SPARSITY_BINARIZE) < 0:
            raise ValueError("SPARSITY_BINARIZE debe ser no negativo.")
        if not str(self.TRAIN_REPORT_LABEL).strip():
            raise ValueError("TRAIN_REPORT_LABEL no puede estar vacío.")
        if (
            self.CLASS_GRAPH_TARGET_EDGES is not None
            and int(self.CLASS_GRAPH_TARGET_EDGES) <= 0
        ):
            raise ValueError(
                "CLASS_GRAPH_TARGET_EDGES debe ser positivo o None."
            )
        graph_xh_mode = str(self.GRAPH_LEARNING_XH_MODE).lower()
        if graph_xh_mode != "zero":
            raise ValueError("El protocolo V8 exige GRAPH_LEARNING_XH_MODE='zero'.")
        self.GRAPH_LEARNING_XH_MODE = graph_xh_mode
        if str(self.CLASS_GRAPH_XH_MODE).lower() not in {"zero", "orig"}:
            raise ValueError(
                "CLASS_GRAPH_XH_MODE debe ser 'zero' u 'orig'."
            )
        self.CLASS_GRAPH_XH_MODE = str(
            self.CLASS_GRAPH_XH_MODE
        ).lower()

        src = str(self.GRAPH_SOURCE).lower()
        presets = {
            "huri_omnipath":   (True,  True,  False),
            "string_omnipath": (False, True,  True),
            "all":             (True,  True,  True),
            "huri":            (True,  False, False),
            "omnipath":        (False, True,  False),
            "string":          (False, False, True),
        }
        if src == "custom":
            pass                      # respeta USE_HURI / USE_OMNIPATH / USE_STRING
        elif src in presets:
            self.USE_HURI, self.USE_OMNIPATH, self.USE_STRING = presets[src]
        else:
            raise ValueError(
                f"GRAPH_SOURCE={self.GRAPH_SOURCE!r} no reconocido. Opciones: "
                f"{sorted(presets)} o 'custom'."
            )
        self.GRAPH_SOURCE = src
        if not (self.USE_HURI or self.USE_OMNIPATH or self.USE_STRING):
            raise ValueError("No hay ninguna fuente de grafo activada.")

    # ------------------------------------------------------------------
    def refresh_options(self) -> "InterGATEConfig":
        """Aplica y valida overrides hechos después de instanciar el config."""
        self._sync_endpoint_from_data_prepare()
        self._resolve_task_and_sources()
        return self

    # ------------------------------------------------------------------
    @property
    def MULTICLASS_LABELS(self) -> Tuple[str, ...]:
        """Espacio de etiquetas del modo multiclase, segun INCLUDE_CONTROLS."""
        return (self.MULTICLASS_LABELS_WITH_CONTROL if self.INCLUDE_CONTROLS
                else self.MULTICLASS_LABELS_NO_CONTROL)

    @property
    def DROP_LABELS(self) -> Tuple[str, ...]:
        """Etiquetas que se eliminan del dataset antes de definir la tarea."""
        return () if self.INCLUDE_CONTROLS else (self.CONTROL_LABEL_NAME,)


# Singleton – import this in notebooks
CFG = InterGATEConfig()


# Campos que definen la identidad de un modelo entrenado. Se excluyen rutas,
# dispositivo y opciones puramente operativas, porque no cambian el modelo.
MODEL_CONFIG_FIELDS = (
    "MODEL_VERSION", "TRANSPORT_PROTOCOL_VERSION",
    "MULTICLASS_LABELS_NO_CONTROL", "TRAIN_COHORTS", "TEST_COHORTS",
    "GRAPH_SOURCE", "USE_HURI", "USE_OMNIPATH", "USE_STRING",
    "PER_LABEL_GRAPHS", "CENTER_CLASS_DELTAS", "PER_LABEL_DELTA_L1",
    "PER_LABEL_DELTA_DIVERSITY_LAMBDA", "PER_LABEL_DELTA_LR_MULT",
    "CLASS_GRAPH_TARGET_EDGES", "CLASS_GRAPH_STABILITY_FIXED_BUDGET",
    "SELECT_METRIC", "GRAPH_LEARNING_XH_MODE", "CLASS_GRAPH_XH_MODE",
    "ADD_SELF_LOOPS_IN_GRAPH", "CONNECTED_ONLY",
    "ADD_REGULATOR_FEATURES", "REG_STATS", "REG_MIN_GENES",
    "REG_MAX_REGULATORS",
    "SCALE_MODE", "USE_QUANTILE", "VAL_SIZE", "SEED",
    "GRAPH_BATCH_SIZE", "GRAPH_ACCUM_STEPS", "BATCH_SIZE", "ACCUM_STEPS",
    "PATIENCE", "LR", "WEIGHT_DECAY", "FOCAL_GAMMA", "GATE_TAU_SCHEDULE",
    "SPARSITY_MODE", "SPARSITY_LAMBDA", "SPARSITY_BINARIZE",
    "EDGE_L1_PER_EDGE", "AUX_LAMBDA", "KEEP_MIN",
    "HIDDEN", "NUM_LAYERS", "DROPOUT", "NUM_HEADS", "POOL_HEADS",
    "XGRAPH_DROPOUT", "KEEP_SELF_LOOPS", "BLOCK_USE_SELF", "BLOCK_RESIDUAL",
    "EDGE_TYPE_GATING", "SIGNED_CHANNELS", "SIGNED_CHANNELS_MODE",
    "ADD_CONNECTIVITY_PENALTY", "CONNECTIVITY_LAMBDA",
    "CONNECTIVITY_MIN_DEG", "CONNECTIVITY_USE_ABS",
    "USE_MIXUP", "MIXUP_ALPHA", "MIXUP_P",
    "EPOCHS1", "FT_EPOCHS_A", "FT_EPOCHS_B",
    "DO_PRETRAIN", "DO_STABILITY_SELECTION",
    "STABILITY_SEEDS", "OUTER_RUNS", "STAB_RUNS", "STAB_SEED_OFFSETS",
    "STAB_EPOCHS", "STAB_KEEP_FINAL", "STAB_FREQ_THR",
    "STAB_STRICT_FREQ_THR", "OUTER_GENE_CORE_THR",
    "OUTER_GENE_SENSITIVITY_THR", "STABILITY_DATA_RESAMPLING",
    "STABILITY_SCOPE",
    "TRANSPORT_EPISODES", "TRANSPORT_INNER_LR", "TRANSPORT_FIRST_ORDER",
    "TRANSPORT_META_TRAIN_WEIGHT", "TRANSPORT_META_VALIDATION_WEIGHT",
    "TRANSPORT_TOPOLOGY_LAMBDA", "TRANSPORT_SOFT_TOPK_TAU",
    "TRANSPORT_QUOTA_END_FRACTION", "TRANSPORT_PATIENCE",
    "TRANSPORT_BALANCED_SAMPLING", "TRANSPORT_STEPS_PER_EPISODE",
    "TRANSPORT_CHECKPOINT_METRIC", "TRANSPORT_SUPPORT_STEPS",
    "TRANSPORT_SUPPORT_BATCHES", "TRANSPORT_MIN_COHORT_SAMPLES",
    "BIOLOGY_FIRST", "BIOLOGY_EVIDENCE_METHOD",
    "BIOLOGY_MIN_SAMPLES_PER_GROUP", "BIOLOGY_MIN_COHORTS",
    "BIOLOGY_EDGE_BATCH_SIZE", "BIOLOGY_META_Z_CLIP",
    "BIOLOGY_HETEROGENEITY_PENALTY",
    "BIOLOGY_CONTEXT_LAMBDA", "BIOLOGY_CONTEXT_WARMUP_EPISODES",
    "BIOLOGY_SCORE_CONTEXT_WEIGHT", "BIOLOGY_SCORE_SUPPORT_WEIGHT",
    "BIOLOGY_MODULE_POOL_MULTIPLIER", "BIOLOGY_MODULE_TARGET_COMPONENTS",
    "BIOLOGY_MODULE_MIN_EDGES",
    "BIOLOGY_MATCHED_NULL_DRAWS", "BIOLOGY_REWIRING_NULL_DRAWS",
    "BIOLOGY_NONINFERIORITY_MARGIN",
    "USE_HYBRID_MODEL", "PHASE2_A_MODE",
    "PHASE2_FIXED_GRAPH_BYPASS_GATES", "PHASE2_FREEZE_STRUCTURAL_PARAMETERS",
    "HYBRID_TAB_LAYERS", "HYBRID_TAB_DROPOUT", "HYBRID_FUSION_DROPOUT",
    "HYBRID_BLEND_LOGITS", "HYBRID_BLEND_INIT",
    "USE_NODE_IDENTITY", "EXPERT_DEEP_SUPERVISION_LAMBDA",
    "TAB_FULL_GENE_UNIVERSE", "HYBRID_TAB_HIDDEN",
    "BRANCH_DROPOUT_TAB", "BRANCH_DROPOUT_GNN", "GNN_AUX_LOSS_LAMBDA",
    "ALLOW_EXTERNAL_DURING_TRAINING",
)

IMPLEMENTATION_FILES = (
    "config.py", "model.py", "training.py", "transport.py", "ablation.py",
    "graph.py", "graph_cache.py", "pruning.py", "biology.py",
    "runtime_setup.py", "prepared_data.py", "data.py", "losses.py",
    "stability.py", "utils.py",
)


def _jsonable_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def model_config_snapshot(cfg: InterGATEConfig = CFG) -> dict[str, Any]:
    """Return the canonical V8 configuration recorded in every bundle."""
    return {
        name: _jsonable_config_value(getattr(cfg, name))
        for name in MODEL_CONFIG_FIELDS
    }


def model_config_fingerprint(cfg: InterGATEConfig = CFG) -> str:
    """SHA-256 identity used to reject stale or incompatible checkpoints."""
    payload = json.dumps(
        model_config_snapshot(cfg), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def implementation_fingerprint() -> str:
    """SHA-256 of the source files that define training and graph inference."""
    digest = hashlib.sha256()
    module_dir = Path(__file__).resolve().parent
    for filename in IMPLEMENTATION_FILES:
        path = module_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
