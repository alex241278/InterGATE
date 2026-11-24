"""Strict reader for the immutable outputs of ``data_prepare``.

The contract deliberately has two expression matrices and two metadata files.
No function in this module creates a new external split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read_expr(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    expr = pd.read_csv(path, index_col=0)
    expr.index = expr.index.astype(str).str.strip()
    expr.columns = expr.columns.astype(str).str.strip()
    if expr.index.duplicated().any():
        raise ValueError(f"Genes duplicados en {path.name}.")
    if expr.columns.duplicated().any():
        raise ValueError(f"Muestras duplicadas en {path.name}.")
    expr = expr.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(expr.to_numpy()).all():
        raise ValueError(f"NaN/inf en {path.name}.")
    return expr


def _read_meta(path: Path, sample_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    meta = pd.read_csv(path)
    meta.columns = [str(c).strip().replace("\ufeff", "") for c in meta.columns]
    if sample_col not in meta.columns:
        # Las salidas pandas pueden guardar el índice sin nombre en la primera
        # columna. Solo se acepta si sus valores son únicos.
        first = meta.columns[0]
        if not str(first).startswith("Unnamed"):
            raise ValueError(f"Falta {sample_col!r} en {path.name}.")
        meta = meta.rename(columns={first: sample_col})
    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    if meta[sample_col].duplicated().any():
        raise ValueError(f"Muestras duplicadas en {path.name}.")
    return meta.set_index(sample_col, drop=False)


def _align_meta_exact(meta: pd.DataFrame, samples: pd.Index, name: str) -> pd.DataFrame:
    """Require a one-to-one, exact expression/metadata sample contract."""
    samples = pd.Index(samples.astype(str))
    observed = set(meta.index.astype(str))
    wanted = set(samples)
    if observed != wanted:
        raise ValueError(
            f"Muestras de expresión/metadata {name} no coinciden: "
            f"faltan_meta={sorted(wanted-observed)[:10]}, "
            f"sobran_meta={sorted(observed-wanted)[:10]}"
        )
    return meta.loc[samples].copy()


@dataclass(frozen=True)
class PreparedDataBundle:
    train_expr: pd.DataFrame       # genes × samples
    test_expr: pd.DataFrame        # genes × samples
    train_meta: pd.DataFrame
    test_meta: pd.DataFrame
    sample_col: str
    cohort_col: str
    label_col: str

    @property
    def genes(self) -> list[str]:
        return self.train_expr.index.astype(str).tolist()

    @property
    def active_labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.train_meta[self.label_col].astype(str).unique()))

    def expression_samples(self) -> pd.DataFrame:
        """Development followed by external test, samples × genes."""
        return pd.concat([self.train_expr, self.test_expr], axis=1).T

    def metadata(self) -> pd.DataFrame:
        tr = self.train_meta.copy()
        te = self.test_meta.copy()
        tr["split"] = "train"
        te["split"] = "test"
        return pd.concat([tr, te], axis=0)


@dataclass(frozen=True)
class DevelopmentDataBundle:
    """Development-only view used before the external protocol is unlocked."""

    train_expr: pd.DataFrame       # genes × samples
    train_meta: pd.DataFrame
    sample_col: str
    cohort_col: str
    label_col: str

    @property
    def genes(self) -> list[str]:
        return self.train_expr.index.astype(str).tolist()

    @property
    def active_labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.train_meta[self.label_col].astype(str).unique()))

    def expression_samples(self) -> pd.DataFrame:
        return self.train_expr.T

    def metadata(self) -> pd.DataFrame:
        out = self.train_meta.copy()
        out["split"] = "train"
        return out


def validate_development_bundle(bundle: DevelopmentDataBundle, *,
                                train_cohorts: tuple[str, ...],
                                active_labels: tuple[str, ...],
                                min_labels_per_cohort: int = 2) -> None:
    """Validate development without opening either external-test CSV."""
    expr, meta = bundle.train_expr, bundle.train_meta
    sample_col, cohort_col, label_col = (
        bundle.sample_col, bundle.cohort_col, bundle.label_col
    )
    required_cols = {sample_col, cohort_col, label_col}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata train: faltan {sorted(missing)}.")
    if set(expr.columns) != set(meta.index):
        raise ValueError("Muestras de expresión/metadata train no coinciden.")

    observed_cohorts = set(meta[cohort_col].astype(str))
    expected_cohorts = set(train_cohorts)
    if observed_cohorts != expected_cohorts:
        raise ValueError(
            "Cohortes train distintas de data_prepare: "
            f"faltan={sorted(expected_cohorts-observed_cohorts)}, "
            f"sobran={sorted(observed_cohorts-expected_cohorts)}"
        )
    observed_labels = set(meta[label_col].astype(str))
    expected_labels = set(active_labels)
    if observed_labels != expected_labels:
        raise ValueError(
            "Clases train incompatibles con el endpoint: "
            f"faltan={sorted(expected_labels-observed_labels)}, "
            f"sobran={sorted(observed_labels-expected_labels)}"
        )
    n_labels = meta.groupby(cohort_col)[label_col].nunique()
    bad = n_labels[n_labels < int(min_labels_per_cohort)]
    if len(bad):
        raise ValueError(
            f"Cohortes train con <{min_labels_per_cohort} etiquetas: "
            f"{bad.to_dict()}"
        )


def load_development_bundle(cfg=None, *,
                            train_expr_csv: Optional[Path] = None,
                            train_meta_csv: Optional[Path] = None,
                            validate: bool = True) -> DevelopmentDataBundle:
    """Read development only; external expression and labels remain unopened."""
    if cfg is None:
        from .config import CFG as cfg
    expr_path = Path(train_expr_csv or cfg.TRAIN_EXPR_CSV)
    meta_path = Path(train_meta_csv or cfg.TRAIN_META_CSV)
    missing = [str(path) for path in (expr_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Ejecuta data_prepare/00→03 antes del modelo. Faltan:\n- "
            + "\n- ".join(missing)
        )
    expr = _read_expr(expr_path)
    meta = _align_meta_exact(
        _read_meta(meta_path, cfg.SAMPLE_COL), expr.columns, "train"
    )
    bundle = DevelopmentDataBundle(
        train_expr=expr, train_meta=meta, sample_col=cfg.SAMPLE_COL,
        cohort_col=cfg.COHORT_COL, label_col=cfg.LABEL_COL,
    )
    if validate:
        validate_development_bundle(
            bundle,
            train_cohorts=tuple(cfg.TRAIN_COHORTS),
            active_labels=tuple(cfg.MULTICLASS_LABELS),
            min_labels_per_cohort=int(cfg.MIN_LABELS_PER_COHORT),
        )
    return bundle


def validate_prepared_bundle(bundle: PreparedDataBundle, *,
                             train_cohorts: tuple[str, ...],
                             test_cohorts: tuple[str, ...],
                             active_labels: tuple[str, ...],
                             min_labels_per_cohort: int = 2) -> None:
    """Fail-fast validation of sample, gene, cohort and endpoint boundaries."""
    tr, te = bundle.train_expr, bundle.test_expr
    mt, me = bundle.train_meta, bundle.test_meta
    sample_col, cohort_col, label_col = (
        bundle.sample_col, bundle.cohort_col, bundle.label_col
    )
    required_cols = {sample_col, cohort_col, label_col}
    for name, meta in (("train", mt), ("test", me)):
        missing = required_cols - set(meta.columns)
        if missing:
            raise ValueError(f"Metadata {name}: faltan {sorted(missing)}.")

    if set(tr.index) != set(te.index):
        raise ValueError("Train y test no contienen el mismo universo de genes.")
    if tr.index.tolist() != te.index.tolist():
        raise ValueError("Train y test contienen los genes en distinto orden.")
    if set(tr.columns) != set(mt.index):
        raise ValueError("Muestras de expresión/metadata train no coinciden.")
    if set(te.columns) != set(me.index):
        raise ValueError("Muestras de expresión/metadata test no coinciden.")
    if set(tr.columns) & set(te.columns):
        raise ValueError("Hay muestras compartidas entre train y test.")

    observed_train = set(mt[cohort_col].astype(str))
    observed_test = set(me[cohort_col].astype(str))
    expected_train, expected_test = set(train_cohorts), set(test_cohorts)
    if observed_train != expected_train:
        raise ValueError(
            f"Cohortes train distintas de data_prepare: "
            f"faltan={sorted(expected_train-observed_train)}, "
            f"sobran={sorted(observed_train-expected_train)}"
        )
    if observed_test != expected_test:
        raise ValueError(
            f"Cohortes test distintas de data_prepare: "
            f"faltan={sorted(expected_test-observed_test)}, "
            f"sobran={sorted(observed_test-expected_test)}"
        )
    if observed_train & observed_test:
        raise ValueError("Hay cohortes compartidas entre train y test.")

    wanted = set(active_labels)
    for name, meta in (("train", mt), ("test", me)):
        observed = set(meta[label_col].astype(str))
        if observed != wanted:
            raise ValueError(
                f"Clases {name} incompatibles con el endpoint: "
                f"faltan={sorted(wanted-observed)}, sobran={sorted(observed-wanted)}"
            )
        n_labels = meta.groupby(cohort_col)[label_col].nunique()
        bad = n_labels[n_labels < int(min_labels_per_cohort)]
        if len(bad):
            raise ValueError(
                f"Cohortes {name} con <{min_labels_per_cohort} etiquetas: "
                f"{bad.to_dict()}"
            )


def load_prepared_bundle(cfg=None, *,
                         train_expr_csv: Optional[Path] = None,
                         test_expr_csv: Optional[Path] = None,
                         train_meta_csv: Optional[Path] = None,
                         test_meta_csv: Optional[Path] = None,
                         validate: bool = True) -> PreparedDataBundle:
    """Load the exact fixed 4/3 M-ComBat outputs declared by the config."""
    if cfg is None:
        from .config import CFG as cfg
    paths = {
        "train_expr": Path(train_expr_csv or cfg.TRAIN_EXPR_CSV),
        "test_expr": Path(test_expr_csv or cfg.TEST_EXPR_CSV),
        "train_meta": Path(train_meta_csv or cfg.TRAIN_META_CSV),
        "test_meta": Path(test_meta_csv or cfg.TEST_META_CSV),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Ejecuta data_prepare/00→03 antes del modelo. Faltan:\n- "
            + "\n- ".join(missing)
        )
    tr = _read_expr(paths["train_expr"])
    te = _read_expr(paths["test_expr"])
    mt = _align_meta_exact(
        _read_meta(paths["train_meta"], cfg.SAMPLE_COL), tr.columns, "train"
    )
    me = _align_meta_exact(
        _read_meta(paths["test_meta"], cfg.SAMPLE_COL), te.columns, "test"
    )
    bundle = PreparedDataBundle(
        train_expr=tr, test_expr=te, train_meta=mt, test_meta=me,
        sample_col=cfg.SAMPLE_COL, cohort_col=cfg.COHORT_COL,
        label_col=cfg.LABEL_COL,
    )
    if validate:
        validate_prepared_bundle(
            bundle,
            train_cohorts=tuple(cfg.TRAIN_COHORTS),
            test_cohorts=tuple(cfg.TEST_COHORTS),
            active_labels=tuple(cfg.MULTICLASS_LABELS),
            min_labels_per_cohort=int(cfg.MIN_LABELS_PER_COHORT),
        )
    return bundle
