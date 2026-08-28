"""Post-hoc probability calibration fitted exclusively on development validation.

Rationale
---------
InterGATE assembles its four-class logit vector from the diagonal of four
per-class experts (``model.HybridGNNTabular.forward``), so the softmax is taken
over logits produced by different embeddings and is not the output of any single
normalized classifier.  The consequence is visible in the locked external run:
per-class F1 ranges from 0.953-0.961 (Basal-like) to 0.691-0.743
(HER2-enriched).  A monotone, low-parameter recalibration of the assembled
logits corrects that scale mismatch without touching the network, the graph or
the protocol.

Everything here is fitted on ``val_idx`` only.  No external sample, label or
class prior enters the fit; ``apply_to_frame`` merely applies a frozen
transformation.  This is the same discipline as the paired tabular benchmark:
the models are frozen on development and applied unchanged.

Families implemented
--------------------
``temperature``   single scalar T, softmax(log p / T).  Guo et al. (2017).
``vector``        diagonal Dirichlet calibration, softmax(a * log p + b).
                  Kull et al. (2019).
``matrix``        full Dirichlet calibration, softmax(W log p + b), with ODIR
                  L2 regularization on off-diagonal terms.  Kull et al. (2019).
``macro_f1_bias`` per-class additive bias chosen by coordinate search on
                  macro-F1 directly; the family already sketched in
                  ``intergate.bootstrap.tune_class_biases``.
``identity``      no calibration, kept as the reference arm.

Log-loss-optimal calibration does not maximize macro-F1, which is a
thresholded metric, so both objectives are represented and the choice between
them is made by repeated stratified cross-validation *inside* validation.

References
----------
Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural
networks. ICML 2017; PMLR 70:1321-1330.
Kull M, Perello-Nieto M, Kängsepp M, Silva Filho T, Song H, Flach P. Beyond
temperature scaling: obtaining well-calibrated multi-class probabilities with
Dirichlet calibration. NeurIPS 2019; 32:12295-12305.
Menon AK, Jayasumana S, Rawat AS, Jain H, Veit A, Kumar S. Long-tail learning
via logit adjustment. ICLR 2021.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold

__all__ = [
    "log_probabilities",
    "Calibrator",
    "IdentityCalibrator",
    "TemperatureCalibrator",
    "VectorCalibrator",
    "MatrixCalibrator",
    "MacroF1BiasCalibrator",
    "CALIBRATOR_FAMILIES",
    "build_calibrator",
    "select_family_by_cv",
    "fit_per_seed",
    "apply_to_frame",
    "probability_columns",
]


# ── helpers ────────────────────────────────────────────────────────────


def probability_columns(class_names: Sequence[str]) -> list[str]:
    """Column names used by the prediction frames."""
    return [f"p_{name}" for name in class_names]


def log_probabilities(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalized log-probabilities.

    Calibration operates on log p rather than on raw logits because the frozen
    bundles only persist probabilities.  For the temperature family this is
    exact: softmax is shift-invariant, so scaling ``log p`` and scaling the
    original logits give the same result.  For the vector and matrix families
    ``log p`` is the canonical input anyway; that is precisely how Dirichlet
    calibration is defined (Kull et al., 2019).
    """
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"proba debe ser 2-D, no {p.shape}.")
    if np.any(~np.isfinite(p)):
        raise ValueError("proba contiene valores no finitos.")
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return out


def _macro_f1(y_true: np.ndarray, logp: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            logp.argmax(axis=1),
            labels=np.arange(logp.shape[1]),
            average="macro",
            zero_division=0,
        )
    )


def _log_loss(y_true: np.ndarray, logp: np.ndarray) -> float:
    return float(
        log_loss(
            y_true,
            softmax(logp, axis=1),
            labels=list(range(logp.shape[1])),
        )
    )


# ── calibrators ────────────────────────────────────────────────────────


class Calibrator:
    """Base class.  ``transform`` returns calibrated log-probabilities."""

    name = "base"

    def __init__(self, n_classes: int):
        self.n_classes = int(n_classes)
        self.fitted = False

    def fit(self, logp: np.ndarray, y: np.ndarray) -> "Calibrator":
        raise NotImplementedError

    def transform(self, logp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, logp: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(logp, y).transform(logp)

    def parameters(self) -> dict:
        return {}

    def _check(self, logp: np.ndarray) -> np.ndarray:
        arr = np.asarray(logp, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_classes:
            raise ValueError(
                f"Se esperaban {self.n_classes} columnas, no {arr.shape}."
            )
        if not self.fitted:
            raise RuntimeError("El calibrador no está ajustado.")
        return arr


class IdentityCalibrator(Calibrator):
    """Reference arm: leaves the probabilities untouched."""

    name = "identity"

    def fit(self, logp: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        self.fitted = True
        return self

    def transform(self, logp: np.ndarray) -> np.ndarray:
        return np.asarray(logp, dtype=np.float64)


class TemperatureCalibrator(Calibrator):
    """softmax(log p / T), a single positive scalar (Guo et al., 2017)."""

    name = "temperature"

    def __init__(self, n_classes: int):
        super().__init__(n_classes)
        self.log_T_ = 0.0

    def fit(self, logp: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        z = np.asarray(logp, dtype=np.float64)
        target = _one_hot(y, self.n_classes)

        def objective(theta: np.ndarray) -> float:
            scaled = z * np.exp(-theta[0])
            return -float((target * log_softmax(scaled, axis=1)).sum()) / len(z)

        result = minimize(
            objective, x0=np.zeros(1), method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.log_T_ = float(result.x[0])
        self.fitted = True
        return self

    def transform(self, logp: np.ndarray) -> np.ndarray:
        return self._check(logp) * np.exp(-self.log_T_)

    def parameters(self) -> dict:
        return {"temperature": float(np.exp(self.log_T_))}


class VectorCalibrator(Calibrator):
    """softmax(a * log p + b): diagonal Dirichlet calibration.

    2C parameters.  This is the family that directly repairs a per-class scale
    mismatch, which is what the expert-diagonal assembly produces.
    """

    name = "vector"

    def __init__(self, n_classes: int, l2: float = 1e-3):
        super().__init__(n_classes)
        self.l2 = float(l2)
        self.a_ = np.ones(self.n_classes)
        self.b_ = np.zeros(self.n_classes)

    def fit(self, logp: np.ndarray, y: np.ndarray) -> "VectorCalibrator":
        z = np.asarray(logp, dtype=np.float64)
        target = _one_hot(y, self.n_classes)
        C = self.n_classes

        def objective(theta: np.ndarray) -> float:
            a = theta[:C]
            b = theta[C:]
            scaled = z * a[None, :] + b[None, :]
            nll = -float((target * log_softmax(scaled, axis=1)).sum()) / len(z)
            penalty = self.l2 * (
                float(((a - 1.0) ** 2).sum()) + float((b ** 2).sum())
            )
            return nll + penalty

        x0 = np.concatenate([np.ones(C), np.zeros(C)])
        result = minimize(
            objective, x0=x0, method="L-BFGS-B", options={"maxiter": 1000},
        )
        self.a_ = result.x[:C].copy()
        self.b_ = result.x[C:].copy()
        self.fitted = True
        return self

    def transform(self, logp: np.ndarray) -> np.ndarray:
        z = self._check(logp)
        return z * self.a_[None, :] + self.b_[None, :]

    def parameters(self) -> dict:
        return {"a": self.a_.tolist(), "b": self.b_.tolist()}


class MatrixCalibrator(Calibrator):
    """softmax(W log p + b): full Dirichlet calibration with ODIR.

    C^2 + C parameters (20 for four classes).  With 743 validation samples this
    is estimable but it is the arm most exposed to selection noise, which is why
    the off-diagonal L2 (ODIR, Kull et al., 2019) is on by default.
    """

    name = "matrix"

    def __init__(self, n_classes: int, odir: float = 1e-2):
        super().__init__(n_classes)
        self.odir = float(odir)
        self.W_ = np.eye(self.n_classes)
        self.b_ = np.zeros(self.n_classes)

    def fit(self, logp: np.ndarray, y: np.ndarray) -> "MatrixCalibrator":
        z = np.asarray(logp, dtype=np.float64)
        target = _one_hot(y, self.n_classes)
        C = self.n_classes
        off = ~np.eye(C, dtype=bool)

        def objective(theta: np.ndarray) -> float:
            W = theta[: C * C].reshape(C, C)
            b = theta[C * C:]
            scaled = z @ W.T + b[None, :]
            nll = -float((target * log_softmax(scaled, axis=1)).sum()) / len(z)
            penalty = self.odir * (
                float((W[off] ** 2).sum()) + float((b ** 2).sum())
            )
            return nll + penalty

        x0 = np.concatenate([np.eye(C).ravel(), np.zeros(C)])
        result = minimize(
            objective, x0=x0, method="L-BFGS-B", options={"maxiter": 2000},
        )
        self.W_ = result.x[: C * C].reshape(C, C).copy()
        self.b_ = result.x[C * C:].copy()
        self.fitted = True
        return self

    def transform(self, logp: np.ndarray) -> np.ndarray:
        z = self._check(logp)
        return z @ self.W_.T + self.b_[None, :]

    def parameters(self) -> dict:
        return {"W": self.W_.tolist(), "b": self.b_.tolist()}


class MacroF1BiasCalibrator(Calibrator):
    """Per-class additive bias maximizing macro-F1 by coordinate search.

    Same family as ``intergate.bootstrap.tune_class_biases``, reimplemented here
    so that the whole comparison lives behind one interface.  The reference
    class is held at zero to remove the shift indeterminacy.
    """

    name = "macro_f1_bias"

    def __init__(
        self,
        n_classes: int,
        ref_class: int = 0,
        n_passes: int = 3,
        grid_half_width: float = 1.5,
        grid_points: int = 61,
    ):
        super().__init__(n_classes)
        self.ref_class = int(ref_class)
        self.n_passes = int(n_passes)
        self.coarse = np.linspace(
            -float(grid_half_width), float(grid_half_width), int(grid_points)
        )
        self.bias_ = np.zeros(self.n_classes)

    def fit(self, logp: np.ndarray, y: np.ndarray) -> "MacroF1BiasCalibrator":
        z = np.asarray(logp, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        bias = np.zeros(self.n_classes)
        best = _macro_f1(y, z + bias[None, :])

        for _ in range(self.n_passes):
            improved = False
            for c in range(self.n_classes):
                if c == self.ref_class:
                    continue
                grid = np.concatenate([
                    self.coarse,
                    np.linspace(bias[c] - 0.05, bias[c] + 0.05, 21),
                ])
                for value in grid:
                    candidate = bias.copy()
                    candidate[c] = float(value)
                    score = _macro_f1(y, z + candidate[None, :])
                    if score > best + 1e-9:
                        best = score
                        bias = candidate
                        improved = True
            if not improved:
                break

        self.bias_ = bias
        self.fitted = True
        return self

    def transform(self, logp: np.ndarray) -> np.ndarray:
        return self._check(logp) + self.bias_[None, :]

    def parameters(self) -> dict:
        return {"bias": self.bias_.tolist(), "ref_class": int(self.ref_class)}


CALIBRATOR_FAMILIES: tuple[str, ...] = (
    "identity",
    "temperature",
    "vector",
    "matrix",
    "macro_f1_bias",
)

_REGISTRY = {
    "identity": IdentityCalibrator,
    "temperature": TemperatureCalibrator,
    "vector": VectorCalibrator,
    "matrix": MatrixCalibrator,
    "macro_f1_bias": MacroF1BiasCalibrator,
}


def build_calibrator(family: str, n_classes: int, **options) -> Calibrator:
    if family not in _REGISTRY:
        raise KeyError(f"Familia desconocida: {family!r}. Use {CALIBRATOR_FAMILIES}.")
    calibrator = _REGISTRY[family](n_classes=int(n_classes))
    for key, value in options.items():
        if not hasattr(calibrator, key):
            raise AttributeError(f"{family} no acepta la opción {key!r}.")
        setattr(calibrator, key, value)
    return calibrator


# ── model selection inside validation ──────────────────────────────────


def select_family_by_cv(
    logp: np.ndarray,
    y: np.ndarray,
    *,
    families: Iterable[str] = CALIBRATOR_FAMILIES,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 20260825,
    options: Mapping[str, Mapping] | None = None,
) -> pd.DataFrame:
    """Repeated stratified CV *within* validation.

    The fold structure protects against reading a gain that is only the
    calibrator memorizing 743 points.  Report this table before applying
    anything to the external block; the honest statement about the expected
    external gain is the held-out fold mean, not the in-sample gain.
    """
    logp = np.asarray(logp, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n_classes = logp.shape[1]
    options = dict(options or {})

    splitter = RepeatedStratifiedKFold(
        n_splits=int(n_splits), n_repeats=int(n_repeats),
        random_state=int(random_state),
    )

    rows: list[dict] = []
    for family in families:
        for fold_id, (fit_rows, held_rows) in enumerate(splitter.split(logp, y)):
            calibrator = build_calibrator(
                family, n_classes, **options.get(family, {})
            )
            calibrator.fit(logp[fit_rows], y[fit_rows])
            held = calibrator.transform(logp[held_rows])
            rows.append({
                "family": family,
                "fold": int(fold_id),
                "macro_f1": _macro_f1(y[held_rows], held),
                "log_loss": _log_loss(y[held_rows], held),
                "n_held_out": int(len(held_rows)),
            })

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("family", sort=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_sd=("log_loss", "std"),
            n_folds=("fold", "count"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary.attrs["detail"] = detail
    return summary


def fit_per_seed(
    validation_frame: pd.DataFrame,
    *,
    class_names: Sequence[str],
    family: str,
    seed_column: str = "seed",
    truth_column: str = "y_true",
    options: Mapping | None = None,
) -> dict[int, Calibrator]:
    """Fit one calibrator per seed on that seed's validation predictions.

    Pairing the calibrator with its own seed keeps the five runs exchangeable,
    which is what the ensemble average and the paired bootstrap assume.
    """
    columns = probability_columns(class_names)
    missing = [column for column in columns if column not in validation_frame]
    if missing:
        raise KeyError(f"Faltan columnas de probabilidad: {missing}.")

    calibrators: dict[int, Calibrator] = {}
    for seed, group in validation_frame.groupby(seed_column, sort=True):
        logp = log_probabilities(group[columns].to_numpy(dtype=np.float64))
        y = group[truth_column].to_numpy(dtype=int)
        calibrator = build_calibrator(
            family, len(class_names), **dict(options or {})
        )
        calibrator.fit(logp, y)
        calibrators[int(seed)] = calibrator
    return calibrators


def apply_to_frame(
    frame: pd.DataFrame,
    calibrators: Mapping[int, Calibrator],
    *,
    class_names: Sequence[str],
    seed_column: str = "seed",
    prediction_column: str = "y_pred",
) -> pd.DataFrame:
    """Apply frozen per-seed calibrators to a prediction frame.

    Works unchanged on ``external_predictions.csv.gz`` and on the paired
    benchmark frames, since both use the ``p_<class>`` layout.
    """
    columns = probability_columns(class_names)
    out = frame.copy()
    for seed, group in frame.groupby(seed_column, sort=True):
        seed = int(seed)
        if seed not in calibrators:
            raise KeyError(f"No hay calibrador ajustado para la semilla {seed}.")
        logp = log_probabilities(group[columns].to_numpy(dtype=np.float64))
        calibrated = softmax(calibrators[seed].transform(logp), axis=1)
        out.loc[group.index, columns] = calibrated
        out.loc[group.index, prediction_column] = calibrated.argmax(axis=1)
    out[prediction_column] = out[prediction_column].astype(int)
    return out
