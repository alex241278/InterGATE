#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
combat_ref.py — M-ComBat de referencia ajustado SOLO en desarrollo y
                proyectado sobre las cohortes externas, con inmoose.
===============================================================================

Contenido
---------
  Bloque 1 : carga, réplicas técnicas, filtrado de cohortes
  Bloque 2 : diagnóstico y corrección del suelo de pseudoconteo
  Bloque 3 : partición fija 4/3 preespecificada a nivel de cohorte
  Bloque 4 : ReferenceMComBat (fit / transform congelado)
  Bloque 5 : verificación de invariancia

Por qué el esquema no tiene fuga
--------------------------------
En inmoose 0.9.x, con ``ref_batch`` especificado:

  * grand_mean = B_hat[ref]                (media del lote de referencia)
  * var_pooled = varianza calculada SOLO con las muestras de la referencia
  * gamma_bar, t2, a_prior, b_prior        se estiman POR LOTE promediando
                                           A TRAVÉS DE GENES, no a través de lotes

Consecuencia verificada: con covar_mod=None, la corrección de una cohorte
no-referencia depende únicamente de (i) esa cohorte y (ii) la referencia.
Añadir o quitar otras cohortes del run no cambia el resultado (~1e-15).
El esquema es por tanto un fit/transform congelado real.

La propiedad se rompe si se introducen covariables estimadas conjuntamente.
Por ello ``covar_mod`` es siempre ``None`` y las etiquetas no participan en
la armonización.

Requisitos: pip install inmoose pandas numpy scikit-learn
===============================================================================
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from inmoose.pycombat import pycombat_norm

warnings.filterwarnings("ignore")
logging.getLogger("inmoose").setLevel(logging.ERROR)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class Config:
    # --- entrada: salidas de 00_union_datasets, antes de M-ComBat ---
    expr_path: str = "./output_prepared/expr_prepared.csv"
    meta_path: str = "./output_prepared/metadata_prepared.csv"

    sample_col: str = "sample"
    cohort_col: str = "dataset"
    label_col: str = "label"
    platform_col: str = "platform"

    # --- partición externa DEFINITIVA (no se vuelve a sortear) ---
    train_cohorts_fixed: tuple = (
        "GSE96058", "GSE25066", "GSE162228", "GSE20711",
    )
    test_cohorts_fixed: tuple = (
        "TCGA-BRCA", "GSE21653", "GSE1456",
    )
    ref_batch: str = "GSE96058"

    # Único endpoint: cuatro subtipos intrínsecos basados en expresión.
    # Basal-like no se interpreta como TNBC y HER2-enriched no se interpreta
    # como positividad HER2 clínica.
    active_labels: tuple = (
        "LumA", "LumB", "HER2-enriched", "Basal-like",
    )
    require_all_active_labels_in_each_split: bool = True

    # --- reglas de inclusión ---
    min_labels_per_cohort: int = 2
    min_samples_per_cohort: int = 2        # mínimo requerido por la proyección

    # --- réplicas técnicas ---
    drop_replicates: bool = True
    replicate_suffix: str = "repl"

    # --- suelo de pseudoconteo ---
    fix_pseudocount: bool = True
    pseudo_destino: float = 1.0            # log2(x+1), como TCGA
    max_frac_suelo_ref: float = 0.50       # filtro de detección en la referencia

    var_eps: float = 1e-8
    out_dir: str = "./output_mcombat"

    train_cohorts: list = field(default_factory=list)
    test_cohorts: list = field(default_factory=list)


# =============================================================================
# BLOQUE 1 — CARGA Y LIMPIEZA
# =============================================================================

def cargar(cfg: Config):
    """Carga la matriz combinada y exige alineación exacta de muestras."""
    print("\n[1] Cargando matriz combinada...")
    expr = pd.read_csv(cfg.expr_path, index_col=0)
    meta = pd.read_csv(cfg.meta_path)
    meta.columns = [str(c).strip().replace("\ufeff", "") for c in meta.columns]

    expr.index = expr.index.astype(str).str.strip()
    expr.columns = expr.columns.astype(str).str.strip()

    if cfg.sample_col not in meta.columns:
        meta = pd.read_csv(cfg.meta_path, index_col=0)
        meta.index = meta.index.astype(str).str.strip()
        meta[cfg.sample_col] = meta.index.astype(str)
    else:
        meta[cfg.sample_col] = meta[cfg.sample_col].astype(str).str.strip()
        meta = meta.set_index(cfg.sample_col, drop=False)

    if meta[cfg.sample_col].duplicated().any():
        duplicadas = meta.loc[
            meta[cfg.sample_col].duplicated(keep=False), cfg.sample_col
        ].astype(str).unique().tolist()
        raise ValueError(
            "La metadata contiene muestras duplicadas: "
            f"{duplicadas[:10]}"
        )

    solo_expr = expr.columns.difference(meta.index)
    solo_meta = meta.index.difference(expr.columns)
    if len(solo_expr) or len(solo_meta):
        raise ValueError(
            "Expresión y metadata preparadas no coinciden exactamente. "
            f"Solo expresión={list(solo_expr[:10])}; "
            f"solo metadata={list(solo_meta[:10])}."
        )
    meta = meta.loc[expr.columns].copy()

    required = {cfg.sample_col, cfg.cohort_col, cfg.label_col}
    missing_columns = sorted(required - set(meta.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")
    if meta[[cfg.cohort_col, cfg.label_col]].isna().any().any():
        raise ValueError("Hay cohortes o etiquetas ausentes en la metadata.")

    if expr.index.duplicated().any():
        print(f"    {int(expr.index.duplicated().sum())} genes duplicados -> media")
        expr = expr.groupby(expr.index).mean()

    print(f"    {expr.shape[0]} genes x {expr.shape[1]} muestras")
    return expr, meta


def auditar_replicas(expr, meta, cfg: Config):
    """
    GSE96058 se distribuye como '3273 samples and 136 replicates'.
    Las réplicas técnicas (sufijo 'repl') son la MISMA paciente medida dos
    veces: inflan las estadísticas del lote de referencia y, si se hace CV
    dentro de train, generan fuga entre folds.
    """
    print("\n[2] Auditando réplicas técnicas...")
    s = meta[cfg.sample_col].astype(str)
    es_repl = s.str.endswith(cfg.replicate_suffix)
    n = int(es_repl.sum())

    if n == 0:
        print("    ninguna réplica detectada")
        return expr, meta, pd.DataFrame()

    base = s[es_repl].str[: -len(cfg.replicate_suffix)]
    audit = pd.DataFrame({
        "replica": s[es_repl].values,
        "muestra_base": base.values,
        "cohorte": meta.loc[es_repl, cfg.cohort_col].values,
        "base_presente": base.isin(set(s)).values,
    })
    print(f"    {n} réplicas ({int(audit['base_presente'].sum())} con su base "
          f"presente) — {audit['cohorte'].value_counts().to_dict()}")

    if cfg.drop_replicates:
        keep = meta.index[~es_repl.values]
        expr, meta = expr.loc[:, keep], meta.loc[keep]
        print(f"    eliminadas -> {expr.shape[1]} muestras")
    else:
        print("    CONSERVADAS: no hagas CV dentro de train")
    return expr, meta, audit


def resumen_cohortes(meta, cfg: Config) -> pd.DataFrame:
    g = meta.groupby(cfg.cohort_col)
    res = pd.DataFrame({
        "n": g.size(),
        "n_labels": g[cfg.label_col].nunique(dropna=True),
        "labels": g[cfg.label_col].apply(
            lambda x: ",".join(sorted(x.dropna().astype(str).unique()))),
    })
    if cfg.platform_col in meta.columns:
        res["platform"] = g[cfg.platform_col].agg(
            lambda x: x.astype(str).mode().iat[0] if len(x) else "unknown")
    return res.sort_values("n", ascending=False)


def filtrar_cohortes(expr, meta, cfg: Config):
    """Aplica el endpoint y conserva únicamente la partición preespecificada.

    El mínimo de dos etiquetas se evalúa después de fijar el endpoint. Todas
    las cohortes declaradas son obligatorias: si una no cumple el contrato se
    detiene el pipeline en vez de sustituirla o volver a sortear el split.
    """
    print("\n[3] Filtrando cohortes...")
    train_fixed = tuple(map(str, cfg.train_cohorts_fixed))
    test_fixed = tuple(map(str, cfg.test_cohorts_fixed))
    if set(train_fixed) & set(test_fixed):
        raise ValueError("La partición fija contiene cohortes solapadas.")
    if cfg.ref_batch not in train_fixed:
        raise ValueError("La cohorte de referencia debe pertenecer a train.")

    active = tuple(map(str, cfg.active_labels))
    expected_labels = {"LumA", "LumB", "HER2-enriched", "Basal-like"}
    if len(active) != 4 or set(active) != expected_labels:
        raise ValueError(
            "active_labels debe ser exactamente LumA, LumB, HER2-enriched "
            "y Basal-like."
        )

    observed_labels = set(meta[cfg.label_col].astype(str))
    if observed_labels != expected_labels:
        raise ValueError(
            "La entrada preparada no coincide con intrinsic4. "
            f"Faltan={sorted(expected_labels-observed_labels)}; "
            f"sobran={sorted(observed_labels-expected_labels)}"
        )

    # Endpoint antes del filtro por número de etiquetas.
    label_mask = meta[cfg.label_col].astype(str).isin(active)
    n_out = int((~label_mask).sum())
    if n_out:
        print(f"    {n_out} muestras fuera del endpoint {list(active)} -> excluidas")
    keep_label = meta.index[label_mask]
    expr, meta = expr.loc[:, keep_label], meta.loc[keep_label].copy()

    # No se incorporan cohortes adicionales aunque también sean elegibles.
    allowed = set(train_fixed) | set(test_fixed)
    extra = sorted(set(meta[cfg.cohort_col].astype(str)) - allowed)
    if extra:
        raise ValueError(
            "Hay cohortes fuera de la partición preespecificada: "
            + ", ".join(extra)
        )
    keep_fixed = meta.index[meta[cfg.cohort_col].astype(str).isin(allowed)]
    expr, meta = expr.loc[:, keep_fixed], meta.loc[keep_fixed].copy()

    res = resumen_cohortes(meta, cfg)
    print(res.to_string())

    fuera_lab = res.index[res["n_labels"] < cfg.min_labels_per_cohort].tolist()
    fuera_n = res.index[res["n"] < cfg.min_samples_per_cohort].tolist()
    fuera = sorted(set(fuera_lab) | set(fuera_n))

    for c in fuera_lab:
        print(f"    ✗ {c}: solo {int(res.loc[c,'n_labels'])} etiqueta "
              f"({res.loc[c,'labels']})")
    for c in fuera_n:
        if c not in fuera_lab:
            print(f"    ✗ {c}: n={int(res.loc[c,'n'])} < {cfg.min_samples_per_cohort}")

    missing = sorted(allowed - set(res.index.astype(str)))
    if missing:
        raise ValueError(
            "Faltan cohortes de la partición preespecificada: " + ", ".join(missing)
        )
    invalid_fixed = sorted(allowed & set(fuera))
    if invalid_fixed:
        raise ValueError(
            "Estas cohortes fijas no cumplen min_labels/min_samples: "
            + ", ".join(invalid_fixed)
        )

    keep = meta.index[~meta[cfg.cohort_col].isin(fuera)]
    expr, meta = expr.loc[:, keep], meta.loc[keep]
    print(f"    -> {meta[cfg.cohort_col].nunique()} cohortes, {len(meta)} muestras")
    return expr, meta, res


# =============================================================================
# BLOQUE 2 — SUELO DE PSEUDOCONTEO
# =============================================================================
#
# GSE96058 se distribuye como log2(FPKM + 0.1).  El suelo es log2(0.1) =
# -3.321928, y todo gen no detectado cae ahí.  Dos consecuencias:
#
#   (a) Genes 100 % en el suelo tienen varianza exactamente 0.  Un solo gen
#       así hace que gamma_bar / t2 / a_prior / b_prior de ESE lote sean NaN,
#       y como esos hiperparámetros se promedian a través de genes, el NaN se
#       propaga a TODAS las columnas del lote.
#
#   (b) El pseudoconteo 0.1 amplifica el ruido del extremo bajo: pasar de
#       FPKM 0 a FPKM 0.1 son 1.000 unidades log2, frente a 0.138 con
#       pseudoconteo 1.  En genes parcialmente censurados eso hace que ~81 %
#       de la varianza por gen sea artefacto de censura (vs ~56 % con +1).
#
# Y esto importa especialmente aquí: con ref_batch, var_pooled y grand_mean
# salen EXCLUSIVAMENTE de la referencia.  Si la escala de la referencia está
# dominada por la censura, esa distorsión se propaga a todas las cohortes
# proyectadas.
#
# Solución en dos pasos:
#   1. Reparametrizar:  x_nuevo = log2(2^x - c + c_destino)   [exacto, ~1e-15]
#      Deja GSE96058 en log2(FPKM+1), la misma convención que TCGA.
#   2. Filtro de detección: descartar los genes que estén en el suelo en más
#      del 50 % de las muestras de la REFERENCIA.  Un gen que la referencia
#      no detecta no puede anclar la escala de nadie.
# =============================================================================

# Pseudoconteos "de catálogo": los que realmente usan los pipelines publicados.
PSEUDOCONTEOS = (0.01, 0.05, 0.1, 0.125, 0.25, 0.5, 1.0)
TOL_PSEUDO = 1e-3          # tolerancia RELATIVA — deliberadamente estricta


def _detectar_pseudoconteo(vmin: float, frac_suelo: float):
    """
    Un suelo de pseudoconteo deja una firma inconfundible:
      (1) mínimo negativo,
      (2) masa puntual apreciable en ese mínimo,
      (3) 2^min coincide con un pseudoconteo de catálogo con precisión ~1e-3.

    Cualquier otra cosa (el suelo de RMA en microarrays, un recorte arbitrario)
    NO se toca: se reporta y se deja como está. Reparametrizar un mínimo que
    no es un pseudoconteo destroza los datos silenciosamente.
    """
    if not (vmin < 0 and frac_suelo > 0.01):
        return None
    c = 2.0 ** vmin
    for cand in PSEUDOCONTEOS:
        if abs(c - cand) <= TOL_PSEUDO * cand:
            return cand
    return None


def diagnosticar_suelo(expr, meta, cfg: Config) -> pd.DataFrame:
    """Diagnostica el suelo de cada cohorte e identifica pseudoconteos reales."""
    print("\n[2b] Diagnóstico del suelo de pseudoconteo...")
    filas = []
    for coh, grp in meta.groupby(cfg.cohort_col):
        X = expr.loc[:, grp.index].values
        vmin = float(np.nanmin(X))
        frac = float((X <= vmin + 1e-9).mean())
        pseudo = _detectar_pseudoconteo(vmin, frac)
        filas.append({
            "cohorte": coh, "n": len(grp), "min": vmin, "max": float(np.nanmax(X)),
            "2^min": float(2.0 ** vmin), "frac_en_suelo": frac,
            "genes_varianza_0": int((np.nanvar(X, axis=1) <= cfg.var_eps).sum()),
            "pseudoconteo": pseudo if pseudo is not None else np.nan,
            "corregible": pseudo is not None,
        })
    df = pd.DataFrame(filas).sort_values("n", ascending=False).set_index("cohorte")
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))

    ok = df.index[df["corregible"]].tolist()
    sospechosas = df.index[(~df["corregible"]) & (df["min"] < 0)].tolist()
    if ok:
        print(f"    ✓ pseudoconteo identificado en: {ok}")
    if sospechosas:
        print(f"    ⚠️  mínimo negativo SIN pseudoconteo reconocible: {sospechosas}")
        print("        (no se tocan; revísalas a mano o usa pseudo_manual)")
    return df


def corregir_suelo(expr, meta, cfg: Config, diag: pd.DataFrame | None = None,
                   pseudo_manual: dict | None = None):
    """
    Reparametriza a log2(x + pseudo_destino) las cohortes cuyo pseudoconteo
    se ha identificado sin ambigüedad:

        x = log2(v + c)  ->  v = 2^x - c  ->  x' = log2(v + c')

    Exacto hasta precisión de máquina (~1e-15). El suelo pasa de log2(c) a
    log2(c'), y con c'=1 queda en 0, la misma convención que TCGA.

    `pseudo_manual={'GSE96058': 0.1}` fuerza el valor si la detección falla.
    """
    if not cfg.fix_pseudocount:
        print("\n[2c] Corrección del suelo DESACTIVADA (fix_pseudocount=False)")
        return expr, pd.DataFrame()

    print("\n[2c] Corrigiendo el suelo de pseudoconteo...")
    diag = diag if diag is not None else diagnosticar_suelo(expr, meta, cfg)
    pseudo_manual = pseudo_manual or {}
    expr = expr.copy()
    registro = []

    objetivo = {c: float(diag.loc[c, "pseudoconteo"])
                for c in diag.index[diag["corregible"]]}
    objetivo.update({k: float(v) for k, v in pseudo_manual.items()})
    cd = cfg.pseudo_destino

    for coh, c in objetivo.items():
        if abs(c - cd) < 1e-12:
            print(f"    {coh}: ya está en log2(x+{cd}) -> sin cambios")
            continue
        cols = meta.index[meta[cfg.cohort_col] == coh]
        X = expr.loc[:, cols].values
        V = np.clip(2.0 ** X - c, 0.0, None)          # FPKM/TPM recuperado
        expr.loc[:, cols] = np.log2(V + cd)
        registro.append({
            "cohorte": coh, "pseudo_detectado": c, "pseudo_nuevo": cd,
            "suelo_antes": float(np.log2(c)), "suelo_despues": float(np.log2(cd)),
            "salto_0_a_0.1_antes": float(np.log2(0.1 + c) - np.log2(c)),
            "salto_0_a_0.1_despues": float(np.log2(0.1 + cd) - np.log2(cd)),
            "manual": coh in pseudo_manual,
        })
        print(f"    {coh}: log2(x+{c}) -> log2(x+{cd}) | "
              f"suelo {np.log2(c):+.3f} -> {np.log2(cd):+.3f} | "
              f"salto 0→0.1 FPKM: {np.log2(0.1+c)-np.log2(c):.3f} -> "
              f"{np.log2(0.1+cd)-np.log2(cd):.3f} unidades log2")

    if not registro:
        print("    ninguna cohorte corregida")
    return expr, pd.DataFrame(registro)


def filtro_deteccion(expr, meta, cfg: Config):
    """
    Descarta genes que estén en el suelo en más de `max_frac_suelo_ref` de las
    muestras de la REFERENCIA. Con ref_batch, la referencia fija grand_mean y
    var_pooled para todo el mundo: un gen que ella no detecta no debe anclar
    la escala de las demás cohortes.
    """
    print("\n[2d] Filtro de detección sobre la referencia...")
    cols = meta.index[meta[cfg.cohort_col] == cfg.ref_batch]
    if len(cols) == 0:
        raise ValueError(f"La referencia '{cfg.ref_batch}' no está en los datos.")

    X = expr.loc[:, cols].values
    suelo = float(np.nanmin(X))
    frac = (X <= suelo + 1e-9).mean(axis=1)
    keep = expr.index[frac <= cfg.max_frac_suelo_ref]

    print(f"    suelo de {cfg.ref_batch} = {suelo:.4f}")
    print(f"    genes en suelo >{cfg.max_frac_suelo_ref:.0%} de la referencia: "
          f"{expr.shape[0] - len(keep)} -> descartados")
    print(f"    -> {len(keep)} genes")
    return expr.loc[keep], expr.index.difference(keep).tolist()


# =============================================================================
# BLOQUE 3 — PARTICIÓN 4/3 PREESPECIFICADA A NIVEL DE COHORTE
# =============================================================================

def split_cohortes(meta, cfg: Config) -> pd.DataFrame:
    """Valida y materializa la partición 4/3 declarada en ``Config``.

    No contiene aleatoriedad ni lógica de optimización del reparto. De este
    modo, volver a ejecutar el notebook nunca puede cambiar qué cohortes son
    desarrollo y cuáles son evaluación externa.
    """
    print("\n[4] Partición 4/3 preespecificada a nivel de cohorte...")
    res = resumen_cohortes(meta, cfg)
    train = list(map(str, cfg.train_cohorts_fixed))
    test = list(map(str, cfg.test_cohorts_fixed))
    observed = set(res.index.astype(str))
    declared = set(train) | set(test)
    if observed != declared:
        raise ValueError(
            "Las cohortes observadas no coinciden con la partición fija. "
            f"Faltan={sorted(declared-observed)}; sobran={sorted(observed-declared)}"
        )
    if not train or not test:
        raise ValueError("Train y test deben contener al menos una cohorte.")
    if cfg.ref_batch not in train:
        raise ValueError(f"La referencia '{cfg.ref_batch}' no está en train.")
    cfg.train_cohorts, cfg.test_cohorts = train, test

    print(f"    TRAIN ({len(train)} cohortes, {int(res.loc[train,'n'].sum())} muestras): {train}")
    print(f"    TEST  ({len(test)} cohortes, {int(res.loc[test,'n'].sum())} muestras): {test}")
    print(f"    referencia: {cfg.ref_batch}")

    lab_tr = set(meta.loc[meta[cfg.cohort_col].isin(train), cfg.label_col].dropna())
    lab_te = set(meta.loc[meta[cfg.cohort_col].isin(test), cfg.label_col].dropna())
    required = set(map(str, cfg.active_labels))
    if cfg.require_all_active_labels_in_each_split:
        missing_train = sorted(required - lab_tr)
        missing_test = sorted(required - lab_te)
        if missing_train or missing_test:
            raise ValueError(
                "El endpoint no está completo en ambos bloques: "
                f"faltan_train={missing_train}; faltan_test={missing_test}"
            )
    if (lab_tr | lab_te) - required:
        raise ValueError("Se detectaron etiquetas fuera del endpoint activo.")
    print(f"    ✓ clases activas presentes en train y test: {list(cfg.active_labels)}")

    if "platform" in res.columns:
        print(f"    plataformas train: {res.loc[train,'platform'].value_counts().to_dict()}")
        print(f"    plataformas test : {res.loc[test,'platform'].value_counts().to_dict()}")

    tabla = res.copy()
    tabla["split"] = np.where(tabla.index.isin(train), "train",
                              np.where(tabla.index.isin(test), "test", "excluida"))
    tabla["es_referencia"] = tabla.index == cfg.ref_batch
    return tabla


# =============================================================================
# BLOQUE 4 — TRANSFORMADOR M-ComBat CON LOTE DE REFERENCIA
# =============================================================================

class ReferenceMComBat:
    """
    Transformador estilo scikit-learn.

        cb   = ReferenceMComBat(ref_batch="GSE96058")
        Xtr  = cb.fit_transform(expr_train, batch_train)
        Xte  = cb.transform(expr_test, batch_test)
        cb.save("artefacto/")            # congelado, reutilizable

    `fit`       congela la matriz cruda de la referencia y la lista de genes.
    `transform` proyecta cada cohorte ejecutando ComBat sobre
                [referencia | cohorte] y devolviendo solo las columnas de la
                cohorte. El train NUNCA se recalcula.
    """

    def __init__(self, ref_batch: str, var_eps: float = 1e-8, verbose: bool = True):
        self.ref_batch = ref_batch
        self.var_eps = var_eps
        self.verbose = verbose
        self.ref_raw_: pd.DataFrame | None = None
        self.genes_: pd.Index | None = None
        self.genes_descartados_: list = []
        self.contract_: dict = {}

    def _log(self, m):
        if self.verbose:
            print(m)

    def _genes_validos(self, X: pd.DataFrame, batch: pd.Series) -> pd.Index:
        """Genes con varianza > eps EN TODOS los lotes (ver nota del Bloque 2)."""
        ok = pd.Series(True, index=X.index)
        for b in pd.unique(batch):
            cols = batch.index[batch == b]
            ok &= (X.loc[:, cols].var(axis=1) > self.var_eps)
        ok &= np.isfinite(X.values).all(axis=1)
        return X.index[ok]

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, batch: pd.Series) -> "ReferenceMComBat":
        batch = pd.Series(batch, index=X.columns).astype(str)
        if self.ref_batch not in set(batch):
            raise ValueError(f"'{self.ref_batch}' no aparece en batch.")
        genes = self._genes_validos(X, batch)
        if len(genes) < X.shape[0]:
            self.genes_descartados_ = X.index.difference(genes).tolist()
            self._log(f"    {X.shape[0]-len(genes)} genes con varianza nula "
                      f"en algún lote -> descartados")
        self.genes_ = genes
        ref_cols = batch.index[batch == self.ref_batch]
        self.ref_raw_ = X.loc[genes, ref_cols].copy()
        self._log(f"    referencia congelada: {self.ref_batch} "
                  f"({len(ref_cols)} muestras, {len(genes)} genes)")
        return self

    def fit_transform(self, X: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
        self.fit(X, batch)
        batch = pd.Series(batch, index=X.columns).astype(str)
        Xg = X.loc[self.genes_]
        self._log(f"    corrigiendo train ({batch.nunique()} lotes)...")
        out = pycombat_norm(counts=Xg, batch=batch.values,
                            covar_mod=None,        # sin label: evita la fuga
                            ref_batch=self.ref_batch)
        out = pd.DataFrame(np.asarray(out), index=Xg.index, columns=Xg.columns)
        ref_cols = batch.index[batch == self.ref_batch]
        d = np.abs(out[ref_cols].values - Xg[ref_cols].values).max()
        assert d < 1e-9, f"La referencia fue modificada (max|dif|={d:.2e})"
        self._log(f"    ✓ referencia intacta (max|dif| = {d:.1e})")
        return out

    # ------------------------------------------------------------ transform
    def transform_cohorte(self, X: pd.DataFrame) -> pd.DataFrame:
        """Proyecta UNA cohorte nueva sobre la escala de la referencia."""
        if self.ref_raw_ is None:
            raise RuntimeError("Llama a fit() primero.")
        if X.shape[1] < 2:
            raise ValueError("inmoose no admite lotes de 1 muestra (delta_hat "
                             "indefinido). El despliegue de una sola muestra "
                             "queda fuera del protocolo preespecificado.")
        faltan = self.genes_.difference(X.index)
        if len(faltan):
            raise ValueError(f"Faltan {len(faltan)} genes del artefacto "
                             f"(p.ej. {list(faltan[:5])}).")
        Xg = X.loc[self.genes_]

        malos = self.genes_[Xg.var(axis=1) <= self.var_eps]
        if len(malos):
            self._log(f"      {len(malos)} genes constantes en la cohorte -> "
                      f"ruido mínimo para no perder el artefacto")
            rng = np.random.default_rng(0)
            Xg = Xg.copy()
            Xg.loc[malos] += rng.normal(0, 1e-6, size=(len(malos), Xg.shape[1]))

        joint = pd.concat([self.ref_raw_, Xg], axis=1)
        b = np.array([self.ref_batch] * self.ref_raw_.shape[1]
                     + ["__NEW__"] * Xg.shape[1])
        out = pycombat_norm(counts=joint, batch=b, covar_mod=None,
                            ref_batch=self.ref_batch)
        out = pd.DataFrame(np.asarray(out), index=joint.index, columns=joint.columns)
        return out.loc[:, Xg.columns]

    def transform(self, X: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
        """Proyecta varias cohortes, una a una (son independientes entre sí)."""
        batch = pd.Series(batch, index=X.columns).astype(str)
        piezas = []
        for b in pd.unique(batch):
            cols = batch.index[batch == b]
            self._log(f"    proyectando {b} ({len(cols)} muestras)")
            piezas.append(self.transform_cohorte(X.loc[:, cols]))
        return pd.concat(piezas, axis=1).loc[:, X.columns]

    # --------------------------------------------------------- persistencia
    def save(self, path: str | Path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "referencia_cruda.npz",
                            X=self.ref_raw_.values.astype(np.float64))
        with (path / "artefacto.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "ref_batch": self.ref_batch, "var_eps": self.var_eps,
                "n_genes": len(self.genes_), "genes": list(self.genes_),
                "ref_samples": list(self.ref_raw_.columns),
                "genes_descartados": self.genes_descartados_,
                "input_contract": self.contract_,
            }, handle, indent=2, ensure_ascii=False)
        self._log(f"    💾 artefacto congelado -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceMComBat":
        path = Path(path)
        with (path / "artefacto.json").open(encoding="utf-8") as handle:
            m = json.load(handle)
        obj = cls(ref_batch=m["ref_batch"], var_eps=m["var_eps"])
        obj.genes_ = pd.Index(m["genes"])
        obj.genes_descartados_ = m.get("genes_descartados", [])
        obj.contract_ = dict(m.get("input_contract", {}))
        obj.ref_raw_ = pd.DataFrame(np.load(path / "referencia_cruda.npz")["X"],
                                    index=obj.genes_, columns=m["ref_samples"])
        return obj


# =============================================================================
# BLOQUE 5 — VERIFICACIÓN DE INVARIANCIA
# =============================================================================

def verificar_invariancia(expr, meta, cfg: Config) -> pd.DataFrame:
    """
    Comprueba lo que hace defendible el esquema:
      A) el train corregido es idéntico se haya visto o no el test
      B) una cohorte de test se proyecta igual sola que junto a las demás
      C) el artefacto recargado de disco reproduce la proyección
      D) el conjunto de genes congelado no cambia al ver el test

    OJO con (A): los hiperparámetros del empirical Bayes (gamma_bar, t2,
    a_prior, b_prior) se estiman promediando A TRAVÉS DE GENES. Por eso EL
    CONJUNTO DE GENES FORMA PARTE DEL ARTEFACTO: si cambia, cambia la
    corrección de todos los genes. La comparación (A) sólo tiene sentido
    fijando la misma lista de genes en ambos runs — que es exactamente lo que
    hace el pipeline, porque el filtro se decide únicamente con train.
    """
    print("\n[6] Verificando ausencia de fuga...")
    import tempfile
    b = meta[cfg.cohort_col].astype(str)
    ctr = meta.index[b.isin(cfg.train_cohorts)]
    cte = meta.index[b.isin(cfg.test_cohorts)]
    filas = []

    cb = ReferenceMComBat(cfg.ref_batch, cfg.var_eps, verbose=False)
    tr_solo = cb.fit_transform(expr.loc[:, ctr], b[ctr])
    genes = cb.genes_                       # el artefacto: fijado sólo con train

    # (A) mismo run con el test presente, PERO con la lista de genes congelada.
    # Se aplica la misma salvaguarda determinista para genes constantes que en
    # ``transform_cohorte``; esto evita que una cohorte externa degenerada haga
    # fallar una comprobación que no forma parte del ajuste.
    cols = list(ctr) + list(cte)
    Xj = expr.loc[genes, cols].copy()
    rng = np.random.default_rng(0)
    for coh_test in cfg.test_cohorts:
        cc = b.index[b == coh_test]
        const = Xj.index[Xj.loc[:, cc].var(axis=1) <= cfg.var_eps]
        if len(const):
            Xj.loc[const, cc] += rng.normal(
                0, 1e-6, size=(len(const), len(cc))
            )
    junto = pycombat_norm(counts=Xj, batch=b[cols].values,
                          covar_mod=None, ref_batch=cfg.ref_batch)
    junto = pd.DataFrame(np.asarray(junto), index=genes, columns=cols)
    filas.append({"prueba": "A) train solo vs train+test en el mismo run",
                  "max_dif": float(np.abs(tr_solo.loc[genes, ctr].values
                                          - junto.loc[genes, ctr].values).max())})

    coh = cfg.test_cohorts[0]
    c = meta.index[b == coh]
    sola = cb.transform_cohorte(expr.loc[:, c])
    todas = cb.transform(expr.loc[:, cte], b[cte])
    filas.append({"prueba": f"B) {coh} sola vs con las demás cohortes de test",
                  "max_dif": float(np.abs(sola.values
                                          - todas.loc[sola.index, c].values).max())})

    # (D) proyectar test no puede mutar el universo aprendido en desarrollo.
    filas.append({"prueba": "D) genes del artefacto alterados tras proyectar test",
                  "max_dif": float(len(genes.symmetric_difference(cb.genes_)))})

    with tempfile.TemporaryDirectory() as td:
        cb.save(Path(td) / "art")
        cb3 = ReferenceMComBat.load(Path(td) / "art")
        filas.append({"prueba": f"C) {coh} con artefacto recargado de disco",
                      "max_dif": float(np.abs(sola.values
                                              - cb3.transform_cohorte(
                                                  expr.loc[:, c]).values).max())})

    df = pd.DataFrame(filas).sort_values("prueba").reset_index(drop=True)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    if df["max_dif"].max() > 1e-9:
        print("    ⚠️  alguna prueba supera 1e-9: revisa covar_mod y el filtro de genes")
    else:
        print("    ✓ todas por debajo de la precisión numérica")
    return df


# =============================================================================
# AJUSTE/PROYECCIÓN SOBRE LA SALIDA PREPARADA
# =============================================================================

def fit_project_prepared(expr: pd.DataFrame, meta: pd.DataFrame,
                         cfg: Config | None = None) -> dict:
    """Ajusta M-ComBat sobre train y proyecta test sin recomputar preparación."""
    cfg = cfg or Config()
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  M-ComBat de referencia — ajuste en train, proyección en test")
    print("=" * 78)
    expr, meta, _ = filtrar_cohortes(expr, meta, cfg)

    tabla = split_cohortes(meta, cfg)
    tabla.to_csv(out / "split_cohortes.csv")

    b = meta[cfg.cohort_col].astype(str)
    ctr = meta.index[b.isin(cfg.train_cohorts)]
    cte = meta.index[b.isin(cfg.test_cohorts)]

    print("\n[5] M-ComBat...")
    cb = ReferenceMComBat(ref_batch=cfg.ref_batch, var_eps=cfg.var_eps)
    Xtr = cb.fit_transform(expr.loc[:, ctr], b[ctr])
    Xte = cb.transform(expr.loc[:, cte], b[cte])
    cb.contract_ = {
        "method": "reference_mcombat",
        "implementation": "inmoose.pycombat.pycombat_norm",
        "covariates": None,
        "active_labels": list(map(str, cfg.active_labels)),
        "development_cohorts": list(map(str, cfg.train_cohorts)),
        "external_cohorts": list(map(str, cfg.test_cohorts)),
        "pseudocount_target": float(cfg.pseudo_destino),
        "reference_floor_filter_max_fraction": float(cfg.max_frac_suelo_ref),
        "new_cohort_min_samples": 2,
        "new_cohort_requirements": (
            "same ordered genes and the same documented continuous log-scale "
            "preparation used before M-ComBat"
        ),
    }
    cb.save(out / "artefacto_mcombat")
    with (out / "preprocessing_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(cb.contract_, handle, indent=2, ensure_ascii=False)

    ver = verificar_invariancia(expr, meta, cfg)
    ver.to_csv(out / "verificacion_invariancia.csv", index=False)

    print("\n[7] Guardando...")
    Xtr.to_csv(out / "expr_train_mcombat.csv")
    Xte.to_csv(out / "expr_test_mcombat.csv")
    meta.loc[ctr].to_csv(out / "meta_train.csv", index=False)
    meta.loc[cte].to_csv(out / "meta_test.csv", index=False)
    print(f"    train: {Xtr.shape[0]} genes x {Xtr.shape[1]} muestras")
    print(f"    test : {Xte.shape[0]} genes x {Xte.shape[1]} muestras")
    print(f"    📁 {out.resolve()}")

    return dict(expr_train=Xtr, expr_test=Xte, meta_train=meta.loc[ctr],
                meta_test=meta.loc[cte], mcombat=cb, split=tabla,
                verificacion=ver, cfg=cfg)


def run(cfg: Config | None = None) -> dict:
    """Carga ``output_prepared`` y ejecuta únicamente M-ComBat."""
    cfg = cfg or Config()
    expr, meta = cargar(cfg)
    return fit_project_prepared(expr, meta, cfg)


if __name__ == "__main__":
    run()
