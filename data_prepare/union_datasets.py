#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
union_datasets.py — construcción de la matriz combinada multi-plataforma
===============================================================================

Produce los dos ficheros auditados (`expr_raw_combined.csv` y
`metadata_combined.csv`) que consume el notebook 02 de preparación:

  1. Diagnóstico EXPLÍCITO de escala en lugar de heurística silenciosa.
  2. Normalización de símbolos HGNC (SEPT→SEPTIN, MARCH→MARCHF, …) y
     detección de corrupción por Excel (SEPT9 → '9-Sep').
  3. Colapso de símbolos duplicados con criterio elegible (mean / max_iqr).
  4. SIN imputación por mediana global sobre desarrollo y externo a la vez.
  5. Informe de cobertura de genes por cohorte, y comparación entre la
     intersección total y la intersección sólo-train.
  6. Contrato semántico estricto del endpoint intrínseco: LumA, LumB,
     HER2-enriched y Basal-like. No convierte HER2 clínico en
     HER2-enriched ni TNBC clínico en Basal-like.
  7. Detección de muestras duplicadas ENTRE cohortes (misma paciente en dos
     series ⇒ fuga a través del split por cohorte).
  8. Namespacing de identificadores y comprobación de colisiones.

Requisitos: pip install pandas numpy
===============================================================================
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class ConfigUnion:
    # --- fuentes ---
    micro_matrix: str = "./Data/expr_genes_microarrays.csv"
    micro_meta: str = "./Data/microarrays_meta.csv"
    micro_sample_col: str = "sample"
    micro_batch_col: str = "batch"

    tcga_matrix: str = "./Data/counts_symbol_TCGA.csv"
    tcga_meta: str = "./Data/TCGA_meta.csv"
    tcga_sample_col: str = "sample"

    gse_matrix: str = "./Data/counts_symbol_GSE96058.csv"
    gse_meta: str = "./Data/GSE96058_meta.csv"
    gse_sample_col: str = "sample"

    # --- genes ---
    gene_mode: str = "interseccion"      # "interseccion" | "union"
    collapse: str = "max_iqr"            # "mean" | "max_iqr" | "max_mean"
    normalizar_simbolos: bool = True

    # --- muestras ---
    namespace_samples: bool = True       # prefija el ID con la cohorte
    min_labels_per_cohort: int = 2       # se aplica en la primera unión
    detectar_duplicados: bool = True
    umbral_duplicado: float = 0.98
    n_genes_duplicados: int = 2000

    out_dir: str = "./output_union"
    platform_map: dict = field(default_factory=lambda: {
        "GSE1456": "GPL96", "GSE25066": "GPL96",
        "GSE162228": "GPL570",  "GSE20711": "GPL570",
        "GSE21653": "GPL570",
        "GSE96058": "RNAseq", "TCGA-BRCA": "RNAseq",
    })

    # La salida de los dos notebooks de descarga debe contener exactamente
    # estas siete cohortes. Una cohorte ausente o adicional detiene el flujo.
    expected_cohorts: tuple = (
        "GSE1456", "GSE162228", "GSE20711", "GSE21653", "GSE25066",
        "GSE96058", "TCGA-BRCA",
    )


# =============================================================================
# SEMÁNTICA DE LAS ETIQUETAS  ← leer esto antes de nada
# =============================================================================
#
# Los notebooks 00a/00b ya excluyen tanto tejido sano como Normal-like. Esta
# tabla se conserva solo para detectar una salida antigua o contaminada y
# detener el flujo con un mensaje informativo.
# =============================================================================

SEMANTICA_NORMAL = {
    "GSE96058": "normal_like", "GSE25066": "normal_like",
    "GSE21653": "normal_like", "GSE20711": "normal_like",
    "TCGA-BRCA": "tejido_sano",
}

INTRINSIC4_LABELS = ("LumA", "LumB", "HER2-enriched", "Basal-like")

CANON_LABEL = {
    "luma": "LumA", "luminala": "LumA", "luminal a": "LumA", "lum-a": "LumA",
    "lumb": "LumB", "luminalb": "LumB", "luminal b": "LumB", "lum-b": "LumB",
    "her2-enriched": "HER2-enriched", "her2 enriched": "HER2-enriched",
    "her2e": "HER2-enriched",
    "basal-like": "Basal-like", "basal like": "Basal-like",
    "normal": "Normal", "normal-like": "Normal", "healthy": "Normal",
    "normal breast tissue": "Normal", "control": "Normal",
}

# Renombramientos sistemáticos del HGNC que rompen la intersección de símbolos
_PREFIJOS_HGNC = {"SEPT": "SEPTIN", "MARCH": "MARCHF", "DEC": "DELEC"}
_RE_FECHA_EXCEL = re.compile(r"^(\d{1,2})[-/](Sep|Mar|Dec|Sept|Apr|Oct|Nov|Feb|Jan)$", re.I)
_MES2GEN = {"sep": "SEPTIN", "sept": "SEPTIN", "mar": "MARCHF", "dec": "DELEC"}


# =============================================================================
# UTILIDADES
# =============================================================================

def _limpiar_columnas(df):
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def diagnosticar_escala(expr: pd.DataFrame, nombre: str) -> dict:
    """
    Reporta la escala en vez de adivinarla. Sólo aplica log2 si es
    inequívocamente lineal (mínimo >= 0 y p99 muy grande).

    Escalas esperadas en este proyecto:
      microarrays RMA        log2, mín ~0.2, máx ~14      -> sin transformar
      TCGA Xena HiSeqV2      log2(norm_count+1), mín 0    -> sin transformar
      GSE96058               log2(FPKM+0.1), mín -3.322   -> sin transformar
                                                             (el suelo se
                                                              arregla en el nb 01)
    """
    v = expr.values
    fin = v[np.isfinite(v)]
    d = {"dataset": nombre, "min": float(fin.min()), "max": float(fin.max()),
         "p99": float(np.percentile(fin, 99)), "mediana": float(np.median(fin)),
         "frac_negativos": float((fin < 0).mean()),
         "frac_en_minimo": float((fin <= fin.min() + 1e-9).mean())}
    d["escala"] = "lineal" if (d["min"] >= 0 and d["p99"] > 50) else "log2"
    d["log2_aplicado"] = d["escala"] == "lineal"
    print(f"    [{nombre}] escala={d['escala']:6s} min={d['min']:8.3f} "
          f"p99={d['p99']:9.2f} max={d['max']:8.2f} "
          f"neg={d['frac_negativos']:.1%} masa_en_min={d['frac_en_minimo']:.1%}")
    if d["log2_aplicado"]:
        print(f"      -> aplicando log2(x+1)")
    return d


def normalizar_simbolos(idx: pd.Index, nombre: str) -> pd.Index:
    """
    Corrige (a) los renombramientos sistemáticos del HGNC y (b) la corrupción
    por Excel, que convierte SEPT9 en '9-Sep' y MARCH1 en '1-Mar'. Sin esto la
    intersección de símbolos pierde decenas de genes en silencio.
    """
    s = pd.Index([str(g).strip() for g in idx])
    n_fecha = 0
    out = []
    for g in s:
        m = _RE_FECHA_EXCEL.match(g)
        if m:
            n_fecha += 1
            out.append(f"{_MES2GEN[m.group(2).lower()]}{m.group(1)}")
            continue
        for viejo, nuevo in _PREFIJOS_HGNC.items():
            mm = re.match(rf"^{viejo}(\d+)$", g)
            if mm:
                g = f"{nuevo}{mm.group(1)}"
                break
        out.append(g)
    if n_fecha:
        print(f"    ⚠️ [{nombre}] {n_fecha} símbolos corrompidos por Excel "
              f"(tipo '9-Sep') -> reconstruidos. Revisa el origen del CSV.")
    return pd.Index(out)


def colapsar_duplicados(expr: pd.DataFrame, metodo: str, nombre: str) -> pd.DataFrame:
    """
    Colapsa símbolos duplicados.

      mean      media aritmética en escala log2 (= media geométrica lineal)
      max_mean  se queda con la fila de mayor media
      max_iqr   se queda con la fila de mayor rango intercuartílico
                (criterio de collapseRows / WGCNA: conserva la sonda con más
                 señal variable, no la más brillante)
    """
    dup = expr.index.duplicated(keep=False)
    if not dup.any():
        return expr
    n = int(expr.index[dup].nunique())
    print(f"    [{nombre}] {n} símbolos duplicados -> colapso por '{metodo}'")
    if metodo == "mean":
        return expr.groupby(expr.index).mean()
    crit = (expr.mean(axis=1) if metodo == "max_mean"
            else expr.quantile(.75, axis=1) - expr.quantile(.25, axis=1))
    orden = crit.groupby(expr.index).rank(ascending=False, method="first")
    return expr.loc[(orden == 1).values]


def alinear(expr, meta, sample_col, nombre):
    """Exige correspondencia exacta entre expresión y metadata.

    Los notebooks de descarga ya generan tablas filtradas y alineadas. Una
    diferencia aquí indica que se han mezclado salidas de versiones distintas
    o que una descarga quedó incompleta; continuar con la intersección
    silenciosa cambiaría el experimento.
    """
    meta = _limpiar_columnas(meta)
    if sample_col not in meta.columns:
        raise ValueError(f"[{nombre}] falta '{sample_col}'. Hay: {list(meta.columns)[:15]}")
    expr = expr.copy()
    expr.columns = pd.Index(expr.columns.astype(str).str.strip())
    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    if meta[sample_col].duplicated().any():
        duplicadas = meta.loc[
            meta[sample_col].duplicated(keep=False), sample_col
        ].astype(str).unique().tolist()
        raise ValueError(
            f"[{nombre}] muestras duplicadas en metadata: {duplicadas[:10]}"
        )
    meta = meta.set_index(sample_col, drop=False)
    solo_expr = expr.columns.difference(meta.index)
    solo_meta = meta.index.difference(expr.columns)
    if len(solo_expr) or len(solo_meta):
        raise ValueError(
            f"[{nombre}] expresión y metadata no coinciden exactamente. "
            f"Solo expresión={list(solo_expr[:10])}; "
            f"solo metadata={list(solo_meta[:10])}. "
            "Vuelve a ejecutar el notebook de descarga correspondiente."
        )
    return expr.loc[:, meta.index], meta.copy()


def canonicalizar_label(s: pd.Series) -> pd.Series:
    """Unifica solo variantes inequívocas de las etiquetas intrínsecas.

    De forma deliberada, ``HER2``/``ERBB2`` y ``TNBC`` no se convierten aquí:
    sin la procedencia de la llamada no son equivalentes a HER2-enriched y
    Basal-like. Los notebooks de descarga realizan ese mapeo donde sí conocen
    el sistema de etiquetas y guardan su trazabilidad.
    """
    raw = s.astype(str).str.strip()
    out = raw.str.lower().map(CANON_LABEL)
    desconocidas = sorted(set(raw[out.isna() & raw.notna()
                                  & ~raw.isin(["nan", "None", ""])]))
    if desconocidas:
        print(f"    ⚠️ etiquetas no reconocidas: {desconocidas[:12]}")
    return out


# =============================================================================
# CARGA POR FUENTE
# =============================================================================

def _cargar_generico(matrix_path, meta_path, sample_col, nombre, cfg, dataset=None):
    print(f"\n  ▸ {nombre}")
    expr = pd.read_csv(matrix_path, index_col=0)
    meta = pd.read_csv(meta_path)

    expr.index = expr.index.astype(str).str.strip()
    if cfg.normalizar_simbolos:
        expr.index = normalizar_simbolos(expr.index, nombre)
    expr = colapsar_duplicados(expr, cfg.collapse, nombre)
    print(f"    {expr.shape[0]} genes x {expr.shape[1]} muestras")

    d = diagnosticar_escala(expr, nombre)
    if d["log2_aplicado"]:
        expr = np.log2(expr.clip(lower=0) + 1)

    expr, meta = alinear(expr, meta, sample_col, nombre)
    if dataset is not None:
        meta["dataset"] = dataset
    return expr, meta, d


def cargar_microarrays(cfg: ConfigUnion):
    expr, meta, d = _cargar_generico(cfg.micro_matrix, cfg.micro_meta,
                                     cfg.micro_sample_col, "Microarrays", cfg)
    if cfg.micro_batch_col not in meta.columns:
        raise ValueError(f"[Microarrays] falta '{cfg.micro_batch_col}'")

    # Si los IDs de muestra vienen prefijados ``GSE:GSM``, el prefijo es la
    # fuente auditada para corregir cualquier discrepancia de ``batch``.
    pref = meta[cfg.micro_sample_col].astype(str).str.extract(r"^(GSE\d+):")[0]
    bat = meta[cfg.micro_batch_col].astype(str).str.strip()
    if pref.notna().any():
        discrepa = pref.notna() & (pref != bat)
        if discrepa.any():
            print(f"    ⚠️ {int(discrepa.sum())} muestras con 'batch' distinto del "
                  f"prefijo del ID: {sorted(set(zip(bat[discrepa], pref[discrepa])))[:6]}")
            print(f"       -> se usa el prefijo del ID (el 'batch' de los "
                  f"notebooks de descarga tiene erratas conocidas)")
        bat = pref.fillna(bat)
    meta["dataset"] = bat
    meta["data_type"] = "microarray"
    print(f"    cohortes: {meta['dataset'].value_counts().to_dict()}")
    return expr, meta, d


def cargar_tcga(cfg: ConfigUnion):
    expr, meta, d = _cargar_generico(cfg.tcga_matrix, cfg.tcga_meta,
                                     cfg.tcga_sample_col, "TCGA-BRCA", cfg,
                                     dataset="TCGA-BRCA")
    if len(expr.index) and str(expr.index[0]).startswith("ENSG"):
        print("    IDs ENSEMBL -> quitando versión")
        expr.index = expr.index.str.split(".").str[0]
        expr = colapsar_duplicados(expr, cfg.collapse, "TCGA-BRCA")
    meta["data_type"] = "rnaseq"
    # código de muestra del barcode: 01-09 tumor, 11 tejido sano
    cod = meta[cfg.tcga_sample_col].astype(str).str.extract(r"-(\d{2})[A-Z]?(?:-|$)")[0]
    if "sample_code" in meta.columns:
        cod = cod.fillna(
            meta["sample_code"].astype("string").str.zfill(2)
        )
    meta["tcga_sample_code"] = cod
    if cod.notna().any():
        print(f"    códigos de barcode: {cod.value_counts().to_dict()}")
    if not cod.eq("01").all():
        raise ValueError(
            "TCGA-BRCA contiene muestras que no son tumores primarios "
            f"(códigos={cod.value_counts(dropna=False).to_dict()})."
        )
    return expr, meta, d


def cargar_gse96058(cfg: ConfigUnion):
    expr, meta, d = _cargar_generico(cfg.gse_matrix, cfg.gse_meta,
                                     cfg.gse_sample_col, "GSE96058", cfg,
                                     dataset="GSE96058")
    meta["data_type"] = "rnaseq"
    n_rep = int(meta[cfg.gse_sample_col].astype(str).str.endswith("repl").sum())
    print(f"    réplicas técnicas ('repl'): {n_rep}  (se auditan en el nb 01)")
    return expr, meta, d


# =============================================================================
# COBERTURA DE GENES
# =============================================================================

def informe_cobertura(dfs: dict, metas: dict) -> pd.DataFrame:
    """
    Qué genes aporta cada fuente y qué se pierde al intersectar. Útil para
    justificar `gene_mode` en el manuscrito en lugar de dar un número suelto.
    """
    print("\n  ▸ Cobertura de genes")
    sets = {k: set(v.index) for k, v in dfs.items()}
    inter = set.intersection(*sets.values())
    union = set.union(*sets.values())
    filas = [{"fuente": k, "n_genes": len(s),
              "en_interseccion": len(s & inter),
              "exclusivos": len(s - set.union(*[t for j, t in sets.items() if j != k]))}
             for k, s in sets.items()]
    df = pd.DataFrame(filas)
    print(df.to_string(index=False))
    print(f"    intersección: {len(inter)}   |   unión: {len(union)}")
    return df


# =============================================================================
# DUPLICADOS ENTRE COHORTES
# =============================================================================

def detectar_duplicados_cruzados(expr, meta, cfg: ConfigUnion) -> pd.DataFrame:
    """
    Busca la MISMA muestra presente en dos cohortes distintas.

    Es un riesgo real: varias series de GEO reanalizan pacientes de otras. Si
    una paciente cae en una cohorte de train y en otra de test, el split por
    cohorte no la separa y hay fuga directa que ninguna corrección de lote
    detecta.

    Compara sólo DENTRO de cada plataforma (entre plataformas la correlación
    está dominada por la tecnología y el umbral no significa lo mismo).
    """
    print("\n  ▸ Duplicados entre cohortes")
    pares = []
    for plat, grp in meta.groupby("platform"):
        if grp["dataset"].nunique() < 2:
            continue
        cols = grp.index
        X = expr.loc[:, cols]
        top = X.var(axis=1).nlargest(min(cfg.n_genes_duplicados, X.shape[0])).index
        Z = X.loc[top].values
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        C = (Z.T @ Z) / Z.shape[0]
        ds = grp["dataset"].values
        iu = np.triu_indices(len(cols), k=1)
        m = (C[iu] >= cfg.umbral_duplicado) & (ds[iu[0]] != ds[iu[1]])
        for a, b, r in zip(iu[0][m], iu[1][m], C[iu][m]):
            pares.append({"muestra_a": cols[a], "cohorte_a": ds[a],
                          "muestra_b": cols[b], "cohorte_b": ds[b],
                          "r": float(r), "platform": plat})
    df = pd.DataFrame(pares).sort_values("r", ascending=False) if pares else pd.DataFrame()
    if len(df):
        print(f"    ⚠️ {len(df)} pares con r >= {cfg.umbral_duplicado} entre cohortes distintas")
        print(df.head(10).to_string(index=False))
        print("    -> resuélvelos ANTES del split: rompen la separación por cohorte")
    else:
        print(f"    ✓ ningún par con r >= {cfg.umbral_duplicado}")
    return df


# =============================================================================
# AUDITORÍA SEMÁNTICA DE ETIQUETAS
# =============================================================================

def auditar_semantica(meta, cfg: ConfigUnion) -> pd.DataFrame:
    """
    Comprueba si 'Normal' significa lo mismo en todas las cohortes.
    Ver la nota SEMANTICA_NORMAL arriba: no lo significa.
    """
    print("\n  ▸ Auditoría semántica de la clase 'Normal'")
    filas = []
    for coh, grp in meta.groupby("dataset"):
        n = int((grp["label"] == "Normal").sum())
        if n:
            filas.append({"cohorte": coh, "n_Normal": n,
                          "semantica": SEMANTICA_NORMAL.get(coh, "SIN DECLARAR")})
    # Con columnas explícitas, el caso correcto (filas=[]) conserva el
    # esquema y no lanza KeyError('n_Normal') al ordenar.
    df = pd.DataFrame(
        filas, columns=["cohorte", "n_Normal", "semantica"]
    )
    if df.empty:
        print("    ✓ ninguna cohorte con clase Normal")
        return df
    df = df.sort_values("n_Normal", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
    tipos = set(df["semantica"]) - {"SIN DECLARAR"}
    if len(tipos) > 1:
        print("\n    ⚠️  'Normal' MEZCLA DOS COSAS DISTINTAS:")
        for t in sorted(tipos):
            sub = df[df.semantica == t]
            print(f"        {t:12s}: {sub['cohorte'].tolist()} "
                  f"(n={int(sub['n_Normal'].sum())})")
        print("    PAM50 Normal-like es un subtipo TUMORAL; tejido sano no lo es.")
        print("    El endpoint intrínseco no permite ninguna de las dos.")
    if (df["semantica"] == "SIN DECLARAR").any():
        print(f"    ⚠️ sin declarar: "
              f"{df.loc[df.semantica=='SIN DECLARAR','cohorte'].tolist()} "
              f"-> añádelas a SEMANTICA_NORMAL")
    raise ValueError(
        "La salida de descarga contiene la clase Normal. Vuelve a ejecutar "
        "00a/00b corregidos: intrinsic4 excluye tejido sano y tumores "
        "Normal-like."
    )


def etiquetas_endpoint(mode: str = "intrinsic4") -> tuple[str, ...]:
    """Devuelve el espacio de etiquetas preespecificado.

    ``intrinsic4`` es el único endpoint ejecutable. ``cuatro_clases`` se
    acepta como alias de compatibilidad, pero devuelve exactamente las mismas
    cuatro etiquetas intrínsecas y nunca HER2/TNBC clínicos.
    """
    modes = {
        "intrinsic4": INTRINSIC4_LABELS,
        "cuatro_clases": INTRINSIC4_LABELS,
    }
    key = str(mode).strip().lower()
    if key not in modes:
        raise ValueError(f"mode debe ser uno de {sorted(modes)}, no {mode!r}")
    return modes[key]


def aplicar_endpoint(expr: pd.DataFrame, meta: pd.DataFrame,
                      mode: str = "intrinsic4"):
    """Filtra expresión y metadata al endpoint, conservando su alineación."""
    labels = etiquetas_endpoint(mode)
    if "label" not in meta.columns:
        raise ValueError("La metadata no contiene la columna 'label'.")
    keep = meta.index[meta["label"].astype(str).isin(labels)]
    out_expr = expr.loc[:, expr.columns.intersection(keep)].copy()
    out_meta = meta.loc[out_expr.columns].copy()
    missing = sorted(set(labels) - set(out_meta["label"].astype(str)))
    if missing:
        raise ValueError(f"El endpoint {mode!r} no contiene {missing}.")
    print(
        f"Endpoint {mode}: {len(out_meta)} muestras, "
        f"clases={list(labels)}; excluidas={len(meta)-len(out_meta)}"
    )
    return out_expr, out_meta, labels


# =============================================================================
# UNIÓN
# =============================================================================

def unir(cfg: ConfigUnion | None = None) -> dict:
    cfg = cfg or ConfigUnion()
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  UNIÓN MULTI-PLATAFORMA — cáncer de mama")
    print("=" * 78)

    print("\n[1] Cargando fuentes...")
    e_m, m_m, d_m = cargar_microarrays(cfg)
    e_t, m_t, d_t = cargar_tcga(cfg)
    e_g, m_g, d_g = cargar_gse96058(cfg)
    escalas = pd.DataFrame([d_m, d_t, d_g])

    dfs = {"microarrays": e_m, "TCGA-BRCA": e_t, "GSE96058": e_g}
    metas = {"microarrays": m_m, "TCGA-BRCA": m_t, "GSE96058": m_g}

    print("\n[2] Genes")
    cob = informe_cobertura(dfs, metas)

    if cfg.gene_mode == "interseccion":
        genes = sorted(set.intersection(*[set(v.index) for v in dfs.values()]))
        print(f"    modo 'interseccion' -> {len(genes)} genes, sin NaN")
    elif cfg.gene_mode == "union":
        genes = sorted(set.union(*[set(v.index) for v in dfs.values()]))
        print(f"    modo 'union' -> {len(genes)} genes; los ausentes quedan NaN.")
        print("      NO se imputan: el filtro de genes del nb 02 los descarta")
        print("      automáticamente (exige varianza finita en todos los lotes de train).")
    else:
        raise ValueError("gene_mode debe ser 'interseccion' o 'union'")

    if len(genes) < 500:
        print("    ⚠️ muy pocos genes comunes: revisa que las tres matrices "
              "usen Gene Symbol")

    print("\n[3] Concatenando...")
    if cfg.namespace_samples:
        for k in dfs:
            m = metas[k]
            pref = m["dataset"].astype(str) + ":"
            nuevos = np.where(m.index.astype(str).str.contains(":"),
                              m.index.astype(str), pref + m.index.astype(str))
            dfs[k] = dfs[k].rename(columns=dict(zip(m.index, nuevos)))
            m.index = pd.Index(nuevos, name="sample")
            m["sample"] = nuevos

    expr = pd.concat([v.reindex(genes) for v in dfs.values()], axis=1)
    meta = pd.concat(list(metas.values()), axis=0, join="outer")

    if expr.columns.duplicated().any():
        d = expr.columns[expr.columns.duplicated()].tolist()
        raise ValueError(f"IDs de muestra colisionando entre fuentes: {d[:10]}")
    meta = meta.reindex(expr.columns)

    meta["platform"] = meta["dataset"].map(cfg.platform_map).fillna("unknown")
    if (meta["platform"] == "unknown").any():
        print(f"    ⚠️ sin plataforma: "
              f"{sorted(meta.loc[meta.platform=='unknown','dataset'].unique())}")

    print("\n[4] Etiquetas")
    meta["label_original"] = meta["label"]
    meta["label"] = canonicalizar_label(meta["label"])
    bad_labels = meta["label"].isna()
    if bad_labels.any():
        detail = (
            meta.loc[bad_labels, ["dataset", "label_original"]]
            .value_counts().head(20).to_dict()
        )
        raise ValueError(
            "Hay etiquetas ambiguas o no reconocidas. No se convierten "
            f"automáticamente a subtipos intrínsecos: {detail}"
        )

    # Las cohortes de una sola etiqueta salen en la primera unión y no pueden
    # reaparecer en notebooks posteriores. El contrato se vuelve a comprobar
    # después de elegir el endpoint, porque excluir Normal puede reducir el
    # número de etiquetas de una cohorte.
    cohort_audit = meta.groupby("dataset").agg(
        n_samples=("label", "size"),
        n_labels=("label", "nunique"),
        labels=("label", lambda x: ",".join(sorted(x.dropna().unique()))),
    )
    drop_one_label = cohort_audit.index[
        cohort_audit["n_labels"] < int(cfg.min_labels_per_cohort)
    ].astype(str).tolist()
    if drop_one_label:
        print(f"    cohortes con <{cfg.min_labels_per_cohort} etiquetas "
              f"eliminadas en la unión: {drop_one_label}")
        keep = ~meta["dataset"].astype(str).isin(drop_one_label)
        meta = meta.loc[keep].copy()
        expr = expr.loc[:, meta.index].copy()

    observed_cohorts = set(meta["dataset"].astype(str))
    expected_cohorts = set(map(str, cfg.expected_cohorts))
    if observed_cohorts != expected_cohorts:
        raise ValueError(
            "Las cohortes no coinciden con el contrato intrinsic4. "
            f"Faltan={sorted(expected_cohorts-observed_cohorts)}; "
            f"sobran={sorted(observed_cohorts-expected_cohorts)}"
        )
    observed_labels = set(meta["label"].astype(str))
    expected_labels = set(INTRINSIC4_LABELS)
    if observed_labels != expected_labels:
        raise ValueError(
            "Las clases no coinciden con intrinsic4. "
            f"Faltan={sorted(expected_labels-observed_labels)}; "
            f"sobran={sorted(observed_labels-expected_labels)}"
        )

    sem = auditar_semantica(meta, cfg)
    meta["normal_semantica"] = np.where(
        meta["label"].eq("Normal"),
        meta["dataset"].map(SEMANTICA_NORMAL).fillna("SIN DECLARAR"), "")

    na = float(expr.isna().mean().mean())
    print(f"\n[5] Matriz: {expr.shape[0]} genes x {expr.shape[1]} muestras "
          f"| NaN {na:.3%}  (sin imputar, a propósito)")
    print(f"\n{meta['dataset'].value_counts().to_string()}")
    print(f"\n{meta['platform'].value_counts().to_string()}")
    print(f"\n{meta['label'].value_counts(dropna=False).to_string()}")

    dup = (detectar_duplicados_cruzados(expr, meta, cfg)
           if cfg.detectar_duplicados else pd.DataFrame())

    print("\n[6] Guardando...")
    cols = [c for c in ["sample", "dataset", "platform", "data_type", "label",
                        "label_original", "normal_semantica", "tcga_sample_code"]
            if c in meta.columns]
    expr.to_csv(out / "expr_raw_combined.csv")
    # ``sample`` ya es una columna explícita; no se serializa además el índice
    # para evitar dos cabeceras homónimas al volver a leer el CSV.
    meta[cols].to_csv(out / "metadata_combined.csv", index=False)
    escalas.to_csv(out / "informe_escalas.csv", index=False)
    cob.to_csv(out / "informe_cobertura_genes.csv", index=False)
    cohort_audit.to_csv(out / "auditoria_cohortes_union.csv")
    sem.to_csv(out / "auditoria_semantica_normal.csv", index=False)
    if len(dup): dup.to_csv(out / "duplicados_entre_cohortes.csv", index=False)
    print(f"    📁 {out.resolve()}")

    return dict(expr=expr, meta=meta[cols], escalas=escalas, cobertura=cob,
                cohort_audit=cohort_audit,
                semantica=sem, duplicados=dup, cfg=cfg)


if __name__ == "__main__":
    unir()
