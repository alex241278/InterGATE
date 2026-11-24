"""
Enrichment analysis: gProfiler, Enrichr (gseapy), KEGG map overlay,
and symbol-to-entrez mapping via mygene.

All functions are standalone — no module-level side effects.
"""

import re
from io import BytesIO
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import xml.etree.ElementTree as ET


# ═══════════════════════════════════════════════════════════════════
#  gProfiler enrichment
# ═══════════════════════════════════════════════════════════════════

def run_gprofiler(
    gene_list: List[str],
    sources: List[str] = ("GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"),
    organism: str = "hsapiens",
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Run g:Profiler enrichment.  Returns result DataFrame (empty on error)."""
    try:
        from gprofiler import GProfiler
    except ImportError:
        print("[WARN] gprofiler-official not installed.  pip install gprofiler-official")
        return pd.DataFrame()

    gp = GProfiler(return_dataframe=True)
    res = gp.profile(organism=organism, query=gene_list, sources=list(sources))
    if save_csv and not res.empty:
        res.to_csv(save_csv, index=False)
        print(f"[gprofiler] Saved: {save_csv}")
    return res


# ═══════════════════════════════════════════════════════════════════
#  Enrichr (via gseapy)
# ═══════════════════════════════════════════════════════════════════

def _detect_kegg_library() -> str:
    """Find the most recent KEGG_YYYY_Human library available in Enrichr."""
    try:
        import gseapy as gp
        libs = gp.get_library_name()
        kegg_libs = [l for l in libs if re.match(r"^KEGG_\d{4}_Human$", l)]
        if kegg_libs:
            return sorted(kegg_libs, key=lambda x: int(x.split("_")[1]))[-1]
    except Exception:
        pass
    return "KEGG_2021_Human"


def run_enrichr(
    gene_list: List[str],
    gene_sets: str = "KEGG_2021_Human",
    organism: str = "human",
    cutoff: float = 0.05,
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Run Enrichr via gseapy.  Returns result DataFrame (empty on error)."""
    try:
        import gseapy as gp
    except ImportError:
        print("[WARN] gseapy not installed.  pip install gseapy")
        return pd.DataFrame()

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=[gene_sets] if isinstance(gene_sets, str) else gene_sets,
        organism=organism,
        outdir=None,
        cutoff=cutoff,
    )
    df = enr.results.copy() if hasattr(enr, "results") else pd.DataFrame()
    if save_csv and not df.empty:
        df.to_csv(save_csv, index=False)
        print(f"[enrichr] Saved: {save_csv}")
    return df


def run_enrichr_battery(
    gene_list: List[str],
    out_prefix: str = "enrich",
    save_dir: str = "./Results",
    cutoff: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Run Enrichr for KEGG, Reactome, GO:BP/MF/CC.  Returns dict of DataFrames."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    kegg_lib = _detect_kegg_library()
    targets = [
        (kegg_lib, "KEGG"),
        ("Reactome_2022", "REACTOME"),
        ("GO_Biological_Process_2023", "GO_BP"),
        ("GO_Molecular_Function_2023", "GO_MF"),
        ("GO_Cellular_Component_2023", "GO_CC"),
    ]
    results = {}
    for gs, tag in targets:
        csv = os.path.join(save_dir, f"{out_prefix}_{tag}.csv")
        df = run_enrichr(gene_list, gene_sets=gs, cutoff=cutoff, save_csv=csv)
        results[tag] = df
    return results


# ═══════════════════════════════════════════════════════════════════
#  Symbol → Entrez mapping (via mygene)
# ═══════════════════════════════════════════════════════════════════

def build_symbol_entrez_maps(
    symbols: List[str],
) -> tuple:
    """
    Map gene symbols to Entrez IDs using mygene.

    Returns (symbol_to_entrez, entrez_to_symbol) dicts.
    """
    try:
        import mygene
    except ImportError:
        print("[WARN] mygene not installed.  pip install mygene")
        return {}, {}

    symbols_u = [str(s).strip().upper() for s in symbols if str(s).strip()]
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        symbols_u,
        scopes="symbol",
        fields="entrezgene,symbol",
        species="human",
        as_dataframe=True,
        returnall=False,
        verbose=False,
    )
    res = res.reset_index().rename(columns={"query": "query"})
    res["query"] = res["query"].astype(str).str.upper()
    res = res[pd.notna(res["entrezgene"])].copy()
    res["entrezgene"] = res["entrezgene"].astype(int)

    symbol_to_entrez = dict(zip(res["query"], res["entrezgene"]))
    entrez_to_symbol = {}
    if "symbol" in res.columns:
        for e, sym in zip(res["entrezgene"], res["symbol"]):
            if pd.notna(sym):
                entrez_to_symbol[int(e)] = str(sym)

    return symbol_to_entrez, entrez_to_symbol


# ═══════════════════════════════════════════════════════════════════
#  KEGG pathway overlay
# ═══════════════════════════════════════════════════════════════════

def _fetch_kegg_kgml(pathway_id: str) -> str:
    url = f"https://rest.kegg.jp/get/{pathway_id}/kgml"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def _fetch_kegg_png(pathway_id: str):
    from PIL import Image
    url = f"https://rest.kegg.jp/get/{pathway_id}/image"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGBA")


def _parse_kgml_gene_boxes(kgml_xml: str) -> Dict[str, list]:
    """Parse KGML XML → dict of gene_id('hsa:####') → list of (x0, y0, w, h)."""
    root = ET.fromstring(kgml_xml)
    boxes: Dict[str, list] = {}
    for entry in root.findall("entry"):
        if entry.attrib.get("type", "") not in ("gene", "group"):
            continue
        name = entry.attrib.get("name", "")
        g = entry.find("graphics")
        if g is None:
            continue
        try:
            x = float(g.attrib.get("x"))
            y = float(g.attrib.get("y"))
            w = float(g.attrib.get("width"))
            h = float(g.attrib.get("height"))
        except Exception:
            continue
        x0, y0 = x - w / 2.0, y - h / 2.0
        gene_ids = [tok.strip() for tok in name.split() if tok.strip().startswith("hsa:")]
        for gid in gene_ids:
            boxes.setdefault(gid, []).append((x0, y0, w, h))
    return boxes


def plot_kegg_map_overlay(
    pathway_id: str,
    selected_genes=None,
    symbol_to_entrez: Optional[Dict] = None,
    entrez_to_symbol: Optional[Dict] = None,
    show_labels: bool = True,
    box_lw: float = 2.0,
):
    """
    Download KEGG pathway image and overlay red boxes on selected genes.

    Parameters
    ----------
    pathway_id : str   e.g. "hsa05224"
    selected_genes : list   Symbols, Entrez IDs, or "hsa:####" strings.
    """
    kgml = _fetch_kegg_kgml(pathway_id)
    img = _fetch_kegg_png(pathway_id)
    boxes = _parse_kgml_gene_boxes(kgml)

    selected_hsa = set()
    if selected_genes is not None:
        for g in selected_genes:
            if isinstance(g, int):
                selected_hsa.add(f"hsa:{g}")
            else:
                s = str(g).strip()
                if s.startswith("hsa:"):
                    selected_hsa.add(s)
                elif symbol_to_entrez is not None and s.upper() in symbol_to_entrez:
                    selected_hsa.add(f"hsa:{int(symbol_to_entrez[s.upper()])}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{pathway_id} (overlay: rojo = genes seleccionados)")

    for gid in selected_hsa:
        if gid not in boxes:
            continue
        for (x0, y0, w, h) in boxes[gid]:
            rect = patches.Rectangle(
                (x0, y0), w, h, fill=False, linewidth=box_lw, edgecolor="crimson"
            )
            ax.add_patch(rect)
            if show_labels:
                if entrez_to_symbol is not None:
                    entrez = int(gid.split(":")[1])
                    lab = entrez_to_symbol.get(entrez, gid)
                else:
                    lab = gid
                ax.text(x0, y0 - 2, lab, fontsize=7, color="crimson")

    plt.tight_layout()
    plt.show()

    in_map = sorted(selected_hsa & set(boxes.keys()))
    if entrez_to_symbol is not None:
        overlap_sym = [entrez_to_symbol.get(int(g.split(":")[1]), g) for g in in_map]
        print("Overlap símbolos:", overlap_sym[:80])
    return fig, ax
