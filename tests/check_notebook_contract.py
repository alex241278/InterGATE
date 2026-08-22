"""Static guard for the supported data and analysis notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = [
    "00a_download_microarrays.ipynb",
    "00b_download_rnaseq.ipynb",
    "01_union_datasets.ipynb",
    "02_preparacion_y_suelo.ipynb",
    "03_split_y_mcombat_ref.ipynb",
    "04_qc_batch_y_fuga.ipynb",
]
ANALYSIS = [
    "01_validar_runtime_y_evidencia.ipynb",
    "02_entrenar_INTERGATE_V6.ipynb",
    "03_estabilidad_y_auditoria_V6.ipynb",
    "04_baselines_y_ablacion_minima.ipynb",
    "05_validacion_biologica_y_depmap.ipynb",
    "06_congelar_protocolo_externo.ipynb",
    "07_test_externo_bloqueado.ipynb",
    "08_concordancia_estructural_externa.ipynb",
    "09_ablaciones_biologia_score_0075.ipynb",
    "10_figuras_definitivas.ipynb",
    "11_benchmark_tabular_pareado_S12.ipynb",
]


def source_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    assert notebook.get("cells")
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    chunks = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        chunks.append(source)
        if cell.get("cell_type") == "code":
            assert cell.get("outputs") == [], path
            assert cell.get("execution_count") is None, path
    return "\n".join(chunks)


assert sorted(path.name for path in (ROOT / "data_prepare").glob("*.ipynb")) == sorted(DATA)
assert sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb")) == sorted(ANALYSIS)

all_sources = {}
for folder, names in (("data_prepare", DATA), ("notebooks", ANALYSIS)):
    for name in names:
        path = ROOT / folder / name
        text = source_text(path)
        all_sources[name] = text
        for forbidden in (
            "/home/",
            "/Users/",
            "Escritorio",
            "C:\\",
            "code_snapshot",
            "requirements-v6.txt",
        ):
            assert forbidden not in text, (path, forbidden)

for name in ANALYSIS[:6] + ANALYSIS[8:10]:
    assert "include_external=True" not in all_sources[name], name
for name in (ANALYSIS[6], ANALYSIS[7], ANALYSIS[10]):
    assert "include_external=True" in all_sources[name], name

assert "DEFINITIVE_BIOLOGY_SCORE_SUPPORT_WEIGHT" in all_sources[ANALYSIS[3]]
assert all_sources[ANALYSIS[10]].lstrip().startswith("# 11")
print("Notebook contract: OK")
