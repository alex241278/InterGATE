"""Static guard for the supported data and analysis notebooks."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA = [
    "00a_download_microarrays.ipynb",
    "00b_download_rnaseq.ipynb",
    "01_union_datasets.ipynb",
    "02_preparacion_y_suelo.ipynb",
    "03_split_y_mcombat_ref.ipynb",
    "04_qc_batch_y_fuga.ipynb",
    "05_supplementary_figure_mcombat_qc.ipynb",
]
ANALYSIS = [
    "00_comprobar_instalacion.ipynb",
    "01_validar_runtime_y_evidencia.ipynb",
    "02_entrenar_INTERGATE_V8.ipynb",
    "03_estabilidad_y_auditoria_V8.ipynb",
    "04_baselines_y_ablacion_minima.ipynb",
    "05_validacion_biologica_y_depmap.ipynb",
    "06_congelar_protocolo_externo.ipynb",
    "07_test_externo_bloqueado.ipynb",
    "07b_calibracion_posthoc.ipynb",
    "08a_benchmark_tabular_pareado_S12.ipynb",
    "08b_concordancia_estructural_externa.ipynb",
    "08c_xgboost_vs_intergate_biology.ipynb",
    "09_ablaciones_biologia_score_0075.ipynb",
    "10_figuras_definitivas.ipynb",
    "12_controles_revision_q_permutaciones.ipynb",
    "12c_sensibilidad_kappa_con_reentrenamiento.ipynb",
    "12d_generate_kappa_retraining_figure.ipynb",
    "13_generate_supplementary_figures_S2_S3.ipynb",
]
ABSOLUTE_PATH = re.compile(r"(?:/home/|/Users/|/workspace/|[A-Za-z]:\\\\Users\\\\)")


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def check_notebook(path: Path) -> tuple[int, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        text = source_text(cell)
        if ABSOLUTE_PATH.search(text):
            raise AssertionError(f"absolute path in {path}:{index}")
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        if cell.get("execution_count") is not None:
            raise AssertionError(f"execution count in {path}:{index}")
        if cell.get("outputs"):
            raise AssertionError(f"embedded output in {path}:{index}")
        compile(text, f"{path.name}:cell-{index}", "exec")
    return len(notebook.get("cells", [])), code_cells


def main() -> None:
    expected = {
        *(ROOT / "data_prepare" / name for name in DATA),
        *(ROOT / "notebooks" / name for name in ANALYSIS),
    }
    missing = sorted(str(path.relative_to(ROOT)) for path in expected if not path.is_file())
    if missing:
        raise AssertionError("missing notebooks: " + ", ".join(missing))

    observed_analysis = {
        path.name for path in (ROOT / "notebooks").glob("*.ipynb")
    }
    if observed_analysis != set(ANALYSIS):
        raise AssertionError(
            "analysis inventory mismatch: "
            f"extra={sorted(observed_analysis - set(ANALYSIS))}, "
            f"missing={sorted(set(ANALYSIS) - observed_analysis)}"
        )

    cell_total = 0
    code_total = 0
    for path in sorted(expected):
        cells, code = check_notebook(path)
        cell_total += cells
        code_total += code
    print(
        f"Notebook contract: OK ({len(expected)} notebooks, "
        f"{cell_total} cells, {code_total} code cells)"
    )


if __name__ == "__main__":
    main()
