#!/usr/bin/env python3
"""Execute the supported InterGATE notebooks in the audited order."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NotebookStep:
    key: str
    path: Path
    completion_marker: Path | None = None


DATA_STEPS = (
    NotebookStep("00a", ROOT / "data_prepare/00a_download_microarrays.ipynb"),
    NotebookStep("00b", ROOT / "data_prepare/00b_download_rnaseq.ipynb"),
    NotebookStep("01", ROOT / "data_prepare/01_union_datasets.ipynb"),
    NotebookStep("02", ROOT / "data_prepare/02_preparacion_y_suelo.ipynb"),
    NotebookStep("03", ROOT / "data_prepare/03_split_y_mcombat_ref.ipynb"),
    NotebookStep("04", ROOT / "data_prepare/04_qc_batch_y_fuga.ipynb"),
)

ANALYSIS_STEPS = (
    NotebookStep("01", ROOT / "notebooks/01_validar_runtime_y_evidencia.ipynb"),
    NotebookStep("02", ROOT / "notebooks/02_entrenar_INTERGATE_V6.ipynb"),
    NotebookStep("03", ROOT / "notebooks/03_estabilidad_y_auditoria_V6.ipynb"),
    NotebookStep("04", ROOT / "notebooks/04_baselines_y_ablacion_minima.ipynb"),
    NotebookStep("05", ROOT / "notebooks/05_validacion_biologica_y_depmap.ipynb"),
    NotebookStep("06", ROOT / "notebooks/06_congelar_protocolo_externo.ipynb"),
    NotebookStep(
        "07",
        ROOT / "notebooks/07_test_externo_bloqueado.ipynb",
        ROOT / "results_v6/07_external_locked/external_evaluation_COMPLETE.json",
    ),
    NotebookStep(
        "08",
        ROOT / "notebooks/08_concordancia_estructural_externa.ipynb",
        ROOT / "results_v6/08_external_structural_concordance/external_structural_concordance_manifest.json",
    ),
    NotebookStep("09", ROOT / "notebooks/09_ablaciones_biologia_score_0075.ipynb"),
    NotebookStep("10", ROOT / "notebooks/10_figuras_definitivas.ipynb"),
    NotebookStep(
        "11",
        ROOT / "notebooks/11_benchmark_tabular_pareado_S12.ipynb",
        ROOT / "results_v6/11_paired_predictive_benchmark/summary/predictive_benchmark_manifest.json",
    ),
)


def _slice_steps(
    steps: tuple[NotebookStep, ...],
    first: str | None,
    last: str | None,
) -> tuple[NotebookStep, ...]:
    keys = [step.key for step in steps]
    start = keys.index(first) if first is not None else 0
    stop = keys.index(last) + 1 if last is not None else len(steps)
    if start >= stop:
        raise ValueError("--from-step must not follow --through-step")
    return steps[start:stop]


def _preflight(stage: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/preflight.py"), "--stage", stage],
        cwd=ROOT,
        check=True,
    )


def _execute(
    stage: str,
    steps: tuple[NotebookStep, ...],
    *,
    kernel: str,
    dry_run: bool,
) -> None:
    output_dir = ROOT / "results/executed_notebooks" / stage
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    working_dir = ROOT / "data_prepare" if stage == "data" else ROOT

    for position, step in enumerate(steps, start=1):
        if step.completion_marker is not None and step.completion_marker.is_file():
            print(
                f"[skip {position}/{len(steps)}] {step.path.name}: "
                f"completion marker found at {step.completion_marker.relative_to(ROOT)}"
            )
            continue
        output_name = f"{step.path.stem}.executed.ipynb"
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(step.path),
            "--output",
            output_name,
            "--output-dir",
            str(output_dir),
            f"--ExecutePreprocessor.kernel_name={kernel}",
            "--ExecutePreprocessor.timeout=-1",
        ]
        print(f"[run {position}/{len(steps)}] {step.path.name}")
        if dry_run:
            print("  ", " ".join(command))
            continue
        subprocess.run(command, cwd=working_dir, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the audited InterGATE data and analysis notebooks."
    )
    parser.add_argument("stage", choices=("data", "analysis", "all", "list"))
    parser.add_argument("--from-step", choices=("00a", "00b", *[f"{i:02d}" for i in range(1, 12)]))
    parser.add_argument("--through-step", choices=("00a", "00b", *[f"{i:02d}" for i in range(1, 12)]))
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "list":
        for name, steps in (("data", DATA_STEPS), ("analysis", ANALYSIS_STEPS)):
            print(f"{name}:")
            for step in steps:
                print(f"  {step.key}  {step.path.relative_to(ROOT)}")
        return 0

    if args.stage == "all" and (
        args.from_step is not None or args.through_step is not None
    ):
        parser.error(
            "--from-step and --through-step are stage-specific; use "
            "'data' or 'analysis' instead of 'all' when resuming"
        )

    if args.stage in {"data", "all"}:
        if not args.skip_preflight and not args.dry_run:
            _preflight("data")
        selected = _slice_steps(DATA_STEPS, args.from_step, args.through_step)
        _execute("data", selected, kernel=args.kernel, dry_run=args.dry_run)

    if args.stage in {"analysis", "all"}:
        if not args.skip_preflight and not args.dry_run:
            _preflight("analysis")
        selected = _slice_steps(ANALYSIS_STEPS, args.from_step, args.through_step)
        _execute("analysis", selected, kernel=args.kernel, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
