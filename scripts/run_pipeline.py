#!/usr/bin/env python3
"""Execute the supported InterGATE notebooks in dependency order."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
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
    NotebookStep("05", ROOT / "data_prepare/05_supplementary_figure_mcombat_qc.ipynb"),
)

ANALYSIS_STEPS = (
    NotebookStep("00", ROOT / "notebooks/00_comprobar_instalacion.ipynb"),
    NotebookStep("01", ROOT / "notebooks/01_validar_runtime_y_evidencia.ipynb"),
    NotebookStep("02", ROOT / "notebooks/02_entrenar_INTERGATE_V8.ipynb"),
    NotebookStep("03", ROOT / "notebooks/03_estabilidad_y_auditoria_V8.ipynb"),
    NotebookStep("04", ROOT / "notebooks/04_baselines_y_ablacion_minima.ipynb"),
    NotebookStep("05", ROOT / "notebooks/05_validacion_biologica_y_depmap.ipynb"),
    NotebookStep("06", ROOT / "notebooks/06_congelar_protocolo_externo.ipynb"),
    NotebookStep(
        "07",
        ROOT / "notebooks/07_test_externo_bloqueado.ipynb",
        ROOT / "results_v8/07_external_locked/external_evaluation_COMPLETE.json",
    ),
    NotebookStep(
        "07b",
        ROOT / "notebooks/07b_calibracion_posthoc.ipynb",
        ROOT / "results_v8/07b_calibration_posthoc/calibration_COMPLETE.json",
    ),
    NotebookStep(
        "08a",
        ROOT / "notebooks/08a_benchmark_tabular_pareado_S12.ipynb",
        ROOT / "results_v8/08a_paired_predictive_benchmark/summary/predictive_benchmark_manifest.json",
    ),
    NotebookStep(
        "08b",
        ROOT / "notebooks/08b_concordancia_estructural_externa.ipynb",
        ROOT / "results_v8/08b_external_structural_concordance/external_structural_concordance_manifest.json",
    ),
    NotebookStep("09", ROOT / "notebooks/09_ablaciones_biologia_score_0075.ipynb"),
    NotebookStep("12", ROOT / "notebooks/12_controles_revision_q_permutaciones.ipynb"),
    NotebookStep("10", ROOT / "notebooks/10_figuras_definitivas.ipynb"),
    NotebookStep("13", ROOT / "notebooks/13_generate_supplementary_figures_S2_S3.ipynb"),
)

OPTIONAL_STEPS = (
    NotebookStep("08c", ROOT / "notebooks/08c_xgboost_vs_intergate_biology.ipynb"),
)


def _slice_steps(
    steps: tuple[NotebookStep, ...], first: str | None, last: str | None
) -> tuple[NotebookStep, ...]:
    keys = [step.key for step in steps]
    if first is not None and first not in keys:
        raise ValueError(f"step {first!r} is not part of this stage: {keys}")
    if last is not None and last not in keys:
        raise ValueError(f"step {last!r} is not part of this stage: {keys}")
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
    include_label_permutations: bool,
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
        env = os.environ.copy()
        if step.key == "12":
            env["INTERGATE_RUN_LABEL_PERMUTATIONS"] = (
                "1" if include_label_permutations else "0"
            )
        print(f"[run {position}/{len(steps)}] {step.path.name}")
        if step.key == "12" and not include_label_permutations:
            print("  label permutations: disabled (factorial and k/R remain enabled)")
        if dry_run:
            print("  ", " ".join(command))
            continue
        subprocess.run(command, cwd=working_dir, env=env, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the audited InterGATE data and analysis notebooks."
    )
    parser.add_argument(
        "stage", choices=("data", "analysis", "optional", "all", "list")
    )
    valid_steps = (
        "00", "00a", "00b", "01", "02", "03", "04", "05", "06",
        "07", "07b", "08a", "08b", "08c", "09", "10", "12", "13",
    )
    parser.add_argument("--from-step", choices=valid_steps)
    parser.add_argument("--through-step", choices=valid_steps)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--include-label-permutations",
        action="store_true",
        help="Enable the extended 49×2 full-retraining label-permutation block.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "list":
        for name, steps in (
            ("data", DATA_STEPS),
            ("analysis", ANALYSIS_STEPS),
            ("optional", OPTIONAL_STEPS),
        ):
            print(f"{name}:")
            for step in steps:
                print(f"  {step.key:>3}  {step.path.relative_to(ROOT)}")
        return 0

    if args.stage == "all" and (
        args.from_step is not None or args.through_step is not None
    ):
        parser.error(
            "--from-step and --through-step are stage-specific; choose data "
            "or analysis when resuming"
        )

    try:
        if args.stage in {"data", "all"}:
            if not args.skip_preflight and not args.dry_run:
                _preflight("data")
            selected = _slice_steps(DATA_STEPS, args.from_step, args.through_step)
            _execute(
                "data", selected, kernel=args.kernel, dry_run=args.dry_run,
                include_label_permutations=False,
            )

        if args.stage in {"analysis", "all"}:
            if not args.skip_preflight and not args.dry_run:
                _preflight("analysis")
            selected = _slice_steps(
                ANALYSIS_STEPS, args.from_step, args.through_step
            )
            _execute(
                "analysis", selected, kernel=args.kernel, dry_run=args.dry_run,
                include_label_permutations=args.include_label_permutations,
            )

        if args.stage == "optional":
            if not args.skip_preflight and not args.dry_run:
                _preflight("analysis")
            selected = _slice_steps(
                OPTIONAL_STEPS, args.from_step, args.through_step
            )
            _execute(
                "optional", selected, kernel=args.kernel, dry_run=args.dry_run,
                include_label_permutations=False,
            )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
