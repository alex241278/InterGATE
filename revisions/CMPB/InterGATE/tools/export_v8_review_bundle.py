#!/usr/bin/env python3
"""Export the small, development-only evidence needed to review V8.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v8"
RUN_ROOT = ROOT / "artifacts" / "intrinsic4" / "INTERGATE_V8"
ALLOWED_SUFFIXES = {
    ".csv", ".gz", ".json", ".md", ".txt", ".png", ".pdf", ".svg",
}
MAX_FILE_BYTES = 50 * 1024 * 1024
EXPECTED_PROTOCOL = "intergate-v8-biology-first-v2-matched-budget"
EXPECTED_SEEDS = (1234, 2025, 7, 99, 31337)


def validate_matched_budget() -> None:
    """Refuse to export a run made with the incomplete first-V8 budget."""
    manifest_path = RESULTS / "02_training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != EXPECTED_PROTOCOL:
        raise RuntimeError(
            "El manifiesto no pertenece al protocolo V8.1 pareado."
        )
    expected_config = {
        "GRAPH_BATCH_SIZE": 2,
        "GRAPH_ACCUM_STEPS": 120,
        "TRANSPORT_STEPS_PER_EPISODE": 192,
        "PHASE1_SAMPLES_PER_FLOW": 384,
        "PHASE1_OPTIMIZER_STEPS": 2,
        "TRANSPORT_SUPPORT_BATCHES": 64,
        "SUPPORT_SAMPLES_PER_STEP": 128,
    }
    observed_config = manifest.get("config", {})
    bad = {
        key: (observed_config.get(key), wanted)
        for key, wanted in expected_config.items()
        if observed_config.get(key) != wanted
    }
    if bad:
        raise RuntimeError(f"Presupuesto incorrecto en el manifiesto 02: {bad}")

    for seed in EXPECTED_SEEDS:
        seed_dir = RUN_ROOT / f"seed_{seed}"
        history_path = seed_dir / "transport_training_history.csv"
        summary_path = seed_dir / "transport_support_summary.json"
        if not history_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"Semilla {seed}: faltan historial o resumen de soporte."
            )
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"Semilla {seed}: historial de fase 1 vacío.")
        history_expected = {
            "steps_completed": 192,
            "microbatch_size": 2,
            "accumulation_steps": 120,
            "samples_per_meta_train_flow": 384,
            "samples_per_meta_validation_flow": 384,
            "optimizer_steps_expected": 2,
        }
        for row_number, row in enumerate(rows, 1):
            row_bad = {
                key: (row.get(key), wanted)
                for key, wanted in history_expected.items()
                if row.get(key) is None or int(float(row[key])) != wanted
            }
            if row_bad:
                raise RuntimeError(
                    f"Semilla {seed}, episodio {row_number}: contrato de "
                    f"fase 1 incorrecto: {row_bad}"
                )
        support = json.loads(summary_path.read_text(encoding="utf-8"))
        support_expected = {
            "protocol_version": EXPECTED_PROTOCOL,
            "support_batches_per_step": 64,
            "support_microbatch_size": 2,
            "support_samples_per_step": 128,
        }
        support_bad = {
            key: (support.get(key), wanted)
            for key, wanted in support_expected.items()
            if support.get(key) != wanted
        }
        if support_bad:
            raise RuntimeError(
                f"Semilla {seed}: soporte no pareado: {support_bad}"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def candidate_files() -> list[Path]:
    static = [
        ROOT / "README.md",
        ROOT / "METHODS_V8.md",
        ROOT / "V7_V8_COMPARISON_PLAN.md",
        ROOT / "configs" / "protocol_v8.json",
    ]
    required = [
        RESULTS / "02_training_manifest.json",
        RESULTS / "02_ablacion_ramas_validacion.csv",
        RESULTS / "02_caida_por_bloque_validacion.csv",
        RUN_ROOT / "development_results.csv",
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    stability = RESULTS / "03_stability"
    if not stability.is_dir():
        missing.append(relative(stability))
    if missing:
        raise FileNotFoundError(
            "Falta completar los notebooks 02–03:\n- " + "\n- ".join(missing)
        )

    selected = static + required
    selected.extend(
        path for path in stability.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )
    for seed_dir in sorted(RUN_ROOT.glob("seed_*")):
        selected.extend(
            path for path in seed_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".gz"}
        )

    unique: dict[str, Path] = {}
    for path in selected:
        if not path.is_file():
            continue
        name = relative(path)
        lowered = name.lower()
        if "07_external" in lowered or "/external" in lowered:
            raise RuntimeError(f"El candidato contiene una salida externa: {name}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError(
                f"{name} supera 50 MiB; revise que no sea un artefacto binario."
            )
        unique[name] = path
    return [unique[name] for name in sorted(unique)]


def source_fingerprints() -> dict[str, str]:
    sources = sorted((ROOT / "intergate").glob("*.py"))
    sources += sorted((ROOT / "notebooks").glob("*.ipynb"))
    sources += sorted((ROOT / "configs").glob("*.json"))
    return {relative(path): sha256(path) for path in sources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "InterGATE_V8_1_review_bundle.zip",
        help="Ruta del ZIP de salida.",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    files = candidate_files()
    validate_matched_budget()
    manifest = {
        "bundle": "InterGATE_V8.1 matched-budget development-only review",
        "external_results_included": False,
        "files": {
            relative(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        },
        "source_sha256": source_fingerprints(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=relative(path))
        archive.writestr(
            "BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
    print(f"Paquete de revisión: {output}")
    print(f"Archivos: {len(files) + 1}; tamaño: {output.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
