"""Extend the original five seeds with fifteen consecutive prespecified seed IDs."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "InterGATE"
ADDITIONAL = tuple(range(10001, 10016))
OUT = HERE / "training"
OUT.mkdir(exist_ok=True)
writer = (OUT / "writer.lock").open("a")
fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.environ["INTERGATE_CACHE_ROOT"] = str(OUT / "cache")
os.environ["INTERGATE_ARTIFACTS_ROOT"] = str(OUT / "artifacts")
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
sys.path.insert(0, str(PROJECT))
from intergate.config import CFG
from intergate.runtime_setup import assert_transport_protocol, assert_v8_protocol, prepare_runtime
from intergate.ablation import CFG_FULL, cfg_to_dict
from intergate.transport import TransportConfig, run_transport_seed
import torch

def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA device is required for this full-protocol fifteen-seed run.")
assert_v8_protocol(CFG)
assert_transport_protocol(CFG)
cache = Path(CFG.PIPELINE_CACHE_DIR)
for source in (PROJECT / "cache/pipeline_cache").iterdir():
    # The second, larger X_h cache contains external inputs and is unnecessary.
    if source.name.endswith("ff3f4ef6c1949632.npz"):
        continue
    destination = cache / source.name
    if source.is_file() and not destination.exists():
        shutil.copyfile(source, destination)
runtime = prepare_runtime(CFG, include_external=False)
if runtime["external_opened"] or len(runtime["test_idx"]):
    raise RuntimeError("Training runtime contains external observations")
reference = json.loads((PROJECT / "artifacts/intrinsic4/INTERGATE_V8/seed_1234/meta.json").read_text())
current = cfg_to_dict(CFG_FULL)
for field in ("implementation_sha256", "model_config_sha256", "splits", "dataset_signature"):
    if current[field] != reference[field]:
        raise RuntimeError(f"Mismatch against original training bundle in {field}: {str(current[field])[:1200]}")
contract = {"original": list(CFG.STABILITY_SEEDS), "new": ADDITIONAL,
            "nested_sizes": [3, 5, 10, 20], "protocol": CFG.TRANSPORT_PROTOCOL_VERSION,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": sys.version,
            "implementation": current["implementation_sha256"],
            "configuration": current["model_config_sha256"],
            "dataset": current["dataset_signature"], "external_opened": False}
atomic_json(OUT / "seed_contract.json", contract)
run_name = "INTERGATE_REVISION_SEEDS"
transport = TransportConfig.from_global(CFG)
for seed in ADDITIONAL:
    print(f"Starting or resuming seed {seed}", flush=True)
    start = time.monotonic()
    row = run_transport_seed(seed, save_root=CFG.ARTIFACTS_ROOT, artifact_name=run_name,
                             transport=transport, resume=True, force=False)
    bundle = Path(CFG.ARTIFACTS_ROOT) / run_name / f"seed_{seed}"
    if not (bundle / "transport_checkpoint_COMPLETE.txt").is_file() or row["transport_all_parameter_deltas_zero"]:
        raise RuntimeError(f"Incomplete or inactive fit for seed {seed}")
    atomic_json(bundle / "revision_result.json", {"seed": seed, "metrics": row,
                                                 "current_invocation_seconds": time.monotonic()-start})
    print(f"Finished seed {seed}", flush=True)
