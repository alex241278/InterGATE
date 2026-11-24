"""Write 10- and 20-seed summaries as soon as each nested set completes.

The structural summaries use development-only inputs. This observer checks
the original training completion markers and validates the dense score files.
"""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "training"
JOBS = TRAIN / "artifacts/INTERGATE_REVISION_SEEDS"
lock = (TRAIN / "aggregate.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
for n in (10, 20):
    total = n - 5
    seeds = [JOBS / f"seed_{seed}" for seed in range(10001,10001+total)]
    while not all((p/"transport_checkpoint_COMPLETE.txt").is_file() and (p/"revision_result.json").is_file() for p in seeds):
        time.sleep(45)
    for path in seeds:
        assert (path/"transport_support_dense.npz").is_file()
        assert json.loads((path/"revision_result.json").read_text())["metrics"]["evaluation_split"]=="development_validation"
    print(f"{datetime.now(timezone.utc).isoformat()} Computing {n}-seed sensitivity",flush=True)
    subprocess.run([sys.executable,"-Bu",str(HERE/"structural_sensitivity.py"),"--extended",str(n)],check=True)
