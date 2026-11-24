"""Bounded local supervisor for this revision's three checkpointed workflows.

The trainer/aggregator locks are the source of truth for live sessions,
including existing processes owned by Codex. A free lock and an incomplete
output set trigger the finite retry policy below. No other job is stopped.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STATE = HERE / "supervision"
STATE.mkdir(exist_ok=True)
TRAIN = HERE / "training"
STRING = HERE / "string_training"
ORIGINAL_SEEDS = (1234, 2025, 7, 99, 31337)
MAX_RESTARTS = 4
MAX_HOURS = 72

def atomic_json(path, values):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)

def in_use(path):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False

def completed_fits(root, seeds):
    complete = []
    for seed in seeds:
        folder = root / f"seed_{seed}"
        if not (folder/"transport_checkpoint_COMPLETE.txt").is_file():
            continue
        if not (folder/"revision_result.json").is_file():
            continue
        result = json.loads((folder/"revision_result.json").read_text())
        meta = json.loads((folder/"meta.json").read_text())
        if result["metrics"]["evaluation_split"]!="development_validation" or not meta["transport_protocol"]["completed"]:
            raise RuntimeError(f"Invalid completion record at {folder}")
        complete.append(seed)
    return complete

def aggregation_ready():
    for size in (10,20):
        for suffix in ("summary.csv","core_edges.csv.gz","seed_panels.csv.gz"):
            path = HERE / f"extended_{size}_{suffix}"
            if not path.is_file() or path.stat().st_size==0:
                return False
    return True

def string_results_ready():
    return all((STRING/file).is_file() and (STRING/file).stat().st_size>0 for file in
               ("five_seed_summary.csv","five_seed_core_edges.csv.gz","seed_panels.csv.gz",
                "pairwise_overlap.csv","development_performance.csv"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("string_source",type=Path)
    parser.add_argument("structural_python",type=Path)
    parser.add_argument("--status-only",action="store_true")
    args = parser.parse_args()
    commands = {
        "primary":[sys.executable,"-Bu",str(HERE/"train_additional_seeds.py")],
        "aggregation":[str(args.structural_python),"-Bu",str(HERE/"aggregate_when_ready.py")],
        "string":[sys.executable,"-Bu",str(HERE/"train_string_prior.py"),"queued-training",str(args.string_source)],
    }
    locks = {"primary":TRAIN/"writer.lock","aggregation":TRAIN/"aggregate.lock","string":STRING/"worker.lock"}
    guard = (STATE/"supervisor.lock").open("a")
    if not args.status_only:
        fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = STATE/"status.json"
    prev = json.loads(status_path.read_text()) if status_path.exists() else {}
    attempts = prev.get("automatic_restarts",dict.fromkeys(commands,0))
    # Remember the absolute deadline across a restart of the supervisor.
    started = prev.get("started_epoch",time.time())
    children = {}
    process_env = os.environ.copy()
    process_env.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1")
    while True:
        now=time.time()
        clock=datetime.fromtimestamp(now,timezone.utc).isoformat()
        primary=completed_fits(TRAIN/"artifacts/INTERGATE_REVISION_SEEDS",range(10001,10016))
        secondary=completed_fits(STRING/"artifacts/INTERGATE_STRING_PHYSICAL",ORIGINAL_SEEDS)
        done = {"primary":len(primary)==15, "aggregation":aggregation_ready(),
                "string":len(secondary)==5 and string_results_ready()}
        busy = {name:in_use(path) for name,path in locks.items()}
        for name,child in list(children.items()):
            code=child.poll()
            if code is not None:
                print(clock,name,"exited",code,flush=True)
                children.pop(name)
        status = {"updated_utc":clock,"started_epoch":started,"maximum_hours":MAX_HOURS,
                  "maximum_automatic_restarts_per_stage":MAX_RESTARTS,
                  "primary_completed":primary,"string_completed":secondary,"stage_completed":done,
                  "stage_lock_held":busy,"automatic_restarts":attempts,
                  "supervised_child_pids":{k:p.pid for k,p in children.items()}}
        if args.status_only:
            print(json.dumps(status,indent=2));return
        atomic_json(status_path,status)
        if all(done.values()): return
        if now-started>MAX_HOURS*3600: raise RuntimeError("Bounded revision supervision window elapsed")
        for name in commands:
            if done[name] or busy[name] or name in children: continue
            if attempts[name]>=MAX_RESTARTS:
                raise RuntimeError(f"{name} exceeded the automatic recovery limit. Inspect its revision log.")
            # Each worker independently obtains an exclusive lock as well,
            # including if a person manually restarts it between this check and launch.
            attempts[name]+=1
            print(clock,"Recovering",name,"attempt",attempts[name],flush=True)
            with (STATE/f"{name}.log").open("a") as log:
                log.write(f"\n{clock} automatic recovery {attempts[name]}\n")
                log.flush()
                children[name]=subprocess.Popen(commands[name],cwd=HERE,env=process_env,stdin=subprocess.DEVNULL,
                                                stdout=log,stderr=subprocess.STDOUT)
        atomic_json(status_path,{**status,"automatic_restarts":attempts,
                                "supervised_child_pids":{k:p.pid for k,p in children.items()}})
        time.sleep(45)
if __name__=="__main__": main()
