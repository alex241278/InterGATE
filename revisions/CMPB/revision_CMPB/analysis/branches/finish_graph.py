"""Finish all five isolated graph fits, evaluate frozen checkpoints, and check pairing.

Existing workers hold an advisory lock per seed. If one is interrupted, the
supervisor resumes its saved epoch with the captured optimizer and RNG state.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / 'InterGATE'
SEEDS = (1234, 2025, 7, 99, 31337)


def verify_fit(path, seed):
    if not (path/'fit.json').exists():
        return False
    fit = json.loads((path/'fit.json').read_text())
    hist = pd.read_csv(path/'history.csv')
    assert fit['model']=='graph' and fit['seed']==seed
    assert fit['n_train']==2970 and fit['n_validation']==743 and fit['external_used_for_fitting'] is False
    assert fit['completed_steps']==hist.step.max() and 6<=fit['best_step']<=fit['completed_steps']
    assert np.isclose(hist.loc[hist.step==fit['best_step'],'val_macro_f1'].item(),fit['best_validation_macro_f1'])
    assert (path/'best.pt').stat().st_size>1000
    return True


def is_training(stage):
    stage.mkdir(exist_ok=True,parents=True)
    with (stage/'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def main():
    base=HERE/'runs'/'graph'; isolated=HERE/'aligned_runs'/'graph'
    workers={}; failures={s:0 for s in SEEDS}
    while True:
        completed=[]
        for seed in SEEDS:
            stage=isolated/str(seed)
            if verify_fit(stage,seed):
                completed.append(seed)
                continue
            proc=workers.get(seed)
            if proc is not None:
                code=proc.poll()
                if code is None:
                    continue
                if code:
                    failures[seed]+=1
                    if failures[seed]>=3:
                        raise RuntimeError(f'Graph worker {seed} failed thrice; inspect aligned_graph_{seed}.log')
                del workers[seed]
            if is_training(stage):
                continue
            device='cuda' if seed==2025 else 'cpu'
            threads=2 if seed==2025 else (4 if seed==1234 else 3)
            cmd=[sys.executable,'-Bu',str(HERE/'train_branches.py'),'graph','fit','--seeds',str(seed),
                 '--threads',str(threads),'--device',device,'--runs-root',str(isolated.parent)]
            print('Recovering graph-only worker',seed,flush=True)
            with (HERE/f'aligned_graph_{seed}.log').open('a') as log:
                workers[seed]=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        if len(completed)==len(SEEDS):
            break
        time.sleep(30)
    for seed in SEEDS:
        staged=isolated/str(seed); dest=base/str(seed)
        if dest.is_dir() and not dest.is_symlink():
            if (dest/'fit.json').exists():
                raise RuntimeError(f'Existing canonical fit for {seed}: manual comparison required')
            preserved=HERE/'superseded_pilot'/str(seed)
            preserved.parent.mkdir(exist_ok=True)
            dest.rename(preserved)
        if not dest.exists():
            dest.symlink_to(os.path.relpath(staged,base),target_is_directory=True)
        assert dest.resolve()==staged.resolve()
    cmd=[sys.executable,'-Bu',str(HERE/'train_branches.py'),'graph','evaluate','--threads','4','--device','cpu']
    with (HERE/'graph_external.log').open('a') as log:
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    with (HERE/'all_branches_summary.log').open('a') as log:
        subprocess.run([sys.executable,'-Bu',str(HERE/'summarize.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    ext=pd.read_csv(PROJECT/'results_v8/07_external_locked/external_predictions.csv.gz')
    rows=[]
    for seed in SEEDS:
        root=base/str(seed)
        new=pd.read_csv(root/'external.csv.gz').sort_values('test_index')
        old=ext[ext.seed.eq(seed)].sort_values('test_index')
        pd.testing.assert_frame_equal(new[['test_index','cohort','y_true']].reset_index(drop=True),
                                      old[['test_index','cohort','y_true']].reset_index(drop=True))
        prob=new.filter(regex='^p_').to_numpy()
        assert prob.shape==(1143,4) and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
        src=PROJECT/f'artifacts/intrinsic4/INTERGATE_V8/seed_{seed}'
        def digest(file): return hashlib.sha256(file.read_bytes()).hexdigest()
        rows.append(dict(seed=seed,external_cases=len(new),
                         ordered_external_indices_sha256=hashlib.sha256(new.test_index.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
                         phase1_sha256=digest(src/'phase1_best_state.pt'),topology_sha256=digest(src/'graph_bundle.pt'),
                         trained_graph_only_sha256=digest(root/'best.pt')))
    pd.DataFrame(rows).to_csv(HERE/'graph_branch_audit.csv',index=False)
    summary=pd.read_csv(HERE/'paired_external.csv')
    assert set(summary.model_or_difference)=={'fusion','graph','tabular','fusion minus graph','fusion minus tabular'}
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
