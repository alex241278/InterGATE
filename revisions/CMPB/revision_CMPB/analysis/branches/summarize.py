"""Evaluate five-seed ensembles on identical patients, with paired conditional bootstrap intervals."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "InterGATE"
SEEDS = (1234,2025,7,99,31337)
PROBS = ["p_"+s for s in ("LumA","LumB","HER2-enriched","Basal-like")]


def ensemble(frame):
    assert not frame.duplicated(["test_index","seed"]).any()
    group=frame.groupby("test_index",sort=True)
    assert group.seed.nunique().eq(5).all() and group.y_true.nunique().eq(1).all() and group.cohort.nunique().eq(1).all()
    summary=group[PROBS].mean()
    assert len(summary)==1143 and np.allclose(summary.sum(axis=1),1,atol=1e-6)
    return group[["cohort","y_true"]].first(),summary.to_numpy().argmax(1)


def metrics(truth,pred,draw):
    code=truth[draw]*4+pred[draw]
    cm=np.bincount(code,minlength=16).reshape(4,4)
    f1=2*np.diag(cm)/(cm.sum(0)+cm.sum(1))
    return np.array([f1.mean(),f1[3]])


def main():
    patients,pred=ensemble(pd.read_csv(PROJECT/"results_v8/07_external_locked/external_predictions.csv.gz"))
    models={"fusion":pred}
    for name in ("tabular","graph"):
        files=[HERE/f"runs/{name}/{s}/external.csv.gz" for s in SEEDS]
        if not all(f.exists() for f in files):
            continue
        stack=pd.concat([pd.read_csv(f).assign(seed=s) for s,f in zip(SEEDS,files)])
        rows,models[name]=ensemble(stack)
        pd.testing.assert_frame_equal(patients,rows)
    pool=patients.reset_index(drop=True)
    strata=list(pool.groupby(["cohort","y_true"]).indices.values())
    rng=np.random.default_rng(20261123)
    draws=[np.arange(len(pool))]+[np.concatenate([rng.choice(idx,len(idx),replace=True) for idx in strata]) for _ in range(2000)]
    scores={m:np.stack([metrics(pool.y_true.to_numpy(),p,d) for d in draws]) for m,p in models.items()}
    for name in list(models)[1:]:
        scores["fusion minus "+name]=scores["fusion"]-scores[name]
    records=[]
    for name,v in scores.items():
        for j,metric in enumerate(("macro_F1","Basal_like_F1")):
            a,b=np.quantile(v[1:,j],[.025,.975])
            records.append(dict(model_or_difference=name,metric=metric,estimate=v[0,j],CI025=a,CI975=b,n=1143))
    out=pd.DataFrame(records)
    out.to_csv(HERE/"paired_external.csv.tmp",index=False)
    (HERE/"paired_external.csv.tmp").replace(HERE/"paired_external.csv")
    print(out.to_string(index=False))

if __name__=="__main__":
    main()
