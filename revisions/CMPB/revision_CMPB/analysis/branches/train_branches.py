"""Refit the published tabular trunk or the fixed predictive GNN without the other branch."""
import argparse
import hashlib
import fcntl
import random
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "InterGATE"
sys.path.insert(0, str(PROJECT))
os.environ["INTERGATE_CACHE_ROOT"] = tempfile.mkdtemp(prefix="intergate_branch_cache_")
from intergate.ablation import build_model_from_cfg, CFG_FULL
from intergate.config import CFG
from intergate.model import _make_mlp, build_pruned_model_from
from intergate.runtime_setup import prepare_runtime
from intergate.training import train_one_epoch, predict_proba_xh_mode
from intergate.transport import _class_weight_loss, _make_loader
from intergate.utils import set_all_seeds

SEEDS = (1234, 2025, 7, 99, 31337)


class Tabular(nn.Module):
    """Published full-gene trunk, input dropout and native tabular four-class head."""
    def __init__(self, n):
        super().__init__()
        self.tab_drop = nn.Dropout(CFG.HYBRID_TAB_DROPOUT)
        self.tab_trunk = _make_mlp(n, 256, 256, CFG.HYBRID_TAB_LAYERS, CFG.HYBRID_TAB_DROPOUT)
        self.tab_head4 = nn.Linear(256, 4)

    def forward(self, x, adj=None, x_graph=None, **unused):
        return self.tab_head4(self.tab_trunk(self.tab_drop(x))), None, None


def get_model(mode, seed, runtime, weights):
    if mode == "tabular":
        return Tabular(len(runtime["genes"])).to(CFG.DEVICE), runtime["X_scaled"]
    source = PROJECT / f"artifacts/intrinsic4/INTERGATE_V8/seed_{seed}"
    bundle = torch.load(source/"graph_bundle.pt", map_location="cpu", weights_only=False)
    if list(bundle["genes_full"]) != list(runtime["genes"]):
        raise ValueError("Inconsistent feature order")
    dense = build_model_from_cfg(CFG_FULL, weights)
    dense.load_state_dict(torch.load(source/"phase1_best_state.pt", map_location=CFG.DEVICE, weights_only=True))
    nodes = bundle["nodes_used_full"]
    model = build_pruned_model_from(dense, bundle["edge_index_comp"], bundle["edge_weight_comp"],
        bundle["edge_type_comp"], num_nodes=len(nodes), hidden=CFG.HIDDEN, n_classes=4,
        graph_feat_dim=runtime["X_h"].shape[1], num_layers=CFG.NUM_LAYERS,
        dropout=CFG.DROPOUT, num_heads=CFG.POOL_HEADS, xgraph_dropout=CFG.XGRAPH_DROPOUT,
        block_use_self=CFG.BLOCK_USE_SELF, block_residual=CFG.BLOCK_RESIDUAL,
        device=CFG.DEVICE, class_edge_ids=bundle["class_edge_ids_comp"],
        class_names=runtime["classes"], per_label_graphs=True,
        use_hybrid=False, use_node_identity=True, nodes_used_full=nodes)
    return model, np.ascontiguousarray(runtime["X_scaled"][:,nodes])


def run_one(mode, seed, runtime, stage, output):
    output.mkdir(exist_ok=True, parents=True)
    checkpoint = output / "best.pt"
    y = runtime["y"]
    train, valid = runtime["train_idx"], runtime["val_idx"]
    set_all_seeds(seed)
    loss, weights = _class_weight_loss(y, train, 4, CFG.DEVICE)
    model, x = get_model(mode, seed, runtime, weights)
    xh = runtime["X_h"] if mode == "graph" else np.zeros((len(y),1), dtype=np.float32)
    def loader(indices, batch_size, shuffle, offset):
        return _make_loader(x, xh, y, indices, batch_size=batch_size, shuffle=shuffle, seed=seed+offset)
    if stage == "evaluate":
        if not (output / "fit.json").exists():
            raise RuntimeError("A completed training run is required")
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=CFG.DEVICE))
        verification = {}
        if mode == "graph":
            model.assert_fixed_graph_phase2()
            source_graph = torch.load(PROJECT/f"artifacts/intrinsic4/INTERGATE_V8/seed_{seed}/graph_bundle.pt",
                                      map_location="cpu",weights_only=False)
            assert torch.equal(model.edge_index.cpu(),source_graph["edge_index_comp"].cpu())
            sizes=[]
            for cid, expected in enumerate(source_graph["class_edge_ids_comp"]):
                retained=getattr(model,f"fixed_class_edge_ids_{cid}").cpu()
                assert torch.equal(retained,torch.as_tensor(expected).cpu()) and retained.numel()==529
                sizes.append(retained.numel())
            assert not any(name.startswith(("tab_trunk", "tab_head", "fusion")) for name,_ in model.named_parameters())
            verification={"fixed_edges_per_subtype":sizes,"direct_tabular_branch":False,"n_regulator_features":xh.shape[1]}
        vp, vy = predict_proba_xh_mode(model,None,loader(valid,8,False,930000),CFG.DEVICE)
        assert np.array_equal(vy,y[valid])
        archived = json.loads((output/"fit.json").read_text())["best_validation_macro_f1"]
        replayed = float(f1_score(vy,vp.argmax(axis=1),average="macro"))
        if abs(replayed-archived) > 0.003:
            raise RuntimeError(f"Frozen validation checkpoint reconciliation failed: {archived} vs {replayed}")
        (output/"validation_replay.json").write_text(json.dumps(dict(fit=archived,replay=replayed,
            n_validation=len(valid),evaluation_device=str(CFG.DEVICE),**verification),indent=2))
        ids = runtime["test_idx"]
        proba, truth = predict_proba_xh_mode(model, None, loader(ids,8,False,930000),CFG.DEVICE)
        assert np.array_equal(truth, y[ids])
        result = pd.DataFrame({"test_index":ids, "cohort":runtime["cohort"][ids], "y_true":truth})
        result[["p_"+name for name in runtime["classes"]]] = proba
        result.to_csv(output/"external.tmp", index=False, compression="gzip")
        (output/"external.tmp").replace(output/"external.csv.gz")
        return
    if (output / "fit.json").exists():
        return
    assert len(runtime["test_idx"]) == 0
    graph = mode == "graph"
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0004 if graph else .0008,weight_decay=.0001)
    tr, va = loader(train,8,True,920000), loader(valid,8,False,930000)
    history = []
    best = -1
    best_step = 0
    start = 1
    stale = 0
    warmup = 5 if graph else 0
    last = output / "resume.pt"
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best, best_step, stale = (state[k] for k in ("history", "best", "best_step", "stale"))
        start = history[-1]["step"]+1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"])
        tr.generator.set_state(state["train_rng"])
        va.generator.set_state(state["val_rng"])
    step = start-1
    for step in range(start, warmup+81) if stale < 20 else ():
        warming = step <= warmup
        score_loss = train_one_epoch(model, None, tr, optimizer, CFG.DEVICE, loss_fn=loss,
            use_mixup=not warming, mixup_alpha=.15, mixup_prob=.30, accum_steps=20,
            structural_regularizers=False, xh_mode="zero" if warming else "orig",
            expert_lambda=.2 if graph and not warming else 0)
        p, truth = predict_proba_xh_mode(model,None,va,CFG.DEVICE,xh_mode="zero" if warming else "orig")
        score = f1_score(truth,p.argmax(axis=1),average="macro")
        history.append(dict(step=step,warmup=warming,loss=score_loss,val_macro_f1=score))
        pd.DataFrame(history).to_csv(output/"history.csv",index=False)
        print(mode, seed, step, score,flush=True)
        if not warming and score > best+.0001:
            best, best_step, stale = score, step, 0
            torch.save(model.state_dict(),output/"best.tmp")
            (output/"best.tmp").replace(checkpoint)
        elif not warming:
            stale += 1
        if graph:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                best=best, best_step=best_step, stale=stale, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if CFG.DEVICE.type=="cuda" else None,
                train_rng=tr.generator.get_state(), val_rng=va.generator.get_state()), output/"resume.tmp")
            (output/"resume.tmp").replace(last)
        if stale >= 20:
            break
    (output/"fit.tmp").write_text(json.dumps(dict(seed=seed,model=mode,device=str(CFG.DEVICE),
        best_validation_macro_f1=best,best_step=best_step,completed_steps=step,
        n_train=len(train),n_validation=len(valid),external_used_for_fitting=False),indent=2))
    (output/"fit.tmp").replace(output/"fit.json")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tabular","graph"))
    parser.add_argument("stage", choices=("fit","evaluate"))
    parser.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS))
    parser.add_argument("--threads",type=int,default=3)
    parser.add_argument("--runs-root",type=Path,default=HERE/"runs")
    parser.add_argument("--device",choices=("cpu","cuda"))
    args=parser.parse_args()
    if args.device:
        CFG.DEVICE = torch.device(args.device)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_per_process_memory_fraction(.42)
    if any(s not in SEEDS for s in args.seeds):
        raise ValueError("Unknown seed")
    torch.set_num_threads(args.threads)
    try:
        for f in (PROJECT/"cache/pipeline_cache").iterdir():
            if f.is_file() and (args.stage=="evaluate" or "ff3f4e" not in f.name):
                shutil.copyfile(f, Path(CFG.PIPELINE_CACHE_DIR)/f.name)
        runtime=prepare_runtime(CFG,include_external=args.stage=="evaluate")
        split = json.loads((PROJECT/"results_v8/08a_paired_predictive_benchmark/development_benchmark_manifest.json").read_text())
        for key, label in [("train_idx","train_index_sha256"),("val_idx","validation_index_sha256")]:
            actual = hashlib.sha256(np.asarray(runtime[key], dtype="int64").tobytes()).hexdigest()
            if actual != split[label]:
                raise RuntimeError(f"Archived split mismatch for {key}")
        for s in args.seeds:
            output = args.runs_root/args.mode/str(s)
            output.mkdir(parents=True, exist_ok=True)
            with (output/"worker.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                run_one(args.mode,s,runtime,args.stage,output)
    finally:
        shutil.rmtree(os.environ["INTERGATE_CACHE_ROOT"])

if __name__ == "__main__":
    main()
