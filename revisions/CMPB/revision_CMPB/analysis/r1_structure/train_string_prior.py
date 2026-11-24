"""Train the submitted model on the human STRING v12.0 physical prior.

Preflight is CPU-compatible. In queue mode the primary fifteen additional
fits must finish before the single GPU worker can start this five-seed arm.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "InterGATE"
OUT = HERE / "string_training"
OUT.mkdir(exist_ok=True)
os.environ["INTERGATE_CACHE_ROOT"] = str(OUT / "cache")
os.environ["INTERGATE_ARTIFACTS_ROOT"] = str(OUT / "artifacts")
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import torch
from intergate.config import CFG, model_config_snapshot
from intergate.runtime_setup import prepare_runtime, assert_transport_protocol
from intergate.ablation import CFG_FULL, cfg_to_dict, register_runtime, resolve_class_keep_ratio, build_model_from_cfg
from intergate.biology import DifferentialEvidenceConfig, get_or_build_differential_evidence
from intergate.biology_validation import candidate_edge_table
from intergate.graph import dedup_edges
from string_prior import read_resource
from intergate.graph_cache import get_or_build_Xh
from intergate.transport import StructuralBudget, TransportConfig, run_transport_seed, assert_transport_runtime

RUN_NAME = "INTERGATE_STRING_PHYSICAL"
K = 529

def atomic(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str)+"\n")
    tmp.replace(path)

def wait_for_primary():
    paths = HERE/"training/artifacts/INTERGATE_REVISION_SEEDS"
    for seed in range(10001,10016):
        directory = paths/f"seed_{seed}"
        while not all((directory/name).is_file() for name in ("revision_result.json","transport_checkpoint_COMPLETE.txt")):
            time.sleep(45)
        if not json.loads((directory/"meta.json").read_text())["transport_protocol"]["completed"]:
            raise RuntimeError(f"Primary training record incomplete at {seed}")
    # Serialise the two complete GPU arms through the primary writer's lock.
    guard = (HERE/"training/writer.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    return guard

def prepare(source):
    files = json.loads((HERE/"string_manifest.json").read_text())["sha256"]
    for name,digest in files.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Public resource differs from audited STRING input: {name}")
    ref_path = PROJECT/"artifacts/intrinsic4/INTERGATE_V8/seed_1234/meta.json"
    ref = json.loads(ref_path.read_text())
    assert_transport_protocol(CFG)
    cache = Path(CFG.PIPELINE_CACHE_DIR)
    for original in (PROJECT/"cache/pipeline_cache").iterdir():
        if not original.is_file() or original.name.endswith("ff3f4ef6c1949632.npz"):
            continue
        target = cache/original.name
        if not target.exists(): shutil.copyfile(original,target)
    # Reconstruct the reference gene and patient matrices to verify the full
    # original content hash before replacing graph-dependent inputs.
    base = prepare_runtime(CFG, include_external=False)
    baseline_cfg = cfg_to_dict(CFG_FULL)
    for field in ("implementation_sha256","model_config_sha256","dataset_signature","splits"):
        if baseline_cfg[field]!=ref[field]:
            raise RuntimeError(f"Baseline reconstruction mismatch: {field}")
    if base["external_opened"] or len(base["test_idx"]): raise RuntimeError("External rows in training runtime")
    genes = base["genes"]
    assert len(genes)==7131
    original_config = model_config_snapshot(CFG)
    CFG.GRAPH_SOURCE = "string"
    # The ordered gene input was already fixed from the validated reference
    # run. Do not shrink the set of measured tabular features under STRING.
    CFG.CONNECTED_ONLY = False
    CFG.CLASS_GRAPH_TARGET_EDGES = K
    CFG.STRING_INFO_FILE = source/"9606.protein.info.v12.0.txt.gz"
    CFG.STRING_LINKS_FILE = source/"9606.protein.physical.links.v12.0.txt.gz"
    CFG.STRING_ALIAS_FILE = None
    CFG.STRING_CACHE_DIR = cache/"string_physical"
    CFG.refresh_options()
    alt_config = model_config_snapshot(CFG)
    delta = {key:[original_config[key],alt_config[key]] for key in original_config if original_config[key]!=alt_config[key]}
    allowed = {"GRAPH_SOURCE","USE_HURI","USE_OMNIPATH","USE_STRING","CLASS_GRAPH_TARGET_EDGES","CONNECTED_ONLY"}
    if set(delta)!=allowed: raise RuntimeError(f"Unexpected model configuration changes: {delta}")
    # STRING preferred names are mapped without synonym or first-token
    # expansion, identically to the independent comparison. The original
    # deduplicator returns lexicographically ordered directed candidates.
    table = read_resource(source, genes, save_manifest=False)
    high = table.loc[table.string_score>=700]
    ij = high[["i","j"]].to_numpy(dtype=np.int64).T
    weight = high.string_score.to_numpy(dtype=np.float32)/1000
    ei,ew,et = dedup_edges(np.hstack([ij,ij[::-1]]),np.r_[weight,weight],
                           np.full(2*len(high),CFG.STRING_EDGE_TYPE,dtype=np.int64))
    if (ei[0]==ei[1]).any() or np.unique(et).tolist()!=[3]:
        raise RuntimeError("Unexpected loops or types in physical network")
    directed_codes = ei[0]*len(genes)+ei[1]
    reverse_codes = ei[1]*len(genes)+ei[0]
    if not np.array_equal(directed_codes,np.sort(reverse_codes)) or not np.array_equal(directed_codes,np.unique(directed_codes)):
        raise RuntimeError("The directed arc list must be unique, ordered by endpoints and reciprocal")
    if ei.shape[1]!=56894 or np.unique(ei).size!=5067:
        raise RuntimeError("Runtime graph differs from the independently audited STRING graph")
    alt_cache=cache/"string_physical"; alt_cache.mkdir(exist_ok=True)
    bio_cfg = DifferentialEvidenceConfig(min_samples_per_group=CFG.BIOLOGY_MIN_SAMPLES_PER_GROUP,
        min_cohorts=CFG.BIOLOGY_MIN_COHORTS,edge_batch_size=CFG.BIOLOGY_EDGE_BATCH_SIZE,
        meta_z_clip=CFG.BIOLOGY_META_Z_CLIP,heterogeneity_penalty=CFG.BIOLOGY_HETEROGENEITY_PENALTY)
    evidence = get_or_build_differential_evidence(cache_path=alt_cache/"physical_context.npz",
        X=base["X_scaled"],genes=genes,y=base["y"],cohort=base["cohort"],train_idx=base["train_idx"],
        edge_index=ei,class_names=base["classes"],cohort_order=CFG.TRAIN_COHORTS,config=bio_cfg)
    xh,_ = get_or_build_Xh(base["X_scaled"],genes,ei,cache_dir=alt_cache,stats=CFG.REG_STATS,
                           min_targets=CFG.REG_MIN_GENES,max_regulators=CFG.REG_MAX_REGULATORS)
    edge_tensor=torch.as_tensor(ei,dtype=torch.long,device=CFG.DEVICE)
    register_runtime(X_h=xh,edge_index_t=edge_tensor,
        edge_weight_t=torch.as_tensor(ew,dtype=torch.float32,device=CFG.DEVICE),
        edge_type_t=torch.as_tensor(et,dtype=torch.long,device=CFG.DEVICE),
        biology_evidence=evidence["context_score"],biology_evidence_bundle=evidence)
    assert_transport_runtime(base["cohort"],base["train_idx"],base["val_idx"],base["test_idx"],CFG)
    snapshot=cfg_to_dict(CFG_FULL)
    if snapshot["splits"]!=ref["splits"]: raise RuntimeError("Sample split changed")
    for key,value in ref["dataset_signature"].items():
        if key!="content_sha256" and snapshot["dataset_signature"][key]!=value:
            raise RuntimeError(f"Unexpected patient or gene change: {key}")
    if snapshot["dataset_signature"]["content_sha256"]==ref["dataset_signature"]["content_sha256"]:
        raise RuntimeError("Alternative-prior content unexpectedly equals reference")
    kappa=resolve_class_keep_ratio(edge_tensor,CFG.KEEP_MIN)
    budget=StructuralBudget.from_core(SimpleNamespace(edge_index=edge_tensor),kappa)
    if budget.paired or budget.k_units!=K or budget.first_ids.numel()!=56894:
        raise RuntimeError("The trainer did not retain the requested 529 directed structural units")
    candidate_edge_table(ei,ew,et,genes,n_degree_bins=10).to_csv(OUT/"candidates.csv.gz",index=False)
    # Build the model and run actual structural logits on one training sample.
    model=build_model_from_cfg(CFG_FULL,None)
    model.eval()
    sample=[base["train_idx"][0]]
    with torch.no_grad():
        xx=torch.as_tensor(base["X_scaled"][sample],device=CFG.DEVICE)
        x0=torch.zeros((1,xh.shape[1]),dtype=torch.float32,device=CFG.DEVICE)
        output=model(xx,adj=None,x_graph=x0)
        if isinstance(output,tuple): output=output[0]
        if tuple(output.shape)!=(1,4) or not torch.isfinite(output).all():
            raise RuntimeError("Invalid model logits on the development preflight sample")
    manifest = {"protocol":CFG.TRANSPORT_PROTOCOL_VERSION,"review_time":True,"raw_external_rows_opened":False,
        "model_changes":delta,"seeds":list(CFG.STABILITY_SEEDS),"n_train":len(base["train_idx"]),
        "n_validation":len(base["val_idx"]),"n_external":len(base["test_idx"]),"measured_genes":len(genes),
        "connected_STRING_genes":int(np.unique(ei).size),"eligible_directed_edges":ei.shape[1],
        "unique_physical_pairs":ei.shape[1]//2,"target_directed_edges":K,"effective_kappa":kappa,
        "reciprocal_order":"full directed list sorted by source then target; one arc per eligible structural unit",
        "certainty":"directional arcs are computational orientations of undirected physical associations",
        "STRING_input_sha256":files,"STRING_minimum_score":700,"edge_types":np.unique(et).tolist(),
        "source_weights":"combined_score / 1000","implementation":snapshot["implementation_sha256"],
        "model_configuration":snapshot["model_config_sha256"],"dataset":snapshot["dataset_signature"],
        "original_model_configuration":ref["model_config_sha256"],"original_dataset":ref["dataset_signature"],
        "graph_feature_dimension":xh.shape[1],"forward_check_dimensions":list(output.shape)}
    # CPU/GPU configurations must render identical statistical inputs.
    path=OUT/"string_training_contract.json"
    if path.is_file() and json.loads(path.read_text())!=manifest:
        raise RuntimeError("Existing STRING arm contract differs from actual runtime")
    atomic(path,manifest)
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["preflight","queued-training"])
    parser.add_argument("source",type=Path)
    args=parser.parse_args()
    if args.mode=="queued-training":
        lock=(OUT/"worker.lock").open("a")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print("Queued to follow the fifteen additional primary-prior seeds.",flush=True)
        primary_guard=wait_for_primary()
        if not torch.cuda.is_available(): raise RuntimeError("GPU unavailable when STRING arm became ready")
    manifest=prepare(args.source.resolve())
    print(json.dumps(manifest,indent=2),flush=True)
    if args.mode=="preflight": return
    transport=TransportConfig.from_global(CFG)
    for seed in CFG.STABILITY_SEEDS:
        start=time.monotonic()
        row=run_transport_seed(int(seed),save_root=CFG.ARTIFACTS_ROOT,transport=transport,
                               artifact_name=RUN_NAME,resume=True,force=False)
        folder=Path(CFG.ARTIFACTS_ROOT)/RUN_NAME/f"seed_{seed}"
        if not (folder/"transport_checkpoint_COMPLETE.txt").is_file() or row["transport_all_parameter_deltas_zero"]:
            raise RuntimeError(f"STRING seed {seed} incomplete or inactive")
        atomic(folder/"revision_result.json",{"seed":int(seed),"metrics":row,"current_invocation_seconds":time.monotonic()-start})
    from analyze_string_fits import summarize
    summarize()
if __name__=="__main__": main()
