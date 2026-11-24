"""Re-fit the published GeSubNet modules on the prespecified common domain.

Run prepare once with a fresh working directory, then fit once per protocol seed.
All study files are read from --study. The upstream source is read from --upstream.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd


PROTOCOL = json.loads(Path(__file__).with_name("gesubnet_protocol.json").read_text())


def read_expression(path, genes):
    """Select named rows without loading the large transcriptome into memory."""
    symbols = {s.encode() for s in genes}
    with path.open("rb") as f:
        header = next(f)
        rows = [line for line in f if line.split(b",", 1)[0].strip(b'"') in symbols]
    frame = pd.read_csv(io.BytesIO(header + b"".join(rows)), index_col=0)
    assert set(frame.index) == set(genes) and frame.index.is_unique
    return frame.loc[genes].T


def verify_revision(repo):
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != PROTOCOL["git_revision"]:
        raise RuntimeError("The comparison requires the recorded upstream revision")


def prepare(study, upstream, work):
    verify_revision(upstream)
    with zipfile.ZipFile(upstream / "data/BRCA.zip") as archive:
        rows = csv.reader(io.TextIOWrapper(archive.open("BRCA_node.csv")))
        next(rows)
        symbols = {row[0] for row in rows}
    prior = pd.read_csv(study / "results_v8/05_biology_validation/candidate_prior_for_null.csv.gz")
    common = prior[prior.source_gene.isin(symbols) & prior.target_gene.isin(symbols)].copy()
    pairs = sorted({tuple(sorted(pair)) for pair in zip(common.source_gene, common.target_gene)})
    genes = sorted({g for pair in pairs for g in pair})
    gene_id = {g: i for i, g in enumerate(genes)}
    edges = np.array([[gene_id[g] for g in pair] for pair in pairs], dtype=int)
    meta = json.loads((study / "artifacts/intrinsic4/INTERGATE_V8/seed_1234/meta.json").read_text())
    labels = meta["label_map"]
    train = meta["splits"]["train_idx"]
    folder = study / "data_prepare/output_mcombat"
    df = read_expression(folder / "expr_train_mcombat.csv", genes)
    metadata = pd.read_csv(folder / "meta_train.csv").set_index("sample").loc[df.index]
    y = metadata.label.map(labels).to_numpy(np.int64)
    cohorts = metadata.dataset.to_numpy(str)
    val = pd.read_csv(study / "results_v8/07b_calibration_posthoc/development_validation_predictions_by_seed.csv.gz")
    val = val[val.seed == 1234]
    assert (y[val.sample_index] == val.y_true).all() and (cohorts[val.sample_index] == val.cohort).all()
    assert set(train).isdisjoint(val.sample_index) and len(train) == 2970
    x_train = df.iloc[train].to_numpy(np.float32)
    x_train = (x_train - x_train.mean(axis=0)) / np.maximum(x_train.std(axis=0), 1e-8)
    assert np.isfinite(x_train).all()
    work.mkdir(parents=True, exist_ok=True)
    np.savez(work / "train.npz", X=x_train, y=y[train], edges=edges,
             genes=np.array(genes), basal=labels["Basal-like"])


def fit(study, upstream, work, seed):
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    from torch_geometric.data import Data
    from torch_geometric.transforms import RandomLinkSplit
    from torch_geometric.utils import negative_sampling, to_undirected
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.sparse import csr_matrix

    verify_revision(upstream)
    sys.path.insert(0, str(upstream))
    import model.SELM_NN as patient_module
    from model.NeoGNN import NeoGNN
    patient_module.rearrange = rearrange  # required by the released module
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    data = np.load(work / "train.npz")
    x = torch.from_numpy(data["X"])
    y = torch.from_numpy(data["y"])
    n_samples, n_genes = x.shape
    patient = patient_module.SELM(n_genes, 32, 256, 32, 32, 1, 0.0)
    patient._vq_vae.to_cls[0] = torch.nn.Linear(32 * 32, int(torch.unique(y).numel()))
    optimizer = torch.optim.Adam(patient.parameters(), lr=1e-4, amsgrad=True)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    for _ in range(100):
        for batch, targets in loader:
            loss_vq, reconstruction, _, _, logits, _ = patient(batch)
            loss = F.mse_loss(reconstruction, batch) + F.cross_entropy(logits, targets) + loss_vq
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    patient.eval()
    basal = y == int(data["basal"])
    with torch.no_grad():
        # Continuous patient encoder output z is the released 32-dimensional
        # representation compatible with the graph embedding and Eq. 3.
        z_patient = patient._encoder(x[basal]).T.contiguous()
    source_edges = torch.from_numpy(data["edges"].T.copy())
    all_edges = to_undirected(source_edges, num_nodes=n_genes)
    bootstrap_features = torch.randperm(n_samples)[:int(basal.sum())]
    train_data, _, _ = RandomLinkSplit(num_val=0.05, num_test=0.05, is_undirected=True,
                                        add_negative_train_samples=False)(
        Data(x=x[bootstrap_features].T.contiguous(), edge_index=all_edges))
    graph = NeoGNN(int(basal.sum()), 64, 32)
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.01)
    # Use only the training prior in Stage 2 for both feature paths.
    train_adj = csr_matrix((np.ones(train_data.edge_index.shape[1]), train_data.edge_index.numpy()),
                           shape=(n_genes, n_genes))
    for _ in range(200):
        z = graph.encode(train_data.x, train_data.edge_index)
        pos = train_data.edge_label_index
        neg = negative_sampling(edge_index=all_edges, num_nodes=n_genes, num_neg_samples=pos.shape[1])
        links = torch.cat((pos, neg), dim=1)
        target = torch.cat((torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])))
        logit = graph.decode(z, links, train_adj).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logit, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    optimizer = torch.optim.Adam(graph.convs.parameters(), lr=0.01)
    x_basal = x[basal].T.contiguous()
    for _ in range(200):
        z = graph.encode(x_basal, all_edges)
        loss = F.mse_loss(z @ z_patient, x_basal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    full_adj = csr_matrix((np.ones(all_edges.shape[1]), all_edges.numpy()), shape=(n_genes, n_genes))
    graph.eval()
    with torch.no_grad():
        z = graph.encode(x_basal, all_edges)
        # The published NeoGNN decoder broadcasts the structural contribution
        # from the first scored pair over the whole batch. All pairs are passed
        # together in fixed alphabetical order to preserve its exact ranking.
        score = graph.decode(z, source_edges, full_adj).reshape(-1).numpy()
    assert np.isfinite(score).all() and score.shape == (source_edges.shape[1],)
    table = pd.DataFrame({"gene_a": data["genes"][data["edges"][:, 0]],
                          "gene_b": data["genes"][data["edges"][:, 1]],
                          "gesubnet_logit": score})
    table.to_csv(work / f"gesubnet_seed_{seed}.csv", index=False)
    print(f"Completed GeSubNet seed {seed}: {n_genes} genes, {len(table)} pairs", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "fit"])
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=PROTOCOL["training_seeds"], default=1234)
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.study, args.upstream, args.work)
    else:
        fit(args.study, args.upstream, args.work, args.seed)


if __name__ == "__main__":
    main()
