"""
cells_as_nodes.py
=================

GeneGCN main script — cell-graph formulation.

Cells are placed at the nodes of a kNN similarity graph built in PCA space of
control samples; patient cells are projected through the same control-fit PCA
so the graph topology carries patient-vs-control signal. A two-layer graph
convolutional autoencoder is trained on the control graph only, and per-gene
reconstruction-error deviation (RED) between patient and control cells gives
the gene-level perturbation score.

This file is the headline contribution of:
    Ba Mahel et al. "GeneGCN: A Graph Convolutional Network Uncovers
    Novel Cell-Type-Specific Gene Perturbations and Biomarkers in
    Autism Spectrum Disorder" (2025).

Self-contained — depends only on scanpy, torch, torch_geometric, sklearn,
and the local gene_filter.py module.

Usage
-----
# Subset of cell types (fast)
python cells_as_nodes.py --celltypes ALL_CELLS,IN-SST,Oligodendrocytes,L2_3 --seeds 5

# All 18 cell-type contexts (full benchmark)
python cells_as_nodes.py --celltypes all --seeds 5

# Low-VRAM mode (≤ 2 GB GPU)
python cells_as_nodes.py --celltypes all --seeds 5 --max_cells 3000 --hidden 64
"""

from __future__ import annotations
import argparse, os, time, gc
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wilcoxon as wilcoxon_test

from gene_filter import filter_adata, FilterConfig


# --------------------------------------------------------------------------- #
# CONFIG — edit paths or override on CLI
# --------------------------------------------------------------------------- #
CONTROL_PATH      = "data/ASD_control.h5ad"
PATIENT_PATH      = "data/ASD_patient.h5ad"
META_PATH         = "data/meta.tsv"
SFARI_PATH        = "SFARI-Gene_genes.csv"
SFARI_SYMBOL_COL  = "gene-symbol"
OUT_DIR           = "cellgcn_results"

CELLTYPE_COL      = "cluster"        # column in meta.tsv with cell-type labels
SAMPLE_COL        = "cell"           # column matching adata.obs_names
DIAGNOSIS_COL     = "diagnosis"      # "Control" or "ASD"

# Pipeline hyperparameters
TOP_N_GENES       = 3000             # HVGs per cell type
N_PCS             = 50
K_NEIGHBORS       = 15
HIDDEN_DIM        = 64               # CLI override --hidden
MAX_CELLS         = 20000            # CLI override --max_cells (lower for less VRAM)
MIN_CELLS         = 100              # skip a cell type if either group has fewer cells

# Training
EPOCHS            = 300
LR                = 1e-3
DROPOUT           = 0.0

# Validation
DEPTHS            = [50, 100, 150, 200, 250]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
def data_normalize(adata):
    """log1p(CPM). Idempotent: skips if already log-normalized."""
    if "log1p" in adata.uns:
        return adata
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata


def load_meta(path):
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def attach_celltype_from_meta(adata, meta, group_label):
    """Merge cell-type labels from meta.tsv onto adata.obs."""
    meta = meta.copy()
    if SAMPLE_COL not in meta.columns:
        meta[SAMPLE_COL] = meta.iloc[:, 0]
    if DIAGNOSIS_COL in meta.columns:
        target = "Control" if group_label == "control" else "ASD"
        meta = meta[meta[DIAGNOSIS_COL].astype(str).str.upper().isin(
                    [target.upper(), {"Control": "CTRL"}.get(target, target).upper()])]
    meta_idx = meta.set_index(SAMPLE_COL)
    # Inner join on cell barcode
    common = adata.obs_names.intersection(meta_idx.index)
    adata = adata[common].copy()
    adata.obs[CELLTYPE_COL] = meta_idx.loc[adata.obs_names, CELLTYPE_COL].values
    n_ct = adata.obs[CELLTYPE_COL].nunique()
    print(f"  [{group_label}] {adata.n_obs} cells labelled across {n_ct} cell types.")
    return adata


def select_highly_variable_genes(top_n, ctrl_ad, pat_ad):
    """Combine controls + patients, run HVG selection, return both groups subset."""
    combined = ctrl_ad.concatenate(pat_ad, batch_key="group", batch_categories=["control", "patient"])
    gene_means = np.mean(combined_X, axis=0)
    gene_vars = np.var(combined_X, axis=0)
    dispersion = gene_vars / (gene_means + 1e-8)
    top_indices = np.argsort(dispersion)[-top_n_genes:]
    # Order-preserving HVG gene list
    hvg_genes = [gene_names[i] for i in np.sort(top_indices)]

    # Intersect with patient genes WHILE PRESERVING ORDER (the original used a
    # set() here, which silently scrambled label-to-column alignment).
    patient_gene_set = set(adata_patient.var_names)
    common_genes = [g for g in hvg_genes if g in patient_gene_set]

    adata_control_hvg = adata_control[:, common_genes]
    adata_patient_hvg = adata_patient[:, common_genes]

    assert list(adata_control_hvg.var_names) == list(adata_patient_hvg.var_names), \
        "Gene names do not match between control and patient datasets."
    return adata_control_hvg, adata_patient_hvg, common_genes
   


# --------------------------------------------------------------------------- #
# MODEL
# --------------------------------------------------------------------------- #
class CellGCN(nn.Module):
    """Two-layer GCN autoencoder over a cell-cell graph.

    Input  : [n_cells, n_genes] expression matrix
    Output : [n_cells, n_genes] reconstructed expression
    """
    def __init__(self, n_genes: int, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(n_genes, hidden)
        self.conv2 = GCNConv(hidden, n_genes)
        self.dropout = dropout

    def forward(self, data):
        h = self.conv1(data.x, data.edge_index, data.edge_attr)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, data.edge_index, data.edge_attr)


# --------------------------------------------------------------------------- #
# GRAPH CONSTRUCTION
# --------------------------------------------------------------------------- #
def build_cell_graph(X_cells_by_genes, seed=0, pca_model=None):
    """Symmetric kNN graph in PCA space.

    Returns
    -------
    edge_index : torch.LongTensor of shape [2, n_edges]
    edge_weight: torch.FloatTensor of shape [n_edges]
    pca_model  : fitted sklearn PCA (returned so patients can reuse the control PCA)
    """
    n_cells = X_cells_by_genes.shape[0]
    k = min(K_NEIGHBORS, n_cells - 1)
    n_pcs = min(N_PCS, n_cells - 1, X_cells_by_genes.shape[1] - 1)

    if pca_model is None:
        pca_model = PCA(n_components=n_pcs, random_state=seed)
        Z = pca_model.fit_transform(X_cells_by_genes)
    else:
        Z = pca_model.transform(X_cells_by_genes)

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dists, idxs = nbrs.kneighbors(Z)
    dists, idxs = dists[:, 1:], idxs[:, 1:]      # drop self-loop

    src = np.repeat(np.arange(n_cells), k)
    dst = idxs.flatten()
    d = dists.flatten()
    # Symmetrize
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    d_sym = np.concatenate([d, d])
    # Gaussian-like edge weights normalised by median distance
    med = np.median(d_sym) if np.median(d_sym) > 0 else 1.0
    w = np.exp(-d_sym / med)

    edge_index  = torch.tensor(np.stack([src_sym, dst_sym]), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight, pca_model


# --------------------------------------------------------------------------- #
# TRAINING
# --------------------------------------------------------------------------- #
def train_cell_gcn(data_train, n_genes, device, hidden, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CellGCN(n_genes, hidden=hidden, dropout=DROPOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(data_train)
        loss = F.mse_loss(out, data_train.x)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"    epoch {ep+1:4d}  loss={loss.item():.4f}")
    return model


def _to_device(ctrl_X, pat_X, ei_c, ew_c, ei_p, ew_p, device):
    return (torch.tensor(ctrl_X, dtype=torch.float).to(device),
            torch.tensor(pat_X,  dtype=torch.float).to(device),
            ei_c.to(device), ew_c.to(device),
            ei_p.to(device), ew_p.to(device))


# --------------------------------------------------------------------------- #
# SCORING (with GPU -> CPU OOM fallback)
# --------------------------------------------------------------------------- #
def cells_as_nodes_score(ctrl_X, pat_X, gene_names, seed=0, verbose=False):
    """Returns (gene_names, |RED|, signed_RED). All arrays length = n_genes."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Subsample if too many cells (for low-VRAM GPUs / small models)
    if ctrl_X.shape[0] > MAX_CELLS:
        idx = np.random.RandomState(seed).choice(ctrl_X.shape[0], MAX_CELLS, replace=False)
        ctrl_X = ctrl_X[idx]
    if pat_X.shape[0] > MAX_CELLS:
        idx = np.random.RandomState(seed + 1).choice(pat_X.shape[0], MAX_CELLS, replace=False)
        pat_X = pat_X[idx]

    n_genes = ctrl_X.shape[1]
    if verbose:
        print(f"    ctrl: {ctrl_X.shape[0]} cells, pat: {pat_X.shape[0]} cells, "
              f"genes: {n_genes}, hidden: {HIDDEN_DIM}")

    # Graphs (fit PCA on controls, reuse for patients)
    ei_c, ew_c, pca = build_cell_graph(ctrl_X, seed=seed)
    ei_p, ew_p, _   = build_cell_graph(pat_X, seed=seed, pca_model=pca)

    # Try GPU first; fall back to CPU on OOM
    used_device = DEVICE
    try:
        x_c, x_p, ei_c_d, ew_c_d, ei_p_d, ew_p_d = _to_device(
            ctrl_X, pat_X, ei_c, ew_c, ei_p, ew_p, used_device)
        data_ctrl = Data(x=x_c, edge_index=ei_c_d, edge_attr=ew_c_d)
        data_pat  = Data(x=x_p, edge_index=ei_p_d, edge_attr=ew_p_d)
        model = train_cell_gcn(data_ctrl, n_genes, used_device, HIDDEN_DIM,
                                seed=seed, verbose=verbose)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower(): raise
        print("    [OOM on GPU] falling back to CPU for this cell type.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        used_device = torch.device("cpu")
        x_c, x_p, ei_c_d, ew_c_d, ei_p_d, ew_p_d = _to_device(
            ctrl_X, pat_X, ei_c, ew_c, ei_p, ew_p, used_device)
        data_ctrl = Data(x=x_c, edge_index=ei_c_d, edge_attr=ew_c_d)
        data_pat  = Data(x=x_p, edge_index=ei_p_d, edge_attr=ew_p_d)
        model = train_cell_gcn(data_ctrl, n_genes, used_device, HIDDEN_DIM,
                                seed=seed, verbose=verbose)

    # Inference
    model.eval()
    with torch.no_grad():
        ctrl_recon = model(data_ctrl).cpu().numpy()
        pat_recon  = model(data_pat).cpu().numpy()

    ctrl_err = ((ctrl_recon - ctrl_X) ** 2).mean(axis=0)
    pat_err  = ((pat_recon  - pat_X)  ** 2).mean(axis=0)
    red = pat_err - ctrl_err

    # Direction (independent of RED, from log2FC on raw log1p(CPM))
    ctrl_mean = ctrl_X.mean(axis=0)
    pat_mean  = pat_X.mean(axis=0)
    log2fc = (pat_mean - ctrl_mean) / np.log(2)     # already on log1p scale

    # Cleanup
    del model, data_ctrl, data_pat, x_c, x_p, ei_c_d, ew_c_d, ei_p_d, ew_p_d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gene_names, np.abs(red), red, log2fc


# --------------------------------------------------------------------------- #
# WRAPPER
# --------------------------------------------------------------------------- #
def run_cells_as_nodes(ctrl_ad, pat_ad, seed=0, apply_filter=True, verbose=False):
    """End-to-end: filter → HVG → cell-graph GCN → ranked gene list."""
    if apply_filter:
        ctrl_ad = filter_adata(ctrl_ad, FilterConfig(), write_audit=(seed == 0))
        pat_ad  = pat_ad[:, list(ctrl_ad.var_names)].copy()

    ctrl_ad, pat_ad, _ = select_highly_variable_genes(TOP_N_GENES, ctrl_ad, pat_ad)

    genes = list(ctrl_ad.var_names)
    ctrl_X = ctrl_ad.X.toarray().astype(np.float32) if hasattr(ctrl_ad.X, "toarray") else ctrl_ad.X.astype(np.float32)
    pat_X  = pat_ad.X.toarray().astype(np.float32)  if hasattr(pat_ad.X,  "toarray") else pat_ad.X.astype(np.float32)

    gene_names, abs_red, signed_red, log2fc = cells_as_nodes_score(
        ctrl_X, pat_X, genes, seed=seed, verbose=verbose)

    order = np.argsort(-abs_red)
    ranked = [gene_names[i] for i in order]
    direction = {gene_names[i]: ("Up" if log2fc[i] > 0 else "Down")
                  for i in range(len(gene_names))}
    return ranked, len(genes), direction, signed_red


# --------------------------------------------------------------------------- #
# SFARI VALIDATION
# --------------------------------------------------------------------------- #
def sfari_overlap(ranked, sfari_set, depths=DEPTHS):
    return {k: 100.0 * sum(g.upper() in sfari_set for g in ranked[:k]) / max(k, 1)
            for k in depths}


def load_sfari():
    df = pd.read_csv(SFARI_PATH)
    return set(df[SFARI_SYMBOL_COL].astype(str).str.upper())


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    global MAX_CELLS, HIDDEN_DIM, OUT_DIR
    ap = argparse.ArgumentParser(description="GeneGCN cell-graph pipeline.")
    ap.add_argument("--celltypes", default="ALL_CELLS,IN-SST,Oligodendrocytes,L2_3",
                    help="comma-separated, or 'all' for every cell type in the data")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_cells", type=int, default=MAX_CELLS,
                    help="cap cells per group (lower = less VRAM)")
    ap.add_argument("--hidden", type=int, default=HIDDEN_DIM,
                    help="GCN hidden dim (lower = less VRAM)")
    ap.add_argument("--no_filter", action="store_true",
                    help="skip the gene-symbol filter (for ablation)")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    
    MAX_CELLS  = args.max_cells
    HIDDEN_DIM = args.hidden
    OUT_DIR    = args.out_dir
    print(f"Settings: max_cells={MAX_CELLS}, hidden={HIDDEN_DIM}, device={DEVICE}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    ctrl_full = data_normalize(sc.read(CONTROL_PATH))
    pat_full  = data_normalize(sc.read(PATIENT_PATH))
    meta = load_meta(META_PATH)
    ctrl_full = attach_celltype_from_meta(ctrl_full, meta, "control")
    pat_full  = attach_celltype_from_meta(pat_full,  meta, "patient")

    sfari = load_sfari()
    print(f"SFARI genes loaded: {len(sfari)}")

    if args.celltypes == "all":
        shared = sorted(set(ctrl_full.obs[CELLTYPE_COL].unique()) &
                        set(pat_full.obs[CELLTYPE_COL].unique()))
        cts = ["ALL_CELLS"] + shared
    else:
        cts = args.celltypes.split(",")

    rows: list = []
    ranking_rows: list = []

    for ct in cts:
        print(f"\n=== {ct} ===")
        if ct == "ALL_CELLS":
            ctrl = ctrl_full.copy(); pat = pat_full.copy()
        else:
            ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
            pat  = pat_full[pat_full.obs[CELLTYPE_COL]  == ct].copy()
            if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
                print(f"  SKIP (cells < {MIN_CELLS})"); continue

        for seed in range(args.seeds):
            t0 = time.time()
            try:
                ranked, n_genes, direction, signed_red = run_cells_as_nodes(
                    ctrl.copy(), pat.copy(), seed=seed,
                    apply_filter=(not args.no_filter), verbose=args.verbose)
            except Exception as e:
                print(f"  [skip] seed={seed}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            ov = sfari_overlap(ranked, sfari)
            for k, v in ov.items():
                rows.append({"CellType": ct, "Method": "CELLS_AS_NODES",
                             "Seed": seed, "Depth": k,
                             "SFARI_pct": v, "N_Genes": n_genes})

            # Save top-250 ranking from the first seed of each cell type
            if seed == 0:
                gene_to_red = dict(zip(ranked,
                                        sorted(np.abs(signed_red), reverse=True)[:len(ranked)]))
                for rank, g in enumerate(ranked[:250], 1):
                    ranking_rows.append({
                        "CellType": ct, "Rank": rank, "Gene": g,
                        "Direction": direction.get(g, ""),
                        "SFARI": g.upper() in sfari,
                    })

            print(f"  seed={seed} top50={ov[50]:.1f}% top250={ov[250]:.1f}% "
                  f"({time.time()-t0:.1f}s)")

    # ----- save -----
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_results.csv"), index=False)
    summ = df.groupby(["CellType", "Method", "Depth"])["SFARI_pct"]\
             .agg(["mean", "std", "count"]).reset_index()
    summ.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_summary.csv"), index=False)
    rdf = pd.DataFrame(ranking_rows)
    rdf.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_rankings.csv"), index=False)

    print(f"\nResults   : {OUT_DIR}/cells_as_nodes_results.csv ({len(df)} rows)")
    print(f"Summary   : {OUT_DIR}/cells_as_nodes_summary.csv")
    print(f"Rankings  : {OUT_DIR}/cells_as_nodes_rankings.csv (top-250, seed=0)")

    # ----- optional comparison against prior ablation summary -----
    prior_path = "ablation_results/ablation_summary.csv"
    if os.path.exists(prior_path):
        _compare_with_ablation(summ, prior_path)


def _compare_with_ablation(summ, prior_path):
    """Print paired Wilcoxon comparisons against prior ablation methods."""
    print(f"\nComparing against {prior_path}")
    prior = pd.read_csv(prior_path).rename(columns={"Ablation": "Method"})
    full = pd.concat([prior, summ], ignore_index=True)

    print("\nMean SFARI overlap (%) across cell types, per method × depth:")
    across = full.groupby(["Method", "Depth"])["mean"].mean().reset_index()
    print(across.pivot(index="Method", columns="Depth", values="mean").round(2).to_string())

    pv = full[full.Depth == 250].pivot_table(index="CellType", columns="Method", values="mean")
    if "CELLS_AS_NODES" not in pv.columns:
        return

    print("\nPaired Wilcoxon signed-rank, CELLS_AS_NODES vs others at top-250:")
    for m in pv.columns:
        if m == "CELLS_AS_NODES": continue
        paired = pv[["CELLS_AS_NODES", m]].dropna()
        if len(paired) < 5: continue
        try:
            stat, pval = wilcoxon_test(paired["CELLS_AS_NODES"], paired[m])
            a = paired["CELLS_AS_NODES"].mean()
            b = paired[m].mean()
            sign = "+" if a > b else "-"
            wins = int((paired["CELLS_AS_NODES"] > paired[m]).sum())
            print(f"  CAN vs {m:14s}: W={stat:.2f}  p={pval:.4g}  "
                  f"mean_CAN={a:.2f}%  mean_{m}={b:.2f}%  "
                  f"delta={sign}{abs(a-b):.2f}%  wins={wins}/{len(paired)}")
        except Exception as e:
            print(f"  CAN vs {m}: {e}")


if __name__ == "__main__":
    main()
