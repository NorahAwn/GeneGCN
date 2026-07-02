"""
run_ablations.py
================

Benchmark driver for the GeneGCN ablation study (Table 1 / Figure 3 of the
paper). Compares five gene-graph variants and a Wilcoxon baseline on the
same filtered HVG universe:

    FULL          : gene-node GCN + 64-quantile pool + filter   (the "GeneGCN(gene)" variant)
    NO_GRAPH      : MLP autoencoder, same shape, no graph       (Does the graph help?)
    MEAN_POOL     : gene-node GCN + mean/std pool               (Does quantile pooling help?)
    NO_FILTER     : gene-node GCN, no gene-symbol filter        (Does filtering help?)
    WILCOXON_FAIR : Scanpy rank_genes_groups, same gene universe (fair classical baseline)

Run cells_as_nodes.py separately for the headline cell-graph method; this
script auto-merges results when you run cells_as_nodes.py afterwards.

Usage
-----
python run_ablations.py --celltypes all --seeds 5
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

from scipy.stats import wilcoxon as wilcoxon_test

from gene_filter import filter_adata, FilterConfig

# Reuse data loading from the main script
from cells_as_nodes import (
    data_normalize, load_meta, attach_celltype_from_meta,
    select_highly_variable_genes,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    SFARI_PATH, SFARI_SYMBOL_COL, DEPTHS, MIN_CELLS,
    TOP_N_GENES, EPOCHS, LR, DROPOUT, DEVICE,
    sfari_overlap, load_sfari,
)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
OUT_DIR        = "ablation_results"
CORR_THRESHOLD = 0.3       # gene-gene co-expression threshold for FULL/MEAN_POOL/NO_FILTER
N_POOL         = 64        # quantile bins for the gene-node pooling


# --------------------------------------------------------------------------- #
# MODELS
# --------------------------------------------------------------------------- #
class GeneGCN(nn.Module):
    """Original gene-node GCN (input → 64 → input). Used in FULL / MEAN_POOL / NO_FILTER."""
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv3 = GCNConv(64, in_channels)
        self.dropout = dropout
    def forward(self, data):
        h = self.conv1(data.x, data.edge_index, data.edge_attr)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv3(h, data.edge_index, data.edge_attr)


class GeneMLP(nn.Module):
    """Same-shape MLP, no graph structure. Used in NO_GRAPH."""
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, in_channels)
        self.dropout = dropout
    def forward(self, data):
        h = torch.relu(self.fc1(data.x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc2(h)


# --------------------------------------------------------------------------- #
# GRAPH + POOLING
# --------------------------------------------------------------------------- #
def build_gene_graph(expr_df, thresh=CORR_THRESHOLD):
    """Gene-gene Pearson correlation graph. Edges |r| > thresh, upper triangle."""
    X = expr_df.values
    n_genes = X.shape[1]
    # column-wise correlation
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0.0)
    iu = np.triu_indices(n_genes, k=1)
    keep = np.abs(corr[iu]) >= thresh
    src = iu[0][keep]; dst = iu[1][keep]
    w = np.abs(corr[src, dst]).astype(np.float32)
    # Symmetrize
    edge_index = torch.tensor(np.stack([np.concatenate([src, dst]),
                                         np.concatenate([dst, src])]), dtype=torch.long)
    edge_weight = torch.tensor(np.concatenate([w, w]), dtype=torch.float)
    return edge_index, edge_weight


def pool_quantiles(expr_df, n_pool=N_POOL):
    qs = np.linspace(0, 1, n_pool)
    X = expr_df.values                            # [cells, genes]
    return np.quantile(X, qs, axis=0).T.astype(np.float32)   # [genes, n_pool]


def pool_mean_std(expr_df):
    X = expr_df.values
    return np.stack([X.mean(0), X.std(0)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# SCORERS
# --------------------------------------------------------------------------- #
def train_one(model, data_train, epochs=EPOCHS, lr=LR):
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data_train)
        loss = F.mse_loss(out, data_train.x)
        loss.backward()
        opt.step()
    return model


def score_gene_gcn(ctrl_df, pat_df, n_pool, seed, use_graph=True):
    torch.manual_seed(seed); np.random.seed(seed)

    if n_pool == 2:
        ctrl_prof = pool_mean_std(ctrl_df)
        pat_prof  = pool_mean_std(pat_df)
    else:
        ctrl_prof = pool_quantiles(ctrl_df, n_pool)
        pat_prof  = pool_quantiles(pat_df,  n_pool)

    in_dim = ctrl_prof.shape[1]
    ctrl_x = torch.tensor(ctrl_prof, dtype=torch.float).to(DEVICE)
    pat_x  = torch.tensor(pat_prof,  dtype=torch.float).to(DEVICE)

    if use_graph:
        ei, ew = build_gene_graph(ctrl_df)
        ei = ei.to(DEVICE); ew = ew.to(DEVICE)
        data_ctrl = Data(x=ctrl_x, edge_index=ei, edge_attr=ew)
        data_pat  = Data(x=pat_x,  edge_index=ei, edge_attr=ew)
        model = GeneGCN(in_dim, DROPOUT).to(DEVICE)
    else:
        data_ctrl = Data(x=ctrl_x)
        data_pat  = Data(x=pat_x)
        model = GeneMLP(in_dim, DROPOUT).to(DEVICE)

    model = train_one(model, data_ctrl)
    model.eval()
    with torch.no_grad():
        ctrl_recon = model(data_ctrl).cpu().numpy()
        pat_recon  = model(data_pat).cpu().numpy()

    ctrl_err = ((ctrl_recon - ctrl_prof) ** 2).mean(axis=1)
    pat_err  = ((pat_recon  - pat_prof)  ** 2).mean(axis=1)
    red = pat_err - ctrl_err
    return np.abs(red)


def score_wilcoxon(ctrl_ad, pat_ad, common_genes, seed):
    np.random.seed(seed)
    ctrl = ctrl_ad[:, common_genes].copy()
    pat  = pat_ad[:,  common_genes].copy()
    ctrl.obs["group"] = "control"
    pat.obs["group"]  = "patient"
    combined = ctrl.concatenate(pat, batch_key="batch")
    combined.obs["group"] = combined.obs["group"].astype("category")
    sc.tl.rank_genes_groups(combined, groupby="group", reference="control",
                             method="wilcoxon", n_genes=len(common_genes))
    names  = combined.uns["rank_genes_groups"]["names"]["patient"]
    scores = combined.uns["rank_genes_groups"]["scores"]["patient"]
    score_map = dict(zip(names, np.abs(scores)))
    return np.array([score_map.get(g, 0.0) for g in common_genes])


# --------------------------------------------------------------------------- #
# ONE ABLATION × ONE CELL TYPE × ONE SEED
# --------------------------------------------------------------------------- #
def run_one(ablation, ctrl, pat, seed):
    """Returns (ranked_genes, n_genes_in_universe)."""
    use_filter = (ablation != "NO_FILTER")
    if use_filter:
        ctrl = filter_adata(ctrl, FilterConfig(), write_audit=(seed == 0))
        pat  = pat[:, list(ctrl.var_names)].copy()

    ctrl, pat, common_genes = select_highly_variable_genes(TOP_N_GENES, ctrl, pat)

    ctrl_df = ctrl.to_df()
    pat_df  = pat.to_df()[ctrl_df.columns]
    genes = list(ctrl_df.columns)

    if ablation == "FULL":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=True)
    elif ablation == "NO_GRAPH":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=False)
    elif ablation == "MEAN_POOL":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=2, seed=seed, use_graph=True)
    elif ablation == "NO_FILTER":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=True)
    elif ablation == "WILCOXON_FAIR":
        scores = score_wilcoxon(ctrl, pat, common_genes, seed=seed)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    order = np.argsort(-scores)
    ranked = [genes[i] for i in order]
    return ranked, len(genes)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltypes", default="ALL_CELLS,IN-SST,Oligodendrocytes,L2_3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ablations",
                    default="FULL,NO_GRAPH,MEAN_POOL,NO_FILTER,WILCOXON_FAIR")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

    ablations = args.ablations.split(",")
    rows = []

    for ct in cts:
        print(f"\n=== {ct} ===")
        if ct == "ALL_CELLS":
            ctrl = ctrl_full.copy(); pat = pat_full.copy()
        else:
            ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
            pat  = pat_full[pat_full.obs[CELLTYPE_COL]  == ct].copy()
            if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
                print(f"  SKIP (cells < {MIN_CELLS})"); continue

        for ab in ablations:
            for seed in range(args.seeds):
                t0 = time.time()
                try:
                    ranked, n_genes = run_one(ab, ctrl.copy(), pat.copy(), seed)
                    ov = sfari_overlap(ranked, sfari)
                except Exception as e:
                    print(f"  [skip] {ab} seed={seed}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                for k, v in ov.items():
                    rows.append({"CellType": ct, "Ablation": ab, "Seed": seed,
                                 "Depth": k, "SFARI_pct": v, "N_Genes": n_genes})
                print(f"  {ab:14s} seed={seed} "
                      f"top50={ov[50]:.1f}% top250={ov[250]:.1f}% "
                      f"({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "ablation_results.csv"), index=False)

    summ = df.groupby(["CellType","Ablation","Depth"])["SFARI_pct"]\
             .agg(["mean","std","count"]).reset_index()
    summ.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)

    print(f"\nResults  : {args.out_dir}/ablation_results.csv  ({len(df)} rows)")
    print(f"Summary  : {args.out_dir}/ablation_summary.csv")

    # Paired Wilcoxon, FULL vs each at top-250
    print("\nPaired Wilcoxon signed-rank (FULL vs ablation) on per-cell-type means, top-250:")
    pv = summ[summ.Depth == 250].pivot_table(index="CellType", columns="Ablation", values="mean")
    if "FULL" not in pv.columns:
        return
    for ab in pv.columns:
        if ab == "FULL": continue
        paired = pv[["FULL", ab]].dropna()
        if len(paired) < 5: continue
        try:
            stat, p = wilcoxon_test(paired["FULL"], paired[ab])
            a = paired["FULL"].mean(); b = paired[ab].mean()
            sign = "+" if a > b else "-"
            print(f"  FULL vs {ab:14s}: n={len(paired)}  W={stat:.2f}  p={p:.4g}  "
                  f"meanD={sign}{abs(a-b):.2f}%")
        except Exception as e:
            print(f"  FULL vs {ab}: {e}")


if __name__ == "__main__":
    main()
