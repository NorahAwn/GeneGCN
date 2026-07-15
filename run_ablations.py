"""
run_ablations.py
================

Benchmark driver for the 5 ablation methods. Optimized to run gene filtering
and HVG selection ONCE per cell type instead of inside the ablation/seed loops.
Updated to support modern AnnData (>=0.10) concatenation syntax.

Methods
-------
FULL           : gene-node GCN + 64-quantile pool + filter   ("GeneGCN gene-graph")
NO_GRAPH       : MLP autoencoder, same shape, no graph
MEAN_POOL      : gene-node GCN + mean/std pool
NO_FILTER      : gene-node GCN, no gene-symbol filter
WILCOXON_FAIR  : Scanpy rank_genes_groups on the same filtered HVG universe

Usage
-----
python run_ablations.py --celltypes all --seeds 5
"""

from __future__ import annotations
import argparse, os, time, gc

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from scipy.stats import wilcoxon as wilcoxon_test

from gene_filter import filter_artifact_genes

# Reuse data loading from your working pipeline
from cells_as_nodes import (
    data_normalize, load_meta, attach_celltype_from_meta,
    select_highly_variable_genes,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    SFARI_PATH, SFARI_SYMBOL_COL, DEPTHS, MIN_CELLS,
    TOP_N_GENES, EPOCHS, LR, DROPOUT, DEVICE,
    sfari_overlap, load_sfari,
)

# --------------------------------------------------------------------------- #
# DEVICE RESOLUTION (GPU Auto-Detection)
# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
print(f"--> Active execution device: {device}")

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
OUT_DIR        = "ablation_results"
CORR_THRESHOLD = 0.3       # gene-gene co-expression threshold
N_POOL         = 64        # quantile bins for gene-node pooling


# --------------------------------------------------------------------------- #
# MODELS
# --------------------------------------------------------------------------- #
class GeneGCN(nn.Module):
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
    X = expr_df.values
    n_genes = X.shape[1]
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0.0)
    iu = np.triu_indices(n_genes, k=1)
    keep = np.abs(corr[iu]) >= thresh
    src = iu[0][keep]; dst = iu[1][keep]
    w = np.abs(corr[src, dst]).astype(np.float32)
    edge_index = torch.tensor(np.stack([np.concat([src, dst]),
                                         np.concat([dst, src])]), dtype=torch.long)
    edge_weight = torch.tensor(np.concat([w, w]), dtype=torch.float)
    return edge_index, edge_weight


def pool_quantiles(expr_df, n_pool=N_POOL):
    qs = np.linspace(0, 1, n_pool)
    X = expr_df.values
    return np.quantile(X, qs, axis=0).T.astype(np.float32)


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
    ctrl_x = torch.tensor(ctrl_prof, dtype=torch.float).to(device)
    pat_x  = torch.tensor(pat_prof,  dtype=torch.float).to(device)

    if use_graph:
        ei, ew = build_gene_graph(ctrl_df)
        data_ctrl = Data(x=ctrl_x, edge_index=ei.to(device), edge_attr=ew.to(device)).to(device)
        data_pat  = Data(x=pat_x,  edge_index=ei.to(device), edge_attr=ew.to(device)).to(device)
        model = GeneGCN(in_dim, DROPOUT).to(device)
    else:
        data_ctrl = Data(x=ctrl_x).to(device)
        data_pat  = Data(x=pat_x).to(device)
        model = GeneMLP(in_dim, DROPOUT).to(device)

    model = train_one(model, data_ctrl)
    model.eval()
    with torch.no_grad():
        ctrl_recon = model(data_ctrl).cpu().numpy()
        pat_recon  = model(data_pat).cpu().numpy()

    del model, data_ctrl, data_pat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ctrl_err = ((ctrl_recon - ctrl_prof) ** 2).mean(axis=1)
    pat_err  = ((pat_recon  - pat_prof)  ** 2).mean(axis=1)
    red = pat_err - ctrl_err
    return np.abs(red)


def score_wilcoxon(ctrl_hvg, pat_hvg, seed):
    np.random.seed(seed)
    ctrl = ctrl_hvg.copy()
    pat  = pat_hvg.copy()
    ctrl.obs["group"] = "control"
    pat.obs["group"]  = "patient"

    # Updated to support modern anndata >= 0.10.0 syntax
    combined = ad.concat([ctrl, pat], label="batch")

    combined.obs["group"] = combined.obs["group"].astype("category")
    sc.tl.rank_genes_groups(combined, groupby="group", reference="control",
                             method="wilcoxon", n_genes=combined.n_vars)
    names  = combined.uns["rank_genes_groups"]["names"]["patient"]
    scores = combined.uns["rank_genes_groups"]["scores"]["patient"]
    score_map = dict(zip(names, np.abs(scores)))
    return np.array([score_map.get(g, 0.0) for g in ctrl_hvg.var_names])


# --------------------------------------------------------------------------- #
# CORE ABLATION EXECUTION
# --------------------------------------------------------------------------- #
def run_one_ablation(ablation, ctrl_hvg, pat_hvg, seed):
    ctrl_df = ctrl_hvg.to_df()
    pat_df  = pat_hvg.to_df()[ctrl_df.columns]
    genes = list(ctrl_df.columns)

    # Direction from log2 fold-change on log1p(CPM)
    log2fc = (pat_df.mean(0).values - ctrl_df.mean(0).values) / np.log(2)
    direction = {genes[i]: ("Up" if log2fc[i] > 0 else "Down") for i in range(len(genes))}

    if ablation == "FULL":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=True)
    elif ablation == "NO_GRAPH":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=False)
    elif ablation == "MEAN_POOL":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=2, seed=seed, use_graph=True)
    elif ablation == "NO_FILTER":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=True)
    elif ablation == "WILCOXON_FAIR":
        scores = score_wilcoxon(ctrl_hvg, pat_hvg, seed=seed)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    del ctrl_df, pat_df
    gc.collect()

    order = np.argsort(-scores)
    ranked = [genes[i] for i in order]
    return ranked, len(genes), direction


# --------------------------------------------------------------------------- #
# MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltypes", default="ALL_CELLS,IN-SST,Oligodendrocytes,L2_3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ablations",
                    default="FULL,NO_GRAPH,MEAN_POOL,NO_FILTER,WILCOXON_FAIR")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--max_cells", type=int, default=20000,
                    help="Max cells per condition to prevent Out-Of-Memory errors on large clusters (0 to disable)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading raw datasets...")
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
    ranking_rows = []

    for ct in cts:
        print(f"\n=== CELL TYPE: {ct} ===")
        if ct == "ALL_CELLS":
            ctrl = ctrl_full.copy(); pat = pat_full.copy()
        else:
            ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
            pat  = pat_full[pat_full.obs[CELLTYPE_COL]  == ct].copy()
            if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
                print(f"  SKIP (cells < {MIN_CELLS})"); continue

        # Memory Defense: Subsample huge groups (like ALL_CELLS) down once
        if args.max_cells > 0:
            if ctrl.n_obs > args.max_cells:
                print(f"  Subsampling control from {ctrl.n_obs} down to {args.max_cells} cells...")
                sc.pp.subsample(ctrl, n_obs=args.max_cells, random_state=42)
            if pat.n_obs > args.max_cells:
                print(f"  Subsampling patient from {pat.n_obs} down to {args.max_cells} cells...")
                sc.pp.subsample(pat, n_obs=args.max_cells, random_state=42)

        # ----------------------------------------------------------------- #
        # OPTIMIZATION: PRE-COMPUTE HVG MATRICES ONCE PER CELL TYPE
        # ----------------------------------------------------------------- #
        precomputed_hvgs = {}

        # 1. Check if we need the standard FILTERED branch (FULL, NO_GRAPH, MEAN_POOL, WILCOXON_FAIR)
        has_filtered_ablations = any(ab != "NO_FILTER" for ab in ablations)
        if has_filtered_ablations:
            print(f"  [Pre-processing] Applying artifact filter & selecting top-{TOP_N_GENES} HVGs...")
            kept = filter_artifact_genes(list(ctrl.var_names), verbose=True)
            ctrl_f = ctrl[:, kept].copy()
            pat_f  = pat[:, kept].copy()
            ctrl_hvg_f, pat_hvg_f, _ = select_highly_variable_genes(TOP_N_GENES, ctrl_f, pat_f, method="dispersion")

            precomputed_hvgs["FILTERED"] = (ctrl_hvg_f, pat_hvg_f)
            del ctrl_f, pat_f
            gc.collect()

        # 2. Check if we need the UNFILTERED branch (NO_FILTER)
        if "NO_FILTER" in ablations:
            print(f"  [Pre-processing] Selecting top-{TOP_N_GENES} HVGs without artifact filter...")
            ctrl_hvg_nf, pat_hvg_nf, _ = select_highly_variable_genes(TOP_N_GENES, ctrl, pat, method="dispersion")
            precomputed_hvgs["UNFILTERED"] = (ctrl_hvg_nf, pat_hvg_nf)

        # We can free the raw workspace copies of cell-type data now
        del ctrl, pat
        gc.collect()

        # ----------------------------------------------------------------- #
        # RUN ABLATIONS & SEEDS ON PRE-COMPUTED DATASETS
        # ----------------------------------------------------------------- #
        for ab in ablations:
            # Route to the correct pre-processed matrices
            group_key = "UNFILTERED" if ab == "NO_FILTER" else "FILTERED"
            ctrl_hvg, pat_hvg = precomputed_hvgs[group_key]

            for seed in range(args.seeds):
                t0 = time.time()
                try:
                    ranked, n_genes, direction = run_one_ablation(ab, ctrl_hvg, pat_hvg, seed)
                    ov = sfari_overlap(ranked, sfari)

                    for k, v in ov.items():
                        rows.append({"CellType": ct, "Ablation": ab, "Seed": seed,
                                     "Depth": k, "SFARI_pct": v, "N_Genes": n_genes})

                    if seed == 0:
                        for rank_i, g in enumerate(ranked[:250], 1):
                            ranking_rows.append({
                                "CellType": ct, "Method": ab, "Rank": rank_i,
                                "Gene": g, "Direction": direction.get(g, ""),
                                "SFARI": g.upper() in sfari,
                            })

                    print(f"  {ab:14s} seed={seed} "
                          f"top50={ov[50]:.1f}% top250={ov[250]:.1f}% "
                          f"({time.time()-t0:.1f}s)")

                except Exception as e:
                    print(f"  [skip] {ab} seed={seed}: {e}")

                finally:
                    if 'ranked' in locals(): del ranked
                    if 'direction' in locals(): del direction
                    if 'ov' in locals(): del ov
                    gc.collect()

        # Clean up cell-type specific objects
        del precomputed_hvgs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Save Results -----
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "ablation_results.csv"), index=False)

    summ = df.groupby(["CellType","Ablation","Depth"])["SFARI_pct"]\
             .agg(["mean","std","count"]).reset_index()
    summ.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)

    if ranking_rows:
        rdf = pd.DataFrame(ranking_rows)
        rdf.to_csv(os.path.join(args.out_dir, "ablation_rankings.csv"), index=False)
        print(f"\nRankings : {args.out_dir}/ablation_rankings.csv ({len(rdf)} rows)")

    print(f"Results  : {args.out_dir}/ablation_results.csv")
    print(f"Summary  : {args.out_dir}/ablation_summary.csv")

    # ----- Paired Wilcoxon summary -----
    print("\nPaired Wilcoxon (each vs each) at top-250:")
    pv = summ[summ.Depth == 250].pivot_table(index="CellType", columns="Ablation", values="mean")
    methods_present = list(pv.columns)
    for i, a in enumerate(methods_present):
        for b in methods_present[i+1:]:
            paired = pv[[a, b]].dropna()
            if len(paired) < 5: continue
            try:
                stat, p = wilcoxon_test(paired[a], paired[b])
                ma = paired[a].mean(); mb = paired[b].mean()
                sign = "+" if ma > mb else "-"
                print(f"  {a:14s} vs {b:14s}: n={len(paired)}  W={stat:.2f}  p={p:.4g}  "
                      f"meanD={sign}{abs(ma-mb):.2f}%")
            except Exception as e:
                print(f"  {a} vs {b}: {e}")


if __name__ == "__main__":
    main()