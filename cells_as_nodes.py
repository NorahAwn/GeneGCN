"""
CellGCN: cells-as-nodes alternative to GeneGCN. 2GB GPU-friendly.

Nodes = cells, edges = kNN (k=15) in PCA space of controls; the GCN learns to
reconstruct each cell's gene-expression vector. Per-gene RED is the mean
squared error across patient cells minus mean across control cells. Patient
cells are projected through the control-fitted PCA so the cell-cell graph
carries patient-specific signal that the gene-graph variant lacked.

Drop in next to model_per_celltype.py and gene_filter.py.

Run on a 2GB GPU
----------------
python cells_as_nodes.py --celltypes all --seeds 5 --max_cells 3000 --hidden 64
"""

import argparse, os, time, gc
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

from model_per_celltype import (
    data_normalize, attach_celltype_from_meta, load_meta,
    select_highly_variable_genes, _apply_gene_filter_adata,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    TOP_N_GENES, HVG_METHOD, EPOCHS, LR, DROPOUT, MIN_CELLS, DEVICE,
)

# --------------------------------------------------------------------------- #
# CONFIG (overridable from CLI)
# --------------------------------------------------------------------------- #
SFARI_PATH        = "SFARI-Gene_genes.csv"
SFARI_SYMBOL_COL  = "gene-symbol"
DEPTHS            = [50, 100, 150, 200, 250]
OUT_DIR           = "cellgcn_results"

N_PCS             = 50          # PCA dims for kNN graph
K_NEIGHBORS       = 15          # kNN degree (scanpy default)
HIDDEN_DIM        = 64          # default; CLI override --hidden
MAX_CELLS         = 3000        # default; CLI override --max_cells


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CellGCN(nn.Module):
    """Two-layer GCN autoencoder over a cell-cell graph."""
    def __init__(self, n_genes, hidden, dropout):
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
# Cell graph
# --------------------------------------------------------------------------- #
def build_cell_graph(expr_cells_by_genes, seed=0, pca_model=None):
    """Symmetric kNN in PCA space. Returns (edge_index, edge_weight, pca)."""
    X = expr_cells_by_genes
    n_cells = X.shape[0]
    k_eff = min(K_NEIGHBORS, n_cells - 1)
    n_pcs_eff = min(N_PCS, n_cells - 1, X.shape[1] - 1)

    if pca_model is None:
        pca_model = PCA(n_components=n_pcs_eff, random_state=seed)
        Z = pca_model.fit_transform(X)
    else:
        Z = pca_model.transform(X)

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(Z)
    dists, idxs = nbrs.kneighbors(Z)
    dists, idxs = dists[:, 1:], idxs[:, 1:]

    src = np.repeat(np.arange(n_cells), k_eff)
    dst = idxs.flatten()
    d = dists.flatten()
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    d_sym = np.concatenate([d, d])
    med = np.median(d_sym) if np.median(d_sym) > 0 else 1.0
    w = np.exp(-d_sym / med)

    edge_index = torch.tensor(np.stack([src_sym, dst_sym]), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight, pca_model


# --------------------------------------------------------------------------- #
# Training
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


def _to_device(ctrl_X, pat_X, ei_ctrl, ew_ctrl, ei_pat, ew_pat, device):
    x_ctrl = torch.tensor(ctrl_X, dtype=torch.float).to(device)
    x_pat  = torch.tensor(pat_X,  dtype=torch.float).to(device)
    return (x_ctrl, x_pat,
            ei_ctrl.to(device), ew_ctrl.to(device),
            ei_pat.to(device),  ew_pat.to(device))


# --------------------------------------------------------------------------- #
# Scoring with GPU->CPU fallback
# --------------------------------------------------------------------------- #
def cells_as_nodes_score(ctrl_X, pat_X, gene_names, seed=0, verbose=False):
    """Returns (gene_names, |RED|, signed_RED)."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Safety guardrail (already subsampled earlier, but kept for double defense)
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

    ei_ctrl, ew_ctrl, pca = build_cell_graph(ctrl_X, seed=seed)
    ei_pat,  ew_pat,  _   = build_cell_graph(pat_X, seed=seed, pca_model=pca)

    # Try GPU first, fall back to CPU on OOM
    used_device = DEVICE
    try:
        x_ctrl, x_pat, ei_ctrl_d, ew_ctrl_d, ei_pat_d, ew_pat_d = _to_device(
            ctrl_X, pat_X, ei_ctrl, ew_ctrl, ei_pat, ew_pat, used_device)
        data_ctrl = Data(x=x_ctrl, edge_index=ei_ctrl_d, edge_attr=ew_ctrl_d)
        data_pat  = Data(x=x_pat,  edge_index=ei_pat_d,  edge_attr=ew_pat_d)
        model = train_cell_gcn(data_ctrl, n_genes, used_device, HIDDEN_DIM,
                               seed=seed, verbose=verbose)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower(): raise
        print(f"    [OOM on GPU] falling back to CPU")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        used_device = torch.device("cpu")
        x_ctrl, x_pat, ei_ctrl_d, ew_ctrl_d, ei_pat_d, ew_pat_d = _to_device(
            ctrl_X, pat_X, ei_ctrl, ew_ctrl, ei_pat, ew_pat, used_device)
        data_ctrl = Data(x=x_ctrl, edge_index=ei_ctrl_d, edge_attr=ew_ctrl_d)
        data_pat  = Data(x=x_pat,  edge_index=ei_pat_d,  edge_attr=ew_pat_d)
        model = train_cell_gcn(data_ctrl, n_genes, used_device, HIDDEN_DIM,
                               seed=seed, verbose=verbose)

    model.eval()
    with torch.no_grad():
        ctrl_recon = model(data_ctrl).cpu().numpy()
        pat_recon  = model(data_pat).cpu().numpy()

    ctrl_err = ((ctrl_recon - ctrl_X) ** 2).mean(axis=0)
    pat_err  = ((pat_recon  - pat_X)  ** 2).mean(axis=0)
    red = pat_err - ctrl_err

    del model, data_ctrl, data_pat, x_ctrl, x_pat
    del ei_ctrl_d, ew_ctrl_d, ei_pat_d, ew_pat_d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gene_names, np.abs(red), red


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #
def run_cells_as_nodes(ctrl_ad, pat_ad, seed=0, apply_filter=True, verbose=False):
    if apply_filter:
        ctrl_ad = _apply_gene_filter_adata(ctrl_ad)
        pat_ad  = pat_ad[:, list(ctrl_ad.var_names)].copy()

    ctrl_ad, pat_ad, _ = select_highly_variable_genes(
        TOP_N_GENES, ctrl_ad, pat_ad, method=HVG_METHOD)

    genes = list(ctrl_ad.var_names)
    ctrl_X = ctrl_ad.X.toarray().astype(np.float32) if hasattr(ctrl_ad.X, "toarray") else ctrl_ad.X.astype(np.float32)
    pat_X  = pat_ad.X.toarray().astype(np.float32)  if hasattr(pat_ad.X,  "toarray") else pat_ad.X.astype(np.float32)

    gene_names, abs_red, signed_red = cells_as_nodes_score(
        ctrl_X, pat_X, genes, seed=seed, verbose=verbose)
    order = np.argsort(-abs_red)
    ranked = [gene_names[i] for i in order]
    direction_map = {gene_names[i]: ("Up" if signed_red[i] > 0 else "Down")
                     for i in range(len(gene_names))}
    return ranked, len(genes), direction_map


# --------------------------------------------------------------------------- #
# SFARI
# --------------------------------------------------------------------------- #
def sfari_overlap(ranked, sfari_set, depths=DEPTHS):
    return {k: 100.0 * sum(g.upper() in sfari_set for g in ranked[:k]) / max(k, 1)
            for k in depths}

def load_sfari():
    df = pd.read_csv(SFARI_PATH)
    return set(df[SFARI_SYMBOL_COL].astype(str).str.upper())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltypes", default="ALL_CELLS,IN-SST,Oligodendrocytes,L2_3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_cells", type=int, default=3000,
                    help="cap cells per group (lower = less VRAM); default 3000")
    ap.add_argument("--hidden", type=int, default=64,
                    help="GCN hidden dim (lower = less VRAM); default 64")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global MAX_CELLS, HIDDEN_DIM
    MAX_CELLS  = args.max_cells
    HIDDEN_DIM = args.hidden
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

    rows = []
    ranking_rows = []

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
                # -------------------------------------------------------------
                # THE OPTIMIZATION FIX: Subsample BEFORE processing genes/arrays
                # -------------------------------------------------------------
                ctrl_seed = ctrl.copy()
                pat_seed  = pat.copy()

                if ctrl_seed.n_obs > MAX_CELLS:
                    sc.pp.subsample(ctrl_seed, n_obs=MAX_CELLS, random_state=seed)
                if pat_seed.n_obs > MAX_CELLS:
                    sc.pp.subsample(pat_seed, n_obs=MAX_CELLS, random_state=seed)

                ranked, n_genes, dir_map = run_cells_as_nodes(
                    ctrl_seed, pat_seed, seed=seed,
                    apply_filter=True, verbose=args.verbose)
            except Exception as e:
                print(f"  [skip] seed={seed}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            finally:
                # Force release temporary copies from memory
                if 'ctrl_seed' in locals(): del ctrl_seed
                if 'pat_seed' in locals(): del pat_seed
                gc.collect()

            ov = sfari_overlap(ranked, sfari)
            for k, v in ov.items():
                rows.append({"CellType": ct, "Method": "CELLS_AS_NODES",
                             "Seed": seed, "Depth": k,
                             "SFARI_pct": v, "N_Genes": n_genes})

            if seed == 0:
                for rank, g in enumerate(ranked[:250], 1):
                    ranking_rows.append({
                        "CellType": ct, "Rank": rank, "Gene": g,
                        "Direction": dir_map.get(g, ""),
                        "SFARI": g.upper() in sfari
                    })

            print(f"  seed={seed} top50={ov[50]:.1f}% top250={ov[250]:.1f}% "
                  f"({time.time()-t0:.1f}s)")

    # save
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_results.csv"), index=False)
    summ = df.groupby(["CellType","Method","Depth"])["SFARI_pct"]\
             .agg(["mean","std","count"]).reset_index()
    summ.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_summary.csv"), index=False)
    rdf = pd.DataFrame(ranking_rows)
    rdf.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_rankings.csv"), index=False)

    print(f"\nResults   : {OUT_DIR}/cells_as_nodes_results.csv ({len(df)} rows)")
    print(f"Summary   : {OUT_DIR}/cells_as_nodes_summary.csv")
    print(f"Rankings  : {OUT_DIR}/cells_as_nodes_rankings.csv (top-250, seed=0)")

    # head-to-head comparison with prior ablation results, if present
    prior_path = "ablation_results/ablation_summary.csv"
    if os.path.exists(prior_path):
        print(f"\nComparing against {prior_path}")
        prior = pd.read_csv(prior_path).rename(columns={"Ablation": "Method"})
        full = pd.concat([prior, summ], ignore_index=True)

        print("\nMean SFARI overlap (%) across cell types, per method per depth:")
        across = full.groupby(["Method", "Depth"])["mean"].mean().reset_index()
        print(across.pivot(index="Method", columns="Depth", values="mean").round(2).to_string())

        pv = full[full.Depth == 250].pivot_table(
            index="CellType", columns="Method", values="mean")
        if "CELLS_AS_NODES" not in pv.columns:
            print("  CELLS_AS_NODES missing -- skipping paired test"); return

        print("\nPaired Wilcoxon signed-rank, CELLS_AS_NODES vs others at top-250:")
        for m in pv.columns:
            if m == "CELLS_AS_NODES": continue
            paired = pv[["CELLS_AS_NODES", m]].dropna()
            if len(paired) < 5: continue
            try:
                stat, pval = wilcoxon_test(paired["CELLS_AS_NODES"], paired[m])
                a = paired["CELLS_AS_NODES"].mean(); b = paired[m].mean()
                sign = "+" if a > b else "-"
                wins = (paired["CELLS_AS_NODES"] > paired[m]).sum()
                print(f"  CELLS_AS_NODES vs {m:14s}: n={len(paired)}  "
                      f"W={stat:.2f}  p={pval:.4g}  "
                      f"mean_CAN={a:.2f}%  mean_{m}={b:.2f}%  "
                      f"delta={sign}{abs(a-b):.2f}%  wins={wins}/{len(paired)}")
            except Exception as e:
                print(f"  CELLS_AS_NODES vs {m}: {e}")


if __name__ == "__main__":
    main()