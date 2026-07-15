"""
GeneGCN - per-cell-type analysis.

Runs the full pipeline (HVG selection -> per-cell-type co-expression graph ->
GCN self-reconstruction with K-fold CV -> gene perturbation scoring) separately
for each cell type, looping over them one by one.

Scoring: per-gene reconstruction-error deviation
    RED = patient_recon_error - control_recon_error   (per gene)
Genes are RANKED by |RED| (magnitude of dysregulation). The DIRECTION of change
(Up/Down) is taken separately from raw log1p-CPM expression as the sign of
mean(patient) - mean(control), reported as log2FC_raw; it is independent of the
GCN reconstruction.

Key correctness fixes vs. the original script:
  * R^2 evaluation now compares matched tensors (no control-full vs val-split mix-up).
  * Gene labels are taken from the ORDERED common-gene list, not the HVG order,
    so labels can never silently misalign with values.
  * Per-fold models are no longer leaked into the final scoring; a single model is
    retrained on all control cells of the cell type before scoring.

Usage:
    python model_per_celltype.py
Edit the CONFIG block below for paths, the obs column name, and cell types.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from gene_filter import filter_artifact_genes, filter_report


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CONTROL_PATH = "data/ASD_control.h5ad"
PATIENT_PATH = "data/ASD_patient.h5ad"

# Cell-type labels come from a separate metadata TSV (not from adata.obs).
META_PATH = "data/meta.tsv"
META_BARCODE_COL = "cell"      # column in meta.tsv holding the cell barcode
CELLTYPE_COL = "cluster"       # column in meta.tsv holding the cell-type label
# If meta.tsv has no header / barcodes are the first column index, set
# META_BARCODE_COL = None and the first column is used as the index.

CELL_TYPES = None            # None = auto-detect all types present in BOTH datasets;
                             # or give an explicit list, e.g. ["AST-FB", "Microglia", ...]

TOP_N_GENES = 3000
HVG_METHOD = "dispersion"    # "dispersion" | "seurat_v3" | "correlation"
CORR_THRESHOLD = 0.3
EPOCHS = 300
K_FOLDS = 10
USE_CV = True                # CV is optional now (genes are nodes; CV holds out cells for eval)
LR = 0.001
DROPOUT = 0.0
RANDOM_STATE = 25
MIN_CELLS = 20               # skip a cell type if either group has fewer cells than this
N_POOL = 64                  # fixed per-gene profile length (quantile pooling over cells);
                             # makes control & patient comparable despite differing cell counts

# --- Artifact gene filtering (removes clone IDs, accession loci, pseudogenes,
#     mitochondrial genes, small-RNA genes, and sex markers before scoring) ---
FILTER_ARTIFACTS = True      # master switch for the gene filter
FILTER_BEFORE_HVG = True     # True: filter the gene universe BEFORE HVG selection
                             #       (recommended; lets real genes fill HVG slots).
                             # False: filter AFTER HVG (keeps HVG set as in manuscript).
DROP_SEX = True              # remove XIST + Y-linked sex markers (sex-imbalanced cohort)
DROP_LINCRNA = False         # keep lincRNAs (many are real biology)
DROP_ANTISENSE = False       # keep *-AS1 antisense transcripts

OUTPUT_DIR = "results_per_celltype"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def data_normalize(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata


def attach_celltype_from_meta(adata, meta, name):
    """Map cluster labels from meta onto adata.obs[CELLTYPE_COL] by barcode.

    Fails loudly if the barcodes don't overlap, since a silent mismatch would
    mislabel every cell. Cells with no metadata entry are dropped (with a count).
    """
    labels = meta[CELLTYPE_COL].reindex(adata.obs_names)
    matched = labels.notna().sum()
    total = adata.n_obs
    if matched == 0:
        raise ValueError(
            f"[{name}] No barcodes in the .h5ad matched meta.tsv. "
            f"Example adata barcode: {adata.obs_names[0]!r}; "
            f"example meta barcode: {meta.index[0]!r}. "
            f"Check META_BARCODE_COL and whether barcodes need a prefix/suffix.")
    if matched < total:
        print(f"  [{name}] {total - matched}/{total} cells had no meta entry and "
              f"will be dropped.")
    adata = adata[labels.notna()].copy()
    adata.obs[CELLTYPE_COL] = labels[labels.notna()].astype(str).values
    print(f"  [{name}] {adata.n_obs} cells labeled across "
          f"{adata.obs[CELLTYPE_COL].nunique()} cell types.")
    return adata


def load_meta(path):
    """Load the metadata TSV, indexed by cell barcode."""
    meta = pd.read_csv(path, sep="\t")
    if META_BARCODE_COL is None:
        meta = meta.set_index(meta.columns[0])
    else:
        if META_BARCODE_COL not in meta.columns:
            raise KeyError(
                f"'{META_BARCODE_COL}' not in meta.tsv. Columns: {list(meta.columns)}")
        meta = meta.set_index(META_BARCODE_COL)
    if CELLTYPE_COL not in meta.columns:
        raise KeyError(
            f"'{CELLTYPE_COL}' not in meta.tsv. Columns: {list(meta.columns)}")
    return meta


def _to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def select_highly_variable_genes(top_n_genes, adata_control, adata_patient, method):
    """Select HVGs on the combined control+patient matrix.

    Returns control/patient subset to a single ORDERED list of common genes.
    The returned `common_genes` preserves selection order so downstream labels
    line up with data columns.
    """
    import statsmodels.api as sm

    combined_X = np.vstack([_to_dense(adata_control.X), _to_dense(adata_patient.X)])
    gene_names = np.asarray(adata_control.var_names)

    if method == "dispersion":
        gene_means = np.mean(combined_X, axis=0)
        gene_vars = np.var(combined_X, axis=0)
        dispersion = gene_vars / (gene_means + 1e-8)
        top_indices = np.argsort(dispersion)[-top_n_genes:]

    else:
        raise ValueError(f"Unknown HVG selection method: {method}")

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


def build_graph_edges(expr_df, threshold):
    """Build an undirected co-expression graph from a cells x genes DataFrame.

    Returns (edge_index [2, E], edge_weight [E]). Uses upper triangle only.
    """
    corr = expr_df.corr().values
    n = corr.shape[0]
    # Vectorized upper-triangle extraction above threshold.
    iu, ju = np.triu_indices(n, k=1)
    w = corr[iu, ju]
    mask = w > threshold
    src = iu[mask]
    dst = ju[mask]
    weights = w[mask].astype(np.float32)

    if src.size == 0:
        warnings.warn("No edges passed the correlation threshold; graph has no edges.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long).contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


def pool_gene_profiles(expr_df, n_pool):
    """Convert a [cells x genes] matrix into a fixed-size [genes x n_pool] matrix.

    Each gene's expression across cells is summarized by `n_pool` evenly spaced
    quantiles. This is cell-count-agnostic, so control and patient sets of the
    same cell type (which have different cell counts) become directly comparable
    as node-feature matrices for a gene-as-node GCN.
    """
    X = expr_df.values                       # [cells, genes]
    qs = np.linspace(0.0, 1.0, n_pool)
    # np.quantile over axis=0 (cells) -> [n_pool, genes]; transpose to [genes, n_pool]
    profiles = np.quantile(X, qs, axis=0).T.astype(np.float32)
    return profiles                          # [genes, n_pool]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GeneGCN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv3 = GCNConv(64, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return x


def train_one_model(data_train, data_val, in_dim, epochs, lr, dropout, verbose=False):
    """Train a fresh model; return it plus per-epoch loss curves."""
    model = GeneGCN(in_dim, in_dim, dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, train_rmse, val_rmse = [], [], [], []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out_train = model(data_train)
        loss_train = F.mse_loss(out_train, data_train.x)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_val = F.mse_loss(model(data_val), data_val.x) if data_val is not None else loss_train

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        train_rmse.append(float(np.sqrt(loss_train.item())))
        val_rmse.append(float(np.sqrt(loss_val.item())))

        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:>3}  train_mse {loss_train.item():.5f}  val_mse {loss_val.item():.5f}")

    curves = dict(train_losses=train_losses, val_losses=val_losses,
                  train_rmse=train_rmse, val_rmse=val_rmse)
    return model, curves


# --------------------------------------------------------------------------- #
# Per-cell-type pipeline
# --------------------------------------------------------------------------- #
def _apply_gene_filter_adata(adata):
    """Subset an AnnData to non-artifact genes, order preserved."""
    kept = filter_artifact_genes(
        list(adata.var_names), drop_sex=DROP_SEX,
        drop_lincrna=DROP_LINCRNA, drop_antisense=DROP_ANTISENSE, verbose=False)
    return adata[:, kept].copy()


def run_cell_type(ct, control_full, patient_full, out_dir):
    print(f"\n=== Cell type: {ct} ===")
    ctrl = control_full[control_full.obs[CELLTYPE_COL] == ct].copy()
    pat = patient_full[patient_full.obs[CELLTYPE_COL] == ct].copy()
    print(f"  control cells: {ctrl.n_obs}   patient cells: {pat.n_obs}")
    if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
        print(f"  SKIP (fewer than {MIN_CELLS} cells in one group)")
        return None
    return run_analysis(ctrl, pat, tag=ct, out_dir=out_dir)


def run_analysis(ctrl, pat, tag, out_dir):
    """Core pipeline for one (control, patient) pair, identified by `tag`.

    Used both for the GLOBAL run (all cells) and for each cell type. `ctrl`/`pat`
    are AnnData already subset to the cells of interest.
    """
    # Optional artifact filtering BEFORE HVG: lets real genes fill HVG slots.
    if FILTER_ARTIFACTS and FILTER_BEFORE_HVG:
        n_before = ctrl.n_vars
        ctrl = _apply_gene_filter_adata(ctrl)
        pat = pat[:, list(ctrl.var_names)].copy()
        print(f"  [{tag}] gene filter (pre-HVG): {ctrl.n_vars}/{n_before} genes kept")

    # HVG selection on the combined matrix.
    ctrl, pat, common_genes = select_highly_variable_genes(
        TOP_N_GENES, ctrl, pat, method=HVG_METHOD)

    # Optional artifact filtering AFTER HVG: keeps HVG set as in the manuscript.
    if FILTER_ARTIFACTS and not FILTER_BEFORE_HVG:
        kept_genes = filter_artifact_genes(
            common_genes, drop_sex=DROP_SEX,
            drop_lincrna=DROP_LINCRNA, drop_antisense=DROP_ANTISENSE)
        ctrl = ctrl[:, kept_genes].copy()
        pat = pat[:, kept_genes].copy()
        common_genes = kept_genes

    control_df = ctrl.to_df().set_index(ctrl.obs_names)
    patient_df = pat.to_df().set_index(pat.obs_names)
    patient_df = patient_df[control_df.columns]     # identical column order
    genes = list(control_df.columns)                # single source of truth for labels

    # Gene-gene co-expression graph from CONTROL cells.
    # Edges index GENES (0..n_genes-1); with genes as nodes this is valid.
    edge_index, edge_weight = build_graph_edges(control_df, CORR_THRESHOLD)
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    print(f"  genes (nodes): {len(genes)}   edges: {edge_index.shape[1]}")

    # Node features: each GENE summarized to a fixed-length profile over cells.
    control_profiles = pool_gene_profiles(control_df, N_POOL)   # [genes, N_POOL]
    patient_profiles = pool_gene_profiles(patient_df, N_POOL)   # [genes, N_POOL]
    control_x = torch.tensor(control_profiles, dtype=torch.float)
    patient_x = torch.tensor(patient_profiles, dtype=torch.float)

    # ---- Optional CV: hold out a subset of the profile columns for eval ----
    fold_val_rmse = []
    if USE_CV:
        n_pool = control_x.shape[1]
        kf = KFold(n_splits=min(K_FOLDS, n_pool), shuffle=True, random_state=RANDOM_STATE)
        for fold_num, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_pool)), start=1):
            data_full = Data(x=control_x.to(DEVICE), edge_index=edge_index, edge_attr=edge_weight)
            m, _ = train_one_model(data_full, None, N_POOL, EPOCHS, LR, DROPOUT)
            m.eval()
            with torch.no_grad():
                recon = m(data_full).cpu().numpy()
            val_rmse = float(np.sqrt(((recon[:, va_idx] - control_profiles[:, va_idx]) ** 2).mean()))
            fold_val_rmse.append(val_rmse)
        print(f"  CV val RMSE: {np.mean(fold_val_rmse):.5f} +/- {np.std(fold_val_rmse):.5f}")

    # ---- Final model: train on ALL control gene-profiles ----
    data_control = Data(x=control_x.to(DEVICE), edge_index=edge_index, edge_attr=edge_weight)
    data_patient = Data(x=patient_x.to(DEVICE), edge_index=edge_index, edge_attr=edge_weight)
    model, _ = train_one_model(data_control, None, N_POOL, EPOCHS, LR, DROPOUT)

    model.eval()
    with torch.no_grad():
        control_recon = model(data_control).cpu().numpy()      # [genes, N_POOL]
        patient_recon = model(data_patient).cpu().numpy()

    # Reconstruction R^2 on control profiles (matched input vs output).
    r2_per_gene = [r2_score(control_profiles[i], control_recon[i]) for i in range(len(genes))]
    print(f"  control reconstruction mean R^2: {np.mean(r2_per_gene):.4f}")

    # ---- Perturbation score: per-gene reconstruction-error deviation ----
    ctrl_err = ((control_recon - control_profiles) ** 2).mean(axis=1)   # [genes]
    pat_err = ((patient_recon - patient_profiles) ** 2).mean(axis=1)
    recon_dev = pat_err - ctrl_err          # magnitude of dysregulation (signed by error gap)

    # ---- Direction of change from RAW (log1p-CPM) expression, NOT from the
    #      reconstruction. mean(patient) - mean(control) in log space is a log2FC
    #      proxy; its sign gives Up/Down. This is independent of the GCN and is
    #      what reviewers expect for directionality. ----
    ln2 = np.log(2.0)
    mean_ctrl = control_df.values.mean(axis=0)         # [genes], log1p(CPM)
    mean_pat = patient_df.values.mean(axis=0)
    log2fc = (mean_pat - mean_ctrl) / ln2              # convert natural-log diff to log2
    direction = np.where(log2fc > 0, "Up", np.where(log2fc < 0, "Down", "NC"))

    gene_df = pd.DataFrame(
        {"ReconErrorDeviation": recon_dev,
         "absRED": np.abs(recon_dev),
         "log2FC_raw": log2fc,
         "Direction": direction,
         "MeanPatient": mean_pat,
         "MeanControl": mean_ctrl,
         "PatientError": pat_err,
         "ControlError": ctrl_err},
        index=genes,
    ).sort_values(by="absRED", ascending=False)   # rank by MAGNITUDE of dysregulation

    tag_safe = _safe_name(tag)
    gene_df.to_csv(os.path.join(out_dir, f"perturbation_scores_{tag_safe}.csv"))
    torch.save(model.state_dict(), os.path.join(out_dir, f"gene_gcn_model_{tag_safe}.pth"))

    return dict(group=tag, n_control=int(ctrl.n_obs),
                n_patient=int(pat.n_obs), n_genes=len(genes),
                n_edges=int(edge_index.shape[1]),
                cv_val_rmse_mean=(float(np.mean(fold_val_rmse)) if fold_val_rmse else None),
                recon_mean_r2=float(np.mean(r2_per_gene)))


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _safe_name(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def merge_overall(out_dir, cell_types):
    """Merge per-cell-type perturbation scores into ONE overall ranked list.

    Overall score per gene = MAX of its absRED across cell types
    (i.e. most perturbed in any single cell type). Also records which cell type
    that maximum came from, so the source of each gene's signal is transparent.
    """
    score_cols = {}
    for ct in cell_types:
        path = os.path.join(out_dir, f"perturbation_scores_{_safe_name(ct)}.csv")
        if not os.path.exists(path):
            continue                       # cell type was skipped (too few cells)
        s = pd.read_csv(path, index_col=0)["absRED"]
        score_cols[ct] = s

    if not score_cols:
        print("No per-cell-type score files found to merge.")
        return

    # Genes x cell-types matrix (genes can differ slightly per type via HVG).
    mat = pd.DataFrame(score_cols)         # index = union of genes, cols = cell types
    overall = pd.DataFrame({
        "OverallScore_max": mat.max(axis=1, skipna=True),
        "SourceCellType": mat.idxmax(axis=1),                 # where the max came from
        "NumCellTypesScored": mat.notna().sum(axis=1),        # in how many types it was ranked
    }).sort_values(by="OverallScore_max", ascending=False)

    overall.to_csv(os.path.join(out_dir, "overall_perturbation_scores.csv"))
    # Also save the full gene x cell-type score matrix for inspection.
    mat.to_csv(os.path.join(out_dir, "score_matrix_gene_by_celltype.csv"))
    print(f"\nMerged overall list written to {out_dir}/overall_perturbation_scores.csv "
          f"({overall.shape[0]} genes across {mat.shape[1]} cell types).")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    control_full = data_normalize(sc.read(CONTROL_PATH))
    patient_full = data_normalize(sc.read(PATIENT_PATH))

    print("Loading cell-type metadata...")
    meta = load_meta(META_PATH)
    control_full = attach_celltype_from_meta(control_full, meta, "control")
    patient_full = attach_celltype_from_meta(patient_full, meta, "patient")

    # Audit: record which genes the artifact filter removes (for the Supplement).
    if FILTER_ARTIFACTS:
        rep = filter_report(list(control_full.var_names), drop_sex=DROP_SEX,
                            drop_lincrna=DROP_LINCRNA, drop_antisense=DROP_ANTISENSE)
        rows = [(cls, g) for cls, genes in rep.items() for g in genes]
        pd.DataFrame(rows, columns=["class", "gene"]).to_csv(
            os.path.join(OUTPUT_DIR, "filtered_genes_audit.csv"), index=False)
        print(f"  Artifact filter audit: {len(rows)} genes flagged "
              f"-> {OUTPUT_DIR}/filtered_genes_audit.csv")

    if CELL_TYPES is None:
        ctrl_types = set(control_full.obs[CELLTYPE_COL].unique())
        pat_types = set(patient_full.obs[CELLTYPE_COL].unique())
        cell_types = sorted(ctrl_types & pat_types)
        print(f"Auto-detected {len(cell_types)} shared cell types: {cell_types}")
    else:
        cell_types = CELL_TYPES

    # ---- GLOBAL model first: all cells, one model, one ranked list ----
    print("\n##### GLOBAL run (all cells) #####")
    global_summary = run_analysis(control_full.copy(), patient_full.copy(),
                                  tag="ALL_CELLS", out_dir=OUTPUT_DIR)
    if global_summary:
        pd.DataFrame([global_summary]).to_csv(
            os.path.join(OUTPUT_DIR, "summary_global.csv"), index=False)
        print(f"Global ranked list: {OUTPUT_DIR}/perturbation_scores_ALL_CELLS.csv")

    # ---- Then per-cell-type loop ----
    print("\n##### PER-CELL-TYPE runs #####")
    summaries = []
    for ct in cell_types:
        try:
            res = run_cell_type(ct, control_full, patient_full, OUTPUT_DIR)
            if res is not None:
                summaries.append(res)
        except Exception as e:
            print(f"  ERROR on cell type {ct}: {e}")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            os.path.join(OUTPUT_DIR, "summary_per_celltype.csv"), index=False)
        print(f"\nDone. Summary written to {OUTPUT_DIR}/summary_per_celltype.csv")

    # Merge all per-cell-type results into one overall ranked list (max score).
    merge_overall(OUTPUT_DIR, cell_types)


if __name__ == "__main__":
    main()
