# GeneGCN

**A graph convolutional network that uncovers cell-type-specific gene perturbations and biomarkers in autism spectrum disorder (ASD).**

GeneGCN is a graph-convolutional autoencoder built on a **cell–cell similarity graph** (cells as nodes) rather than the usual gene–gene co-expression graph (genes as nodes). The model is trained **only on healthy control cells**; the per-gene deviation in reconstruction error between patient and control cells provides the ranking statistic. Applied to the Velmeshev ASD cortex cohort across 17 cell types plus a pseudo-bulk (`ALL_CELLS`) context, GeneGCN surfaces interpretable, cell-type-specific gene lists that are enriched for SFARI ASD-risk genes.

This repository contains the source code, the gene-symbol filter, and the benchmark drivers used to produce the per-cell-type ranked gene lists.

---

## Key idea

Conventional differential-expression tools (Scanpy `rank_genes_groups`, Seurat `FindMarkers`, MAST) score one gene at a time and ignore the manifold structure that separates patient from control cells. Recent GCN approaches to single-cell data usually place **genes at the nodes** with gene–gene co-expression as edges — but a co-expression graph built from healthy cells encodes only normal regulatory relationships and carries no patient-specific signal for message-passing to amplify.

GeneGCN instead places **cells at the nodes**:

1. Select the top 3,000 highly variable genes (HVGs) after gene-symbol filtering.
2. Fit a 50-component PCA **on control cells only**, then project both control and patient cells into that PC space.
3. Build a symmetric k-nearest-neighbor graph (k = 15) with weights `w_ij = exp(-d_ij / median(d))` in PC space.
4. Train a two-layer GCN autoencoder to reconstruct per-cell gene-expression vectors — **on the control graph only**.
5. Score each gene by the **reconstruction-error deviation (RED)**:

   ```
   RED_g = e_g(patient) - e_g(control)
   ```

Genes are ranked by `|RED_g|` (magnitude of perturbation). The direction (Up/Down) is reported separately from raw `log1p(CPM)` expression (sign of `log2FC`), independent of the reconstruction error. Because patient cells are projected through the control-fitted PCA, their displacement from the healthy manifold becomes the disease signal that message-passing propagates.

---

## Model architecture

Two-layer graph convolutional autoencoder (PyTorch Geometric):

```
h(1) = ReLU( GCNConv1(X, A) )
X̂   = GCNConv2( Dropout(h(1)), A )
```

- `X ∈ R^{n × g}` — cells-by-genes matrix (n cells, g = 3,000 genes)
- `A` — weighted symmetric kNN adjacency
- hidden dimension = 64, dropout = 0.0
- Adam optimizer, learning rate 1e-3, 300 epochs, MSE reconstruction loss
- trained on the control graph only

---

## Repository contents

| File | Description |
|---|---|
| `gene_filter.py` | Curated gene-symbol filter. Removes clone-library identifiers (RP-, CTD-, AC-, AL-), accession-only loci, mitochondrial transcripts, small/structural RNAs, and pseudogenes, plus optional sex-marker removal (XIST + Y-linked). Includes a whitelist for canonical genes that incidentally match a pattern (e.g. TP53, FOXP1) and an auditing report. |
| `model_per_celltype.py` | The per-cell-type GeneGCN pipeline: HVG selection → co-expression graph → GCN self-reconstruction with K-fold CV → per-gene RED scoring, looped over each cell type. Contains the shared config block and the data-loading / HVG / filter helpers imported by the other scripts. |
| `cells_as_nodes.py` | The cell-graph (cells-as-nodes) GeneGCN model (`CellGCN`) and its benchmark runner. 2 GB GPU-friendly with cell subsampling. |
| `run_ablations.py` | Benchmark driver for the five ablation methods, computing SFARI overlap per method and cell type. |

---

## Ablation methods

`run_ablations.py` benchmarks five alternatives on the same upstream pipeline (same filtered gene universe, same HVGs, same cell partition):

| Method (code name) | Description |
|---|---|
| `FULL` | Gene-node GCN + 64-quantile pooling + filter (the "GeneGCN gene-graph" variant) |
| `NO_GRAPH` | MLP autoencoder, identical input/hidden/output shape (3000→64→3000), no graph structure |
| `MEAN_POOL` | Gene-node GCN with mean/std pooling instead of quantile pooling |
| `NO_FILTER` | Gene-node GCN without the gene-symbol filter |
| `WILCOXON_FAIR` | Scanpy `rank_genes_groups` (method=`wilcoxon`) on the same filtered HVG universe |

Five random seeds are used per method per cell type. All statistical comparisons use the paired Wilcoxon signed-rank test on per-cell-type seed-mean SFARI percentages (n = 18 cell-type contexts).

---

## Validation

Ranked gene lists are evaluated by enrichment for the **1,277 ASD-risk genes in the SFARI Gene database** at five depths (top-50, 100, 150, 200, 250). The random-baseline expectation for uniform sampling of the filtered HVG universe is ~0.25% per ranked gene, so a validation rate of 5% corresponds to roughly 20-fold enrichment over random. SFARI labels are used **only at evaluation time** — never during training, PCA fitting, or graph construction.

---

## Installation

Requires Python with PyTorch and PyTorch Geometric.

```bash
pip install torch
pip install torch_geometric
pip install scanpy anndata scikit-learn scipy pandas numpy
```

> Install the PyTorch / PyTorch Geometric builds that match your CUDA version. See the [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) installation guides.

---

## Data setup

The scripts expect the Velmeshev snRNA-seq cortex cohort and a SFARI reference file. Default paths (editable in the config block of `model_per_celltype.py`):

```
data/ASD_control.h5ad     # control cells (h5ad)
data/ASD_patient.h5ad     # patient cells (h5ad)
data/meta.tsv             # cell-type metadata; barcode col 'cell', cell-type col 'cluster'
SFARI-Gene_genes.csv      # SFARI reference; symbol col 'gene-symbol'
```

- The Velmeshev cohort (104,559 nuclei; 15 ASD patients, 16 controls; prefrontal and anterior cingulate cortex) is publicly available from the original publication and the UCSC Cell Browser.
- SFARI Gene is available at <https://gene.sfari.org>.

UMI counts are normalized to counts-per-million and `log1p`-transformed. XIST and Y-linked transcripts are removed by default to control for sex imbalance in the cohort.

---

## Usage

### Per-cell-type GeneGCN pipeline

Paths, cell types, and the `obs` column name are set in the CONFIG block at the top of the script:

```bash
python model_per_celltype.py
```

### Cell-graph model / benchmark (cells-as-nodes)

Runs on a 2 GB GPU with cell subsampling:

```bash
python cells_as_nodes.py --celltypes all --seeds 5 --max_cells 3000 --hidden 64
```

Arguments: `--celltypes` (comma-separated list or `all`), `--seeds`, `--max_cells`, `--hidden`, `--verbose`.

### Ablation benchmark

```bash
python run_ablations.py --celltypes all --seeds 5
```

Arguments: `--celltypes`, `--seeds`, `--ablations` (comma-separated subset of `FULL,NO_GRAPH,MEAN_POOL,NO_FILTER,WILCOXON_FAIR`), `--out_dir`, `--max_cells`. Results are written as `ablation_results.csv`, `ablation_summary.csv`, and optionally `ablation_rankings.csv`.

### Gene-symbol filter (standalone)

```python
from gene_filter import filter_artifact_genes, filter_report

kept = filter_artifact_genes(list(adata.var_names))          # order-preserving
report = filter_report(list(adata.var_names))                # {class: [removed symbols]}
```

A quick self-test runs on a representative sample of symbols with `python gene_filter.py`.

---

## Results summary

- Across 18 cell-type contexts, the cell-graph formulation reached a mean SFARI overlap of **4.60% at top-250** versus **2.11%** for the gene-graph variant — a 2.2-fold improvement — winning in 17 of 18 cell types (paired Wilcoxon p = 6.3 × 10⁻⁴).
- The cell-graph GCN is statistically indistinguishable from a same-shape MLP (4.55%, p = 0.76) and from a Wilcoxon rank-sum test on the same filtered gene universe (4.13%, p = 0.15).
- GeneGCN is the best-performing method in 8 of 18 contexts, notably oligodendrocytes (6.24%, the highest cell-type-specific score), astrocyte subpopulations, and inhibitory interneurons.
- Removing the gene-symbol filter reduces mean SFARI overlap ~2.3-fold and raises top-50 artifact content from 0% to 56–80%.

Recovered biology includes an oligodendrocyte myelination signature (down in patients), immediate-early gene activation in excitatory layers, an inflammatory chemokine/interleukin axis in glia, and SST-interneuron rankings containing known ASD risk genes.

---

## Citation

Awn N.S., Zhao M., Ba Mahel M.S.M., Bamahel A.S., Tang J. *GeneGCN: A graph convolutional network uncovers cell-type-specific gene perturbations and biomarkers in autism spectrum disorder.*

Code archived at Zenodo: [10.5281/zenodo.21132225](https://doi.org/10.5281/zenodo.21132225).

---

## License

MIT License.

---

## Acknowledgements

We thank the Velmeshev et al. group for making their snRNA-seq cortex dataset publicly available, and the SFARI Gene curators for maintaining the ASD-risk gene reference. This work was supported by the Shenzhen Science and Technology Program (grants JCYJ20241202130212016, KQTD20200820113106007, JCYJ20230807140709020).
