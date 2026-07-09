# GeneGCN

**A graph convolutional network for cell-type-specific gene perturbation analysis in single-nucleus transcriptomes of autism spectrum disorder.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)

---

## Overview

GeneGCN trains a graph convolutional autoencoder on a **cell–cell similarity graph** built from healthy control cells, then scores genes by the per-gene reconstruction-error deviation (RED) of patient cells against the control-derived manifold. The method is designed for single-nucleus RNA-seq (snRNA-seq) cohorts where the question is "*which genes are perturbed in patients, in which cell types?*"

Applied to the Velmeshev *et al.* (2019) autism cortex cohort, GeneGCN:

- Outperforms a gene-node graph variant of the same model in 14/15 cell types (paired Wilcoxon *p* = 1.8 × 10⁻³).
- Matches the best simple baselines (MLP autoencoder, Wilcoxon rank-sum) on average across cell types and outperforms them in 7 of 18 specific cell-type contexts, including oligodendrocytes, somatostatin interneurons, and astrocyte subpopulations.
- Recovers a coherent oligodendrocyte myelination signature, an inflammatory chemokine axis in glia, and immediate-early gene activation in excitatory layers, with several SFARI ASD-risk genes appearing at the top of cell-type-specific rankings.

![Pipeline overview](figures/fig1_overview.png)

---

## Installation

```bash
git clone https://github.com/<your-username>/GeneGCN.git
cd GeneGCN
pip install -r requirements.txt
```

PyTorch Geometric requires PyTorch to be installed first; if `pip install -r requirements.txt` fails on the PyG line, follow the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your CUDA/PyTorch combination, then re-run.

Tested on:
- Linux + CUDA 11.8 + PyTorch 2.1 + PyG 2.4 + NVIDIA A100 (full benchmark, ~6 hours)
- Windows 10 + CUDA 11.8 + PyTorch 2.1 + PyG 2.4 + NVIDIA RTX 3060 6 GB (per-cell-type runs work; ALL_CELLS needs `--max_cells 3000`)

---

## Quick start

### 1. Download the data

The Velmeshev autism cortex cohort is available from the [UCSC Cell Browser](https://cells.ucsc.edu) (search "autism"). Download the three files and place them in `data/`:

```
data/
├── ASD_control.h5ad     # control cells, log-normalized
├── ASD_patient.h5ad     # patient cells, log-normalized
└── meta.tsv             # cell-level metadata with cell-type labels
```

The SFARI Gene CSV is at [https://gene.sfari.org](https://gene.sfari.org) (login required, free). Download as `SFARI-Gene_genes.csv` and place at the repo root.

See `data/README.md` for the exact column requirements.

### 2. Run GeneGCN on a few cell types

```bash
python cells_as_nodes.py \
    --celltypes ALL_CELLS,IN-SST,Oligodendrocytes,L2_3 \
    --seeds 5
```

### 3. Run on all 18 cell-type contexts

```bash
python cells_as_nodes.py --celltypes all --seeds 5
```

### 4. Run on a low-VRAM GPU (≤ 2 GB)

```bash
python cells_as_nodes.py --celltypes all --seeds 5 --max_cells 3000 --hidden 64
```

Outputs land in `cellgcn_results/`:

| File | Contents |
|---|---|
| `cells_as_nodes_results.csv` | One row per (cell type × seed × ranking depth). |
| `cells_as_nodes_summary.csv` | Mean ± SD across seeds. |
| `cells_as_nodes_rankings.csv` | Top-250 ranked genes per cell type, with direction and SFARI annotation. |

---

## Reproducing the manuscript benchmark

The full ablation (GeneGCN cell-graph vs gene-graph vs MLP vs mean/std pool vs no-filter vs Wilcoxon) is in `run_ablations.py`.

```bash
# Step 1: gene-graph baselines (uses model_per_celltype.py)
python run_ablations.py --celltypes all --seeds 5

# Step 2: cell-graph (this paper's method)
python cells_as_nodes.py --celltypes all --seeds 5
```

The cell-graph script automatically merges its results with the prior ablation summary (if present) and prints a paired Wilcoxon signed-rank table for every method comparison.

---

## Repository structure

```
GeneGCN/
├── cells_as_nodes.py          # main method: cell-graph GCN (self-contained)
├── gene_filter.py             # gene-symbol artifact filter
├── model_per_celltype.py      # gene-graph baseline (for ablation reproducibility)
├── run_ablations.py           # ablation driver
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── README.md
├── data/                      # snRNA-seq data (not tracked; see data/README.md)
├── results/                   # example outputs from our benchmark run
│   ├── cells_as_nodes_summary.csv
│   ├── cells_as_nodes_rankings.csv
│   └── ablation_summary.csv
├── examples/
│   └── quick_start.py         # minimal usage example
└── docs/
    └── PIPELINE.md            # step-by-step methodology
```

---

## Method summary

**Inputs**: snRNA-seq counts, cell-type annotations, control/patient labels.

**Pipeline** (per cell type):

1. **Filter**: remove transcript-annotation artifacts (RP-, CTD-, AC-, AL-, MT-, small RNAs, pseudogenes, sex markers) via `gene_filter.py`. After filtering, artifact content in the top-50 ranked genes drops from 56–80% to 0%.
2. **HVG**: select the top 3,000 highly-variable genes on the combined matrix.
3. **PCA**: fit 50-component PCA on controls; project patients through the same basis.
4. **kNN graph**: build a symmetric k=15 nearest-neighbor graph in PC space, separately for controls and patients. Edge weights w_ij = exp(−d_ij / median(d)).
5. **GCN autoencoder**: train a two-layer GCN (hidden=64) on the control graph for 300 epochs to reconstruct each cell's expression vector.
6. **RED scoring**: at inference, compute per-gene mean squared reconstruction error for control and patient cells separately; RED_g = e_g(patient) − e_g(control); rank by |RED_g|.
7. **Direction**: assign Up/Down from the log2 fold-change on log1p(CPM) values.

See `docs/PIPELINE.md` for the full mathematical specification.

---

## Citation

If you use this code in your work, please cite:

```bibtex
@article{Norah2026genegcn,
  title={GeneGCN: A Graph Convolutional Network Uncovers Novel Cell-Type-Specific
         Gene Perturbations and Biomarkers in Autism Spectrum Disorder},
  author={Saeed Awn, Norah and Zhao, Mengyuan and Ba Mahel, Mansoor and Bamahel, Abdulaziz S.
           and Tang, Jijun},
  journal={Bioinformatics},
  year={2026},
  note={Submitted}
}
```

A machine-readable version is in `CITATION.cff`.

---

## Acknowledgements

- Velmeshev et al. for the public snRNA-seq cortex cohort.
- SFARI Gene for the curated ASD-risk gene reference.
- The scanpy and PyTorch Geometric maintainers.

---

## License

MIT. See `LICENSE`.
