"""
gene_filter.py
==============
Remove non-biological "artifact" gene identifiers before co-expression graph
construction and perturbation scoring in the GeneGCN pipeline.

WHY THIS EXISTS
---------------
The reconstruction-error deviation score (pat_err - ctrl_err) rewards genes whose
per-gene quantile profile shifts most between patient and control. Sparse, near-
zero features (unannotated clone IDs, accession-only loci, pseudogenes, small RNA
genes, mitochondrial tRNAs) have tiny profiles, so a small absolute change yields
a large *relative* deviation. The result is that the top-ranked genes per cell
type are dominated by these artifacts rather than ASD biology. Removing them from
the gene universe BEFORE the graph is built makes the ranking interpretable and
removes a clear reviewer objection.

Sex-linked genes (XIST and Y-linked loci) are removed by default because the ASD
single-cell cohort is sex-imbalanced; XIST otherwise tops several cortical layers
purely as a sex marker. Set drop_sex=False to keep them.

USAGE (in run_analysis, immediately after select_highly_variable_genes):

    from gene_filter import filter_artifact_genes
    ctrl = ctrl[:, filter_artifact_genes(list(ctrl.var_names))].copy()
    pat  = pat[:,  list(ctrl.var_names)].copy()   # keep identical column order

or filter the common_genes list directly (see patch_snippet at bottom).
"""

from __future__ import annotations
import re
from typing import Iterable, List

# --------------------------------------------------------------------------- #
# Artifact patterns. Each is anchored and case-sensitive to match GENCODE/HGNC
# symbol conventions as they appear in the ASD dataset var_names.
# --------------------------------------------------------------------------- #

# Genomic clone identifiers: RP11-, RP13-, RP1-, RP3-, RP4-, RP5-, etc.
_RE_RP_CLONE = re.compile(r"^RP\d+-")
# CTD-, CTB-, CTA-, CTC- clone libraries
_RE_CT_CLONE = re.compile(r"^CT[ABCD]-")
# Other BAC/fosmid clone libraries seen in the data: KB-, CH507-, bP-, XX-, AP00x
_RE_MISC_CLONE = re.compile(r"^(KB-|CH\d+|bP-|XX-|XXbac-|XXyac-)")
# Accession-only loci with no gene symbol: AC#####.#, AL#####.#, AP#####.#, Z#####.#
_RE_ACCESSION = re.compile(r"^(AC|AL|AP|Z)\d{4,}\.\d+$")
# Mitochondrial genes/tRNAs
_RE_MITO = re.compile(r"^MT-")
# Small / structural RNA genes: Y_RNA, Metazoa_SRP, 7SK, U#, snoRNA, miRNA precursors
_RE_RNA_GENE = re.compile(r"(_RNA(-\d+)?$|Metazoa_SRP|^7SK|^U\d+$|^SNOR|^MIR\d)")
# Pseudogene suffix. Two GENCODE conventions:
#   (a) parent symbol + 'P' + number:  RPL32P33, NPM1P19, PMS2P6, USP9YP31, TDGF1P7
#   (b) parent symbol + trailing 'P' (often after a digit/letter): FAM204CP, DGAT2L7P
# We require the 'P' to terminate the symbol AND be preceded by a digit, OR be a
# 'P<number>' tail. This spares real genes like TP53 (P not terminal), FOXP1
# (P1 is internal-style but symbol is whitelisted-short), GRINA, etc.
_RE_PSEUDO = re.compile(r".+(\dP\d+$|[A-Z0-9]P\d+$|\dP$)")
# Whitelist of real protein-coding symbols that superficially resemble pseudogenes
# and must never be filtered.
_PSEUDO_WHITELIST = {
    "TP53", "TP63", "TP73", "FOXP1", "FOXP2", "FOXP3", "FOXP4",
    "GRINA", "ATP1A1", "ATP2B2", "SP1", "SP4", "SP8", "NFATC1",
}
# Bare-'P'-tail pseudogenes with no trailing digit (e.g. FAM204CP, ...CP, ...GP)
# are too ambiguous to catch by regex without risking real genes, so list the
# observed ones explicitly. Extend as needed after auditing filter_report().
_PSEUDO_EXPLICIT = {"FAM204CP"}
# Y-linked clones / genes that act as sex markers (TTTY*, *Y pseudogenes, AMELY,
# RPS4Y, USP9Y, UTY, DDX3Y, EIF1AY, KDM5D, NLGN4Y, ZFY, PRKY, etc.)
_RE_Y_GENE = re.compile(r"(^TTTY|^RPS4Y|^USP9Y|^UTY$|^DDX3Y|^EIF1AY|^KDM5D$|"
                        r"^NLGN4Y|^ZFY$|^PRKY|^AMELY$|^TXLNGY|^UTY|^PCDH11Y|"
                        r"^TBL1Y|^TMSB4Y|^NLGN4Y|YP\d+$|Y$)")

_SEX_KEEP_EXCEPTIONS = set()  # add symbols here if a 'Y'-ending real gene is wrongly caught


def _is_artifact(g: str, drop_lincrna: bool, drop_antisense: bool) -> bool:
    if g in _SEX_KEEP_EXCEPTIONS:
        return False
    if _RE_RP_CLONE.match(g):      return True
    if _RE_CT_CLONE.match(g):      return True
    if _RE_MISC_CLONE.match(g):    return True
    if _RE_ACCESSION.match(g):     return True
    if _RE_MITO.match(g):          return True
    if _RE_RNA_GENE.search(g):     return True
    if g in _PSEUDO_EXPLICIT: return True
    if _RE_PSEUDO.match(g) and g not in _PSEUDO_WHITELIST: return True
    if drop_lincrna and g.startswith("LINC"):     return True
    if drop_antisense and g.endswith("-AS1"):     return True
    return False


def filter_artifact_genes(
    genes: Iterable[str],
    drop_sex: bool = True,
    drop_lincrna: bool = False,
    drop_antisense: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Return `genes` with artifact identifiers removed, ORDER PRESERVED.

    Parameters
    ----------
    genes        : iterable of gene symbols (e.g. list(adata.var_names))
    drop_sex     : also remove XIST and Y-linked sex-marker genes (default True;
                   recommended for the sex-imbalanced ASD cohort).
    drop_lincrna : remove LINC#### lncRNAs (default False; many are real biology).
    drop_antisense : remove *-AS1 antisense transcripts (default False).
    verbose      : print a per-class removal tally.

    Notes
    -----
    The default removes clone IDs, accession-only loci, mitochondrial genes,
    small/structural RNA genes, and clearly-named pseudogenes, plus sex markers.
    It deliberately KEEPS lincRNAs and antisense transcripts unless asked, to
    avoid discarding potentially ASD-relevant non-coding signal.
    """
    genes = [str(g) for g in genes]
    kept, removed = [], []
    for g in genes:
        artifact = _is_artifact(g, drop_lincrna, drop_antisense)
        if not artifact and drop_sex and (g == "XIST" or _RE_Y_GENE.search(g)):
            artifact = True
        (removed if artifact else kept).append(g)

    if verbose:
        n0 = len(genes)
        print(f"  [gene_filter] kept {len(kept)}/{n0} genes "
              f"({len(removed)} removed, {100*len(removed)/max(n0,1):.1f}%).")
    return kept


def filter_report(genes: Iterable[str],
                  drop_sex: bool = True,
                  drop_lincrna: bool = False,
                  drop_antisense: bool = False) -> "dict":
    """Return a dict of {class_name: [removed symbols]} for auditing/Methods."""
    genes = [str(g) for g in genes]
    classes = {
        "RP clone":      lambda g: bool(_RE_RP_CLONE.match(g)),
        "CT clone":      lambda g: bool(_RE_CT_CLONE.match(g)),
        "misc clone":    lambda g: bool(_RE_MISC_CLONE.match(g)),
        "accession":     lambda g: bool(_RE_ACCESSION.match(g)),
        "mitochondrial": lambda g: bool(_RE_MITO.match(g)),
        "RNA gene":      lambda g: bool(_RE_RNA_GENE.search(g)),
        "pseudogene":    lambda g: bool(_RE_PSEUDO.match(g)),
    }
    if drop_lincrna:   classes["lincRNA"]   = lambda g: g.startswith("LINC")
    if drop_antisense: classes["antisense"] = lambda g: g.endswith("-AS1")
    if drop_sex:       classes["sex marker"] = lambda g: g == "XIST" or bool(_RE_Y_GENE.search(g))

    report = {}
    for name, fn in classes.items():
        report[name] = [g for g in genes if fn(g)]
    return report


if __name__ == "__main__":
    # quick self-test on a few representative symbols
    sample = ["XIST", "MT-TT", "RP11-76C10.4", "CTD-2319I12.5", "AP000925.2",
              "Metazoa_SRP-108", "RPL32P33", "USP9YP31", "TTTY11", "LINC00453",
              "EIF1AX-AS1", "NPAS4", "MAG", "PLP1", "CNDP1", "APOE", "GRIN2B",
              "SCN2A", "SHANK3", "CDK1", "MIF"]
    kept = filter_artifact_genes(sample, verbose=True)
    print("kept:", kept)
