"""
gene_filter.py
==============

A curated gene-symbol filter for snRNA-seq pipelines. Removes transcript-
annotation artifacts (clone identifiers, accession-only loci, mitochondrial
transcripts, small RNAs, pseudogenes, sex-chromosome markers) before HVG
selection.

Why this matters
----------------
On raw 10x Genomics output for the Velmeshev autism cortex cohort, the top-50
genes by any deviation-based score were 56-80% transcript-annotation artifacts
before filtering. Their sparse near-zero expression amplifies small absolute
differences into large relative ones, which dominates the top of any ranked
list. After filtering, artifact content in the top-50 drops to 0% across all
cell types.

Usage
-----
>>> from gene_filter import filter_artifact_genes
>>> kept, removed = filter_artifact_genes(list_of_gene_symbols)
>>> kept                                     # symbols to keep
>>> removed                                  # dict: {symbol -> filter_class}

Or for an AnnData object:
>>> adata_filtered = filter_adata(adata)

Filter classes (each toggleable via the FilterConfig dataclass):
    DROP_CLONES          : RP, CTD, AC, AL, KB clone library identifiers
    DROP_ACCESSIONS      : AE, AF, AJ, LA, LL accession-only loci
    DROP_MITOCHONDRIAL   : MT-
    DROP_SMALL_RNAS      : SNORD, SNHG, MIR, RNU, SCARNA, RNA5S, RNY
    DROP_PSEUDOGENES     : curated pseudogene patterns + curated list
    DROP_SEX             : XIST, Y-linked transcripts (recommended for sex-imbalanced cohorts)
    DROP_LINCRNA         : LINC- (off by default)
    DROP_ANTISENSE       : -AS1, -AS2, -AS3 (off by default)

A whitelist preserves canonical genes that incidentally match a pattern
(TP53, FOXP1, EGR1, etc.).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Dict, List, Set, Optional
import warnings


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FilterConfig:
    """Filter toggles. Defaults are recommended for ASD snRNA-seq cohorts."""
    DROP_CLONES:        bool = True
    DROP_ACCESSIONS:    bool = True
    DROP_MITOCHONDRIAL: bool = True
    DROP_SMALL_RNAS:    bool = True
    DROP_PSEUDOGENES:   bool = True
    DROP_SEX:           bool = True       # disable if no sex imbalance
    DROP_LINCRNA:       bool = False      # off by default — most LINC- are real lncRNAs
    DROP_ANTISENSE:     bool = False      # off by default — most -AS1 are real antisense
    AUDIT_PATH: Optional[str] = "filtered_genes_audit.csv"


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
# Clone library prefixes (BAC, cosmid, P1, etc.) — these are placeholder IDs
# for transcripts in regions without an assigned curated gene symbol.
CLONE_PATTERNS = [
    re.compile(r"^RP\d+-", re.IGNORECASE),      # RP11-, RP1-, RP4-, ...
    re.compile(r"^CTD-",   re.IGNORECASE),
    re.compile(r"^CTC-",   re.IGNORECASE),
    re.compile(r"^CTA-",   re.IGNORECASE),
    re.compile(r"^CH\d+-", re.IGNORECASE),      # CH17-, CH507-, ...
    re.compile(r"^KB-",    re.IGNORECASE),
    re.compile(r"^XX\w+-", re.IGNORECASE),      # XXyac-, XXbac-
    re.compile(r"^WI\d+-", re.IGNORECASE),
    re.compile(r"^GS1-",   re.IGNORECASE),
]

# Accession-only loci (no curated symbol assigned)
ACCESSION_PATTERNS = [
    re.compile(r"^AC\d{6}\.", re.IGNORECASE),       # AC000123.1
    re.compile(r"^AL\d{6}\.", re.IGNORECASE),       # AL000123.1
    re.compile(r"^AP\d{6}\.", re.IGNORECASE),       # AP000123.1
    re.compile(r"^AE\d{6}",   re.IGNORECASE),       # AE000658
    re.compile(r"^AF\d{6}",   re.IGNORECASE),       # AF127577
    re.compile(r"^AJ\d{6}",   re.IGNORECASE),       # AJ011932
    re.compile(r"^LA\d+c-",   re.IGNORECASE),       # LA16c-...
    re.compile(r"^LL\d+NC\d+-", re.IGNORECASE),     # LL0XNC01-...
    re.compile(r"^FP\d+\.",   re.IGNORECASE),
    re.compile(r"^Z\d{5}\.",  re.IGNORECASE),       # Z83843.1
]

# Mitochondrial transcripts
MITO_PATTERNS = [re.compile(r"^MT-", re.IGNORECASE)]

# Small structural and ribosomal RNAs
SMALL_RNA_PATTERNS = [
    re.compile(r"^SNORD\d+",  re.IGNORECASE),
    re.compile(r"^SNORA\d+",  re.IGNORECASE),
    re.compile(r"^SNHG\d+",   re.IGNORECASE),
    re.compile(r"^MIR\d+",    re.IGNORECASE),       # MIR21, MIR155, etc.
    re.compile(r"^MIRLET\d+", re.IGNORECASE),
    re.compile(r"^RNU\d+",    re.IGNORECASE),       # RNU1-, RNU2-, ...
    re.compile(r"^RN7S",      re.IGNORECASE),
    re.compile(r"^RNY\d+",    re.IGNORECASE),
    re.compile(r"^SCARNA\d+", re.IGNORECASE),
    re.compile(r"^RNA5S\d+",  re.IGNORECASE),
    re.compile(r"^RNA5-8S",   re.IGNORECASE),
    re.compile(r"^RNA18S",    re.IGNORECASE),
    re.compile(r"^RNA28S",    re.IGNORECASE),
    re.compile(r"^VTRNA\d+",  re.IGNORECASE),       # vault RNAs
    re.compile(r"^Y_RNA",     re.IGNORECASE),
]

# Pseudogene patterns + curated list of explicit pseudogenes
# Note: many real genes end with P-digit (e.g., POLR3GL2P would be a pseudogene)
# so we keep the patterns conservative and rely on the whitelist below.
PSEUDOGENE_PATTERNS = [
    re.compile(r".*-PS\d*$",      re.IGNORECASE),   # XYZ-PS, XYZ-PS1
]
# Explicit pseudogene set (extend as needed)
EXPLICIT_PSEUDOGENES = {
    # Common pseudogenes that don't match any pattern
    "GAPDHP", "ACTBP", "HMGB1P", "RPL21P", "HMGN1P",
}

# Sex-chromosome markers (recommended drop for imbalanced cohorts)
SEX_MARKERS = {
    "XIST",
    "TSIX",
    # Y-linked
    "RPS4Y1", "RPS4Y2", "EIF1AY", "DDX3Y", "KDM5D", "USP9Y",
    "UTY", "NLGN4Y", "ZFY", "SRY", "TBL1Y", "AMELY", "PRKY",
    "TMSB4Y", "TXLNGY", "NLGN4Y", "TSPY1", "TSPY2", "TSPY3",
    "TSPY4", "TSPY8", "TSPY10",
}

# LincRNA and antisense patterns (optional)
LINCRNA_PATTERNS  = [re.compile(r"^LINC\d+",   re.IGNORECASE)]
ANTISENSE_PATTERNS = [re.compile(r".*-AS\d+$", re.IGNORECASE)]

# WHITELIST: canonical genes that may incidentally match a pattern but must be kept
WHITELIST: Set[str] = {
    # Tumor suppressors / TFs
    "TP53", "TP63", "TP73", "FOXP1", "FOXP2", "FOXP3", "FOXP4",
    # Immediate-early genes
    "EGR1", "EGR2", "EGR3", "EGR4", "FOS", "FOSB", "JUN", "JUNB", "JUND",
    "NPAS4", "NR4A1", "NR4A2", "NR4A3", "ARC",
    # ASD-relevant canonicals that might trip patterns
    "MECP2", "FMR1", "SHANK3", "NLGN4X", "CHD8", "CHD7", "ADNP", "PTEN",
    "TSC1", "TSC2", "SCN2A", "SYNGAP1",
    # Common kinases / receptors
    "GAPDH", "ACTB", "ACTG1", "ACTA1", "HMGB1", "HMGB2", "HMGB3",
}


# --------------------------------------------------------------------------- #
# Core API
# --------------------------------------------------------------------------- #
def _matches_any(symbol: str, patterns) -> bool:
    return any(p.match(symbol) for p in patterns)


def classify_symbol(symbol: str, cfg: FilterConfig) -> Optional[str]:
    """Return the filter class (e.g. 'clone') if the symbol should be dropped,
    else None.

    Whitelist always wins: a whitelisted symbol is kept regardless of pattern.
    """
    if not isinstance(symbol, str) or not symbol:
        return None
    s = symbol.strip()
    if not s:
        return None

    if s.upper() in WHITELIST:
        return None

    if cfg.DROP_CLONES         and _matches_any(s, CLONE_PATTERNS):       return "clone"
    if cfg.DROP_ACCESSIONS     and _matches_any(s, ACCESSION_PATTERNS):   return "accession"
    if cfg.DROP_MITOCHONDRIAL  and _matches_any(s, MITO_PATTERNS):        return "mitochondrial"
    if cfg.DROP_SMALL_RNAS     and _matches_any(s, SMALL_RNA_PATTERNS):   return "small_rna"
    if cfg.DROP_PSEUDOGENES    and (_matches_any(s, PSEUDOGENE_PATTERNS)
                                     or s.upper() in EXPLICIT_PSEUDOGENES):
        return "pseudogene"
    if cfg.DROP_SEX            and s.upper() in SEX_MARKERS:              return "sex_marker"
    if cfg.DROP_LINCRNA        and _matches_any(s, LINCRNA_PATTERNS):     return "lincrna"
    if cfg.DROP_ANTISENSE      and _matches_any(s, ANTISENSE_PATTERNS):   return "antisense"
    return None


def filter_artifact_genes(
    symbols: Iterable[str],
    cfg: Optional[FilterConfig] = None,
    write_audit: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """Filter a list of gene symbols.

    Returns
    -------
    kept : list of str
        Symbols that pass the filter.
    removed : dict
        Mapping symbol -> filter_class for everything that was removed.
    """
    if cfg is None:
        cfg = FilterConfig()

    symbols = list(symbols)
    kept: List[str] = []
    removed: Dict[str, str] = {}

    for s in symbols:
        cls = classify_symbol(s, cfg)
        if cls is None:
            kept.append(s)
        else:
            removed[s] = cls

    if write_audit and cfg.AUDIT_PATH and removed:
        try:
            import csv
            with open(cfg.AUDIT_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["symbol", "filter_class"])
                for sym, cls in sorted(removed.items()):
                    w.writerow([sym, cls])
        except Exception as e:
            warnings.warn(f"Could not write audit CSV: {e}")

    return kept, removed


def filter_adata(adata, cfg: Optional[FilterConfig] = None, write_audit: bool = True):
    """Subset an AnnData to genes that pass the filter. Returns a new AnnData.

    Adds a `var['filter_class']` column on the kept genes (= "kept") for traceability.
    """
    symbols = list(adata.var_names)
    kept, removed = filter_artifact_genes(symbols, cfg=cfg, write_audit=write_audit)
    keep_mask = adata.var_names.isin(kept)
    out = adata[:, keep_mask].copy()
    out.var["filter_class"] = "kept"
    return out


# --------------------------------------------------------------------------- #
# CLI for sanity-checking a gene list
# --------------------------------------------------------------------------- #
def _cli():
    import argparse, sys
    ap = argparse.ArgumentParser(description="Filter gene symbols.")
    ap.add_argument("input", help="Text file with one gene symbol per line.")
    ap.add_argument("--output", "-o", help="Write kept symbols to this file.")
    ap.add_argument("--audit", default="filtered_genes_audit.csv")
    args = ap.parse_args()

    symbols = [l.strip() for l in open(args.input, encoding="utf-8") if l.strip()]
    cfg = FilterConfig(AUDIT_PATH=args.audit)
    kept, removed = filter_artifact_genes(symbols, cfg=cfg)
    print(f"Input    : {len(symbols)}")
    print(f"Kept     : {len(kept)}")
    print(f"Removed  : {len(removed)}")
    classes: Dict[str, int] = {}
    for cls in removed.values():
        classes[cls] = classes.get(cls, 0) + 1
    for cls, n in sorted(classes.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:18s}: {n}")
    print(f"\nAudit    : {args.audit}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for s in kept:
                f.write(s + "\n")
        print(f"Kept symbols written to {args.output}")


if __name__ == "__main__":
    _cli()
