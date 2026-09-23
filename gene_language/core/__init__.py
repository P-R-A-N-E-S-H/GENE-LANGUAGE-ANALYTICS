"""
Core genomic representations and utilities.
"""

from .sequence import GenomicSequence
from .fasta import read_fasta, write_fasta, clean_fasta, FastaRecord
from .codons import (
    GENETIC_CODE,
    translate_sequence,
    find_orfs,
    calculate_rscu,
    CodonUsageProfile,
)
from .isochore import (
    classify_isochore_family,
    segment_isochores,
    compute_gc3_profile,
    IsochoreSegment,
)
from .crispr import CrisprGuideDesigner, SgRNACandidate
from .repeats import (
    TandemRepeat,
    TandemRepeatScanner,
    PATHOGENIC_REPEAT_MOTIFS,
)

__all__ = [
    "GenomicSequence",
    "FastaRecord",
    "read_fasta",
    "write_fasta",
    "clean_fasta",
    "GENETIC_CODE",
    "translate_sequence",
    "find_orfs",
    "calculate_rscu",
    "CodonUsageProfile",
    "classify_isochore_family",
    "segment_isochores",
    "compute_gc3_profile",
    "IsochoreSegment",
    "CrisprGuideDesigner",
    "SgRNACandidate",
    "TandemRepeat",
    "TandemRepeatScanner",
    "PATHOGENIC_REPEAT_MOTIFS",
]
