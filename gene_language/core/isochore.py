"""
Genomic Isochore structure and GC3 codon compositional bias profiling (Bernardi classification).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .sequence import GenomicSequence

# Bernardi Human Isochore Classifications:
# L1: < 38% GC (Very AT-rich, gene poor, late replicating)
# L2: 38% - 42% GC (AT-rich)
# H1: 42% - 47% GC (Moderate GC, gene dense)
# H2: 47% - 52% GC (High GC, very gene dense)
# H3: > 52% GC (Extremely GC-rich, highest gene & CpG density)

ISOCHORE_FAMILIES = [
    ("L1", 0.0, 38.0, "Very AT-rich, low gene density"),
    ("L2", 38.0, 42.0, "AT-rich, moderate gene density"),
    ("H1", 42.0, 47.0, "Moderate GC, high gene density"),
    ("H2", 47.0, 52.0, "High GC, very high gene density"),
    ("H3", 52.0, 100.0, "Extremely GC-rich, maximum gene & CpG island density"),
]


@dataclass
class IsochoreSegment:
    start: int
    end: int
    length: int
    gc_percent: float
    family: str
    description: str


def classify_isochore_family(gc_percent: float) -> Tuple[str, str]:
    """Returns Isochore family name (L1, L2, H1, H2, H3) and biological description."""
    for name, low, high, desc in ISOCHORE_FAMILIES:
        if low <= gc_percent < high:
            return name, desc
    return "H3", "Extremely GC-rich, maximum gene density"


def segment_isochores(
    sequence: str,
    window_size: int = 1000,
    step_size: int = 500
) -> List[IsochoreSegment]:
    """
    Partitions long genomic contigs into Isochore compositional domains.
    """
    seq = "".join(sequence.split()).upper()
    L = len(seq)
    segments = []

    if L < window_size:
        seq_obj = GenomicSequence(seq)
        gc = seq_obj.gc_content
        fam, desc = classify_isochore_family(gc)
        return [IsochoreSegment(start=0, end=L, length=L, gc_percent=gc, family=fam, description=desc)]

    for start in range(0, L - window_size + 1, step_size):
        end = start + window_size
        chunk = seq[start:end]
        gc = ((chunk.count("G") + chunk.count("C")) / len(chunk)) * 100.0
        fam, desc = classify_isochore_family(gc)
        segments.append(
            IsochoreSegment(
                start=start,
                end=end,
                length=window_size,
                gc_percent=round(gc, 2),
                family=fam,
                description=desc
            )
        )
    return segments


def compute_gc3_profile(sequence: str) -> Dict[str, float]:
    """
    Computes GC content specifically at the 3rd codon positions (GC3 / GC3s).
    GC3 is a classical marker for thermal stability, translation efficiency, and isochore correlation.
    """
    seq = "".join(sequence.split()).upper()
    codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3) if len(seq[i:i + 3]) == 3]
    if not codons:
        return {"gc3_percent": 0.0, "total_codons": 0}

    gc3_count = sum(1 for c in codons if c[2] in "GC")
    gc1_count = sum(1 for c in codons if c[0] in "GC")
    gc2_count = sum(1 for c in codons if c[1] in "GC")
    tot = len(codons)

    return {
        "gc1_percent": round((gc1_count / tot) * 100.0, 2),
        "gc2_percent": round((gc2_count / tot) * 100.0, 2),
        "gc3_percent": round((gc3_count / tot) * 100.0, 2),
        "total_codons": tot
    }
