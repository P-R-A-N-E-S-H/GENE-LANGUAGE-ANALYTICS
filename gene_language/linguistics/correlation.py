"""
Mutual Information across nucleotide distance lags and genomic distance metrics (Jensen-Shannon, Cosine).
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import math
import numpy as np
from collections import Counter
from .kmer import KmerExtractor


def mutual_information_lag(sequence: str, max_lag: int = 20) -> Dict[int, float]:
    """
    Calculates genomic Mutual Information I(X; Y_d) between bases separated by lag distance d:
    I(d) = sum_{x,y} P(x, y; d) * log2( P(x, y; d) / (P(x) * P(y)) )
    Captures long-range periodicities such as the 3-base codon periodicity in coding regions.
    """
    seq = "".join(sequence.split()).upper()
    bases = ["A", "C", "G", "T"]
    L = len(seq)

    if L < max_lag + 2:
        return {d: 0.0 for d in range(1, max_lag + 1)}

    # Base marginal probabilities
    counts = Counter(b for b in seq if b in bases)
    tot = sum(counts.values())
    if tot == 0:
        return {d: 0.0 for d in range(1, max_lag + 1)}
    p_marginal = {b: (counts.get(b, 0) / tot) for b in bases}

    mi_lags = {}

    for d in range(1, max_lag + 1):
        pair_counts = Counter()
        pair_total = 0
        for i in range(L - d):
            b1 = seq[i]
            b2 = seq[i + d]
            if b1 in bases and b2 in bases:
                pair_counts[(b1, b2)] += 1
                pair_total += 1

        if pair_total == 0:
            mi_lags[d] = 0.0
            continue

        mi = 0.0
        for (b1, b2), cnt in pair_counts.items():
            p_joint = cnt / pair_total
            p_indep = p_marginal[b1] * p_marginal[b2]
            if p_joint > 0 and p_indep > 0:
                mi += p_joint * math.log2(p_joint / p_indep)

        mi_lags[d] = round(max(0.0, float(mi)), 5)

    return mi_lags


def jensen_shannon_divergence(seq1: str, seq2: str, k: int = 3) -> float:
    """
    Computes Jensen-Shannon divergence (JSD) between k-mer distributions of two genomic sequences.
    JSD is symmetric, bounded in [0, 1] (when using base 2), and forms a true metric space.
    """
    ext = KmerExtractor(k=k)
    f1 = ext.relative_frequencies(seq1)
    f2 = ext.relative_frequencies(seq2)

    p = np.array([f1[km] for km in ext.vocabulary], dtype=np.float64)
    q = np.array([f2[km] for km in ext.vocabulary], dtype=np.float64)

    # Normalize
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)

    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return round(float(np.sqrt(max(0.0, jsd))), 4)
