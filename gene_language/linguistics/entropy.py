"""
Genomic information theory, Shannon entropy, Renyi entropy, linguistic complexity, and Zipf's law.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
from collections import Counter
import numpy as np
from .kmer import KmerExtractor


def shannon_entropy(sequence: str, k: int = 1, base: float = 2.0) -> Dict[str, float]:
    """
    Computes Shannon Information Entropy H(X) = - sum(p_i * log_base(p_i))
    and Normalized Entropy H_norm = H(X) / (k * log_base(4)).
    """
    extractor = KmerExtractor(k=k)
    counts = extractor.count_kmers(sequence)
    total = sum(counts.values())

    if total == 0:
        return {"entropy": 0.0, "max_entropy": 2.0 * k, "normalized_entropy": 0.0, "k": k}

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * (math.log(p) / math.log(base))

    max_entropy = (2.0 * k) if base == 2.0 else (k * math.log(4.0) / math.log(base))
    normalized = round(entropy / max_entropy, 6) if max_entropy > 0 else 0.0

    return {
        "entropy": round(entropy, 6),
        "max_entropy": round(max_entropy, 6),
        "normalized_entropy": min(1.0, max(0.0, normalized)),
        "k": k,
        "base": base
    }


def renyi_entropy(sequence: str, alpha: float = 2.0, k: int = 1) -> float:
    """
    Computes Renyi Generalized Entropy of order alpha:
    H_alpha = (1 / (1 - alpha)) * log2( sum( p_i^alpha ) )
    """
    if alpha <= 0 or alpha == 1.0:
        # For alpha=1, Renyi converges to Shannon entropy
        return shannon_entropy(sequence, k=k)["entropy"]

    extractor = KmerExtractor(k=k)
    counts = extractor.count_kmers(sequence)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    sum_p_alpha = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            sum_p_alpha += p ** alpha

    if sum_p_alpha <= 0:
        return 0.0

    val = (1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha)
    return round(val, 6)


def linguistic_complexity(sequence: str, max_k: int = 6) -> Dict[str, float]:
    """
    Computes Trifonov & Ulanovsky Linguistic Complexity:
    LC = prod_{k=1}^{K_max} ( V_obs(k) / min(L - k + 1, 4^k) )

    LC is in range [0, 1], where higher values denote maximum vocabulary richness,
    and lower values denote repetitive/low-complexity regions (e.g. microsatellites).
    """
    seq = "".join(sequence.split()).upper()
    L = len(seq)
    if L == 0:
        return {"linguistic_complexity": 0.0, "max_k": max_k}

    product_terms = []
    k_details = {}

    for k in range(1, min(max_k + 1, L + 1)):
        tokens = [seq[i:i + k] for i in range(L - k + 1)]
        v_obs = len(set(tokens))
        v_max = min(L - k + 1, 4 ** k)
        ratio = (v_obs / v_max) if v_max > 0 else 0.0
        product_terms.append(ratio)
        k_details[f"k_{k}"] = {
            "observed_vocab": v_obs,
            "max_possible_vocab": v_max,
            "ratio": round(ratio, 4)
        }

    lc = 1.0
    for term in product_terms:
        lc *= term

    return {
        "linguistic_complexity": round(lc, 6),
        "geometric_mean_richness": round(lc ** (1.0 / len(product_terms)), 6) if product_terms else 0.0,
        "max_k": len(product_terms),
        "k_ratios": k_details
    }


def zipf_power_law_fit(sequence: str, k: int = 3) -> Dict[str, Union[float, List[dict]]]:
    """
    Tests whether genomic k-mer frequency distribution follows Zipf's Law:
    f(r) = C * r^(-alpha) => log(f) = log(C) - alpha * log(r)
    Fits log-log linear model and returns slope alpha, R^2, and top ranked tokens.
    """
    extractor = KmerExtractor(k=k)
    counts = extractor.count_kmers(sequence)
    sorted_items = sorted([c for c in counts.values() if c > 0], reverse=True)

    if len(sorted_items) < 3:
        return {"alpha": 0.0, "r_squared": 0.0, "valid": False, "points": []}

    ranks = np.arange(1, len(sorted_items) + 1)
    freqs = np.array(sorted_items, dtype=np.float64)

    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)

    # Linear regression
    slope, intercept = np.polyfit(log_ranks, log_freqs, deg=1)
    y_pred = slope * log_ranks + intercept

    # Compute R^2
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    ss_res = np.sum((log_freqs - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    points = [
        {"rank": int(r), "freq": int(f), "log_rank": round(float(lr), 4), "log_freq": round(float(lf), 4)}
        for r, f, lr, lf in zip(ranks[:30], freqs[:30], log_ranks[:30], log_freqs[:30])
    ]

    return {
        "alpha": round(-float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r2), 4),
        "valid": True,
        "total_unique_tokens": len(sorted_items),
        "points": points
    }
