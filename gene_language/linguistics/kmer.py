"""
High-performance k-mer tokenization, frequency counting, and linguistic spectra.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Generator, Optional, Union
import itertools
from collections import Counter
import pandas as pd
import numpy as np


class KmerExtractor:
    """
    Genomic k-mer tokenizer and multi-scale feature extractor.
    """

    BASES = "ACGT"

    def __init__(self, k: int = 3, overlapping: bool = True, smoothing_eps: float = 1e-6):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.overlapping = overlapping
        self.smoothing_eps = smoothing_eps
        self._all_kmers = ["".join(p) for p in itertools.product(self.BASES, repeat=self.k)]
        self._kmer_index = {kmer: idx for idx, kmer in enumerate(self._all_kmers)}

    @property
    def vocabulary_size(self) -> int:
        """Total theoretical combinations 4^k."""
        return len(self._all_kmers)

    @property
    def vocabulary(self) -> List[str]:
        return list(self._all_kmers)

    def tokenize(self, sequence: str) -> List[str]:
        """Splits sequence into k-mers (sliding window)."""
        seq = "".join(sequence.split()).upper()
        step = 1 if self.overlapping else self.k
        return [seq[i:i + self.k] for i in range(0, len(seq) - self.k + 1, step)]

    def count_kmers(self, sequence: str) -> Dict[str, int]:
        """Counts raw occurrences of valid 4^k kmers in sequence."""
        tokens = self.tokenize(sequence)
        counts = Counter(t for t in tokens if all(b in self.BASES for b in t))
        return {kmer: counts.get(kmer, 0) for kmer in self._all_kmers}

    def relative_frequencies(self, sequence: str, smoothed: bool = True) -> Dict[str, float]:
        """
        Computes normalized relative frequencies f(kmer) = count / (total + eps).
        """
        raw_counts = self.count_kmers(sequence)
        total_observed = sum(raw_counts.values())
        denom = total_observed + (self.smoothing_eps if smoothed else 0.0)

        if denom == 0:
            return {kmer: 0.0 for kmer in self._all_kmers}

        return {kmer: round(count / denom, 6) for kmer, count in raw_counts.items()}

    def to_vector(self, sequence: str, normalized: bool = True) -> np.ndarray:
        """Converts sequence into a 1D numpy vector of k-mer frequencies."""
        freqs = self.relative_frequencies(sequence) if normalized else self.count_kmers(sequence)
        return np.array([freqs[kmer] for kmer in self._all_kmers], dtype=np.float32)

    def transform_dataframe(
        self,
        sequences: Union[List[str], List[tuple], pd.Series],
        sample_ids: Optional[List[str]] = None,
        prefix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extracts k-mer features across multiple sequences and returns a structured pandas DataFrame.
        """
        col_prefix = prefix if prefix else f"k{self.k}_"
        data = {}

        for idx, item in enumerate(sequences):
            if isinstance(item, tuple) and len(item) >= 2:
                s_id, seq = item[0], item[1]
            else:
                s_id = sample_ids[idx] if sample_ids and idx < len(sample_ids) else f"seq_{idx}"
                seq = str(item)

            freqs = self.relative_frequencies(seq)
            data[s_id] = {f"{col_prefix}{kmer}": val for kmer, val in freqs.items()}

        return pd.DataFrame.from_dict(data, orient="index")

    def top_kmers(self, sequence: str, top_n: int = 10) -> List[Tuple[str, int, float]]:
        """Returns top_n most frequent k-mers with counts and relative percentages."""
        counts = self.count_kmers(sequence)
        total = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            (kmer, count, round((count / total * 100.0) if total > 0 else 0.0, 4))
            for kmer, count in sorted_counts
        ]


def compute_kmer_spectrum(sequence: str, k_range: Tuple[int, ...] = (2, 3, 4, 5)) -> Dict[int, Dict[str, Union[int, float, List]]]:
    """
    Computes a multi-scale k-mer summary across different k lengths.
    """
    spectrum = {}
    for k in k_range:
        extractor = KmerExtractor(k=k)
        counts = extractor.count_kmers(sequence)
        total = sum(counts.values())
        observed = sum(1 for c in counts.values() if c > 0)
        vocab_size = extractor.vocabulary_size
        top = extractor.top_kmers(sequence, top_n=5)
        spectrum[k] = {
            "k": k,
            "total_tokens": total,
            "observed_unique_kmers": observed,
            "vocabulary_size": vocab_size,
            "diversity_ratio": round(observed / vocab_size, 4) if vocab_size > 0 else 0.0,
            "top_5": top
        }
    return spectrum
