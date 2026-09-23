"""
Genomic Sequence Embeddings and Alignment-Free Distance Metrics.
"""

from __future__ import annotations
import math
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from .kmer import KmerExtractor


def _kmer_counts(seq: str, k: int) -> Dict[str, int]:
    """Helper to extract non-zero k-mer counts."""
    extractor = KmerExtractor(k=k)
    return extractor.count_kmers(seq)


def cosine_similarity_dna(seq_a: str, seq_b: str, k: int = 3) -> float:
    """
    Compute Cosine similarity between two DNA sequences based on k-mer frequency vectors.

    Parameters
    ----------
    seq_a : str
        First DNA sequence.
    seq_b : str
        Second DNA sequence.
    k : int
        K-mer size (default: 3).

    Returns
    -------
    float
        Cosine similarity in range [0.0, 1.0].
    """
    counts_a = _kmer_counts(seq_a, k)
    counts_b = _kmer_counts(seq_b, k)

    all_kmers = set(counts_a.keys()).union(set(counts_b.keys()))
    if not all_kmers:
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for kmer in all_kmers:
        val_a = counts_a.get(kmer, 0)
        val_b = counts_b.get(kmer, 0)
        dot_product += val_a * val_b
        norm_a += val_a * val_a
        norm_b += val_b * val_b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b)))


def jaccard_similarity_dna(seq_a: str, seq_b: str, k: int = 3) -> float:
    """
    Compute Jaccard set similarity between k-mer sets of two DNA sequences.

    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    counts_a = _kmer_counts(seq_a, k)
    counts_b = _kmer_counts(seq_b, k)

    set_a = {kmer for kmer, count in counts_a.items() if count > 0}
    set_b = {kmer for kmer, count in counts_b.items() if count > 0}

    union = set_a.union(set_b)
    if not union:
        return 0.0
    intersection = set_a.intersection(set_b)
    return float(len(intersection) / len(union))


def euclidean_distance_dna(seq_a: str, seq_b: str, k: int = 3, normalized: bool = True) -> float:
    """
    Compute Euclidean distance between k-mer frequency vectors.
    """
    counts_a = _kmer_counts(seq_a, k)
    counts_b = _kmer_counts(seq_b, k)

    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    all_kmers = set(counts_a.keys()).union(set(counts_b.keys()))
    if not all_kmers:
        return 0.0

    dist_sq = 0.0
    for kmer in all_kmers:
        val_a = (counts_a.get(kmer, 0) / total_a) if (normalized and total_a > 0) else counts_a.get(kmer, 0)
        val_b = (counts_b.get(kmer, 0) / total_b) if (normalized and total_b > 0) else counts_b.get(kmer, 0)
        diff = val_a - val_b
        dist_sq += diff * diff

    return float(math.sqrt(dist_sq))


def jensen_shannon_divergence(seq_a: str, seq_b: str, k: int = 3) -> float:
    """
    Compute Jensen-Shannon Divergence (JSD) between k-mer probability distributions.
    JSD is symmetric and bounded in [0, 1] (using log2).
    """
    counts_a = _kmer_counts(seq_a, k)
    counts_b = _kmer_counts(seq_b, k)

    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    if total_a == 0 or total_b == 0:
        return 1.0

    all_kmers = sorted(list(set(counts_a.keys()).union(set(counts_b.keys()))))
    
    # Smooth with pseudocount
    eps = 1e-12
    p = np.array([counts_a.get(km, 0) / total_a for km in all_kmers], dtype=np.float64)
    q = np.array([counts_b.get(km, 0) / total_b for km in all_kmers], dtype=np.float64)

    # Replace zeros with epsilon and renormalize
    p = (p + eps) / (1.0 + eps * len(all_kmers))
    q = (q + eps) / (1.0 + eps * len(all_kmers))

    m = 0.5 * (p + q)

    def _kl(dist_x, dist_y):
        return np.sum(dist_x * np.log2(dist_x / dist_y))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.clip(jsd, 0.0, 1.0))


class GenomicEmbedding:
    """
    Embeds DNA sequences into dense k-mer numerical vectors with optional normalization.
    """

    def __init__(self, k: int = 3, normalize: str = "l2"):
        """
        Parameters
        ----------
        k : int
            K-mer word size (e.g. 3 for triplets/codons).
        normalize : str
            Normalization mode: 'l2', 'l1', 'freq', or 'none'.
        """
        self.k = k
        self.normalize = normalize.lower()
        self.extractor = KmerExtractor(k=k)
        self.vocab = self.extractor.generate_kmer_vocab()
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.vocab)}

    @property
    def embedding_dim(self) -> int:
        return len(self.vocab)

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Embed a single DNA sequence into an array of shape (4^k,)."""
        counts = self.extractor.count_kmers(sequence)
        vec = np.zeros(self.embedding_dim, dtype=np.float32)

        for kmer, count in counts.items():
            if kmer in self.kmer_to_idx:
                vec[self.kmer_to_idx[kmer]] = count

        if self.normalize == "l2":
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        elif self.normalize in ("l1", "freq"):
            s = np.sum(vec)
            if s > 0:
                vec = vec / s

        return vec

    def embed_batch(self, sequences: List[str]) -> np.ndarray:
        """Embed a list of DNA sequences into a matrix of shape (N, 4^k)."""
        return np.vstack([self.embed_sequence(seq) for seq in sequences])


class TfidfGenomicEmbedding:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) Embedding for genomic corpora.
    Useful for discovering discriminative k-mer motifs across distinct genomic classes.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.extractor = KmerExtractor(k=k)
        self.vocab = self.extractor.generate_kmer_vocab()
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.vocab)}
        self.idf_ = np.zeros(len(self.vocab), dtype=np.float32)
        self.is_fitted = False

    def fit(self, sequences: List[str]) -> "TfidfGenomicEmbedding":
        """Compute IDF weights across a corpus of sequences."""
        num_docs = len(sequences)
        doc_freq = np.zeros(len(self.vocab), dtype=np.float32)

        for seq in sequences:
            seen_kmers = set()
            seq_clean = seq.upper()
            for i in range(len(seq_clean) - self.k + 1):
                kmer = seq_clean[i:i + self.k]
                if kmer in self.kmer_to_idx:
                    seen_kmers.add(self.kmer_to_idx[kmer])
            for idx in seen_kmers:
                doc_freq[idx] += 1.0

        # Smooth IDF: ln((1 + N) / (1 + DF)) + 1
        self.idf_ = np.log((1.0 + num_docs) / (1.0 + doc_freq)) + 1.0
        self.is_fitted = True
        return self

    def transform(self, sequences: List[str]) -> np.ndarray:
        """Transform sequences into TF-IDF normalized vector matrix."""
        if not self.is_fitted:
            raise ValueError("TfidfGenomicEmbedding must be fitted before transforming.")

        embeddings = []
        for seq in sequences:
            counts = self.extractor.count_kmers(seq)
            tf = np.zeros(len(self.vocab), dtype=np.float32)
            for kmer, cnt in counts.items():
                if kmer in self.kmer_to_idx:
                    tf[self.kmer_to_idx[kmer]] = cnt

            total = np.sum(tf)
            if total > 0:
                tf = tf / total

            tfidf = tf * self.idf_
            norm = np.linalg.norm(tfidf)
            if norm > 0:
                tfidf = tfidf / norm
            embeddings.append(tfidf)

        return np.vstack(embeddings)

    def fit_transform(self, sequences: List[str]) -> np.ndarray:
        return self.fit(sequences).transform(sequences)
