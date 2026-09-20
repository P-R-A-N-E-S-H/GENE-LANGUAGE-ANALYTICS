"""
Genomic linguistic analytics: k-mer tokenization, information entropy, Markov transitions, BPE subwords, vectorization, and mutual information.
"""

from .kmer import KmerExtractor, compute_kmer_spectrum
from .entropy import (
    shannon_entropy,
    renyi_entropy,
    linguistic_complexity,
    zipf_power_law_fit,
)
from .markov import MarkovModelDNA
from .vectorizer import GenomicVectorizer
from .subword import GenomicBpeTokenizer
from .correlation import mutual_information_lag, jensen_shannon_divergence

__all__ = [
    "KmerExtractor",
    "compute_kmer_spectrum",
    "shannon_entropy",
    "renyi_entropy",
    "linguistic_complexity",
    "zipf_power_law_fit",
    "MarkovModelDNA",
    "GenomicVectorizer",
    "GenomicBpeTokenizer",
    "mutual_information_lag",
    "jensen_shannon_divergence",
]
