"""
Gene Language Analytics
=======================
Decoding the Language of Life using Natural Language Processing and Machine Learning.
"""

__version__ = "2.0.0"
__author__ = "Pranesh M"

from .core.sequence import GenomicSequence
from .core.fasta import read_fasta, write_fasta, clean_fasta, FastaRecord
from .core.codons import (
    GENETIC_CODE,
    translate_sequence,
    find_orfs,
    calculate_rscu,
    CodonUsageProfile,
)
from .linguistics.kmer import KmerExtractor, compute_kmer_spectrum
from .linguistics.entropy import (
    shannon_entropy,
    renyi_entropy,
    linguistic_complexity,
    zipf_power_law_fit,
)
from .linguistics.markov import MarkovModelDNA
from .linguistics.vectorizer import GenomicVectorizer
from .models.classifier import ExonIntronClassifier
from .models.motifs import MotifScanner, COMMON_MOTIFS

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
    "KmerExtractor",
    "compute_kmer_spectrum",
    "shannon_entropy",
    "renyi_entropy",
    "linguistic_complexity",
    "zipf_power_law_fit",
    "MarkovModelDNA",
    "GenomicVectorizer",
    "ExonIntronClassifier",
    "MotifScanner",
    "COMMON_MOTIFS",
]
