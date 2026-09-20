import pytest
from gene_language.linguistics.kmer import KmerExtractor, compute_kmer_spectrum


def test_kmer_extractor_vocab():
    ext3 = KmerExtractor(k=3)
    assert ext3.vocabulary_size == 64
    assert len(ext3.vocabulary) == 64

    ext2 = KmerExtractor(k=2)
    assert ext2.vocabulary_size == 16


def test_kmer_counts_and_frequencies():
    ext = KmerExtractor(k=3)
    counts = ext.count_kmers("AAAAA")
    assert counts["AAA"] == 3
    assert counts["AAC"] == 0

    freqs = ext.relative_frequencies("AAAAA")
    assert freqs["AAA"] > 0.99


def test_kmer_spectrum():
    spectrum = compute_kmer_spectrum("ATGCGATCGATCGATC", k_range=(2, 3))
    assert 2 in spectrum
    assert 3 in spectrum
    assert spectrum[2]["observed_unique_kmers"] > 0
