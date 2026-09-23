"""
Tests for genomic sequence embeddings and alignment-free similarity metrics.
"""

import pytest
import numpy as np
from gene_language.linguistics.embeddings import (
    GenomicEmbedding,
    TfidfGenomicEmbedding,
    cosine_similarity_dna,
    jaccard_similarity_dna,
    euclidean_distance_dna,
    jensen_shannon_divergence,
)


def test_cosine_and_jaccard_similarity():
    seq1 = "ATGCATGCATGC"
    seq2 = "ATGCATGCATGC"
    seq3 = "GGGGGGGGGGGG"

    # Identical sequences should yield similarity 1.0
    assert pytest.approx(cosine_similarity_dna(seq1, seq2, k=2), 0.001) == 1.0
    assert pytest.approx(jaccard_similarity_dna(seq1, seq2, k=2), 0.001) == 1.0
    assert pytest.approx(euclidean_distance_dna(seq1, seq2, k=2, normalized=True), 0.001) == 0.0
    assert pytest.approx(jensen_shannon_divergence(seq1, seq2, k=2), 0.001) == 0.0

    # Divergent sequences should have low similarity
    cos_diff = cosine_similarity_dna(seq1, seq3, k=2)
    assert cos_diff < 0.5


def test_genomic_embedding():
    embedder = GenomicEmbedding(k=3, normalize="l2")
    assert embedder.embedding_dim == 64

    seq = "ATGCGATCGATCGATC"
    vec = embedder.embed_sequence(seq)
    assert vec.shape == (64,)
    assert pytest.approx(np.linalg.norm(vec), 0.01) == 1.0

    batch = embedder.embed_batch([seq, "AAAAAAAAGGGGG"])
    assert batch.shape == (2, 64)


def test_tfidf_genomic_embedding():
    corpus = [
        "ATGCATGCATGCATGC",
        "GCGCGCGCGCGCGCGC",
        "AAAAAAAAAAAAATTT",
    ]
    tfidf = TfidfGenomicEmbedding(k=2)
    mat = tfidf.fit_transform(corpus)
    assert mat.shape == (3, 16)
