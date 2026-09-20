import pytest
from gene_language.linguistics.entropy import (
    shannon_entropy,
    renyi_entropy,
    linguistic_complexity,
    zipf_power_law_fit,
)
from gene_language.linguistics.markov import MarkovModelDNA


def test_shannon_entropy():
    # Maximum entropy for uniform A, C, G, T is 2.0 bits
    uniform_seq = "ACGT" * 100
    res = shannon_entropy(uniform_seq, k=1)
    assert abs(res["entropy"] - 2.0) < 0.05
    assert abs(res["normalized_entropy"] - 1.0) < 0.05

    # Low entropy for homopolymer
    low_seq = "AAAA" * 100
    res_low = shannon_entropy(low_seq, k=1)
    assert res_low["entropy"] < 0.1


def test_linguistic_complexity():
    seq = "ATGCGATCGATCGATC"
    lc = linguistic_complexity(seq, max_k=4)
    assert 0.0 <= lc["linguistic_complexity"] <= 1.0


def test_zipf_fit():
    seq = "ATGCGATCGATCGATCGATCGATC" * 5
    fit = zipf_power_law_fit(seq, k=2)
    assert fit["valid"] is True
    assert "alpha" in fit


def test_markov_model():
    seq = "ACGTACGTACGT"
    model = MarkovModelDNA(order=1)
    model.fit(seq)
    matrix = model.to_dict()
    assert "A" in matrix
    assert "C" in matrix["A"]
    assert model.transition_matrix.shape == (4, 4)
