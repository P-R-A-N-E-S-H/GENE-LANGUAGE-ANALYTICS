import pytest
from gene_language.linguistics.subword import GenomicBpeTokenizer


def test_bpe_tokenizer_fit_and_tokenize():
    corpus = [
        "ATGCATGCATGCATGC",
        "ATGCGGCCATGCGGCC",
        "ATGCATGCGGCCGGCC"
    ]
    bpe = GenomicBpeTokenizer(vocab_size=32)
    bpe.fit(corpus, num_merges=5)

    assert len(bpe.merges) > 0

    tokens = bpe.tokenize("ATGCATGC")
    assert len(tokens) <= 8

    ids = bpe.encode("ATGCATGC")
    assert len(ids) == len(tokens)
