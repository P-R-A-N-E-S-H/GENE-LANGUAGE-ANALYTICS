import pytest
from gene_language.core.sequence import GenomicSequence
from gene_language.core.fasta import parse_fasta_string, clean_fasta


def test_genomic_sequence_basic():
    seq = GenomicSequence("ATGCGATCGA", identifier="test_1")
    assert len(seq) == 10
    assert seq.base_counts["A"] == 3
    assert seq.base_counts["T"] == 2
    assert seq.base_counts["G"] == 3
    assert seq.base_counts["C"] == 2
    assert seq.gc_content == 50.0
    assert seq.at_content == 50.0
    assert seq.gc_skew == 0.2


def test_reverse_complement_and_transcribe():
    seq = GenomicSequence("ATGC")
    rev = seq.reverse_complement()
    assert rev.sequence == "GCAT"
    assert seq.transcribe() == "AUGC"


def test_sliding_window_gc():
    seq = GenomicSequence("GGGGCCCCAAAATTTT")
    windows = seq.sliding_window_gc(window_size=8, step_size=4)
    assert len(windows) == 3
    assert windows[0]["gc_content"] == 100.0
    assert windows[2]["gc_content"] == 0.0


def test_cpg_ratio():
    seq = GenomicSequence("CGCG")
    ratio = seq.cpg_observed_expected_ratio()
    assert ratio > 0.0


def test_fasta_parsing():
    raw_fasta = ">seq1 Test Sequence\nATGCATGC\n>seq2 Another\nGGCCGGCC\n"
    records = parse_fasta_string(raw_fasta)
    assert len(records) == 2
    assert records[0].identifier == "seq1"
    assert records[0].sequence.sequence == "ATGCATGC"
    assert records[1].sequence.sequence == "GGCCGGCC"
