import pytest
from gene_language.models.splice_junction import SpliceJunctionScorer


def test_splice_scorer():
    # Construct sequence with consensus donor CAG|GTAAGT
    donor_seq = "NNNCAGGTAAGTNNN"
    scorer = SpliceJunctionScorer(donor_threshold=2.0, acceptor_threshold=2.0)
    donors = scorer.scan_donors(donor_seq)

    assert len(donors) >= 1
    assert donors[0].site_type == "5_DONOR"
    assert donors[0].score > 3.0


def test_splice_acceptor():
    # Construct sequence with polypyrimidine tract and AG: TTTTTTTTTTNCAG|G
    acceptor_seq = "NNNTTTTTTTTTTCAGAGNNN"
    scorer = SpliceJunctionScorer(donor_threshold=2.0, acceptor_threshold=2.0)
    acceptors = scorer.scan_acceptors(acceptor_seq)

    assert len(acceptors) >= 1
    assert acceptors[0].site_type == "3_ACCEPTOR"
