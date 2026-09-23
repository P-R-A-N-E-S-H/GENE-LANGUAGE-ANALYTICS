"""
Tests for microsatellite and Short Tandem Repeat (STR) scanner.
"""

import pytest
from gene_language.core.repeats import TandemRepeatScanner, PATHOGENIC_REPEAT_MOTIFS


def test_tandem_repeat_scanner():
    scanner = TandemRepeatScanner()
    # Sequence with dinucleotide repeat (AT)x5 and trinucleotide repeat (CAG)x6
    seq = "NNNNATATATATATNNNNCAGCAGCAGCAGCAGCAGNNNN"
    repeats = scanner.scan(seq, min_copies=3)

    assert len(repeats) >= 2
    units = [r.unit for r in repeats]
    assert "AT" in units
    assert "CAG" in units


def test_pathogenic_expansion_detection():
    scanner = TandemRepeatScanner()
    # Huntington's CAG expansion tract
    seq = "ATGCGAT" + "CAG" * 40 + "GATCGATC"
    disease_repeats = scanner.scan_pathogenic_expansions(seq, min_copies=4)

    assert len(disease_repeats) >= 1
    cag_rep = disease_repeats[0]
    assert cag_rep.unit == "CAG"
    assert cag_rep.copies >= 40
    assert "Huntington" in cag_rep.disease_associated
    assert cag_rep.is_pathogenic_risk is True


def test_telomeric_profile():
    scanner = TandemRepeatScanner()
    seq = "TTAGGGTTAGGGTTAGGG" + "NNNN" + "CCCTAACCCTAACCCTAA"
    profile = scanner.telomeric_profile(seq)

    assert profile["forward_telomere_count_TTAGGG"] == 3
    assert profile["reverse_telomere_count_CCCTAA"] == 3
    assert profile["total_telomeric_bp"] == 36
    assert profile["telomeric_percentage"] > 0
