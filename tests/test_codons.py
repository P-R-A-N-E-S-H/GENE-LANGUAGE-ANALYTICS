import pytest
from gene_language.core.codons import (
    GENETIC_CODE,
    translate_sequence,
    find_orfs,
    calculate_rscu,
)


def test_translation():
    # ATG = M (Met), GCT = A (Ala), TAA = * (Stop)
    seq = "ATGGCTTAA"
    protein = translate_sequence(seq)
    assert protein == "MA*"


def test_find_orfs():
    # Sequence with ATG ... TAA
    seq = "NNNATG" + ("GCC" * 35) + "TAANNN"
    orfs = find_orfs(seq, min_protein_len=30)
    assert len(orfs) >= 1
    assert orfs[0].length_aa >= 30
    assert orfs[0].strand == "+"


def test_rscu():
    seq = "GCCGCGGCAGCT" * 10
    profile = calculate_rscu(seq)
    assert profile.total_codons == 40
    assert profile.effective_number_of_codons > 0
