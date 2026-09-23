"""
Tests for Core Promoter and TFBS Position Weight Matrix (PWM) scoring.
"""

import pytest
from gene_language.models.promoters import (
    PositionWeightMatrix,
    PromoterArchitectureScanner,
    PROMOTER_PWM_DEFINITIONS,
)


def test_pwm_scoring():
    tata_data = PROMOTER_PWM_DEFINITIONS["TATA_BOX"]["matrix"]
    pwm = PositionWeightMatrix(tata_data)

    # Perfect consensus sequence
    raw, rel = pwm.score_kmer("TATAAAAG")
    assert rel > 0.8
    assert raw > 0

    # Non-consensus GC-rich sequence
    raw_bad, rel_bad = pwm.score_kmer("GCGCGCGC")
    assert rel_bad < 0.4
    assert raw_bad < raw


def test_promoter_architecture_scanner():
    scanner = PromoterArchitectureScanner()
    # Sequence with TATA-box (-30) and Inr (YYANWYY)
    # TATAAAAG ... 25bp spacer ... CCATTTT
    promoter_seq = "GGGG" + "TATAAAAG" + ("N" * 25) + "CCATTTT" + "GGGG"
    elements = scanner.scan(promoter_seq, min_relative_score=0.70)

    assert len(elements) >= 2
    motifs_found = [e.motif_id for e in elements]
    assert "TATA_BOX" in motifs_found

    regions = scanner.identify_putative_promoter_regions(promoter_seq)
    assert len(regions) >= 1
