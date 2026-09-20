import pytest
from gene_language.core.crispr import CrisprGuideDesigner


def test_crispr_guide_designer():
    # Construct sequence with 20nt spacer followed by AGG
    spacer = "GATCGATCGATCGATCGATC"
    pam = "AGG"
    seq = "NNNN" + spacer + pam + "NNNN"

    designer = CrisprGuideDesigner()
    guides = designer.find_guides(seq)

    assert len(guides) >= 1
    found_spacers = [g.spacer_sequence for g in guides]
    assert spacer in found_spacers
    assert guides[0].efficiency_score > 0
