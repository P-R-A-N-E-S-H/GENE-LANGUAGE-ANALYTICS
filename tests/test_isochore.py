import pytest
from gene_language.core.isochore import classify_isochore_family, segment_isochores, compute_gc3_profile


def test_isochore_classification():
    fam_l1, _ = classify_isochore_family(35.0)
    assert fam_l1 == "L1"

    fam_h3, _ = classify_isochore_family(60.0)
    assert fam_h3 == "H3"


def test_isochore_segmentation():
    seq = "AT" * 500 + "GC" * 500
    segments = segment_isochores(seq, window_size=500, step_size=500)
    assert len(segments) == 4
    assert segments[0].family == "L1"
    assert segments[1].family == "L1"
    assert segments[2].family == "H3"
    assert segments[3].family == "H3"


def test_gc3_profile():
    # Sequence where every 3rd base is G or C: ATG GCG CTG
    seq = "ATGGCGCTG"
    res = compute_gc3_profile(seq)
    assert res["gc3_percent"] == 100.0
