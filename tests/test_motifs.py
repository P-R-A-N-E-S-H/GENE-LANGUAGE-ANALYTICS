import pytest
from gene_language.models.motifs import MotifScanner


def test_motif_scanner():
    seq = "NNNTATAAANNNGCCACCATGGNNNGAATTCNNN"
    scanner = MotifScanner()
    matches = scanner.scan(seq)

    names = [m.motif_name for m in matches]
    assert "TATA Box" in names
    assert "Kozak Consensus (Initiation)" in names
    assert "EcoRI Recognition Site" in names
