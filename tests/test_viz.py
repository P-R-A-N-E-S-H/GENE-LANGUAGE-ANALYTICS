"""
Tests for ASCII and terminal visualization functions.
"""

import pytest
from gene_language.visualization.ascii_plots import (
    ascii_sparkline,
    ascii_horizontal_bar,
    ascii_kmer_barplot,
    ascii_composition_gauge,
)


def test_ascii_sparkline():
    vals = [10.0, 20.0, 50.0, 80.0, 100.0, 40.0, 10.0]
    spark = ascii_sparkline(vals)
    assert len(spark) == len(vals)
    assert isinstance(spark, str)


def test_ascii_horizontal_bar():
    bar = ascii_horizontal_bar("ATG", 50.0, 100.0, bar_width=20)
    assert "ATG" in bar
    assert "|" in bar


def test_ascii_kmer_barplot():
    kmers = [("AAA", 100, 50.0), ("TTT", 50, 25.0), ("GGG", 50, 25.0)]
    plot = ascii_kmer_barplot(kmers, top_n=3)
    assert "AAA" in plot
    assert "TTT" in plot


def test_ascii_composition_gauge():
    pcts = {"A": 30.0, "C": 20.0, "G": 20.0, "T": 30.0}
    gauge = ascii_composition_gauge(pcts, total_chars=20)
    assert "A: 30.0%" in gauge
