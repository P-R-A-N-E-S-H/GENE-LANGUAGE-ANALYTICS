"""
Genomic data visualization: terminal ASCII sparklines and publication plots.
"""

from .ascii_plots import (
    ascii_sparkline,
    ascii_horizontal_bar,
    ascii_kmer_barplot,
    ascii_composition_gauge,
)
from .plots import (
    plot_gc_profile,
    plot_kmer_spectrum,
    HAS_MATPLOTLIB,
)

__all__ = [
    "ascii_sparkline",
    "ascii_horizontal_bar",
    "ascii_kmer_barplot",
    "ascii_composition_gauge",
    "plot_gc_profile",
    "plot_kmer_spectrum",
    "HAS_MATPLOTLIB",
]
