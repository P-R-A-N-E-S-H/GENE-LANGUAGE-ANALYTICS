"""
Publication-Quality Genomic Data Visualizations (Matplotlib & Seaborn wrappers).
"""

from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional, Any

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for graphical plotting. Install with 'pip install matplotlib'.")


def plot_gc_profile(
    sliding_gc_values: List[float],
    title: str = "Sliding Window GC-Content Profile",
    window_size: int = 100,
    output_path: Optional[str] = None
) -> Any:
    """
    Plot sliding window GC-content percentage curve with baseline and threshold lines.
    """
    check_matplotlib()
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    ax.plot(sliding_gc_values, color="#00f5d4", linewidth=1.5, label=f"GC% (Window={window_size}bp)")
    ax.axhline(50.0, color="#ffffff", linestyle="--", alpha=0.4, label="Equilibrium 50% GC")
    ax.fill_between(range(len(sliding_gc_values)), sliding_gc_values, 50.0, alpha=0.15, color="#00f5d4")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Window Index", fontsize=10)
    ax.set_ylabel("GC Content (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig


def plot_kmer_spectrum(
    kmer_counts: List[Tuple[str, int, float]],
    title: str = "Top K-mer Frequency Spectrum",
    top_n: int = 15,
    output_path: Optional[str] = None
) -> Any:
    """
    Plot ranked bar chart of top k-mer frequencies.
    """
    check_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    subset = kmer_counts[:top_n]
    kmers = [k[0] for k in subset]
    pcts = [k[2] for k in subset]

    colors = plt.cm.viridis([i / len(subset) for i in range(len(subset))])
    bars = ax.bar(kmers, pcts, color=colors, edgecolor="none", width=0.6)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("K-mer Token", fontsize=10)
    ax.set_ylabel("Relative Frequency (%)", fontsize=10)
    plt.xticks(rotation=45, ha="right")

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.1, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig
