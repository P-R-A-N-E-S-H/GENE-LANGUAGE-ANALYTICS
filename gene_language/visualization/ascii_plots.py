"""
Lightweight Terminal ASCII and Unicode Sparkline Visualizers for Genomics.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any


SPARKLINE_TICKS = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def ascii_sparkline(values: List[float], max_bars: int = 50) -> str:
    """
    Render numerical float series as an ASCII/Unicode sparkline string.
    """
    if not values:
        return ""

    # Downsample if needed
    n = len(values)
    if n > max_bars:
        step = n / float(max_bars)
        downsampled = []
        for i in range(max_bars):
            start_idx = int(i * step)
            end_idx = int((i + 1) * step)
            chunk = values[start_idx:end_idx]
            downsampled.append(sum(chunk) / len(chunk) if chunk else 0.0)
        values = downsampled

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val

    if val_range == 0.0:
        return "▅" * len(values)

    spark = []
    num_ticks = len(SPARKLINE_TICKS)
    for v in values:
        norm = (v - min_val) / val_range
        idx = int(norm * (num_ticks - 1))
        idx = max(0, min(num_ticks - 1, idx))
        spark.append(SPARKLINE_TICKS[idx])

    return "".join(spark)


def ascii_horizontal_bar(label: str, value: float, max_val: float, bar_width: int = 25, char: str = "█") -> str:
    """Render a single horizontal labeled bar."""
    ratio = value / max_val if max_val > 0 else 0.0
    filled = int(round(ratio * bar_width))
    bar_str = char * filled + " " * (bar_width - filled)
    return f"{label:>8} |{bar_str}| {value:.2f}"


def ascii_kmer_barplot(kmer_counts: List[Tuple[str, int, float]], top_n: int = 10, bar_width: int = 25) -> str:
    """
    Generate ASCII bar graph of top k-mers.
    """
    top = kmer_counts[:top_n]
    if not top:
        return "No k-mers."

    max_cnt = max(cnt for _, cnt, _ in top)
    lines = []
    for kmer, cnt, pct in top:
        bar = ascii_horizontal_bar(kmer, cnt, max_cnt, bar_width=bar_width)
        lines.append(f"{bar} ({pct:.1f}%)")
    return "\n".join(lines)


def ascii_composition_gauge(base_pcts: Dict[str, float], total_chars: int = 40) -> str:
    """
    Render stacked nucleotide gauge [A: red, C: blue, G: yellow, T: green].
    """
    a_pct = base_pcts.get("A", 25.0) / 100.0
    c_pct = base_pcts.get("C", 25.0) / 100.0
    g_pct = base_pcts.get("G", 25.0) / 100.0
    t_pct = base_pcts.get("T", 25.0) / 100.0

    a_len = int(round(a_pct * total_chars))
    c_len = int(round(c_pct * total_chars))
    g_len = int(round(g_pct * total_chars))
    t_len = max(0, total_chars - (a_len + c_len + g_len))

    gauge = "A" * a_len + "C" * c_len + "G" * g_len + "T" * t_len
    legend = f"A: {base_pcts.get('A', 0.0):.1f}% | C: {base_pcts.get('C', 0.0):.1f}% | G: {base_pcts.get('G', 0.0):.1f}% | T: {base_pcts.get('T', 0.0):.1f}%"
    return f"[{gauge}]\n{legend}"
