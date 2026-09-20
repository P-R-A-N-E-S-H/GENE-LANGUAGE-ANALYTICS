"""
Example 5: Isochore Family Classification and GC3 Codon Profiling
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language import read_fasta, segment_isochores, compute_gc3_profile

def main():
    fasta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sample_promoters.fasta"))
    records = read_fasta(fasta_path, max_records=1)

    if not records:
        return

    rec = records[0]
    seq_str = rec.sequence.sequence
    print(f"Analyzing Isochore Structure for: {rec.identifier} ({len(seq_str):,} bp)")

    # 1. Isochore Segmentation
    segments = segment_isochores(seq_str, window_size=300, step_size=200)
    print(f"\n[+] Isochore Compositional Domains (300bp windows):")
    for s in segments:
        print(f"  - [{s.start:>4}:{s.end:<4} bp] GC: {s.gc_percent:>5.2f}% -> Family: {s.family} ({s.description})")

    # 2. GC3 Codon Analysis
    gc3 = compute_gc3_profile(seq_str)
    print(f"\n[+] Codon Position GC Profile ({gc3['total_codons']} codons):")
    print(f"    GC at Position 1 (GC1): {gc3['gc1_percent']:.2f}%")
    print(f"    GC at Position 2 (GC2): {gc3['gc2_percent']:.2f}%")
    print(f"    GC at Position 3 (GC3): {gc3['gc3_percent']:.2f}% (Synonymous third codon bias)")

if __name__ == "__main__":
    main()
