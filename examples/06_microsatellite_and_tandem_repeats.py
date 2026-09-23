"""
Example 06: Microsatellite & Short Tandem Repeat (STR) Discovery.
=================================================================
Demonstrates scanning genomic DNA for tandem repeats, detecting pathogenic
trinucleotide expansions (e.g. Huntington's CAG repeat), and telomere profiling.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language.core.repeats import TandemRepeatScanner


def main():
    print("=" * 70)
    print("[*] GENE-LANGUAGE-ANALYTICS: Tandem Repeat & Microsatellite Scanner")
    print("=" * 70)

    # 1. Synthetic sequence containing dinucleotide, trinucleotide, and telomeric repeats
    synthetic_dna = (
        "ATGCGATCGA"
        + "AT" * 8               # (AT)8 dinucleotide repeat
        + "GCGCGCGC"
        + "CAG" * 42             # Pathogenic Huntington's CAG expansion tract
        + "TTTTTT"
        + "TTAGGG" * 6           # Canonical vertebrate telomere tract
        + "GATCGATCGATC"
    )

    print(f"\nAnalyzing Sequence ({len(synthetic_dna)} bp):")
    print(synthetic_dna[:60] + "... [truncated]")

    scanner = TandemRepeatScanner()
    all_repeats = scanner.scan(synthetic_dna, min_copies=3)

    print(f"\n[+] Discovered {len(all_repeats)} Tandem Repeat Tracts:")
    for r in all_repeats:
        dis = f" [! Pathogenic: {r.disease_associated}]" if r.is_pathogenic_risk else ""
        print(f"  * Unit: {r.unit:<6} | Copies: {r.copies:.1f}x | Range: {r.start:>3}-{r.end:<3} | Len: {r.total_length} bp{dis}")

    # 2. Pathogenic Expansions Filter
    expansions = scanner.scan_pathogenic_expansions(synthetic_dna, min_copies=4)
    print(f"\n[+] Pathogenic Expansion Candidates ({len(expansions)} found):")
    for exp in expansions:
        print(f"  * Motif: {exp.unit} | Disease: {exp.disease_associated} | Copies: {exp.copies:.1f}x")

    # 3. Telomeric Profile
    telomeres = scanner.telomeric_profile(synthetic_dna)
    print("\n[+] Telomeric Repeat Profile:")
    print(f"  * Forward TTAGGG count: {telomeres['forward_telomere_count_TTAGGG']}")
    print(f"  * Telomeric fraction: {telomeres['telomeric_percentage']:.2f}% of total sequence")


if __name__ == "__main__":
    main()
