"""
Example 07: Core Promoter & Transcription Factor Binding Site (TFBS) Scoring.
==============================================================================
Demonstrates Position Weight Matrix (PWM) scoring for core promoter elements
(TATA-box, Initiator Inr, DPE, BRE, Sp1) and predicting Transcription Start Sites (TSS).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language.models.promoters import PromoterArchitectureScanner


def main():
    print("=" * 70)
    print("[*] GENE-LANGUAGE-ANALYTICS: Core Promoter & TSS Architecture Scanner")
    print("=" * 70)

    # Synthetic eukaryotic promoter with BRE, TATA-box, Inr (+1), and DPE
    # Coordinates relative to predicted TSS at pos 50:
    # BRE (-35): ~pos 15
    # TATA (-30): ~pos 20 (TATAAAAG)
    # Inr (-2 to +4): ~pos 48 (CCATTTT)
    # DPE (+28): ~pos 78 (AGACGTG)
    promoter_dna = (
        "GCGCGC"
        + "GGGCGG"              # Sp1 GC Box (-44)
        + "TATAAAAG"            # TATA Box (-30)
        + "ACGTACGTACGTACGT"
        + "CCATTTT"             # Initiator Inr (TSS anchor)
        + "ACGTACGTACGTACGTACGTACGT"
        + "AGACG"               # Downstream Promoter Element DPE
        + "ATCGATCGATC"
    )

    print(f"\nAnalyzing Promoter Sequence ({len(promoter_dna)} bp):")
    print(promoter_dna)

    scanner = PromoterArchitectureScanner()
    elements = scanner.scan(promoter_dna, min_relative_score=0.70)

    print(f"\n[+] Detected {len(elements)} Core Promoter Elements:")
    for el in elements:
        print(f"  * {el.name:<30} | Pos: {el.start:>2}-{el.end:<2} | PWM Score: {el.raw_score:>5.2f} bits | Match: {el.sequence}")

    architectures = scanner.identify_putative_promoter_regions(promoter_dna)
    print(f"\n[+] Identified {len(architectures)} Putative TSS Architectures:")
    for arch in architectures:
        print(f"  * Predicted TSS: Base +{arch['predicted_tss']}")
        print(f"    Architecture: {arch['architecture_type']}")
        if arch['tata_box']:
            print(f"    TATA-Box: '{arch['tata_box']['sequence']}' at {arch['tata_box']['start']}-{arch['tata_box']['end']}")
        if arch['sp1_gc_boxes']:
            print(f"    Sp1 / GC Boxes: {len(arch['sp1_gc_boxes'])} sites")


if __name__ == "__main__":
    main()
