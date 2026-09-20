"""
Example 3: Scan Genomic Sequences for Regulatory Motifs and Promoters
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language import read_fasta, MotifScanner

def main():
    fasta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sample_promoters.fasta"))
    records = read_fasta(fasta_path, max_records=2)

    scanner = MotifScanner()

    for rec in records:
        print(f"\n=======================================================")
        print(f"Scanning: {rec.identifier}")
        print(f"=======================================================")
        matches = scanner.scan(rec.sequence.sequence)

        print(f"Found {len(matches)} motif occurrences:")
        for m in matches[:10]:
            print(f"  [{m.strand}] pos {m.start:>4}-{m.end:<4} | {m.motif_name:<28} | {m.matched_sequence:<12} | {m.description}")

if __name__ == "__main__":
    main()
