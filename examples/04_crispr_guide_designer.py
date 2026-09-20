"""
Example 4: CRISPR-Cas9 sgRNA Guide Design and On-Target Efficiency Scoring
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language import read_fasta, CrisprGuideDesigner

def main():
    fasta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sample_exons_introns.fasta"))
    records = read_fasta(fasta_path, max_records=2)

    designer = CrisprGuideDesigner()

    for rec in records:
        print(f"\n=======================================================")
        print(f"Target Sequence: {rec.identifier} ({len(rec.sequence)} bp)")
        print(f"=======================================================")
        guides = designer.find_guides(rec.sequence.sequence)

        print(f"Discovered {len(guides)} SpCas9 target sites (PAM: NGG):")
        for rank, g in enumerate(guides[:8], 1):
            term_flag = " [!] Poly-T" if g.has_poly_t else ""
            print(f"  {rank}. [{g.strand}] pos {g.start:>3}-{g.end:<3} | {g.spacer_sequence} ({g.pam_sequence}) | GC: {g.gc_content:>4.1f}% | Score: {g.efficiency_score:>4.1f} ({g.recommendation}){term_flag}")

if __name__ == "__main__":
    main()
