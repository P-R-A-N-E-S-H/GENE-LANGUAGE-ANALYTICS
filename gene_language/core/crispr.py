"""
CRISPR-Cas9 sgRNA guide RNA designer, PAM (NGG) scanner, and on-target efficiency scoring.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
from .sequence import GenomicSequence


@dataclass
class SgRNACandidate:
    spacer_sequence: str       # 20 nt protospacer
    pam_sequence: str          # 3 nt PAM (e.g. NGG)
    strand: str                # '+' or '-'
    start: int                 # 0-indexed start
    end: int                   # 0-indexed end including PAM
    gc_content: float          # GC% of 20nt spacer
    has_poly_t: bool           # Poly-T tract (>= 4 Ts) which causes Pol III termination
    efficiency_score: float    # Heuristic on-target score (0-100)
    recommendation: str        # 'High', 'Moderate', 'Low'


class CrisprGuideDesigner:
    """
    Identifies and evaluates candidate CRISPR single guide RNAs (sgRNAs) with SpCas9 PAM (NGG).
    """

    def __init__(self, pam_pattern: str = r"(?=([ACGT]{20}([ACGT]GG)))"):
        self.pam_pattern = pam_pattern

    @staticmethod
    def _score_efficiency(spacer: str, gc: float, has_poly_t: bool) -> Tuple[float, str]:
        """
        Computes Doench-style heuristic on-target efficiency score based on:
        - Optimal GC range (40% - 70%)
        - Absence of premature Pol III terminator (TTTT)
        - Specific nucleotide preferences at position 20 (G favoured at pos 20, T penalised)
        """
        score = 60.0

        # GC content penalties
        if 40.0 <= gc <= 65.0:
            score += 15.0
        elif gc < 30.0 or gc > 80.0:
            score -= 25.0

        # Terminator penalty
        if has_poly_t:
            score -= 35.0

        # Position-specific preferences (1-indexed relative to PAM)
        # Position 20 (adjacent to PAM): G is strongly preferred
        if len(spacer) == 20:
            if spacer[19] == "G":
                score += 12.0
            elif spacer[19] == "T":
                score -= 10.0

            # Position 1: G is preferred for U6 promoter transcription
            if spacer[0] == "G":
                score += 8.0

        final_score = max(0.0, min(100.0, score))
        if final_score >= 70.0:
            rec = "High Efficiency"
        elif final_score >= 45.0:
            rec = "Moderate Efficiency"
        else:
            rec = "Low Efficiency / Poor Candidate"

        return round(final_score, 1), rec

    def find_guides(self, sequence: str) -> List[SgRNACandidate]:
        """
        Discovers all SpCas9 target sites across both strands.
        """
        seq_obj = GenomicSequence(sequence)
        fwd_seq = seq_obj.sequence
        rev_seq = seq_obj.reverse_complement().sequence
        L = len(fwd_seq)
        guides: List[SgRNACandidate] = []

        def _scan(strand_seq: str, strand_label: str):
            for match in re.finditer(self.pam_pattern, strand_seq):
                full_match = match.group(1)
                spacer = full_match[:20]
                pam = full_match[20:]
                start_pos = match.start()
                end_pos = start_pos + 23

                gc = round(((spacer.count("G") + spacer.count("C")) / 20.0) * 100.0, 2)
                poly_t = "TTTT" in spacer

                score, rec = self._score_efficiency(spacer, gc, poly_t)

                # Real position coordinates on forward strand
                real_start = start_pos if strand_label == "+" else (L - end_pos)
                real_end = end_pos if strand_label == "+" else (L - start_pos)

                guides.append(
                    SgRNACandidate(
                        spacer_sequence=spacer,
                        pam_sequence=pam,
                        strand=strand_label,
                        start=real_start,
                        end=real_end,
                        gc_content=gc,
                        has_poly_t=poly_t,
                        efficiency_score=score,
                        recommendation=rec
                    )
                )

        _scan(fwd_seq, "+")
        _scan(rev_seq, "-")

        # Sort by efficiency score descending
        guides.sort(key=lambda x: x.efficiency_score, reverse=True)
        return guides
