"""
Position Weight Matrix (PWM) and log-odds scoring model for 5' Splice Donors and 3' Splice Acceptors.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import math
from dataclasses import dataclass
from ..core.sequence import GenomicSequence

# Standard Vertebrate Splice Donor Matrix (Positions -3 to +6 around invariant GT junction)
# Context: [Exon 3nt] [G T] [Intron 4nt] (Total 9 bp)
DONOR_PWM = {
    "A": [0.34, 0.60, 0.10, 0.00, 0.00, 0.54, 0.74, 0.10, 0.16],
    "C": [0.36, 0.13, 0.04, 0.00, 0.00, 0.03, 0.08, 0.16, 0.18],
    "G": [0.18, 0.14, 0.80, 1.00, 0.00, 0.35, 0.11, 0.60, 0.21],
    "T": [0.12, 0.13, 0.06, 0.00, 1.00, 0.08, 0.07, 0.14, 0.45],
}

# Standard Vertebrate Splice Acceptor Matrix (Positions -14 to +1 around invariant AG junction)
# Context: [Intron Polypyrimidine tract 12nt] [A G] [Exon 1nt] (Total 15 bp)
ACCEPTOR_PWM = {
    "A": [0.08, 0.09, 0.08, 0.09, 0.07, 0.08, 0.09, 0.04, 0.07, 0.09, 0.03, 0.03, 1.00, 0.00, 0.26],
    "C": [0.38, 0.34, 0.38, 0.35, 0.42, 0.40, 0.38, 0.42, 0.38, 0.40, 0.28, 0.68, 0.00, 0.00, 0.32],
    "G": [0.10, 0.10, 0.09, 0.08, 0.07, 0.08, 0.08, 0.07, 0.09, 0.07, 0.21, 0.01, 0.00, 1.00, 0.30],
    "T": [0.44, 0.47, 0.45, 0.48, 0.44, 0.44, 0.45, 0.47, 0.46, 0.44, 0.48, 0.28, 0.00, 0.00, 0.12],
}


@dataclass
class SpliceJunctionMatch:
    site_type: str      # '5_DONOR' or '3_ACCEPTOR'
    position: int       # 0-indexed position of GT / AG
    score: float        # Log-odds score (bits)
    confidence: float   # Normalized probability (0-1)
    sequence_context: str
    is_canonical: bool


class SpliceJunctionScorer:
    """
    Predicts and scores canonical and non-canonical eukaryotic splice sites.
    """

    def __init__(self, donor_threshold: float = 4.0, acceptor_threshold: float = 4.0):
        self.donor_threshold = donor_threshold
        self.acceptor_threshold = acceptor_threshold

    @staticmethod
    def _score_pwm(subseq: str, pwm: Dict[str, List[float]]) -> float:
        score = 0.0
        bg = 0.25  # Uniform background prior
        for pos, base in enumerate(subseq):
            if base in pwm:
                p = max(0.01, pwm[base][pos])
                score += math.log2(p / bg)
        return round(score, 3)

    def scan_donors(self, sequence: str) -> List[SpliceJunctionMatch]:
        """Scans for 5' splice donor sites (MAG|GTRAGT)."""
        seq = "".join(sequence.split()).upper()
        matches = []

        for i in range(3, len(seq) - 6):
            if seq[i:i + 2] == "GT":
                window = seq[i - 3:i + 6]
                score = self._score_pwm(window, DONOR_PWM)
                if score >= self.donor_threshold:
                    conf = min(1.0, max(0.1, 1.0 / (1.0 + math.exp(-0.4 * (score - 5.0)))))
                    matches.append(
                        SpliceJunctionMatch(
                            site_type="5_DONOR",
                            position=i,
                            score=score,
                            confidence=round(conf, 4),
                            sequence_context=window,
                            is_canonical=True
                        )
                    )
        return matches

    def scan_acceptors(self, sequence: str) -> List[SpliceJunctionMatch]:
        """Scans for 3' splice acceptor sites (YYYYYYYYYYYYNCAG|G)."""
        seq = "".join(sequence.split()).upper()
        matches = []

        for i in range(12, len(seq) - 3):
            if seq[i:i + 2] == "AG":
                window = seq[i - 12:i + 3]
                score = self._score_pwm(window, ACCEPTOR_PWM)
                if score >= self.acceptor_threshold:
                    conf = min(1.0, max(0.1, 1.0 / (1.0 + math.exp(-0.4 * (score - 5.0)))))
                    matches.append(
                        SpliceJunctionMatch(
                            site_type="3_ACCEPTOR",
                            position=i,
                            score=score,
                            confidence=round(conf, 4),
                            sequence_context=window,
                            is_canonical=True
                        )
                    )
        return matches

    def scan_all(self, sequence: str) -> List[SpliceJunctionMatch]:
        """Scans sequence for both donor and acceptor junctions, ordered by position."""
        all_matches = self.scan_donors(sequence) + self.scan_acceptors(sequence)
        all_matches.sort(key=lambda x: x.position)
        return all_matches
