"""
Core Promoter and Transcription Factor Binding Site (TFBS) Matrix Scorer.
========================================================================
Implements Position Weight Matrix (PWM) scoring for eukaryotic and prokaryotic
promoter elements: TATA-box, Initiator (Inr), DPE, BRE, CCAAT-box, GC-box/Sp1, and E-box.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple


# Standard Position Frequency Matrices for canonical core promoter motifs
# Transfac / JASPAR consensus approximations
PROMOTER_PWM_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "TATA_BOX": {
        "name": "TATA Box (TBP Binding)",
        "description": "Consensus TATAWAWR situated ~25-30 bp upstream of transcription start site.",
        "typical_offset": -30,
        "matrix": [
            # T, A, T, A, W(A/T), A, W(A/T), R(A/G)
            {"A": 0.05, "C": 0.05, "G": 0.05, "T": 0.85},
            {"A": 0.85, "C": 0.05, "G": 0.05, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.05, "T": 0.85},
            {"A": 0.85, "C": 0.05, "G": 0.05, "T": 0.05},
            {"A": 0.45, "C": 0.05, "G": 0.05, "T": 0.45},
            {"A": 0.80, "C": 0.05, "G": 0.10, "T": 0.05},
            {"A": 0.45, "C": 0.05, "G": 0.05, "T": 0.45},
            {"A": 0.45, "C": 0.05, "G": 0.45, "T": 0.05},
        ]
    },
    "INR_INITIATOR": {
        "name": "Initiator Element (Inr)",
        "description": "Consensus YYANWYY directly overlapping the TSS (-2 to +4 bp).",
        "typical_offset": 0,
        "matrix": [
            # Y (C/T), Y (C/T), A (+1 TSS), N, W (A/T), Y (C/T), Y (C/T)
            {"A": 0.05, "C": 0.50, "G": 0.05, "T": 0.40},
            {"A": 0.05, "C": 0.55, "G": 0.05, "T": 0.35},
            {"A": 0.85, "C": 0.05, "G": 0.05, "T": 0.05},  # +1 TSS base
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.45, "C": 0.05, "G": 0.05, "T": 0.45},
            {"A": 0.05, "C": 0.50, "G": 0.05, "T": 0.40},
            {"A": 0.05, "C": 0.50, "G": 0.05, "T": 0.40},
        ]
    },
    "DPE": {
        "name": "Downstream Promoter Element (DPE)",
        "description": "Consensus RGWYV situated +28 to +32 bp downstream of TSS in TATA-less promoters.",
        "typical_offset": 28,
        "matrix": [
            # R (A/G), G, W (A/T), Y (C/T), V (A/C/G)
            {"A": 0.45, "C": 0.05, "G": 0.45, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
            {"A": 0.45, "C": 0.05, "G": 0.05, "T": 0.45},
            {"A": 0.05, "C": 0.45, "G": 0.05, "T": 0.45},
            {"A": 0.30, "C": 0.35, "G": 0.30, "T": 0.05},
        ]
    },
    "BRE_TFIIB": {
        "name": "TFIIB Recognition Element (BREu)",
        "description": "Consensus SSRCGCC located immediately upstream of TATA box (-37 to -32 bp).",
        "typical_offset": -35,
        "matrix": [
            # S (G/C), S (G/C), R (A/G), C, G, C, C
            {"A": 0.05, "C": 0.45, "G": 0.45, "T": 0.05},
            {"A": 0.05, "C": 0.45, "G": 0.45, "T": 0.05},
            {"A": 0.45, "C": 0.05, "G": 0.45, "T": 0.05},
            {"A": 0.05, "C": 0.80, "G": 0.10, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
            {"A": 0.05, "C": 0.85, "G": 0.05, "T": 0.05},
            {"A": 0.05, "C": 0.85, "G": 0.05, "T": 0.05},
        ]
    },
    "CCAAT_BOX": {
        "name": "CCAAT Box (NF-Y Binding)",
        "description": "Consensus CCAAT located ~75-80 bp upstream of transcription start site.",
        "typical_offset": -80,
        "matrix": [
            {"A": 0.05, "C": 0.85, "G": 0.05, "T": 0.05},
            {"A": 0.05, "C": 0.85, "G": 0.05, "T": 0.05},
            {"A": 0.85, "C": 0.05, "G": 0.05, "T": 0.05},
            {"A": 0.85, "C": 0.05, "G": 0.05, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.05, "T": 0.85},
        ]
    },
    "GC_BOX_SP1": {
        "name": "GC Box (Sp1 Binding)",
        "description": "Consensus GGGCGG enriched in CpG island promoters and housekeeping genes.",
        "typical_offset": -50,
        "matrix": [
            {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
            {"A": 0.05, "C": 0.85, "G": 0.05, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
        ]
    },
    "E_BOX": {
        "name": "E-Box Motif (bHLH / Myc/Max)",
        "description": "Consensus CACGTG / CANNTG regulatory enhancer binding site.",
        "typical_offset": -100,
        "matrix": [
            {"A": 0.05, "C": 0.85, "G": 0.05, "T": 0.05},
            {"A": 0.85, "C": 0.05, "G": 0.05, "T": 0.05},
            {"A": 0.05, "C": 0.80, "G": 0.10, "T": 0.05},
            {"A": 0.05, "C": 0.10, "G": 0.80, "T": 0.05},
            {"A": 0.05, "C": 0.05, "G": 0.05, "T": 0.85},
            {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
        ]
    }
}


@dataclass
class PromoterElementMatch:
    motif_id: str
    name: str
    start: int
    end: int
    sequence: str
    raw_score: float
    max_possible_score: float
    relative_score: float  # Range [0.0, 1.0]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motif_id": self.motif_id,
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "sequence": self.sequence,
            "raw_score": round(self.raw_score, 3),
            "relative_score": round(self.relative_score, 3),
            "description": self.description,
        }


class PositionWeightMatrix:
    """
    Log-odds Position Weight Matrix (PWM) scoring model with background nucleotide distribution.
    """

    def __init__(self, matrix: List[Dict[str, float]], background: Optional[Dict[str, float]] = None):
        self.matrix = matrix
        self.motif_len = len(matrix)
        self.background = background or {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        self.log_odds_matrix = self._compute_log_odds()
        self.min_score, self.max_score = self._compute_score_bounds()

    def _compute_log_odds(self) -> List[Dict[str, float]]:
        log_odds = []
        for pos_dict in self.matrix:
            pos_lo = {}
            for base in ("A", "C", "G", "T"):
                prob = pos_dict.get(base, 0.001)
                bg = self.background.get(base, 0.25)
                pos_lo[base] = math.log2(prob / bg)
            log_odds.append(pos_lo)
        return log_odds

    def _compute_score_bounds(self) -> Tuple[float, float]:
        min_s = sum(min(p[b] for b in ("A", "C", "G", "T")) for p in self.log_odds_matrix)
        max_s = sum(max(p[b] for b in ("A", "C", "G", "T")) for p in self.log_odds_matrix)
        return min_s, max_s

    def score_kmer(self, kmer: str) -> Tuple[float, float]:
        """Score a k-mer string matching motif length. Returns (raw_score, relative_score [0,1])."""
        if len(kmer) != self.motif_len:
            return -999.0, 0.0

        raw = 0.0
        for i, b in enumerate(kmer.upper()):
            raw += self.log_odds_matrix[i].get(b, -2.0)

        # Map to relative score [0.0, 1.0]
        score_range = self.max_score - self.min_score
        relative = (raw - self.min_score) / score_range if score_range > 0 else 0.0
        return raw, max(0.0, min(1.0, relative))


class PromoterArchitectureScanner:
    """
    Scans genomic sequences for core promoter architectures and regulatory TFBS consensus sites.
    """

    def __init__(self, background_gc: float = 0.5):
        bg_g_or_c = background_gc / 2.0
        bg_a_or_t = (1.0 - background_gc) / 2.0
        self.bg = {"A": bg_a_or_t, "C": bg_g_or_c, "G": bg_g_or_c, "T": bg_a_or_t}
        self.pwms: Dict[str, PositionWeightMatrix] = {
            m_id: PositionWeightMatrix(data["matrix"], self.bg)
            for m_id, data in PROMOTER_PWM_DEFINITIONS.items()
        }

    def scan(self, sequence: str, min_relative_score: float = 0.75) -> List[PromoterElementMatch]:
        """
        Scan DNA sequence for promoter elements exceeding score threshold.
        """
        seq_upper = sequence.upper()
        matches: List[PromoterElementMatch] = []
        n = len(seq_upper)

        for m_id, pwm in self.pwms.items():
            info = PROMOTER_PWM_DEFINITIONS[m_id]
            k = pwm.motif_len
            for i in range(n - k + 1):
                sub = seq_upper[i:i + k]
                if any(b not in "ACGT" for b in sub):
                    continue
                raw, rel = pwm.score_kmer(sub)
                if rel >= min_relative_score:
                    matches.append(PromoterElementMatch(
                        motif_id=m_id,
                        name=info["name"],
                        start=i,
                        end=i + k,
                        sequence=sub,
                        raw_score=raw,
                        max_possible_score=pwm.max_score,
                        relative_score=rel,
                        description=info["description"]
                    ))

        matches.sort(key=lambda m: (m.start, -m.relative_score))
        return matches

    def identify_putative_promoter_regions(self, sequence: str) -> List[Dict[str, Any]]:
        """
        Group discovered promoter elements into putative core promoter architectures (TATA-box + Inr, or CpG/Sp1 + Inr).
        """
        elements = self.scan(sequence, min_relative_score=0.72)
        inr_matches = [e for e in elements if e.motif_id == "INR_INITIATOR"]
        tata_matches = [e for e in elements if e.motif_id == "TATA_BOX"]
        dpe_matches = [e for e in elements if e.motif_id == "DPE"]
        sp1_matches = [e for e in elements if e.motif_id == "GC_BOX_SP1"]

        promoters: List[Dict[str, Any]] = []

        # Find Inr coordinates as putative TSS anchors (+1)
        for inr in inr_matches:
            tss_pos = inr.start + 2  # A at position +1

            # Check for upstream TATA (-40 to -20 relative to TSS)
            associated_tata = [t for t in tata_matches if -45 <= (t.start - tss_pos) <= -15]
            # Check for downstream DPE (+20 to +35 relative to TSS)
            associated_dpe = [d for d in dpe_matches if 15 <= (d.start - tss_pos) <= 40]
            # Check for upstream Sp1 (-120 to -30 relative to TSS)
            associated_sp1 = [s for s in sp1_matches if -150 <= (s.start - tss_pos) <= -20]

            arch_type = "TATA-driven Core Promoter" if associated_tata else (
                "DPE-driven TATA-less Promoter" if associated_dpe else (
                    "CpG / GC-box Housekeeping Promoter" if associated_sp1 else "Dispersed / Inr Core Promoter"
                )
            )

            promoters.append({
                "predicted_tss": tss_pos,
                "inr_site": inr.to_dict(),
                "architecture_type": arch_type,
                "tata_box": associated_tata[0].to_dict() if associated_tata else None,
                "dpe": associated_dpe[0].to_dict() if associated_dpe else None,
                "sp1_gc_boxes": [s.to_dict() for s in associated_sp1],
            })

        return promoters
