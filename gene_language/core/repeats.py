"""
Microsatellites and Short Tandem Repeat (STR) Discovery Engine.
=============================================================
Detects simple sequence repeats (SSRs), telomeric motifs, and
pathogenic trinucleotide/hexanucleotide repeat expansions.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any


# Clinically relevant and well-documented repeat expansion disorders
PATHOGENIC_REPEAT_MOTIFS: Dict[str, Dict[str, Any]] = {
    "CAG": {
        "disease": "Huntington's Disease / Spinocerebellar Ataxias (SCA1, SCA2, SCA3, SCA6)",
        "type": "Polyglutamine (PolyQ)",
        "normal_threshold": 35,
    },
    "CTG": {
        "disease": "Myotonic Dystrophy Type 1 (DM1)",
        "type": "Untranslated 3' UTR",
        "normal_threshold": 50,
    },
    "CGG": {
        "disease": "Fragile X Syndrome (FXS)",
        "type": "5' UTR Hypermethylation",
        "normal_threshold": 45,
    },
    "GAA": {
        "disease": "Friedreich's Ataxia (FRDA)",
        "type": "Intronic Transcriptional Repression",
        "normal_threshold": 66,
    },
    "CCTG": {
        "disease": "Myotonic Dystrophy Type 2 (DM2)",
        "type": "Intronic CCUG Expansion",
        "normal_threshold": 75,
    },
    "GGGGCC": {
        "disease": "C9orf72 Amyotrophic Lateral Sclerosis / Frontotemporal Dementia (ALS/FTD)",
        "type": "Non-coding G-quadruplex",
        "normal_threshold": 30,
    },
    "TTAGGG": {
        "disease": "Canonical Vertebrate Telomere",
        "type": "Chromosomal End Protection",
        "normal_threshold": 1000,
    }
}


@dataclass
class TandemRepeat:
    """Represents a discovered Short Tandem Repeat / Microsatellite tract."""
    unit: str
    unit_length: int
    copies: float
    start: int
    end: int
    total_length: int
    sequence: str
    disease_associated: Optional[str] = None
    is_pathogenic_risk: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "unit_length": self.unit_length,
            "copies": round(self.copies, 2),
            "start": self.start,
            "end": self.end,
            "total_length": self.total_length,
            "sequence": self.sequence,
            "disease_associated": self.disease_associated,
            "is_pathogenic_risk": self.is_pathogenic_risk,
        }


class TandemRepeatScanner:
    """
    High-performance scanner for discovering micro-, mini-, and disease-associated tandem repeats.
    """

    def __init__(self, min_unit_len: int = 1, max_unit_len: int = 6):
        self.min_unit_len = min_unit_len
        self.max_unit_len = max_unit_len

    @staticmethod
    def _is_primitive_unit(unit: str) -> bool:
        """Check if unit is primitive (e.g. 'AT' is primitive, 'ATAT' is not)."""
        n = len(unit)
        for i in range(1, n // 2 + 1):
            if n % i == 0:
                sub = unit[:i]
                if sub * (n // i) == unit:
                    return False
        return True

    def scan(
        self,
        sequence: str,
        min_copies: int = 3,
        min_total_len: int = 6
    ) -> List[TandemRepeat]:
        """
        Scan DNA sequence for all tandem repeats within [min_unit_len, max_unit_len].

        Parameters
        ----------
        sequence : str
            Input DNA string.
        min_copies : int
            Minimum consecutive repetitions.
        min_total_len : int
            Minimum total length of the repeat tract in bp.

        Returns
        -------
        List[TandemRepeat]
            Sorted list of identified tandem repeat tracts.
        """
        seq_upper = sequence.upper()
        results: List[TandemRepeat] = []
        n = len(seq_upper)

        for ulen in range(self.min_unit_len, self.max_unit_len + 1):
            pattern = re.compile(rf"(([ACGT]{{{ulen}}})\2{{{min_copies - 1},}})")
            for match in pattern.finditer(seq_upper):
                full_match = match.group(1)
                unit = match.group(2)

                # Skip non-primitive units to prevent duplicates (e.g. AAAA vs AA)
                if not self._is_primitive_unit(unit):
                    continue

                total_len = len(full_match)
                if total_len < min_total_len:
                    continue

                copies = total_len / len(unit)
                start_pos = match.start()
                end_pos = match.end()

                # Check disease database
                disease_info = PATHOGENIC_REPEAT_MOTIFS.get(unit)
                disease_name = disease_info["disease"] if disease_info else None
                is_risk = False
                if disease_info and copies >= disease_info["normal_threshold"]:
                    is_risk = True

                results.append(TandemRepeat(
                    unit=unit,
                    unit_length=len(unit),
                    copies=copies,
                    start=start_pos,
                    end=end_pos,
                    total_length=total_len,
                    sequence=full_match,
                    disease_associated=disease_name,
                    is_pathogenic_risk=is_risk
                ))

        # Sort by start coordinate then total length descending
        results.sort(key=lambda r: (r.start, -r.total_length))
        return results

    def scan_pathogenic_expansions(self, sequence: str, min_copies: int = 4) -> List[TandemRepeat]:
        """Specifically filter for known disease-associated repeat motifs."""
        all_repeats = self.scan(sequence, min_copies=min_copies, min_total_len=min_copies * 3)
        return [r for r in all_repeats if r.disease_associated is not None]

    def telomeric_profile(self, sequence: str) -> Dict[str, Any]:
        """Quantify vertebrate telomeric repeat (TTAGGG / CCCTAA) frequency."""
        seq_upper = sequence.upper()
        fwd_motif = "TTAGGG"
        rev_motif = "CCCTAA"

        fwd_count = seq_upper.count(fwd_motif)
        rev_count = seq_upper.count(rev_motif)
        total_bases = len(seq_upper)
        telomere_bp = (fwd_count + rev_count) * 6
        pct = (telomere_bp / total_bases * 100.0) if total_bases > 0 else 0.0

        return {
            "forward_telomere_count_TTAGGG": fwd_count,
            "reverse_telomere_count_CCCTAA": rev_count,
            "total_telomeric_bp": telomere_bp,
            "telomeric_percentage": round(pct, 4)
        }
