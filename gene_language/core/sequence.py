"""
Genomic sequence object and biological sequence arithmetic.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
from collections import Counter


class GenomicSequence:
    """
    High-performance genomic sequence representation with nucleotide arithmetic,
    sliding window profiling, GC-skew, and CpG island detection.
    """

    COMPLEMENT_MAP = str.maketrans("ACGTUWSMKRYBDHVNacgtuwsmkrybdhvn", "TGCAAWSKMYRVHDBNtgcaawskmyrvhdbn")

    def __init__(self, sequence: str, identifier: str = "seq_0", description: str = ""):
        self.identifier = identifier
        self.description = description
        self.raw_sequence = sequence.strip()
        self.sequence = "".join(self.raw_sequence.split()).upper()

    @classmethod
    def clean(cls, raw_seq: str, strip_ambiguous: bool = True, identifier: str = "seq_clean") -> GenomicSequence:
        """Sanitizes sequence by converting to uppercase and stripping whitespace and ambiguous 'N' characters."""
        cleaned = "".join(raw_seq.split()).upper()
        if strip_ambiguous:
            cleaned = cleaned.replace("N", "")
        # Retain only valid standard DNA bases if requested or permissible
        return cls(cleaned, identifier=identifier)

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, item) -> str:
        return self.sequence[item]

    def __str__(self) -> str:
        return self.sequence

    def __repr__(self) -> str:
        preview = self.sequence[:20] + "..." + self.sequence[-10:] if len(self.sequence) > 30 else self.sequence
        return f"<GenomicSequence id='{self.identifier}' len={len(self)} seq='{preview}'>"

    @property
    def base_counts(self) -> Dict[str, int]:
        """Returns frequency of standard nucleotide bases."""
        counts = Counter(self.sequence)
        return {
            "A": counts.get("A", 0),
            "C": counts.get("C", 0),
            "G": counts.get("G", 0),
            "T": counts.get("T", 0),
            "N": counts.get("N", 0),
            "Other": sum(v for k, v in counts.items() if k not in "ACGTN")
        }

    @property
    def base_percentages(self) -> Dict[str, float]:
        """Returns base composition in percentage (0-100%)."""
        total = len(self.sequence)
        if total == 0:
            return {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0, "N": 0.0}
        counts = self.base_counts
        return {k: round((v / total) * 100.0, 4) for k, v in counts.items() if k != "Other"}

    @property
    def gc_content(self) -> float:
        """Computes GC percentage (G + C) / (A + C + G + T)."""
        counts = self.base_counts
        canonical_total = counts["A"] + counts["C"] + counts["G"] + counts["T"]
        if canonical_total == 0:
            return 0.0
        return round(((counts["G"] + counts["C"]) / canonical_total) * 100.0, 4)

    @property
    def at_content(self) -> float:
        """Computes AT percentage (A + T) / (A + C + G + T)."""
        counts = self.base_counts
        canonical_total = counts["A"] + counts["C"] + counts["G"] + counts["T"]
        if canonical_total == 0:
            return 0.0
        return round(((counts["A"] + counts["T"]) / canonical_total) * 100.0, 4)

    @property
    def gc_skew(self) -> float:
        """Computes overall GC Skew = (G - C) / (G + C)."""
        counts = self.base_counts
        denom = counts["G"] + counts["C"]
        if denom == 0:
            return 0.0
        return round((counts["G"] - counts["C"]) / denom, 6)

    @property
    def at_skew(self) -> float:
        """Computes overall AT Skew = (A - T) / (A + T)."""
        counts = self.base_counts
        denom = counts["A"] + counts["T"]
        if denom == 0:
            return 0.0
        return round((counts["A"] - counts["T"]) / denom, 6)

    def reverse_complement(self) -> GenomicSequence:
        """Computes the 5' -> 3' reverse complement sequence."""
        rev = self.sequence[::-1].translate(self.COMPLEMENT_MAP)
        return GenomicSequence(rev, identifier=f"{self.identifier}_revcomp")

    def transcribe(self) -> str:
        """Transcribes DNA sequence to RNA sequence (T -> U)."""
        return self.sequence.replace("T", "U")

    def melting_temperature(self) -> float:
        """
        Estimates melting temperature (Tm) in Celsius.
        Uses Wallace Rule for short sequences (< 14 bp) and nearest-neighbor approximation for longer.
        """
        length = len(self.sequence)
        if length == 0:
            return 0.0
        counts = self.base_counts
        wA, wT, wC, wG = counts["A"], counts["T"], counts["C"], counts["G"]
        if length < 14:
            return 2.0 * (wA + wT) + 4.0 * (wG + wC)
        # Marmur-Doty formula for longer sequences
        return round(64.9 + 41.0 * (wG + wC - 16.4) / length, 2)

    def molecular_weight(self, double_stranded: bool = False, circular: bool = False) -> float:
        """
        Calculates molecular weight in Daltons (g/mol).
        Standard single-stranded DNA monophosphate average masses:
        A=313.21, C=289.18, G=329.21, T=304.2
        """
        if len(self.sequence) == 0:
            return 0.0
        counts = self.base_counts
        mw = (
            counts["A"] * 313.21 +
            counts["C"] * 289.18 +
            counts["G"] * 329.21 +
            counts["T"] * 304.20
        )
        if not circular:
            mw += 18.02  # Terminal H and OH
        if double_stranded:
            # Complement strand
            comp_mw = (
                counts["T"] * 313.21 +
                counts["G"] * 289.18 +
                counts["C"] * 329.21 +
                counts["A"] * 304.20
            )
            if not circular:
                comp_mw += 18.02
            return round(mw + comp_mw, 2)
        return round(mw, 2)

    def sliding_window_gc(self, window_size: int = 100, step_size: int = 20) -> List[Dict[str, float]]:
        """
        Computes GC content and GC skew across sliding windows along the sequence.
        """
        seq = self.sequence
        length = len(seq)
        if length < window_size:
            return [{
                "position": 0,
                "gc_content": self.gc_content,
                "gc_skew": self.gc_skew,
                "at_skew": self.at_skew
            }]

        results = []
        for start in range(0, length - window_size + 1, step_size):
            sub = seq[start:start + window_size]
            g = sub.count("G")
            c = sub.count("C")
            a = sub.count("A")
            t = sub.count("T")
            tot = g + c + a + t
            gc_pct = ((g + c) / tot * 100.0) if tot > 0 else 0.0
            gc_sk = ((g - c) / (g + c)) if (g + c) > 0 else 0.0
            at_sk = ((a - t) / (a + t)) if (a + t) > 0 else 0.0
            results.append({
                "position": start,
                "center": start + window_size // 2,
                "gc_content": round(gc_pct, 2),
                "gc_skew": round(gc_sk, 4),
                "at_skew": round(at_sk, 4)
            })
        return results

    def cpg_observed_expected_ratio(self) -> float:
        """
        Calculates CpG Observed/Expected ratio:
        Obs/Exp = (Count(CG) * Length) / (Count(C) * Count(G))
        """
        seq = self.sequence
        length = len(seq)
        if length < 2:
            return 0.0
        c_count = seq.count("C")
        g_count = seq.count("G")
        cg_count = seq.count("CG")
        if c_count == 0 or g_count == 0:
            return 0.0
        return round((cg_count * length) / (c_count * g_count), 4)

    def find_cpg_islands(
        self,
        min_length: int = 200,
        min_gc: float = 50.0,
        min_obs_exp: float = 0.60,
        window_size: int = 200,
        step_size: int = 50,
    ) -> List[Dict[str, float]]:
        """
        Detects putative CpG Islands meeting standard Gardiner-Garden & Frommer criteria:
        Window >= 200 bp, GC% >= 50%, CpG Obs/Exp >= 0.60.
        """
        seq = self.sequence
        length = len(seq)
        islands = []

        if length < window_size:
            # Check whole sequence
            if self.gc_content >= min_gc and self.cpg_observed_expected_ratio() >= min_obs_exp:
                islands.append({
                    "start": 0,
                    "end": length,
                    "length": length,
                    "gc_content": self.gc_content,
                    "cpg_ratio": self.cpg_observed_expected_ratio()
                })
            return islands

        for start in range(0, length - window_size + 1, step_size):
            end = start + window_size
            sub = GenomicSequence(seq[start:end])
            gc = sub.gc_content
            cpg_ratio = sub.cpg_observed_expected_ratio()

            if gc >= min_gc and cpg_ratio >= min_obs_exp:
                # Merge overlapping regions if any
                if islands and islands[-1]["end"] >= start:
                    prev = islands[-1]
                    merged_seq = GenomicSequence(seq[prev["start"]:end])
                    prev["end"] = end
                    prev["length"] = end - prev["start"]
                    prev["gc_content"] = merged_seq.gc_content
                    prev["cpg_ratio"] = merged_seq.cpg_observed_expected_ratio()
                else:
                    islands.append({
                        "start": start,
                        "end": end,
                        "length": window_size,
                        "gc_content": gc,
                        "cpg_ratio": cpg_ratio
                    })

        return islands
