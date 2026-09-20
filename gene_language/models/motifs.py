"""
Biological motif scanner for promoters, splice junctions, Kozak sequences, and restriction sites.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from ..core.sequence import GenomicSequence

IUPAC_DEGENERATE_MAP = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "U",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}

COMMON_MOTIFS = [
    {
        "name": "TATA Box",
        "pattern": "TATAAA|TATAAT|TATATA|TATAAG",
        "category": "Promoter Element",
        "description": "Core promoter binding site for TATA-binding protein (TBP), ~25-30bp upstream of TSS."
    },
    {
        "name": "Pribnow Box (-10 Promoter)",
        "pattern": "TATAAT|TATGAT|TAAAAT",
        "category": "Prokaryotic Promoter",
        "description": "Essential bacterial core promoter region facilitating RNA polymerase unwinding."
    },
    {
        "name": "Kozak Consensus (Initiation)",
        "pattern": "[AG]CCACCATG[G]|GCCGCCACCATG",
        "category": "Translation Initiation",
        "description": "Eukaryotic mRNA ribosome translation initiation consensus surrounding start codon ATG."
    },
    {
        "name": "Shine-Dalgarno (RBS)",
        "pattern": "AGGAGG|AGGA|GGAGG",
        "category": "Ribosome Binding Site",
        "description": "Prokaryotic mRNA ribosomal binding site located 8 bases upstream of start codon."
    },
    {
        "name": "Canonical 5' Splice Donor",
        "pattern": "[AC]AGGT[AG]AGT|CAGGTGAG",
        "category": "Splice Site",
        "description": "Exon-intron 5' splice junction donor motif recognized by U1 snRNP."
    },
    {
        "name": "Canonical 3' Splice Acceptor",
        "pattern": "[CT]{4,8}[CT]N[CT]AG[GC]|TTTTTTTTTTAG",
        "category": "Splice Site",
        "description": "Intron-exon 3' splice junction acceptor consensus ending with invariant AG."
    },
    {
        "name": "Polyadenylation Signal",
        "pattern": "AATAAA|ATTAAA|AGTAAA",
        "category": "mRNA Processing",
        "description": "Cleavage and polyadenylation specificity factor (CPSF) recognition signal."
    },
    {
        "name": "EcoRI Recognition Site",
        "pattern": "GAATTC",
        "category": "Restriction Enzyme",
        "description": "Type II restriction endonuclease cleavage site generating 5' sticky ends."
    },
    {
        "name": "BamHI Recognition Site",
        "pattern": "GGATCC",
        "category": "Restriction Enzyme",
        "description": "Type II restriction endonuclease site for molecular cloning."
    },
    {
        "name": "HindIII Recognition Site",
        "pattern": "AAGCTT",
        "category": "Restriction Enzyme",
        "description": "Common cloning restriction endonuclease cleavage sequence."
    },
]


@dataclass
class MotifMatch:
    motif_name: str
    category: str
    start: int
    end: int
    strand: str
    matched_sequence: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motif_name": self.motif_name,
            "category": self.category,
            "start": self.start,
            "end": self.end,
            "strand": self.strand,
            "matched_sequence": self.matched_sequence,
            "description": self.description
        }


class MotifScanner:
    """
    Scans genomic sequences for regulatory elements, splice sites, promoters, and custom IUPAC motifs.
    """

    def __init__(self, custom_motifs: Optional[List[Dict[str, str]]] = None):
        self.motifs = list(COMMON_MOTIFS)
        if custom_motifs:
            self.motifs.extend(custom_motifs)

    @staticmethod
    def iupac_to_regex(iupac_pattern: str) -> str:
        """Converts IUPAC nucleotide string (with R, Y, N, etc.) to standard regex."""
        pattern = "".join(iupac_pattern.split()).upper()
        res = []
        for char in pattern:
            res.append(IUPAC_DEGENERATE_MAP.get(char, char))
        return "".join(res)

    def scan(self, sequence: str, scan_reverse_strand: bool = True) -> List[MotifMatch]:
        """
        Scans sequence on forward (and optionally reverse complement) strand.
        """
        seq_obj = GenomicSequence(sequence)
        fwd = seq_obj.sequence
        rev = seq_obj.reverse_complement().sequence
        matches: List[MotifMatch] = []

        def _search_strand(s_str: str, strand_label: str):
            for motif in self.motifs:
                regex_pat = motif["pattern"]
                for match in re.finditer(regex_pat, s_str, re.IGNORECASE):
                    start_pos = match.start()
                    end_pos = match.end()
                    matched_txt = match.group(0)
                    matches.append(
                        MotifMatch(
                            motif_name=motif["name"],
                            category=motif.get("category", "Motif"),
                            start=start_pos,
                            end=end_pos,
                            strand=strand_label,
                            matched_sequence=matched_txt,
                            description=motif.get("description", "")
                        )
                    )

        _search_strand(fwd, "+")
        if scan_reverse_strand:
            _search_strand(rev, "-")

        matches.sort(key=lambda m: (m.start, m.motif_name))
        return matches
