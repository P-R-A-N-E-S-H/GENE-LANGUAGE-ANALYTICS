"""
Genetic code translation tables, 6-frame Open Reading Frame (ORF) finding,
and Relative Synonymous Codon Usage (RSCU) analytics.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter
from .sequence import GenomicSequence

# Standard Genetic Code Mapping
GENETIC_CODE: Dict[str, str] = {
    "ATA": "I", "ATC": "I", "ATT": "I", "ATG": "M",
    "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
    "AAC": "N", "AAT": "N", "AAA": "K", "AAG": "K",
    "AGC": "S", "AGT": "S", "AGA": "R", "AGG": "R",
    "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
    "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
    "CAC": "H", "CAT": "H", "CAA": "Q", "CAG": "Q",
    "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
    "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
    "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
    "GAC": "D", "GAT": "D", "GAA": "E", "GAG": "E",
    "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
    "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
    "TTC": "F", "TTT": "F", "TTA": "L", "TTG": "L",
    "TAC": "Y", "TAT": "Y", "TAA": "*", "TAG": "*",
    "TGC": "C", "TGT": "C", "TGA": "*", "TGG": "W",
}

START_CODONS: Set[str] = {"ATG"}
STOP_CODONS: Set[str] = {"TAA", "TAG", "TGA"}

AMINO_ACID_NAMES: Dict[str, Dict[str, str]] = {
    "A": {"name": "Alanine", "3letter": "Ala", "property": "Aliphatic, Hydrophobic"},
    "R": {"name": "Arginine", "3letter": "Arg", "property": "Basic, Positively Charged"},
    "N": {"name": "Asparagine", "3letter": "Asn", "property": "Polar, Uncharged"},
    "D": {"name": "Aspartic Acid", "3letter": "Asp", "property": "Acidic, Negatively Charged"},
    "C": {"name": "Cysteine", "3letter": "Cys", "property": "Polar, Contains Thiol"},
    "E": {"name": "Glutamic Acid", "3letter": "Glu", "property": "Acidic, Negatively Charged"},
    "Q": {"name": "Glutamine", "3letter": "Gln", "property": "Polar, Uncharged"},
    "G": {"name": "Glycine", "3letter": "Gly", "property": "Small, Flexible"},
    "H": {"name": "Histidine", "3letter": "His", "property": "Basic, Weakly Positive"},
    "I": {"name": "Isoleucine", "3letter": "Ile", "property": "Aliphatic, Hydrophobic"},
    "L": {"name": "Leucine", "3letter": "Leu", "property": "Aliphatic, Hydrophobic"},
    "K": {"name": "Lysine", "3letter": "Lys", "property": "Basic, Positively Charged"},
    "M": {"name": "Methionine", "3letter": "Met", "property": "Hydrophobic, Start Codon"},
    "F": {"name": "Phenylalanine", "3letter": "Phe", "property": "Aromatic, Hydrophobic"},
    "P": {"name": "Proline", "3letter": "Pro", "property": "Cyclic, Rigid"},
    "S": {"name": "Serine", "3letter": "Ser", "property": "Polar, Hydroxyl"},
    "T": {"name": "Threonine", "3letter": "Thr", "property": "Polar, Hydroxyl"},
    "W": {"name": "Tryptophan", "3letter": "Trp", "property": "Aromatic, Hydrophobic"},
    "Y": {"name": "Tyrosine", "3letter": "Tyr", "property": "Aromatic, Hydroxyl"},
    "V": {"name": "Valine", "3letter": "Val", "property": "Aliphatic, Hydrophobic"},
    "*": {"name": "Stop Codon", "3letter": "Stop", "property": "Termination Signal"},
}

# Group codons by amino acid
SYNONYMOUS_CODONS: Dict[str, List[str]] = {}
for codon, aa in GENETIC_CODE.items():
    SYNONYMOUS_CODONS.setdefault(aa, []).append(codon)


@dataclass
class ORF:
    """Represents a discovered Open Reading Frame."""
    strand: str            # '+' or '-'
    frame: int             # 1, 2, 3, -1, -2, -3
    start: int             # 0-indexed start on given strand
    end: int               # 0-indexed exclusive end
    length_nt: int
    length_aa: int
    dna_sequence: str
    protein_sequence: str

    def __repr__(self) -> str:
        return f"<ORF frame={self.frame:+d} pos={self.start}:{self.end} ({self.length_nt}nt/{self.length_aa}aa)>"


@dataclass
class CodonUsageProfile:
    """Summary of codon usage frequencies and RSCU values."""
    codon_counts: Dict[str, int]
    rscu_values: Dict[str, float]
    effective_number_of_codons: float
    total_codons: int


def translate_sequence(
    sequence: str,
    frame: int = 0,
    stop_symbol: str = "*",
    unknown_symbol: str = "X"
) -> str:
    """
    Translates a nucleotide sequence to an amino acid string in the specified reading frame (0, 1, or 2).
    """
    seq = "".join(sequence.split()).upper()
    aa_list = []
    for i in range(frame, len(seq) - 2, 3):
        triplet = seq[i:i + 3]
        aa = GENETIC_CODE.get(triplet, unknown_symbol)
        aa_list.append(aa if aa != "*" else stop_symbol)
    return "".join(aa_list)


def find_orfs(
    sequence: str,
    min_protein_len: int = 30,
    include_reverse_frames: bool = True,
    require_start_codon: bool = True
) -> List[ORF]:
    """
    Discovers all Open Reading Frames across 3 forward and 3 reverse reading frames.
    """
    seq_obj = GenomicSequence(sequence)
    fwd_seq = seq_obj.sequence
    rev_seq = seq_obj.reverse_complement().sequence
    orfs: List[ORF] = []

    def _scan_strand(strand_seq: str, strand_label: str, frame_offset_mult: int):
        seq_len = len(strand_seq)
        for offset in range(3):
            current_frame = (offset + 1) * frame_offset_mult
            # Scan in steps of 3
            i = offset
            start_pos = None

            while i < seq_len - 2:
                codon = strand_seq[i:i + 3]
                if require_start_codon:
                    if codon in START_CODONS and start_pos is None:
                        start_pos = i
                    elif codon in STOP_CODONS and start_pos is not None:
                        # Found complete ORF
                        dna_chunk = strand_seq[start_pos:i + 3]
                        protein = translate_sequence(dna_chunk)
                        aa_len = len(protein) - 1  # Excluding stop codon
                        if aa_len >= min_protein_len:
                            orfs.append(ORF(
                                strand=strand_label,
                                frame=current_frame,
                                start=start_pos,
                                end=i + 3,
                                length_nt=len(dna_chunk),
                                length_aa=aa_len,
                                dna_sequence=dna_chunk,
                                protein_sequence=protein.rstrip("*")
                            ))
                        start_pos = None
                else:
                    # Non-strict mode: ORF is between stops
                    if start_pos is None:
                        start_pos = i
                    if codon in STOP_CODONS:
                        dna_chunk = strand_seq[start_pos:i + 3]
                        protein = translate_sequence(dna_chunk)
                        aa_len = len(protein) - 1
                        if aa_len >= min_protein_len:
                            orfs.append(ORF(
                                strand=strand_label,
                                frame=current_frame,
                                start=start_pos,
                                end=i + 3,
                                length_nt=len(dna_chunk),
                                length_aa=aa_len,
                                dna_sequence=dna_chunk,
                                protein_sequence=protein.rstrip("*")
                            ))
                        start_pos = i + 3
                i += 3

    # Scan forward frames (+1, +2, +3)
    _scan_strand(fwd_seq, "+", +1)

    # Scan reverse frames (-1, -2, -3)
    if include_reverse_frames:
        _scan_strand(rev_seq, "-", -1)

    # Sort ORFs by length descending
    orfs.sort(key=lambda x: x.length_nt, reverse=True)
    return orfs


def calculate_rscu(sequence: str) -> CodonUsageProfile:
    """
    Calculates Relative Synonymous Codon Usage (RSCU) and Effective Number of Codons (Nc):
    RSCU = (X_ij) / ( (1 / n_i) * sum_j(X_ij) )
    """
    seq = "".join(sequence.split()).upper()
    # Count codons in frame 0
    codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3) if len(seq[i:i + 3]) == 3]
    counts = Counter(codons)
    total_codons = len(codons)

    rscu: Dict[str, float] = {}

    for aa, syn_codons in SYNONYMOUS_CODONS.items():
        if aa == "*":
            continue  # Exclude stop codons from RSCU calculation
        k = len(syn_codons)
        aa_total = sum(counts.get(c, 0) for c in syn_codons)
        for codon in syn_codons:
            c_count = counts.get(codon, 0)
            if aa_total > 0:
                val = (c_count * k) / aa_total
            else:
                val = 1.0  # Equal usage baseline
            rscu[codon] = round(val, 4)

    # Estimate Wright's Effective Number of Codons (Nc) approximation
    # Nc = 2 + s + (29 / (s^2 + (1-s)^2)) where s is GC3s (GC content at 3rd codon position)
    gc3_count = sum(1 for c in codons if len(c) == 3 and c[2] in "GC")
    gc3s = (gc3_count / total_codons) if total_codons > 0 else 0.5
    s = max(0.01, min(0.99, gc3s))
    nc_est = round(2.0 + s + (29.0 / (s**2 + (1.0 - s)**2)), 2)

    return CodonUsageProfile(
        codon_counts={c: counts.get(c, 0) for c in GENETIC_CODE},
        rscu_values=rscu,
        effective_number_of_codons=nc_est,
        total_codons=total_codons
    )
