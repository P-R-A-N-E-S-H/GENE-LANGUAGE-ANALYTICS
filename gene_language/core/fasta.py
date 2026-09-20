"""
FASTA parsing, writing, streaming, and standardization pipeline.
"""

from __future__ import annotations
import os
from typing import List, Generator, Optional, Union
from dataclasses import dataclass
from .sequence import GenomicSequence


@dataclass
class FastaRecord:
    """Represents a single FASTA sequence entry."""
    identifier: str
    description: str
    sequence: GenomicSequence

    @property
    def header(self) -> str:
        return f">{self.identifier} {self.description}".strip()

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def gc_content(self) -> float:
        return self.sequence.gc_content

    def __repr__(self) -> str:
        return f"<FastaRecord id='{self.identifier}' len={self.length} gc={self.gc_content}%>"


def stream_fasta(
    filepath: str,
    strip_ambiguous: bool = False,
    max_records: Optional[int] = None
) -> Generator[FastaRecord, None, None]:
    """
    Memory-efficient generator yielding FastaRecord objects from a FASTA file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FASTA file not found: {filepath}")

    current_id = ""
    current_desc = ""
    current_seq_chunks = []
    record_count = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq_chunks:
                    seq_str = "".join(current_seq_chunks)
                    if strip_ambiguous:
                        seq_str = seq_str.replace("N", "").replace("n", "")
                    yield FastaRecord(
                        identifier=current_id,
                        description=current_desc,
                        sequence=GenomicSequence(seq_str, identifier=current_id, description=current_desc)
                    )
                    record_count += 1
                    if max_records and record_count >= max_records:
                        return
                    current_seq_chunks = []

                header_parts = line[1:].split(None, 1)
                current_id = header_parts[0] if header_parts else f"seq_{record_count+1}"
                current_desc = header_parts[1] if len(header_parts) > 1 else ""
            else:
                current_seq_chunks.append(line)

        if current_seq_chunks:
            seq_str = "".join(current_seq_chunks)
            if strip_ambiguous:
                seq_str = seq_str.replace("N", "").replace("n", "")
            yield FastaRecord(
                identifier=current_id,
                description=current_desc,
                sequence=GenomicSequence(seq_str, identifier=current_id, description=current_desc)
            )


def read_fasta(
    filepath: str,
    strip_ambiguous: bool = False,
    max_records: Optional[int] = None
) -> List[FastaRecord]:
    """Reads all records from a FASTA file into a list."""
    return list(stream_fasta(filepath, strip_ambiguous=strip_ambiguous, max_records=max_records))


def parse_fasta_string(
    fasta_text: str,
    strip_ambiguous: bool = False
) -> List[FastaRecord]:
    """Parses a multi-record or single-record FASTA string."""
    records = []
    current_id = "seq_1"
    current_desc = ""
    current_seq_chunks = []
    record_count = 0

    lines = fasta_text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq_chunks:
                seq_str = "".join(current_seq_chunks)
                if strip_ambiguous:
                    seq_str = seq_str.replace("N", "").replace("n", "")
                records.append(
                    FastaRecord(
                        identifier=current_id,
                        description=current_desc,
                        sequence=GenomicSequence(seq_str, identifier=current_id, description=current_desc)
                    )
                )
                current_seq_chunks = []

            header_parts = line[1:].split(None, 1)
            current_id = header_parts[0] if header_parts else f"seq_{record_count+1}"
            current_desc = header_parts[1] if len(header_parts) > 1 else ""
            record_count += 1
        else:
            current_seq_chunks.append(line)

    if current_seq_chunks:
        seq_str = "".join(current_seq_chunks)
        if strip_ambiguous:
            seq_str = seq_str.replace("N", "").replace("n", "")
        records.append(
            FastaRecord(
                identifier=current_id,
                description=current_desc,
                sequence=GenomicSequence(seq_str, identifier=current_id, description=current_desc)
            )
        )

    # If raw sequence with no header was provided:
    if not records and fasta_text.strip():
        seq_clean = "".join(fasta_text.split()).upper()
        if strip_ambiguous:
            seq_clean = seq_clean.replace("N", "")
        records.append(
            FastaRecord(
                identifier="input_sequence",
                description="Raw uploaded sequence",
                sequence=GenomicSequence(seq_clean, identifier="input_sequence")
            )
        )

    return records


def write_fasta(
    records: Union[List[FastaRecord], List[GenomicSequence], List[tuple]],
    output_path: str,
    line_wrap: int = 70
) -> int:
    """
    Writes records to a formatted FASTA file with standardized line wrapping.
    Returns the number of written records.
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in records:
            if isinstance(item, FastaRecord):
                header = item.header
                seq = item.sequence.sequence
            elif isinstance(item, GenomicSequence):
                header = f">{item.identifier} {item.description}".strip()
                seq = item.sequence
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                header = f">{item[0]}"
                seq = str(item[1]).upper()
            else:
                continue

            f.write(f"{header}\n")
            for j in range(0, len(seq), line_wrap):
                f.write(f"{seq[j:j+line_wrap]}\n")
            count += 1

    return count


def clean_fasta(
    input_path: str,
    output_path: str,
    strip_ambiguous: bool = True,
    line_wrap: int = 70,
    max_records: Optional[int] = None
) -> Dict[str, Union[int, float]]:
    """
    Reads a raw FASTA file, standardizes formatting, strips invalid/ambiguous characters,
    and writes to output_path. Returns summary statistics.
    """
    total_records = 0
    total_bases = 0
    gc_bases = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for record in stream_fasta(input_path, strip_ambiguous=strip_ambiguous, max_records=max_records):
            total_records += 1
            seq = record.sequence.sequence
            length = len(seq)
            total_bases += length
            gc_bases += seq.count("G") + seq.count("C")

            out.write(f"{record.header}\n")
            for j in range(0, length, line_wrap):
                out.write(f"{seq[j:j+line_wrap]}\n")

    overall_gc = round((gc_bases / total_bases) * 100.0, 2) if total_bases > 0 else 0.0
    return {
        "records_processed": total_records,
        "total_bases": total_bases,
        "overall_gc_percent": overall_gc,
        "output_file": output_path
    }
