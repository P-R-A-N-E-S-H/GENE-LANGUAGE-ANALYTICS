"""
FastAPI REST API Server for GENE-LANGUAGE-ANALYTICS Studio.
"""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language.core.sequence import GenomicSequence
from gene_language.core.fasta import parse_fasta_string
from gene_language.core.codons import find_orfs, calculate_rscu, translate_sequence, GENETIC_CODE, AMINO_ACID_NAMES
from gene_language.linguistics.kmer import KmerExtractor, compute_kmer_spectrum
from gene_language.linguistics.entropy import shannon_entropy, linguistic_complexity, zipf_power_law_fit
from gene_language.linguistics.markov import MarkovModelDNA
from gene_language.models.classifier import ExonIntronClassifier
from gene_language.models.motifs import MotifScanner, COMMON_MOTIFS

app = FastAPI(
    title="GENE-LANGUAGE-ANALYTICS API",
    description="Decoding the Language of Life using Natural Language Processing and Machine Learning.",
    version="2.0.0"
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pre-trained / initialized default classifier
_default_classifier = None


def get_default_classifier() -> ExonIntronClassifier:
    global _default_classifier
    if _default_classifier is None:
        clf = ExonIntronClassifier(model_type="xgboost" if hasattr(ExonIntronClassifier, "HAS_XGBOOST") else "random_forest")
        # Train on embedded default benchmark dataset
        sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sample_exons_introns.fasta"))
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                records = parse_fasta_string(f.read())
            seqs, labels = [], []
            for r in records:
                hdr = (r.identifier + " " + r.description).lower()
                if "intron" in hdr or "non-coding" in hdr or "noncoding" in hdr:
                    labels.append(ExonIntronClassifier.CLASS_INTRON)
                    seqs.append(r.sequence.sequence)
                elif "exon" in hdr or "coding" in hdr:
                    labels.append(ExonIntronClassifier.CLASS_EXON)
                    seqs.append(r.sequence.sequence)
            if len(seqs) >= 4:
                clf.fit(seqs, labels, val_split=0.0)
        _default_classifier = clf
    return _default_classifier


class SequenceRequest(BaseModel):
    sequence: str = Field(..., description="Raw DNA string or FASTA text")
    k: int = Field(3, description="k-mer length for frequency extraction (default: 3)")
    window_size: int = Field(100, description="Sliding window size for GC track")
    step_size: int = Field(20, description="Sliding window step size")


class CleanRequest(BaseModel):
    sequence: str
    strip_ambiguous: bool = True
    line_wrap: int = 70


class MotifRequest(BaseModel):
    sequence: str
    scan_reverse: bool = True


class OrfRequest(BaseModel):
    sequence: str
    min_length_aa: int = 25


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0", "framework": "GENE-LANGUAGE-ANALYTICS"}


@app.get("/api/presets")
async def get_presets():
    """Returns biological preset sequences for instant interactive demo."""
    presets = {
        "human_brca1_exon": {
            "name": "Human BRCA1 Exon 11 Snippet",
            "type": "Exon (Coding)",
            "description": "Breast cancer susceptibility gene 1 protein-coding region with high GC content.",
            "sequence": "GATGCGACCGCTTCCCAAGCCAGAGTCGAGCGCCTAAGGCTACGGAGGCTAAGCGCTCAACCCGTTGGCGCCCAGCCATTGCCGCTCGCCCGGGTGAGTGGCGCGAGACCCTGCTCCAGCGCTCGTACACACGGAGCAGAGGATCCGGGGCGTAGGCCGGAGCTCAGCC"
        },
        "human_brca1_intron": {
            "name": "Human BRCA1 Intron Intervening Region",
            "type": "Intron (Non-Coding)",
            "description": "AT-rich non-coding intron with regulatory repeats and low GC ratio.",
            "sequence": "TCCTGACTACTACATTTTGATACTTAGAGGGTTCTGAAGATTTAACCGAAGAACAAATATCAAAGTATAGCTAAATATGTGAGTAAGATCAAACGTGGTTATATCATGCTAGTAAGGTTGTGCAGTCAAGACAAATTACTTCCTTTTACTCAAGTCGAGAACTAATATTTCATTACATGAAACCTTCGGAAAACTAGGGTGGTTCTCAAGCAATTTTCAAGTTGTATGGATGAGCTTTAGTTTTGATTGTCAGAGCTTAAGGGTCGGAGATTAGCTTGACAGTACCCAACTGAACATTCATT"
        },
        "ecoli_lac_promoter": {
            "name": "E. coli Lac Operon Promoter",
            "type": "Prokaryotic Promoter",
            "description": "Classic bacterial operon promoter containing -10 Pribnow Box and -35 consensus.",
            "sequence": "GACACCATCGAATGGCGCAAAACCTTTCGCGGTATGGCATGATAGCGCCCGGAAGAGAGTCAATTCAGGGTGGTGAATGTGAAACCAGTAACGTTATACGATGTCGCAGAGTATGCCGGTGTCTCTTATCAGACCGTTTCCCGCGTGGTGAACCAGGCCAGCCACGTTTCTGCGAAAACGCGGGAAAAAGTGGAAGCGGCGATGGCGGAGCTGAATTACATTCCCAACCGCGTGGCACAACAACTGGCGGGCAAACAGTCGTTGCTGATTGGCGTTGCC"
        },
        "human_beta_globin": {
            "name": "Human Beta-Globin (HBB) CDS",
            "type": "Exon (Coding)",
            "description": "Hemoglobin subunit beta gene containing canonical Kozak translation initiation site.",
            "sequence": "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACCATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCACTAAGCTCGCTTTCTTGCTGTCCAATTTCTATTAAAGGTTCCTTTGTTCCCTAAGTCCAACTACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGC"
        },
        "sars_cov2_spike_rbd": {
            "name": "SARS-CoV-2 Spike RBD Fragment",
            "type": "Viral Gene CDS",
            "description": "Receptor-binding domain fragment from Spike glycoprotein.",
            "sequence": "AATGTTACAAATTTATGTCCATTTGGTGAAGTTTTTAACGCCACCAGATTTGCATCTGTTTATGCTTGGAACAGGAAGAGAATCAGCAACTGTGTTGCTGATTATTCTGTCCTATATAATTCCGCATCATTTTCCACTTTTAAGTGTTATGGAGTGTCTCCTACTAAATTAAATGATCTCTGCTTTACTAATGTCTATGCAGATTCATTTGTAATTAGAGGTGATGAAGTCAGACAAATCGCTCCAGGGCAAACTGGAAAGATTGCTGATTATAATTATAAATTACCAGATGATTTTACAGGCTGCGTTATAGCTTGGAATTCTAACAATCTTGATTCTAAGGTTGGTGGTAATTATAATTACCTGTATAGATTGTTTAGGAAGTCTAATCTCAAACCTTTTGAGAGAGATATTTCAACTGAAATCTATCAGGCCGGTAGCACACCTTGTAATGGTGTTGAAGGTTTTAATTGTTACTTTCCTTTACAATCATATGGTTTCCAACCCACTAATGGTGTTGGTTACCAACCATACAGAGTAGTAGTACTTTCTTTTGAACTTCTACATGCACCAGCAACTGTTTGTGGACCTAAAAAGTCTACTAATTTGGTTAAAAACAAATGTGTCAATTTCAACTTCAATGGTTTAACAGGCACAGGTGTTCTTACTGAGTCTAACAAAAAGTTTCTGCCTTTCCAACAATTTGGCAGAGACATTGCTGACACTACTGATGCTGTCCGTGATCCACAGACACTTGAGATTCTTGACATTACACCATGTTCTTTTGGTGGTGTCAGTGTTATAACACCAGGAACAAATACTTCTAACCAGGTTGCTGTTCTTTATCAGGATGTTAACTGCACAGAAGTCCCTGTTGCTATTCATGCAGATCAACTTACTCCTACTTGGCGTGTTTATTCTACAGGTTCTAATGTTTTTCAAACACGTGCAGGCTGTTTAATAGGGGCTGAACATGTCAACAACTCATATGAGTGTGACATACCCATTGGTGCAGGTATATGCGCTAGTTATCAGACTCAGACTAATTCTCCTCGGCGGGCACGTAGTGTAGCTAGTCAATCCATCATTGCCTACACTATGTCACTTGGTGCAGAAAATTCAGTTGCTTACTCTAATAACTCTATTGCCATACCCACAAATTTTACTATTAGTGTTACCACAGAAATTCTACCAGTGTCTATGACCAAGACATCAGTAGATTGTACAATGTACATTTGTGGTGATTCAACTGAATGCAGCAATCTTTTGTTGCAATATGGCAGTTTTTGTACACAATTAAACCGTGCTTTAACTGGAATAGCTGTTGAACAAGACAAAAACACCCAAGAAGTTTTTGCACAAGTCAAACAAATTTACAAAACACCACCAATTAAAGATTTTGGTGGTTTTAATTTTTCACAAATATTACCAGATCCATCAAAACCAAGCAAGAGGTCATTTATTGAAGATCTACTTTTCAACAAAGTGACACTTGCAGATGCTGGCTTCATCAAACAATATGGTGATTGCCTTGGTGATATTGCTGCTAGAGACCTCATTTGTGCACAAAAGTTTAACGGCCTTACTGTTTTGCCACCTTTGCTCACAGATGAAATGATTGCTCAATACACTTCTGCACTGTTAGCGGGTACAATCACTTCTGGTTGGACCTTTGGTGCAGGTGCTGCATTACAAATACCATTTGCTATGCAAATGGCTTATAGGTTTAATGGTATTGGAGTTACACAGAATGTTCTCTATGAGAACCAAAAATTGATTGCCAACCAATTTAATAGTGCTATTGGCAAAATTCAAGACTCACTTTCTTCCACAGCAAGTGCACTTGGAAAACTTCAAGATGTGGTCAACCAAAATGCACAAGCTTTAAACACGCTTGTTAAACAACTTAGCTCCAATTTTGGTGCAATTTCAAGTGTTTTAAATGATATCCTTTCACGTCTTGACAAAGTTGAGGCTGAAGTGCAAATTGATAGGTTGATCACAGGCAGACTTCAAAGTTTGCAGACATATGTGACTCAACAATTAATTAGAGCTGCAGAAATCAGAGCTTCTGCTAATCTTGCTGCTACTAAAATGTCAGAGTGTGTACTTGGACAATCAAAAAGAGTTGATTTTTGTGGAAAGGGCTATCATCTTATGTCCTTCCCTCAGTCAGCACCTCATGGTGTAGTCTTCTTGCATGTGACATATGTACCAGCTCAAGAAAAGAACTTCACAACTGCTCCTGCCATTTGTCATGATGGAAAAGCACACTTTCCTCGTGAAGGTGTCTTTGTTTCAAATGGCACACACTGGTTTGTAACACAAAGGAATTTTTATGAACCACAAATCATTACTACAGACAACACATTTGTGTCTGGTAACTGTGATGTTGTAATAGGAATTGTCAACAACACAGTTTATGATCCTTTGCAACCTGAATTAGACTCATTCAAGGAGGAGTTAGATAAATATTTTAAGAATCATACATCACCAGATGTTGATTTAGGTGACATCTCTGGCATTAATGCTTCAGTTGTAAACATTCAAAAAGAAATTGACCGCCTCAATGAGGTCGCCAAGAATTTAAATGAATCTCTCATCGATCTCCAAGAACTTGGAAAGTATGAGCAGTATATAAAATGGCCATGGTACATTTGGCTAGGTTTTATAGCTGGCTTGATTGCCATAGTAATGGTGACAATTATGCTTTGCTGTATGACCAGTTGCTGTAGTTGTCTCAAGGGCTGTTGTTCTTGTGGATCCTGCTGCAAATTTGATGAAGACGACTCTGAGCCAGTGCTCAAAGGAGTCAAATTACATTACACATAA"
        }
    }
    return presets


@app.post("/api/analyze")
async def analyze_sequence(req: SequenceRequest):
    """Full comprehensive genomic NLP and biochemical analysis endpoint."""
    raw_text = req.sequence.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty sequence provided.")

    records = parse_fasta_string(raw_text)
    if not records:
        raise HTTPException(status_code=400, detail="Could not parse any valid sequence.")

    rec = records[0]
    seq_obj = rec.sequence
    k = max(1, min(6, req.k))

    # Basic stats
    length = len(seq_obj)
    if length == 0:
        raise HTTPException(status_code=400, detail="Sequence length is 0.")

    counts = seq_obj.base_counts
    pcts = seq_obj.base_percentages
    gc_val = seq_obj.gc_content
    at_val = seq_obj.at_content
    gc_sk = seq_obj.gc_skew
    at_sk = seq_obj.at_skew
    cpg_ratio = seq_obj.cpg_observed_expected_ratio()
    cpg_islands = seq_obj.find_cpg_islands()
    sliding_gc = seq_obj.sliding_window_gc(window_size=req.window_size, step_size=req.step_size)

    # K-mer analysis
    kmer_ext = KmerExtractor(k=k)
    kmer_counts = kmer_ext.count_kmers(seq_obj.sequence)
    top_kmers = kmer_ext.top_kmers(seq_obj.sequence, top_n=15)
    spectrum = compute_kmer_spectrum(seq_obj.sequence, k_range=(2, 3, 4))

    # Entropy & Linguistics
    shannon = shannon_entropy(seq_obj.sequence, k=1)
    shannon_kmer = shannon_entropy(seq_obj.sequence, k=k)
    complexity = linguistic_complexity(seq_obj.sequence, max_k=5)
    zipf = zipf_power_law_fit(seq_obj.sequence, k=3)

    # Markov model
    markov = MarkovModelDNA(order=1).fit(seq_obj.sequence)
    markov_dict = markov.to_dict()

    # RSCU
    rscu = calculate_rscu(seq_obj.sequence)

    # Classification check
    clf = get_default_classifier()
    clf_res = clf.predict_single(seq_obj.sequence)

    # Motifs & ORFs summary
    scanner = MotifScanner()
    motifs = [m.to_dict() for m in scanner.scan(seq_obj.sequence)[:30]]
    orfs = [
        {
            "strand": o.strand,
            "frame": o.frame,
            "start": o.start,
            "end": o.end,
            "length_nt": o.length_nt,
            "length_aa": o.length_aa,
            "protein": o.protein_sequence[:30] + "..." if len(o.protein_sequence) > 30 else o.protein_sequence
        }
        for o in find_orfs(seq_obj.sequence, min_protein_len=20)[:10]
    ]

    return {
        "header": rec.header,
        "identifier": rec.identifier,
        "description": rec.description,
        "length": length,
        "base_counts": counts,
        "base_percentages": pcts,
        "gc_content": gc_val,
        "at_content": at_val,
        "gc_skew": gc_sk,
        "at_skew": at_sk,
        "melting_temperature_c": seq_obj.melting_temperature(),
        "molecular_weight_da": seq_obj.molecular_weight(double_stranded=True),
        "cpg_ratio": cpg_ratio,
        "cpg_islands": cpg_islands,
        "sliding_gc": sliding_gc[:200],  # Bound payload size
        "k": k,
        "kmer_counts": {kmer: cnt for kmer, cnt in kmer_counts.items() if cnt > 0},
        "top_kmers": top_kmers,
        "kmer_spectrum": spectrum,
        "shannon_entropy": shannon,
        "shannon_entropy_kmer": shannon_kmer,
        "linguistic_complexity": complexity,
        "zipf_fit": zipf,
        "markov_transition_matrix": markov_dict,
        "rscu": {
            "effective_nc": rscu.effective_number_of_codons,
            "total_codons": rscu.total_codons,
            "top_rscu": sorted(rscu.rscu_values.items(), key=lambda x: x[1], reverse=True)[:10]
        },
        "classification": clf_res,
        "motifs_found": motifs,
        "top_orfs": orfs
    }


@app.post("/api/classify")
async def classify_seq(req: SequenceRequest):
    """Predicts Exon vs. Intron classification."""
    records = parse_fasta_string(req.sequence)
    if not records:
        raise HTTPException(status_code=400, detail="Invalid sequence.")

    clf = get_default_classifier()
    res = clf.predict_single(records[0].sequence.sequence)
    return res


@app.post("/api/orfs")
async def get_orfs(req: OrfRequest):
    """6-frame Open Reading Frame analysis."""
    records = parse_fasta_string(req.sequence)
    if not records:
        raise HTTPException(status_code=400, detail="Invalid sequence.")

    seq = records[0].sequence.sequence
    orfs = find_orfs(seq, min_protein_len=req.min_length_aa)
    return [
        {
            "strand": o.strand,
            "frame": o.frame,
            "start": o.start,
            "end": o.end,
            "length_nt": o.length_nt,
            "length_aa": o.length_aa,
            "protein_sequence": o.protein_sequence,
            "dna_sequence": o.dna_sequence
        }
        for o in orfs
    ]


@app.post("/api/motifs")
async def get_motifs(req: MotifRequest):
    """Scans for biological motifs."""
    records = parse_fasta_string(req.sequence)
    if not records:
        raise HTTPException(status_code=400, detail="Invalid sequence.")

    scanner = MotifScanner()
    matches = scanner.scan(records[0].sequence.sequence, scan_reverse_strand=req.scan_reverse)
    return [m.to_dict() for m in matches]


@app.post("/api/clean")
async def clean_seq(req: CleanRequest):
    """Standardizes FASTA sequence."""
    records = parse_fasta_string(req.sequence, strip_ambiguous=req.strip_ambiguous)
    if not records:
        raise HTTPException(status_code=400, detail="Invalid sequence.")

    rec = records[0]
    seq_str = rec.sequence.sequence
    wrapped = [seq_str[i:i + req.line_wrap] for i in range(0, len(seq_str), req.line_wrap)]

    fasta_formatted = f">{rec.identifier} cleaned\n" + "\n".join(wrapped)
    return {
        "identifier": rec.identifier,
        "length": len(seq_str),
        "gc_content": rec.sequence.gc_content,
        "fasta_text": fasta_formatted
    }


# Mount static frontend directory if present
web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../web"))
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
