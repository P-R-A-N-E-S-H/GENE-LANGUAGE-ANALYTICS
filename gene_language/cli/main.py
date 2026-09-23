"""
Rich interactive Command Line Interface for GENE-LANGUAGE-ANALYTICS.
"""

from __future__ import annotations
import sys
import os
import json
import click

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

from ..core.sequence import GenomicSequence
from ..core.fasta import read_fasta, clean_fasta, stream_fasta
from ..core.codons import find_orfs, calculate_rscu, translate_sequence
from ..linguistics.kmer import KmerExtractor, compute_kmer_spectrum
from ..linguistics.entropy import shannon_entropy, linguistic_complexity, zipf_power_law_fit
from ..models.classifier import ExonIntronClassifier
from ..models.evaluator import ModelEvaluator
from ..models.motifs import MotifScanner


@click.group()
@click.version_option(version="2.0.0", prog_name="gene-lang")
def cli():
    """🧬 GENE-LANGUAGE-ANALYTICS: Decoding the Language of Life using NLP & ML."""
    pass


@cli.command("analyze")
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--k", default=3, help="K-mer length for linguistic tokenization (default: 3)")
@click.option("--top", default=10, help="Number of top k-mers to display (default: 10)")
@click.option("--json-out", is_flag=True, help="Output results in JSON format")
def analyze(fasta_file: str, k: int, top: int, json_out: bool):
    """Perform in-depth linguistic, compositional, and information entropy analysis on FASTA sequence."""
    records = read_fasta(fasta_file, max_records=1)
    if not records:
        click.echo("Error: No valid sequences found in FASTA file.", err=True)
        sys.exit(1)

    rec = records[0]
    seq_obj = rec.sequence
    kmer_ext = KmerExtractor(k=k)
    top_kmers = kmer_ext.top_kmers(seq_obj.sequence, top_n=top)
    entropy_info = shannon_entropy(seq_obj.sequence, k=1)
    complexity = linguistic_complexity(seq_obj.sequence, max_k=5)
    cpg_islands = seq_obj.find_cpg_islands()
    rscu = calculate_rscu(seq_obj.sequence)

    results = {
        "identifier": rec.identifier,
        "length_bp": len(seq_obj),
        "gc_content_pct": seq_obj.gc_content,
        "at_content_pct": seq_obj.at_content,
        "gc_skew": seq_obj.gc_skew,
        "at_skew": seq_obj.at_skew,
        "cpg_observed_expected_ratio": seq_obj.cpg_observed_expected_ratio(),
        "cpg_islands_found": len(cpg_islands),
        "shannon_entropy_bits": entropy_info["entropy"],
        "normalized_entropy": entropy_info["normalized_entropy"],
        "linguistic_complexity": complexity["linguistic_complexity"],
        "effective_codons_nc": rscu.effective_number_of_codons,
        f"top_{top}_kmers_k{k}": top_kmers,
    }

    if json_out:
        click.echo(json.dumps(results, indent=2))
        return

    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]🧬 Genomic Linguistic & Composition Profile[/bold cyan]\n"
            f"[bold white]ID:[/bold white] {rec.identifier} | [bold white]Length:[/bold white] {len(seq_obj):,} bp",
            border_style="cyan"
        ))

        # Overview table
        tbl = Table(title="Nucleotide Composition & Skew", show_header=True, header_style="bold magenta")
        tbl.add_column("Metric", style="dim")
        tbl.add_column("Value", justify="right", style="bold green")

        tbl.add_row("Sequence Length", f"{len(seq_obj):,} bp")
        tbl.add_row("GC Content", f"{seq_obj.gc_content:.2f}%")
        tbl.add_row("AT Content", f"{seq_obj.at_content:.2f}%")
        tbl.add_row("GC Skew ((G-C)/(G+C))", f"{seq_obj.gc_skew:+.4f}")
        tbl.add_row("AT Skew ((A-T)/(A+T))", f"{seq_obj.at_skew:+.4f}")
        tbl.add_row("CpG Obs/Exp Ratio", f"{seq_obj.cpg_observed_expected_ratio():.4f}")
        tbl.add_row("Putative CpG Islands", str(len(cpg_islands)))
        tbl.add_row("Shannon Entropy H (k=1)", f"{entropy_info['entropy']:.4f} bits (Norm: {entropy_info['normalized_entropy']:.2%})")
        tbl.add_row("Linguistic Complexity (LC)", f"{complexity['linguistic_complexity']:.6f}")
        tbl.add_row("Effective No. of Codons (Nc)", f"{rscu.effective_number_of_codons:.2f}")
        console.print(tbl)

        # K-mers table
        ktbl = Table(title=f"Top {top} Most Frequent {k}-mers", show_header=True, header_style="bold blue")
        ktbl.add_column("Rank", justify="center")
        ktbl.add_column(f"{k}-mer", style="bold yellow")
        ktbl.add_column("Count", justify="right")
        ktbl.add_column("Frequency (%)", justify="right", style="bold green")

        for rank, (kmer, cnt, pct) in enumerate(top_kmers, 1):
            ktbl.add_row(str(rank), kmer, f"{cnt:,}", f"{pct:.2f}%")
        console.print(ktbl)
    else:
        click.echo(f"=== Analysis for {rec.identifier} ===")
        for k_name, val in results.items():
            click.echo(f"{k_name}: {val}")


@cli.command("clean")
@click.argument("input_fasta", type=click.Path(exists=True))
@click.argument("output_fasta", type=click.Path())
@click.option("--strip-n/--no-strip-n", default=True, help="Remove ambiguous 'N' nucleotides")
@click.option("--wrap", default=70, help="Line wrap width (default: 70)")
def clean(input_fasta: str, output_fasta: str, strip_n: bool, wrap: int):
    """Sanitize and standardize multiline FASTA sequence formatting."""
    stats = clean_fasta(input_fasta, output_fasta, strip_ambiguous=strip_n, line_wrap=wrap)
    if HAS_RICH:
        console.print(f"[bold green]✓[/bold green] Cleaned [bold white]{stats['records_processed']}[/bold white] records.")
        console.print(f"Total bases: [bold cyan]{stats['total_bases']:,}[/bold cyan] bp (GC: [bold yellow]{stats['overall_gc_percent']}%[/bold yellow])")
        console.print(f"Saved to: [bold underline]{output_fasta}[/bold underline]")
    else:
        click.echo(f"Cleaned {stats['records_processed']} records ({stats['total_bases']} bp). Saved to {output_fasta}")


@cli.command("orf")
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--min-len", default=30, help="Minimum protein length in amino acids (default: 30)")
def orf(fasta_file: str, min_len: int):
    """Find all Open Reading Frames (ORFs) across 6 reading frames."""
    records = read_fasta(fasta_file, max_records=1)
    if not records:
        click.echo("Error: FASTA sequence empty.", err=True)
        return

    seq = records[0].sequence.sequence
    orfs = find_orfs(seq, min_protein_len=min_len)

    if HAS_RICH:
        tbl = Table(title=f"Discovered Open Reading Frames (>= {min_len} AA)", header_style="bold green")
        tbl.add_column("Strand", justify="center")
        tbl.add_column("Frame", justify="center")
        tbl.add_column("Coordinates", justify="center")
        tbl.add_column("DNA (bp)", justify="right")
        tbl.add_column("Peptide (AA)", justify="right")
        tbl.add_column("Protein Preview", style="dim")

        for item in orfs[:15]:
            preview = item.protein_sequence[:15] + "..." if len(item.protein_sequence) > 15 else item.protein_sequence
            tbl.add_row(
                item.strand,
                f"{item.frame:+d}",
                f"{item.start:,} - {item.end:,}",
                f"{item.length_nt:,}",
                f"{item.length_aa:,}",
                preview
            )
        console.print(tbl)
        console.print(f"Total ORFs identified: [bold cyan]{len(orfs)}[/bold cyan]")
    else:
        click.echo(f"Found {len(orfs)} ORFs:")
        for o in orfs[:10]:
            click.echo(f"Strand {o.strand} Frame {o.frame}: {o.start}-{o.end} ({o.length_aa} AA): {o.protein_sequence[:20]}")


@cli.command("motifs")
@click.argument("fasta_file", type=click.Path(exists=True))
def motifs(fasta_file: str):
    """Scan FASTA sequence for promoters, Kozak initiation consensus, and splice junctions."""
    records = read_fasta(fasta_file, max_records=1)
    if not records:
        return

    scanner = MotifScanner()
    matches = scanner.scan(records[0].sequence.sequence)

    if HAS_RICH:
        tbl = Table(title="Identified Regulatory Motifs & Consensus Sites", header_style="bold yellow")
        tbl.add_column("Motif Name", style="bold cyan")
        tbl.add_column("Category", style="dim")
        tbl.add_column("Strand", justify="center")
        tbl.add_column("Position", justify="center")
        tbl.add_column("Sequence", style="bold green")

        for m in matches[:25]:
            tbl.add_row(
                m.motif_name,
                m.category,
                m.strand,
                f"{m.start:,} - {m.end:,}",
                m.matched_sequence
            )
        console.print(tbl)
        console.print(f"Total motif matches: [bold cyan]{len(matches)}[/bold cyan]")
    else:
        click.echo(f"Found {len(matches)} motif occurrences.")
        for m in matches[:15]:
            click.echo(f"[{m.motif_name}] ({m.start}-{m.end}): {m.matched_sequence}")


@cli.command("classify")
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--model", default="xgboost", help="Classifier type (xgboost, random_forest, logistic_regression)")
def classify(fasta_file: str, model: str):
    """Train and evaluate Exon vs. Intron classifier on labeled FASTA dataset."""
    records = read_fasta(fasta_file)
    seqs = []
    labels = []

    for r in records:
        # Detect label in header
        header_lower = (r.identifier + " " + r.description).lower()
        if "intron" in header_lower or "non-coding" in header_lower or "noncoding" in header_lower:
            labels.append(ExonIntronClassifier.CLASS_INTRON)
            seqs.append(r.sequence.sequence)
        elif "exon" in header_lower or "coding" in header_lower:
            labels.append(ExonIntronClassifier.CLASS_EXON)
            seqs.append(r.sequence.sequence)

    if len(seqs) < 4:
        click.echo("Error: Need at least 4 labeled sequences (with 'exon' or 'intron' in headers).", err=True)
        return

    clf = ExonIntronClassifier(model_type=model)
    report = ModelEvaluator.evaluate(clf, seqs, labels)

    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]🤖 Exon vs. Intron Machine Learning Classifier ({model})[/bold cyan]\n"
            f"Evaluated on [bold white]{len(seqs)}[/bold white] labeled genomic sequences.",
            border_style="green"
        ))

        tbl = Table(title="Model Performance Metrics", header_style="bold magenta")
        tbl.add_column("Metric", style="dim")
        tbl.add_column("Score", justify="right", style="bold green")

        tbl.add_row("Classification Accuracy", f"{report.accuracy * 100:.2f}%")
        tbl.add_row("Macro Precision", f"{report.precision_macro:.4f}")
        tbl.add_row("Macro Recall", f"{report.recall_macro:.4f}")
        tbl.add_row("Macro F1-Score", f"{report.f1_macro:.4f}")
        tbl.add_row("5-Fold Cross-Validation", f"{report.cv_mean * 100:.2f}% ± {report.cv_std * 100:.2f}%")
        console.print(tbl)

        # Feature importances
        top_feats = clf.get_top_features(top_n=8)
        if top_feats:
            ftbl = Table(title="Top Predictive k-mer Features", header_style="bold blue")
            ftbl.add_column("Feature", style="bold yellow")
            ftbl.add_column("Importance Score", justify="right", style="bold green")
            for f in top_feats:
                ftbl.add_row(f["feature"], f"{f['importance']:.5f}")
            console.print(ftbl)
    else:
        click.echo(f"Accuracy: {report.accuracy * 100:.2f}% | F1: {report.f1_macro:.4f}")
        click.echo(f"5-Fold CV: {report.cv_mean * 100:.2f}% +- {report.cv_std * 100:.2f}%")


@cli.command("crispr")
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--top", default=10, help="Number of top candidates to display (default: 10)")
def crispr(fasta_file: str, top: int):
    """Discover SpCas9 sgRNA guide RNA candidates with PAM (NGG) and on-target efficiency."""
    from ..core.crispr import CrisprGuideDesigner
    records = read_fasta(fasta_file, max_records=1)
    if not records:
        return

    designer = CrisprGuideDesigner()
    guides = designer.find_guides(records[0].sequence.sequence)

    if HAS_RICH:
        tbl = Table(title="Top CRISPR-Cas9 sgRNA Guide RNA Candidates", header_style="bold green")
        tbl.add_column("Rank", justify="center")
        tbl.add_column("Protospacer (20nt)", style="bold cyan")
        tbl.add_column("PAM", style="bold magenta")
        tbl.add_column("Strand", justify="center")
        tbl.add_column("Coordinates", justify="center")
        tbl.add_column("GC%", justify="right")
        tbl.add_column("Score", justify="right", style="bold yellow")
        tbl.add_column("Evaluation", style="dim")

        for rank, g in enumerate(guides[:top], 1):
            tbl.add_row(
                str(rank),
                g.spacer_sequence,
                g.pam_sequence,
                g.strand,
                f"{g.start}:{g.end}",
                f"{g.gc_content:.1f}%",
                f"{g.efficiency_score:.1f}",
                g.recommendation
            )
        console.print(tbl)
        console.print(f"Total guide candidates found: [bold cyan]{len(guides)}[/bold cyan]")
    else:
        click.echo(f"Found {len(guides)} guide sites:")
        for r, g in enumerate(guides[:top], 1):
            click.echo(f"#{r} [{g.strand}] {g.spacer_sequence} {g.pam_sequence} (Score: {g.efficiency_score})")


@cli.command("splice")
@click.argument("fasta_file", type=click.Path(exists=True))
def splice(fasta_file: str):
    """Scan sequence for 5' splice donors (GT) and 3' splice acceptors (AG)."""
    from ..models.splice_junction import SpliceJunctionScorer
    records = read_fasta(fasta_file, max_records=1)
    if not records:
        return

    scorer = SpliceJunctionScorer()
    junctions = scorer.scan_all(records[0].sequence.sequence)

    if HAS_RICH:
        tbl = Table(title="Detected Splice Junctions (PWM Log-Odds Scoring)", header_style="bold yellow")
        tbl.add_column("Type", style="bold cyan")
        tbl.add_column("Position", justify="center")
        tbl.add_column("Context Window", style="bold green")
        tbl.add_column("Score (bits)", justify="right")
        tbl.add_column("Confidence", justify="right", style="bold yellow")

        for j in junctions[:20]:
            tbl.add_row(
                j.site_type,
                str(j.position),
                j.sequence_context,
                f"{j.score:.2f}",
                f"{j.confidence * 100:.1f}%"
            )
        console.print(tbl)
        console.print(f"Total splice junctions identified: [bold cyan]{len(junctions)}[/bold cyan]")
    else:
        click.echo(f"Found {len(junctions)} junctions:")
        for j in junctions[:15]:
            click.echo(f"[{j.site_type}] pos {j.position} ({j.score} bits): {j.sequence_context}")


@cli.command("repeats")
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--min-copies", default=3, help="Minimum consecutive repetitions (default: 3)")
@click.option("--top", default=15, help="Number of repeat tracts to display (default: 15)")
def repeats(fasta_file: str, min_copies: int, top: int):
    """Discover microsatellites, tandem repeats, and pathogenic expansion motifs."""
    from ..core.repeats import TandemRepeatScanner
    records = read_fasta(fasta_file, max_records=1)
    if not records:
        return

    seq = records[0].sequence.sequence
    scanner = TandemRepeatScanner()
    all_repeats = scanner.scan(seq, min_copies=min_copies)
    disease_repeats = scanner.scan_pathogenic_expansions(seq)
    telomere_info = scanner.telomeric_profile(seq)

    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]🔬 Microsatellite & Short Tandem Repeat (STR) Analysis[/bold cyan]\n"
            f"Discovered [bold white]{len(all_repeats)}[/bold white] repeat tracts across [bold white]{len(seq):,}[/bold white] bp.",
            border_style="magenta"
        ))

        tbl = Table(title=f"Top Tandem Repeats (>= {min_copies} copies)", header_style="bold magenta")
        tbl.add_column("Unit", style="bold yellow")
        tbl.add_column("Unit Len", justify="center")
        tbl.add_column("Copies", justify="right", style="bold green")
        tbl.add_column("Coordinates", justify="center")
        tbl.add_column("Total (bp)", justify="right")
        tbl.add_column("Pathogenic / Disease Association", style="bold red")

        for r in all_repeats[:top]:
            dis = r.disease_associated or "—"
            tbl.add_row(
                r.unit,
                str(r.unit_length),
                f"{r.copies:.1f}x",
                f"{r.start:,} - {r.end:,}",
                f"{r.total_length:,}",
                dis
            )
        console.print(tbl)

        if disease_repeats:
            console.print(f"\n[bold red]⚠️  Discovered {len(disease_repeats)} Pathogenic Disease-Associated Repeat Motifs![/bold red]")

        console.print(f"Telomeric Repeat Tracts (TTAGGG/CCCTAA): [bold cyan]{telomere_info['total_telomeric_bp']} bp ({telomere_info['telomeric_percentage']}%) [/bold cyan]")
    else:
        click.echo(f"Found {len(all_repeats)} repeats:")
        for r in all_repeats[:top]:
            click.echo(f"Unit {r.unit} ({r.copies:.1f}x) at {r.start}-{r.end} [{r.disease_associated or 'Normal'}]")


@cli.command("promoters")
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--min-score", default=0.72, help="Minimum PWM relative score threshold (default: 0.72)")
def promoters(fasta_file: str, min_score: float):
    """Scan sequence for core promoter elements (TATA, Inr, DPE, BRE, Sp1) and TSS architectures."""
    from ..models.promoters import PromoterArchitectureScanner
    records = read_fasta(fasta_file, max_records=1)
    if not records:
        return

    seq = records[0].sequence.sequence
    scanner = PromoterArchitectureScanner()
    elements = scanner.scan(seq, min_relative_score=min_score)
    architectures = scanner.identify_putative_promoter_regions(seq)

    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]🎯 Core Promoter & Regulatory Architecture Scanner[/bold cyan]\n"
            f"Found [bold white]{len(elements)}[/bold white] promoter elements; predicted [bold white]{len(architectures)}[/bold white] putative TSS loci.",
            border_style="cyan"
        ))

        tbl = Table(title="Detected Core Promoter Elements (PWM Scoring)", header_style="bold green")
        tbl.add_column("Element", style="bold cyan")
        tbl.add_column("Coordinates", justify="center")
        tbl.add_column("Sequence", style="bold yellow")
        tbl.add_column("PWM Score (bits)", justify="right")
        tbl.add_column("Relative Score", justify="right", style="bold green")

        for el in elements[:20]:
            tbl.add_row(
                el.name,
                f"{el.start:,} - {el.end:,}",
                el.sequence,
                f"{el.raw_score:.2f}",
                f"{el.relative_score * 100:.1f}%"
            )
        console.print(tbl)

        if architectures:
            ptbl = Table(title="Predicted Core Promoter Architectures & Transcription Start Sites", header_style="bold blue")
            ptbl.add_column("Predicted TSS", justify="center", style="bold yellow")
            ptbl.add_column("Architecture Type", style="bold green")
            ptbl.add_column("TATA-box Status", style="dim")
            ptbl.add_column("Sp1 / CpG Boxes", justify="center")

            for arch in architectures[:10]:
                tata_str = f"Present ({arch['tata_box']['sequence']})" if arch["tata_box"] else "TATA-less"
                ptbl.add_row(
                    f"Position +{arch['predicted_tss']}",
                    arch["architecture_type"],
                    tata_str,
                    str(len(arch["sp1_gc_boxes"]))
                )
            console.print(ptbl)
    else:
        click.echo(f"Discovered {len(elements)} promoter elements, {len(architectures)} putative TSS sites.")


@cli.command("isochores")
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--window", default=1000, help="Sliding window size in bp (default: 1000)")
@click.option("--step", default=200, help="Step size in bp (default: 200)")
def isochores(fasta_file: str, window: int, step: int):
    """Segment eukaryotic genomic sequences into isochore families (L1, L2, H1, H2, H3)."""
    from ..core.isochore import classify_isochore_family, segment_isochores, compute_gc3_profile
    from ..visualization.ascii_plots import ascii_sparkline
    records = read_fasta(fasta_file, max_records=1)
    if not records:
        return

    seq = records[0].sequence.sequence
    family, mean_gc = classify_isochore_family(seq)
    segments = segment_isochores(seq, window_size=window, step_size=step)
    gc3 = compute_gc3_profile(seq)

    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold yellow]🌐 Isochore Segmentation & Compositional Profile[/bold yellow]\n"
            f"Overall Family: [bold magenta]{family}[/bold magenta] | Mean GC: [bold green]{mean_gc:.2f}%[/bold green] | GC3: [bold cyan]{gc3['gc3_percent']:.2f}%[/bold cyan]",
            border_style="yellow"
        ))

        gc_vals = [s.gc_percent for s in segments]
        if gc_vals:
            spark = ascii_sparkline(gc_vals, max_bars=60)
            console.print(f"GC% Sparkline: [bold cyan]{spark}[/bold cyan]\n")

        tbl = Table(title="Isochore Regional Windows", header_style="bold yellow")
        tbl.add_column("Segment Coordinates", justify="center")
        tbl.add_column("GC%", justify="right", style="bold green")
        tbl.add_column("Isochore Family", style="bold magenta")

        for s in segments[:15]:
            tbl.add_row(
                f"{s.start:,} - {s.end:,}",
                f"{s.gc_percent:.2f}%",
                s.family
            )
        console.print(tbl)
    else:
        click.echo(f"Isochore Family: {family} (GC: {mean_gc:.2f}%, GC3: {gc3['gc3_percent']:.2f}%)")


@cli.command("distance")
@click.argument("fasta_file1", type=click.Path(exists=True))
@click.argument("fasta_file2", type=click.Path(exists=True))
@click.option("--k", default=3, help="K-mer length for frequency vector distance (default: 3)")
def distance(fasta_file1: str, fasta_file2: str, k: int):
    """Compute alignment-free distance and similarity metrics between two FASTA files."""
    from ..linguistics.embeddings import (
        cosine_similarity_dna,
        jaccard_similarity_dna,
        euclidean_distance_dna,
        jensen_shannon_divergence,
    )
    rec1 = read_fasta(fasta_file1, max_records=1)
    rec2 = read_fasta(fasta_file2, max_records=1)

    if not rec1 or not rec2:
        click.echo("Error: Could not read sequences from one or both files.", err=True)
        return

    seq1 = rec1[0].sequence.sequence
    seq2 = rec2[0].sequence.sequence

    cos_sim = cosine_similarity_dna(seq1, seq2, k=k)
    jac_sim = jaccard_similarity_dna(seq1, seq2, k=k)
    euc_dist = euclidean_distance_dna(seq1, seq2, k=k, normalized=True)
    jsd_div = jensen_shannon_divergence(seq1, seq2, k=k)

    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]📏 Alignment-Free Sequence Distance & Similarity (k={k})[/bold cyan]\n"
            f"Seq 1: [bold white]{rec1[0].identifier}[/bold white] ({len(seq1):,} bp) vs. Seq 2: [bold white]{rec2[0].identifier}[/bold white] ({len(seq2):,} bp)",
            border_style="green"
        ))

        tbl = Table(title="Distance & Similarity Metrics", header_style="bold blue")
        tbl.add_column("Metric", style="dim")
        tbl.add_column("Score", justify="right", style="bold yellow")
        tbl.add_column("Interpretation", style="dim")

        tbl.add_row("Cosine Similarity", f"{cos_sim:.5f}", "1.0 = identical k-mer distribution")
        tbl.add_row("Jaccard Set Similarity", f"{jac_sim:.5f}", "1.0 = identical k-mer vocabulary")
        tbl.add_row("Normalized Euclidean Distance", f"{euc_dist:.5f}", "0.0 = identical frequency vector")
        tbl.add_row("Jensen-Shannon Divergence", f"{jsd_div:.5f}", "0.0 = identical distribution")
        console.print(tbl)
    else:
        click.echo(f"Cosine: {cos_sim:.5f} | Jaccard: {jac_sim:.5f} | Euclidean: {euc_dist:.5f} | JSD: {jsd_div:.5f}")


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Host IP address (default: 127.0.0.1)")
@click.option("--port", default=8000, help="Port number (default: 8000)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Launch the interactive Web Studio & FastAPI server."""
    try:
        import uvicorn
        from ...api.server import app
    except Exception:
        # Import directly
        import uvicorn
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        from api.server import app

    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold green]🚀 Launching GENE-LANGUAGE-ANALYTICS Studio[/bold green]\n"
            f"Serving Web UI at: [bold cyan]http://{host}:{port}[/bold cyan]\n"
            f"Interactive API Docs at: [bold cyan]http://{host}:{port}/docs[/bold cyan]",
            border_style="cyan"
        ))
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()
