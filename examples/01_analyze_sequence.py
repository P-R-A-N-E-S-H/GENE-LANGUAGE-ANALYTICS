"""
Example 1: In-depth Genomic Linguistic and Information Theory Analysis
"""

import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language import (
    GenomicSequence,
    read_fasta,
    KmerExtractor,
    shannon_entropy,
    linguistic_complexity,
    MarkovModelDNA,
    find_orfs,
    calculate_rscu
)

def main():
    fasta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sample_promoters.fasta"))
    print(f"Loading sequence from: {fasta_path}")
    records = read_fasta(fasta_path, max_records=1)

    if not records:
        print("No records found.")
        return

    rec = records[0]
    seq = rec.sequence
    print(f"\n[+] Analyzing: {rec.header}")
    print(f"    Length: {len(seq):,} bp")
    print(f"    GC Content: {seq.gc_content:.2f}% | AT Content: {seq.at_content:.2f}%")
    print(f"    GC Skew: {seq.gc_skew:+.4f} | AT Skew: {seq.at_skew:+.4f}")
    print(f"    CpG Obs/Exp Ratio: {seq.cpg_observed_expected_ratio():.4f}")

    # 1. Linguistic k-mer analysis
    k = 3
    extractor = KmerExtractor(k=k)
    top_kmers = extractor.top_kmers(seq.sequence, top_n=8)
    print(f"\n[+] Top {k}-mers (Codons):")
    for rank, (kmer, count, pct) in enumerate(top_kmers, 1):
        print(f"    {rank}. {kmer}: {count:,} ({pct:.2f}%)")

    # 2. Information Theory
    entropy_info = shannon_entropy(seq.sequence, k=1)
    complexity = linguistic_complexity(seq.sequence, max_k=5)
    print(f"\n[+] Information Theory:")
    print(f"    Shannon Entropy (k=1): {entropy_info['entropy']:.4f} bits (Normalized: {entropy_info['normalized_entropy']:.2%})")
    print(f"    Linguistic Complexity: {complexity['linguistic_complexity']:.6f}")

    # 3. Markov DNA Syntax
    markov = MarkovModelDNA(order=1).fit(seq.sequence)
    print(f"\n[+] 1st-Order Markov Transition Matrix (A, C, G, T):")
    print(markov.to_dataframe())

    # 4. ORFs and Translation
    orfs = find_orfs(seq.sequence, min_protein_len=25)
    print(f"\n[+] Discovered {len(orfs)} ORFs (>= 25 AA):")
    for o in orfs[:3]:
        print(f"    - Frame {o.frame:+d} ({o.start}-{o.end}): {o.length_aa} AA -> {o.protein_sequence[:25]}...")

if __name__ == "__main__":
    main()
