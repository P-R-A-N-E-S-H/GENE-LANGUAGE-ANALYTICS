"""
Example 08: Genomic Sequence Embeddings & Alignment-Free Distance Metrics.
==========================================================================
Demonstrates generating dense k-mer vector embeddings, TF-IDF representations,
and computing Cosine, Jaccard, Euclidean, and Jensen-Shannon divergence across sequences.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language.linguistics.embeddings import (
    GenomicEmbedding,
    TfidfGenomicEmbedding,
    cosine_similarity_dna,
    jaccard_similarity_dna,
    euclidean_distance_dna,
    jensen_shannon_divergence,
)


def main():
    print("=" * 70)
    print("[*] GENE-LANGUAGE-ANALYTICS: Genomic Sequence Embeddings & Distances")
    print("=" * 70)

    seq_exon = "ATGCGACCGCTTCCCAAGCCAGAGTCGAGCGCCTAAGGCTACGGAGGCTAAGCGCTCAACCCGTTGGCGCCCAGCC"
    seq_intron = "TCCTGACTACTACATTTTGATACTTAGAGGGTTCTGAAGATTTAACCGAAGAACAAATATCAAAGTATAGCTAAAT"
    seq_exon_variant = "ATGCGACCGCTTCCCAAGCCAGAGTCGAGCGCCTAAGGCTACGGAGGCTAAGCGCTCAACCCGTTGGCGCCCAGTT"

    print("\n[+] Comparing Pairwise Alignment-Free Distance Metrics (k=3):")

    def compare_pair(label, s1, s2):
        cos = cosine_similarity_dna(s1, s2, k=3)
        jac = jaccard_similarity_dna(s1, s2, k=3)
        euc = euclidean_distance_dna(s1, s2, k=3, normalized=True)
        jsd = jensen_shannon_divergence(s1, s2, k=3)
        print(f"\n--- {label} ---")
        print(f"  * Cosine Similarity:        {cos:.5f}")
        print(f"  * Jaccard Set Similarity:   {jac:.5f}")
        print(f"  * Euclidean Distance:       {euc:.5f}")
        print(f"  * Jensen-Shannon Div (JSD): {jsd:.5f}")

    compare_pair("Exon vs. Near-Identical Exon Variant", seq_exon, seq_exon_variant)
    compare_pair("Coding Exon vs. Non-Coding Intron", seq_exon, seq_intron)

    # 2. Vector Embeddings
    print("\n[+] Dense Vector Embeddings (k=3):")
    embedder = GenomicEmbedding(k=3, normalize="l2")
    vec_exon = embedder.embed_sequence(seq_exon)
    print(f"  * Embedding Dimension: {embedder.embedding_dim}")
    print(f"  * Vector Shape:        {vec_exon.shape}")
    print(f"  * Vector L2 Norm:      {sum(v**2 for v in vec_exon):.4f}")

    # 3. TF-IDF Representation
    print("\n[+] TF-IDF Matrix Across 3 Sequences:")
    tfidf = TfidfGenomicEmbedding(k=2)
    corpus = [seq_exon, seq_intron, seq_exon_variant]
    matrix = tfidf.fit_transform(corpus)
    print(f"  * TF-IDF Matrix Shape: {matrix.shape}")


if __name__ == "__main__":
    main()
