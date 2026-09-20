"""
Example 2: Train and Benchmark XGBoost Exon vs. Intron Classifier
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gene_language import read_fasta, ExonIntronClassifier
from gene_language.models.evaluator import ModelEvaluator

def main():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sample_exons_introns.fasta"))
    print(f"Loading labeled sequences from: {dataset_path}")
    records = read_fasta(dataset_path)

    seqs, labels = [], []
    for r in records:
        hdr = (r.identifier + " " + r.description).lower()
        if "exon" in hdr or "coding" in hdr:
            labels.append(ExonIntronClassifier.CLASS_EXON)
            seqs.append(r.sequence.sequence)
        elif "intron" in hdr or "non-coding" in hdr or "noncoding" in hdr:
            labels.append(ExonIntronClassifier.CLASS_INTRON)
            seqs.append(r.sequence.sequence)

    print(f"Loaded {len(seqs)} sequences (Exons: {labels.count(ExonIntronClassifier.CLASS_EXON)}, Introns: {labels.count(ExonIntronClassifier.CLASS_INTRON)})")

    # Initialize Classifier
    classifier = ExonIntronClassifier(model_type="xgboost" if hasattr(ExonIntronClassifier, "HAS_XGBOOST") else "random_forest")

    # Cross-validation & benchmark
    print("\nRunning Stratified 5-Fold Cross Validation...")
    report = ModelEvaluator.evaluate(classifier, seqs, labels, n_splits=5)

    print(f"\n[+] Results:")
    print(f"    Accuracy:        {report.accuracy * 100:.2f}%")
    print(f"    Macro F1-Score:  {report.f1_macro:.4f}")
    print(f"    5-Fold CV:       {report.cv_mean * 100:.2f}% +- {report.cv_std * 100:.2f}%")
    print("\n[+] Classification Report:\n", report.detailed_report)

    # Top Features
    top_feats = classifier.get_top_features(top_n=10)
    if top_feats:
        print("[+] Top 10 Informative k-mer Features:")
        for f in top_feats:
            print(f"    - {f['feature']:<12}: {f['importance']:.5f}")

if __name__ == "__main__":
    main()
