"""
Scikit-learn compatible Genomic Feature Vectorizer (Multi-kmer, TF-IDF, and Biochemical composition).
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .kmer import KmerExtractor
from ..core.sequence import GenomicSequence


class GenomicVectorizer(BaseEstimator, TransformerMixin):
    """
    Transforms genomic DNA strings into multi-scale linguistic & compositional feature matrices.
    """

    def __init__(
        self,
        k_values: Tuple[int, ...] = (3, 4),
        use_tfidf: bool = False,
        include_composition_features: bool = True,
        smoothing_eps: float = 1e-6
    ):
        self.k_values = tuple(k_values)
        self.use_tfidf = use_tfidf
        self.include_composition_features = include_composition_features
        self.smoothing_eps = smoothing_eps
        self._extractors = [KmerExtractor(k=k, smoothing_eps=smoothing_eps) for k in self.k_values]
        self._idf_weights: Dict[str, float] = {}
        self.feature_names_: List[str] = []

    def _build_feature_names(self):
        names = []
        for ext in self._extractors:
            names.extend([f"k{ext.k}_{kmer}" for kmer in ext.vocabulary])
        if self.include_composition_features:
            names.extend([
                "comp_gc_content",
                "comp_at_content",
                "comp_gc_skew",
                "comp_at_skew",
                "comp_cpg_ratio",
                "comp_melting_temp",
            ])
        self.feature_names_ = names

    def fit(self, X: Sequence[str], y: Optional[Sequence] = None) -> GenomicVectorizer:
        """Fits vocabulary and computes IDF weights if use_tfidf is enabled."""
        self._build_feature_names()
        if self.use_tfidf:
            N = len(X)
            # Count document frequency for each k-mer
            doc_freq = {feat: 0 for feat in self.feature_names_ if not feat.startswith("comp_")}
            for seq in X:
                seen = set()
                for ext in self._extractors:
                    counts = ext.count_kmers(seq)
                    for kmer, count in counts.items():
                        if count > 0:
                            seen.add(f"k{ext.k}_{kmer}")
                for feat in seen:
                    doc_freq[feat] = doc_freq.get(feat, 0) + 1

            # Standard smooth IDF formula: ln((1 + N) / (1 + df)) + 1
            self._idf_weights = {
                feat: float(np.log((1.0 + N) / (1.0 + df)) + 1.0)
                for feat, df in doc_freq.items()
            }
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Transforms sequences into a 2D float feature matrix."""
        if not self.feature_names_:
            self._build_feature_names()

        matrix = []
        for raw_seq in X:
            row = []
            seq_obj = GenomicSequence(raw_seq)

            # K-mer features
            for ext in self._extractors:
                freqs = ext.relative_frequencies(seq_obj.sequence)
                for kmer in ext.vocabulary:
                    feat_name = f"k{ext.k}_{kmer}"
                    val = freqs[kmer]
                    if self.use_tfidf and feat_name in self._idf_weights:
                        val *= self._idf_weights[feat_name]
                    row.append(val)

            # Composition features
            if self.include_composition_features:
                row.extend([
                    seq_obj.gc_content / 100.0,
                    seq_obj.at_content / 100.0,
                    seq_obj.gc_skew,
                    seq_obj.at_skew,
                    seq_obj.cpg_observed_expected_ratio(),
                    seq_obj.melting_temperature() / 100.0,
                ])

            matrix.append(row)

        return np.array(matrix, dtype=np.float32)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns feature names for interpretability."""
        if not self.feature_names_:
            self._build_feature_names()
        return list(self.feature_names_)

    def transform_to_dataframe(self, X: Sequence[str], sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Transforms sequences and returns a labeled DataFrame."""
        mat = self.transform(X)
        ids = sample_ids if sample_ids is not None else [f"seq_{i}" for i in range(len(X))]
        return pd.DataFrame(mat, index=ids, columns=self.get_feature_names_out())
