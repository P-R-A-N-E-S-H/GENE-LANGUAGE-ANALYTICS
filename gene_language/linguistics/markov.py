"""
Genomic Markov chains and DNA syntax transition matrix models.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class MarkovModelDNA:
    """
    1st and 2nd Order Markov Model for DNA Sequence Grammar and Transition Dynamics.
    """

    BASES = ["A", "C", "G", "T"]

    def __init__(self, order: int = 1, pseudocount: float = 1.0):
        if order not in (1, 2):
            raise ValueError("Markov order must be 1 or 2.")
        self.order = order
        self.pseudocount = pseudocount
        self._states = (
            self.BASES if order == 1 else [f"{b1}{b2}" for b1 in self.BASES for b2 in self.BASES]
        )
        self._state_idx = {st: i for i, st in enumerate(self._states)}
        self.transition_counts: np.ndarray = np.zeros((len(self._states), 4), dtype=np.float64)
        self.transition_matrix: np.ndarray = np.zeros((len(self._states), 4), dtype=np.float64)

    def fit(self, sequence: str) -> MarkovModelDNA:
        """Estimates transition probabilities from a given DNA sequence."""
        seq = "".join(sequence.split()).upper()
        # Initialize with pseudocounts (Laplace smoothing)
        counts = np.full((len(self._states), 4), fill_value=self.pseudocount, dtype=np.float64)

        if self.order == 1:
            for i in range(len(seq) - 1):
                from_base = seq[i]
                to_base = seq[i + 1]
                if from_base in self.BASES and to_base in self.BASES:
                    r = self._state_idx[from_base]
                    c = self.BASES.index(to_base)
                    counts[r, c] += 1.0
        else:
            for i in range(len(seq) - 2):
                context = seq[i:i + 2]
                next_base = seq[i + 2]
                if context in self._state_idx and next_base in self.BASES:
                    r = self._state_idx[context]
                    c = self.BASES.index(next_base)
                    counts[r, c] += 1.0

        self.transition_counts = counts
        # Normalize each row to sum to 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0
        )
        return self

    def log_likelihood(self, sequence: str) -> float:
        """Calculates total log-likelihood (base 2) of a sequence under this model."""
        seq = "".join(sequence.split()).upper()
        log_prob = 0.0

        if self.order == 1:
            for i in range(len(seq) - 1):
                from_base, to_base = seq[i], seq[i + 1]
                if from_base in self.BASES and to_base in self.BASES:
                    r = self._state_idx[from_base]
                    c = self.BASES.index(to_base)
                    p = max(1e-12, self.transition_matrix[r, c])
                    log_prob += np.log2(p)
        else:
            for i in range(len(seq) - 2):
                context, next_base = seq[i:i + 2], seq[i + 2]
                if context in self._state_idx and next_base in self.BASES:
                    r = self._state_idx[context]
                    c = self.BASES.index(next_base)
                    p = max(1e-12, self.transition_matrix[r, c])
                    log_prob += np.log2(p)

        return round(float(log_prob), 4)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns transition probability matrix as a readable DataFrame."""
        return pd.DataFrame(
            self.transition_matrix,
            index=self._states,
            columns=self.BASES
        ).round(4)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Exports transition matrix as nested JSON-serializable dictionary."""
        res = {}
        for r_idx, state in enumerate(self._states):
            res[state] = {
                base: round(float(self.transition_matrix[r_idx, c_idx]), 4)
                for c_idx, base in enumerate(self.BASES)
            }
        return res
