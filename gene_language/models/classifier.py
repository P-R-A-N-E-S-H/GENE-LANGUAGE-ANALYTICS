"""
Exon vs. Intron classification model pipeline using genomic NLP representations.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from ..linguistics.vectorizer import GenomicVectorizer
from ..core.sequence import GenomicSequence


class ExonIntronClassifier:
    """
    End-to-end Machine Learning classifier for predicting whether genomic DNA sequences
    correspond to Coding (Exon) or Non-Coding (Intron) regions.
    """

    CLASS_EXON = "Coding (Exon)"
    CLASS_INTRON = "Non-Coding (Intron)"

    def __init__(
        self,
        model_type: str = "xgboost",
        k_values: Tuple[int, ...] = (3, 4),
        use_tfidf: bool = False,
        include_composition: bool = True,
        random_state: int = 42
    ):
        self.model_type = model_type.lower()
        self.k_values = tuple(k_values)
        self.use_tfidf = use_tfidf
        self.include_composition = include_composition
        self.random_state = random_state

        self.vectorizer = GenomicVectorizer(
            k_values=self.k_values,
            use_tfidf=self.use_tfidf,
            include_composition_features=self.include_composition
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.estimator = None

    def _init_estimator(self, scale_pos_weight: float = 1.0):
        if self.model_type == "xgboost" and HAS_XGBOOST:
            self.estimator = XGBClassifier(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "random_forest" or (self.model_type == "xgboost" and not HAS_XGBOOST):
            self.estimator = RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.estimator = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=self.random_state
            )
        elif self.model_type == "logistic_regression":
            self.estimator = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        sequences: List[str],
        labels: List[str],
        val_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Trains the classifier on a set of labeled genomic sequences.
        """
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have identical length.")

        # Fit vectorizer & transform
        self.vectorizer.fit(sequences)
        X_mat = self.vectorizer.transform(sequences)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_mat)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)

        # Compute pos_weight
        num_pos = sum(y_encoded)
        num_neg = len(y_encoded) - num_pos
        scale_pos_weight = (num_neg / max(1, num_pos)) if num_pos > 0 else 1.0

        self._init_estimator(scale_pos_weight=scale_pos_weight)

        metrics = {}
        if val_split > 0 and len(sequences) >= 10:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_scaled, y_encoded, test_size=val_split, stratify=y_encoded, random_state=self.random_state
            )
            self.estimator.fit(X_tr, y_tr)
            val_acc = self.estimator.score(X_val, y_val)
            metrics["validation_accuracy"] = round(float(val_acc), 4)
            # Refit on full dataset
            self.estimator.fit(X_scaled, y_encoded)
        else:
            self.estimator.fit(X_scaled, y_encoded)
            metrics["training_accuracy"] = round(float(self.estimator.score(X_scaled, y_encoded)), 4)

        self.is_fitted = True
        return metrics

    def predict(self, sequences: Union[str, List[str]]) -> List[str]:
        """Predicts class labels for one or more sequences."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict.")
        seq_list = [sequences] if isinstance(sequences, str) else sequences
        X_mat = self.vectorizer.transform(seq_list)
        X_scaled = self.scaler.transform(X_mat)
        preds = self.estimator.predict(X_scaled)
        return list(self.label_encoder.inverse_transform(preds))

    def predict_proba(self, sequences: Union[str, List[str]]) -> List[Dict[str, float]]:
        """Predicts calibrated class probabilities for one or more sequences."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict_proba.")
        seq_list = [sequences] if isinstance(sequences, str) else sequences
        X_mat = self.vectorizer.transform(seq_list)
        X_scaled = self.scaler.transform(X_mat)
        probas = self.estimator.predict_proba(X_scaled)

        classes = list(self.label_encoder.classes_)
        results = []
        for row in probas:
            results.append({cls_name: round(float(p), 4) for cls_name, p in zip(classes, row)})
        return results

    def predict_single(self, sequence: str) -> Dict[str, Any]:
        """Convenience method for comprehensive single-sequence classification and diagnostics."""
        pred = self.predict([sequence])[0]
        probas = self.predict_proba([sequence])[0]
        seq_obj = GenomicSequence(sequence)

        # Baseline GC heuristic
        gc = seq_obj.gc_content

        return {
            "prediction": pred,
            "confidence": probas[pred],
            "probabilities": probas,
            "gc_content": gc,
            "gc_heuristic": "Coding-leaning (GC >= 50%)" if gc >= 50 else "NonCoding-leaning (GC < 50%)",
            "length": len(seq_obj),
            "cpg_ratio": seq_obj.cpg_observed_expected_ratio()
        }

    def get_feature_importances(self, top_n: int = 20) -> List[Dict[str, Union[str, float]]]:
        """Returns top feature importances from the fitted model."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted to retrieve feature importances.")

        feat_names = self.vectorizer.get_feature_names_out()
        if hasattr(self.estimator, "feature_importances_"):
            importances = self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):
            importances = np.abs(self.estimator.coef_[0])
        else:
            return []

        sorted_idx = np.argsort(importances)[::-1][:top_n]
        return [
            {"feature": feat_names[i], "importance": round(float(importances[i]), 6)}
            for i in sorted_idx
        ]

    def get_top_features(self, top_n: int = 20) -> List[Dict[str, Union[str, float]]]:
        """Alias for get_feature_importances."""
        return self.get_feature_importances(top_n=top_n)

    def save(self, filepath: str):
        """Serializes the fitted pipeline to a file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> ExonIntronClassifier:
        """Loads a serialized classifier pipeline."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
