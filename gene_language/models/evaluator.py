"""
Comprehensive model evaluation metrics, cross-validation benchmarks, and reports.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
from .classifier import ExonIntronClassifier


@dataclass
class EvaluationReport:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: List[List[int]]
    classes: List[str]
    cv_mean: float
    cv_std: float
    detailed_report: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "confusion_matrix": self.confusion_matrix,
            "classes": self.classes,
            "cv_mean_percent": round(self.cv_mean * 100, 2),
            "cv_std_percent": round(self.cv_std * 100, 2),
        }


class ModelEvaluator:
    """
    Evaluator performing cross-validation, confusion matrix calculation, and classification reporting.
    """

    @staticmethod
    def evaluate(
        classifier: ExonIntronClassifier,
        sequences: List[str],
        labels: List[str],
        n_splits: int = 5
    ) -> EvaluationReport:
        if not classifier.is_fitted:
            classifier.fit(sequences, labels, val_split=0.0)

        preds = classifier.predict(sequences)
        classes = list(classifier.label_encoder.classes_)

        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(labels, preds, labels=classes)
        rep = classification_report(labels, preds, target_names=classes)

        # Cross-validation
        X_mat = classifier.vectorizer.transform(sequences)
        X_scaled = classifier.scaler.transform(X_mat)
        y_enc = classifier.label_encoder.transform(labels)

        cv = StratifiedKFold(n_splits=min(n_splits, len(labels)), shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in cv.split(X_scaled, y_enc):
            cls_copy = ExonIntronClassifier(
                model_type=classifier.model_type,
                k_values=classifier.k_values,
                use_tfidf=classifier.use_tfidf,
                include_composition=classifier.include_composition
            )
            cls_copy.fit([sequences[i] for i in train_idx], [labels[i] for i in train_idx], val_split=0.0)
            fold_pred = cls_copy.predict([sequences[i] for i in test_idx])
            fold_acc = accuracy_score([labels[i] for i in test_idx], fold_pred)
            scores.append(fold_acc)

        scores_arr = np.array(scores)

        return EvaluationReport(
            accuracy=round(float(acc), 4),
            precision_macro=round(float(p), 4),
            recall_macro=round(float(r), 4),
            f1_macro=round(float(f1), 4),
            confusion_matrix=cm.tolist(),
            classes=classes,
            cv_mean=round(float(scores_arr.mean()), 4),
            cv_std=round(float(scores_arr.std()), 4),
            detailed_report=rep
        )
