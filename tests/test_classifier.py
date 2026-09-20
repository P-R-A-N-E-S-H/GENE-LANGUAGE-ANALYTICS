import pytest
from gene_language.models.classifier import ExonIntronClassifier
from gene_language.models.evaluator import ModelEvaluator


def test_classifier_training_and_prediction():
    # Synthetic exons (GC-rich) vs introns (AT-rich)
    exons = ["GCGCGCGCCCGGGCGCGC" * 5 for _ in range(6)]
    introns = ["ATATATATTTTAAATTAT" * 5 for _ in range(6)]

    seqs = exons + introns
    labels = [ExonIntronClassifier.CLASS_EXON] * 6 + [ExonIntronClassifier.CLASS_INTRON] * 6

    clf = ExonIntronClassifier(model_type="random_forest")
    metrics = clf.fit(seqs, labels, val_split=0.0)

    assert clf.is_fitted is True

    # Test predictions
    preds = clf.predict(["GCGCGCGCCCGGGCGCGC" * 5])
    assert preds[0] == ExonIntronClassifier.CLASS_EXON

    single = clf.predict_single("ATATATATTTTAAATTAT" * 5)
    assert single["prediction"] == ExonIntronClassifier.CLASS_INTRON
    assert "confidence" in single


def test_evaluator():
    exons = ["GCGCGCGCCCGGGCGCGC" * 5 for _ in range(6)]
    introns = ["ATATATATTTTAAATTAT" * 5 for _ in range(6)]
    seqs = exons + introns
    labels = [ExonIntronClassifier.CLASS_EXON] * 6 + [ExonIntronClassifier.CLASS_INTRON] * 6

    clf = ExonIntronClassifier(model_type="random_forest")
    report = ModelEvaluator.evaluate(clf, seqs, labels, n_splits=2)
    assert report.accuracy >= 0.8
