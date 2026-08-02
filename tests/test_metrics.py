"""Tests for the shared metrics, which every method's numbers flow through."""

import numpy as np
import pytest

from eventfm.training.metrics import (
    average_precision,
    binary_auc_score,
    classification_metrics,
    macro_f1,
    tpp_metrics,
)


def test_auc_is_one_for_a_perfect_ranking():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])

    assert binary_auc_score(labels, scores) == pytest.approx(1.0)


def test_auc_is_half_when_every_score_ties():
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.5, 0.5, 0.5, 0.5])

    assert binary_auc_score(labels, scores) == pytest.approx(0.5)


def test_auc_is_nan_for_a_single_class():
    assert np.isnan(binary_auc_score(np.asarray([1, 1, 1]), np.asarray([0.1, 0.5, 0.9])))


def test_average_precision_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)

    assert average_precision(labels, scores) == pytest.approx(
        sklearn_metrics.average_precision_score(labels, scores), abs=1e-6
    )


def test_macro_f1_is_unweighted_across_classes():
    labels = np.asarray([0] * 9 + [1])
    # Predicting the majority everywhere scores 0.9 accuracy but only ~0.47
    # macro F1, which is why the skewed-mark benchmarks report both.
    predictions = np.zeros(10, dtype=np.int64)

    # class 0: F1 = 2*0.9*1.0/1.9 = 0.947; class 1: F1 = 0; macro = 0.474.
    assert macro_f1(labels, predictions, num_classes=2) == pytest.approx(0.473684, abs=1e-5)


def test_macro_f1_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 5, size=300)
    predictions = rng.integers(0, 5, size=300)

    assert macro_f1(labels, predictions, num_classes=5) == pytest.approx(
        sklearn_metrics.f1_score(labels, predictions, average="macro"), abs=1e-6
    )


def test_classification_metrics_reads_two_column_logits():
    logits = np.asarray([[2.0, -2.0], [-2.0, 2.0], [-1.0, 1.0], [1.0, -1.0]])
    labels = np.asarray([0, 1, 1, 0])

    metrics = classification_metrics((logits, labels))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["positive_rate"] == pytest.approx(0.5)


def test_tpp_metrics_scores_marks_and_times_separately():
    type_logits = np.asarray([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    delta_log = np.asarray([1.0, 2.0, 3.0])
    time_nll = np.asarray([0.5, 0.5, 0.5])
    labels = (np.asarray([0, 1, 1]), np.asarray([1.0, 2.0, 4.0]))

    metrics = tpp_metrics(((type_logits, delta_log, time_nll), labels))

    assert metrics["next_type_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["delta_log_rmse"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics["delta_log_mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["time_nll"] == pytest.approx(0.5)


def test_tpp_metrics_ignores_a_non_vector_third_output():
    """A model that emits pooled embeddings must not be read as a likelihood."""

    type_logits = np.asarray([[5.0, 0.0], [0.0, 5.0]])
    delta_log = np.asarray([1.0, 2.0])
    embeddings = np.zeros((2, 8))
    labels = (np.asarray([0, 1]), np.asarray([1.0, 2.0]))

    metrics = tpp_metrics(((type_logits, delta_log, embeddings), labels))

    assert "time_nll" not in metrics
    assert metrics["next_type_accuracy"] == pytest.approx(1.0)
