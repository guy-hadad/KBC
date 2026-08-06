"""Metrics shared by every method, so the benchmark tables compare like with like.

Classification reports accuracy, ROC-AUC and average precision. Training may be
rebalanced, while current validation/test splits retain natural prevalence; AUC
and AP are the headline metrics and accuracy is context.

The TPP task is scored separately for *what* and *when*, which the neural-TPP
review specifically recommends over a single likelihood that hides which half
of the problem the model is actually solving:

``next_type_accuracy``  mark prediction
``next_type_macro_f1``  mark prediction, unweighted across marks
``delta_log_rmse``      time prediction, RMSE on ``log1p(seconds)``
``delta_log_mae``       time prediction, MAE on ``log1p(seconds)``
``time_nll``            mean negative log-likelihood, where the model defines one
"""

from typing import Dict, Optional, Tuple

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def binary_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC that needs no scikit-learn and handles ties."""

    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, order.shape[0] + 1)
    # Average ranks within tied score groups.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, sorted_scores.shape[0] + 1):
        if index == sorted_scores.shape[0] or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    pos = float(positives.sum())
    neg = float(negatives.sum())
    return float((ranks[positives].sum() - pos * (pos + 1.0) / 2.0) / (pos * neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0 or labels.sum() == labels.shape[0]:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_hits = np.cumsum(sorted_labels)
    precision = cumulative_hits / np.arange(1, sorted_labels.shape[0] + 1)
    return float((precision * sorted_labels).sum() / max(1.0, float(labels.sum())))


def macro_f1(
    labels: np.ndarray, predictions: np.ndarray, num_classes: Optional[int] = None
) -> float:
    labels = np.asarray(labels).astype(np.int64)
    predictions = np.asarray(predictions).astype(np.int64)
    classes = (
        np.arange(int(num_classes))
        if num_classes is not None
        else np.unique(np.concatenate([labels, predictions]))
    )
    scores = []
    for class_id in classes:
        true_positive = float(((predictions == class_id) & (labels == class_id)).sum())
        predicted = float((predictions == class_id).sum())
        actual = float((labels == class_id).sum())
        if actual == 0 and predicted == 0:
            continue
        precision = true_positive / predicted if predicted > 0 else 0.0
        recall = true_positive / actual if actual > 0 else 0.0
        harmonic = (
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
        scores.append(harmonic)
    return float(np.mean(scores)) if scores else float("nan")


def classification_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    logits = np.asarray(logits[0] if isinstance(logits, tuple) else logits)
    labels = np.asarray(labels)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        scores = logits.reshape(-1)
        predictions = (scores >= 0.0).astype(np.int64)
    else:
        probabilities = _softmax(logits)
        predictions = probabilities.argmax(axis=-1)
        scores = probabilities[:, 1] if probabilities.shape[-1] > 1 else probabilities[:, 0]

    binary = set(np.unique(labels).tolist()).issubset({0, 1})
    return {
        "accuracy": float((predictions == labels).mean()),
        "auc": binary_auc_score(labels, scores) if binary else float("nan"),
        "average_precision": average_precision(labels, scores) if binary else float("nan"),
        "macro_f1": macro_f1(labels, predictions),
        "positive_rate": float(labels.mean()) if binary else float("nan"),
    }


def _unpack_tpp_predictions(predictions) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if isinstance(predictions, (tuple, list)):
        type_logits = np.asarray(predictions[0])
        delta_log = np.asarray(predictions[1]) if len(predictions) > 1 else None
        time_nll = np.asarray(predictions[2]) if len(predictions) > 2 else None
        # Guard against a model that emits an extra 2-D tensor (e.g. pooled
        # embeddings) where the likelihood is expected.
        if time_nll is not None and time_nll.ndim != 1:
            time_nll = None
    else:
        type_logits = np.asarray(predictions)
        delta_log = None
        time_nll = None
    if delta_log is None:
        delta_log = np.zeros(type_logits.shape[0], dtype=np.float32)
    return type_logits, delta_log, time_nll


def tpp_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    type_logits, delta_log, time_nll = _unpack_tpp_predictions(predictions)

    if isinstance(labels, (tuple, list)):
        type_labels = np.asarray(labels[0]).reshape(-1).astype(np.int64)
        delta_labels = np.asarray(labels[1]).reshape(-1).astype(np.float64)
    else:
        labels = np.asarray(labels)
        if labels.ndim == 2 and labels.shape[-1] == 2:
            type_labels = labels[:, 0].astype(np.int64)
            delta_labels = labels[:, 1].astype(np.float64)
        else:
            return {"next_type_accuracy": float("nan"), "delta_log_rmse": float("nan")}

    predicted_types = type_logits.argmax(axis=-1).reshape(-1)
    residual = delta_log.reshape(-1).astype(np.float64) - delta_labels
    metrics = {
        "next_type_accuracy": float((predicted_types == type_labels).mean()),
        "next_type_macro_f1": macro_f1(type_labels, predicted_types, type_logits.shape[-1]),
        "delta_log_rmse": float(np.sqrt(np.mean(residual**2))),
        "delta_log_mae": float(np.mean(np.abs(residual))),
    }
    if time_nll is not None and time_nll.size:
        metrics["time_nll"] = float(np.mean(time_nll))
    return metrics


def unpack_tpp_labels(next_event_type_labels: np.ndarray, next_delta_log: np.ndarray):
    return np.asarray(next_event_type_labels), np.asarray(next_delta_log)
