"""Numpy metrics for Hugging Face Trainer callbacks."""

from typing import Dict, Tuple

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def binary_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive_rank_sum = ranks[positives].sum()
    pos = float(positives.sum())
    neg = float(negatives.sum())
    return float((positive_rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg))


def classification_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        scores = logits.reshape(-1)
        predictions = (scores >= 0.0).astype(np.int64)
    else:
        probs = _softmax(logits)
        predictions = probs.argmax(axis=-1)
        scores = probs[:, 1] if probs.shape[-1] > 1 else probs[:, 0]
    return {
        "accuracy": float((predictions == labels).mean()),
        "auc": binary_auc_score(labels, scores) if set(np.unique(labels)).issubset({0, 1}) else float("nan"),
    }


def tpp_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        type_logits = np.asarray(predictions[0])
        delta_log = np.asarray(predictions[1])
    else:
        type_logits = np.asarray(predictions)
        delta_log = np.zeros(type_logits.shape[0], dtype=np.float32)

    if isinstance(labels, tuple):
        type_labels = np.asarray(labels[0])
        delta_labels = np.asarray(labels[1])
    else:
        labels = np.asarray(labels)
        if labels.ndim == 2 and labels.shape[-1] == 2:
            type_labels = labels[:, 0].astype(np.int64)
            delta_labels = labels[:, 1].astype(np.float32)
        else:
            return {"next_type_accuracy": float("nan"), "delta_log_rmse": float("nan")}

    predictions = type_logits.argmax(axis=-1)
    rmse = np.sqrt(np.mean((delta_log.reshape(-1) - delta_labels.reshape(-1)) ** 2))
    return {
        "next_type_accuracy": float((predictions == type_labels).mean()),
        "delta_log_rmse": float(rmse),
    }


def unpack_tpp_labels(next_event_type_labels: np.ndarray, next_delta_log: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(next_event_type_labels), np.asarray(next_delta_log)
