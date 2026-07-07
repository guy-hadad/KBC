"""Classic baselines for the synthetic benchmark.

These are intentionally dependency-light so a quick comparison is possible even
before installing scikit-learn.
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np

from eventfm.data.io import load_jsonl_sequences
from eventfm.data.schema import EventSequence


def _type_index(event_type: str) -> int:
    if not event_type.startswith("type_"):
        return -1
    return int(event_type.split("_", 1)[1])


def count_features(sequences: Iterable[EventSequence], num_event_types: int) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    labels: List[int] = []
    for sequence in sequences:
        counts = np.zeros(num_event_types, dtype=np.float32)
        timestamps = []
        for event in sequence.events:
            idx = _type_index(event.event_type)
            if 0 <= idx < num_event_types:
                counts[idx] += 1.0
            timestamps.append(event.timestamp)
        total = max(1.0, counts.sum())
        features = counts / total
        if len(timestamps) > 1:
            deltas = np.diff(np.asarray(timestamps, dtype=np.float32))
            extra = np.asarray(
                [np.mean(np.log1p(deltas)), np.std(np.log1p(deltas)), float((deltas < 7200).mean())],
                dtype=np.float32,
            )
        else:
            extra = np.zeros(3, dtype=np.float32)
        rows.append(np.concatenate([features, extra], axis=0))
        labels.append(int(sequence.label or 0))
    return np.stack(rows, axis=0), np.asarray(labels, dtype=np.int64)


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    lr: float = 0.5,
    epochs: int = 400,
    l2: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = 0.0
    y = y_train.astype(np.float64)
    for _ in range(epochs):
        logits = x_train @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = probs - y
        grad_w = x_train.T @ error / float(len(x_train)) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights.astype(np.float32), float(bias)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _binary_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = float(positives.sum())
    neg = float(negatives.sum())
    positive_rank_sum = ranks[positives].sum()
    return float((positive_rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg))


def _best_accuracy_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    if thresholds.size == 0:
        return 0.5
    return float(
        max(
            thresholds,
            key=lambda threshold: ((scores >= threshold).astype(np.int64) == labels).mean(),
        )
    )


def evaluate_logistic(
    weights: np.ndarray,
    bias: float,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    logits = x @ weights + bias
    probs = _sigmoid(logits)
    preds = (probs >= threshold).astype(np.int64)
    return {
        "accuracy": float((preds == y).mean()),
        "positive_rate": float(preds.mean()),
        "auc": _binary_auc_score(y, probs),
        "threshold": float(threshold),
    }


def run_classification_baseline(
    train_path: str,
    validation_path: str,
    test_path: str,
    num_event_types: int,
) -> Dict[str, Dict[str, float]]:
    train = load_jsonl_sequences(train_path)
    validation = load_jsonl_sequences(validation_path)
    test = load_jsonl_sequences(test_path)
    x_train, y_train = count_features(train, num_event_types)
    x_val, y_val = count_features(validation, num_event_types)
    x_test, y_test = count_features(test, num_event_types)
    weights, bias = train_logistic_regression(x_train, y_train)
    validation_probs = _sigmoid(x_val @ weights + bias)
    threshold = _best_accuracy_threshold(validation_probs, y_val)
    return {
        "validation": evaluate_logistic(weights, bias, x_val, y_val, threshold=threshold),
        "test": evaluate_logistic(weights, bias, x_test, y_test, threshold=threshold),
    }


class MarkovTppBaseline:
    def __init__(self, num_event_types: int, smoothing: float = 1.0) -> None:
        self.num_event_types = int(num_event_types)
        self.smoothing = float(smoothing)
        self.transitions = np.full((num_event_types, num_event_types), smoothing, dtype=np.float64)
        self.delta_sum = np.zeros(num_event_types, dtype=np.float64)
        self.delta_count = np.zeros(num_event_types, dtype=np.float64)

    def fit(self, sequences: Iterable[EventSequence]) -> "MarkovTppBaseline":
        for sequence in sequences:
            events = sequence.events
            for prev, nxt in zip(events[:-1], events[1:]):
                prev_idx = _type_index(prev.event_type)
                next_idx = _type_index(nxt.event_type)
                if 0 <= prev_idx < self.num_event_types and 0 <= next_idx < self.num_event_types:
                    self.transitions[prev_idx, next_idx] += 1.0
                    self.delta_sum[prev_idx] += np.log1p(max(0.0, nxt.timestamp - prev.timestamp))
                    self.delta_count[prev_idx] += 1.0
        return self

    def evaluate(self, sequences: Iterable[EventSequence]) -> Dict[str, float]:
        correct = 0
        total = 0
        sq_error = []
        probs = self.transitions / self.transitions.sum(axis=1, keepdims=True)
        default_delta = np.log1p(6.0 * 3600.0)
        mean_delta = np.divide(
            self.delta_sum,
            np.maximum(1.0, self.delta_count),
            out=np.full_like(self.delta_sum, default_delta),
            where=self.delta_count > 0,
        )
        for sequence in sequences:
            for prev, nxt in zip(sequence.events[:-1], sequence.events[1:]):
                prev_idx = _type_index(prev.event_type)
                next_idx = _type_index(nxt.event_type)
                if not (0 <= prev_idx < self.num_event_types and 0 <= next_idx < self.num_event_types):
                    continue
                pred_type = int(probs[prev_idx].argmax())
                pred_delta = float(mean_delta[prev_idx])
                true_delta = float(np.log1p(max(0.0, nxt.timestamp - prev.timestamp)))
                correct += int(pred_type == next_idx)
                total += 1
                sq_error.append((pred_delta - true_delta) ** 2)
        return {
            "next_type_accuracy": float(correct / max(1, total)),
            "delta_log_rmse": float(np.sqrt(np.mean(sq_error))) if sq_error else float("nan"),
        }


def run_tpp_baseline(
    train_path: str,
    validation_path: str,
    test_path: str,
    num_event_types: int,
) -> Dict[str, Dict[str, float]]:
    baseline = MarkovTppBaseline(num_event_types=num_event_types).fit(load_jsonl_sequences(train_path))
    return {
        "validation": baseline.evaluate(load_jsonl_sequences(validation_path)),
        "test": baseline.evaluate(load_jsonl_sequences(test_path)),
    }
