"""Non-neural controls.

Every scaling curve needs a floor that costs nothing to fit. ``count-logistic``
is a bag-of-marks logistic regression with a few inter-arrival summaries, and
``markov`` is a first-order marked Markov chain with per-mark mean log
inter-arrival times. If a foundation model cannot beat these on a benchmark,
that is the most informative result the table can carry.
"""

from typing import Dict, List

import numpy as np

from eventfm.data.io import load_jsonl_sequences
from eventfm.data.schema import EventSequence
from eventfm.methods.base import STATUS_IMPLEMENTED, Method, MethodSpec, register_method
from eventfm.training.metrics import average_precision, binary_auc_score, macro_f1


def _mark_index(tokenizer, event_type: str) -> int:
    return tokenizer.event_type_index(event_type)


def _count_features(
    sequences: List[EventSequence], tokenizer, num_event_types: int
) -> tuple:
    rows: List[np.ndarray] = []
    labels: List[int] = []
    for sequence in sequences:
        counts = np.zeros(num_event_types, dtype=np.float64)
        timestamps: List[float] = []
        for event in sequence.events:
            index = _mark_index(tokenizer, event.event_type)
            if 0 <= index < num_event_types:
                counts[index] += 1.0
            timestamps.append(event.timestamp)
        shares = counts / max(1.0, counts.sum())
        if len(timestamps) > 1:
            deltas = np.log1p(np.maximum(0.0, np.diff(np.asarray(timestamps, dtype=np.float64))))
            summary = np.asarray(
                [deltas.mean(), deltas.std(), float((deltas < np.log1p(7200.0)).mean())],
                dtype=np.float64,
            )
        else:
            summary = np.zeros(3, dtype=np.float64)
        rows.append(np.concatenate([shares, summary, [np.log1p(len(sequence.events))]]))
        labels.append(int(sequence.label or 0))
    return np.stack(rows, axis=0), np.asarray(labels, dtype=np.int64)


class CountLogistic(Method):
    name = "count-logistic"

    def run(self) -> Dict[str, float]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        context = self.context
        num_event_types = max(2, context.num_event_types)
        x_train, y_train = _count_features(
            load_jsonl_sequences(context.train_path), context.tokenizer, num_event_types
        )
        x_test, y_test = _count_features(
            load_jsonl_sequences(context.test_path), context.tokenizer, num_event_types
        )

        scaler = StandardScaler().fit(x_train)
        if len(np.unique(y_train)) < 2:
            probabilities = np.full(y_test.shape, float(y_train.mean()))
        else:
            model = LogisticRegression(max_iter=2000, C=1.0)
            model.fit(scaler.transform(x_train), y_train)
            probabilities = model.predict_proba(scaler.transform(x_test))[:, 1]

        predictions = (probabilities >= 0.5).astype(np.int64)
        return {
            "accuracy": float((predictions == y_test).mean()),
            "auc": binary_auc_score(y_test, probabilities),
            "average_precision": average_precision(y_test, probabilities),
            "macro_f1": macro_f1(y_test, predictions, 2),
            "positive_rate": float(y_test.mean()),
            "num_parameters": float(x_train.shape[1] + 1),
        }


class MarkovChain(Method):
    name = "markov"

    def run(self) -> Dict[str, float]:
        context = self.context
        num_event_types = max(2, context.num_event_types)
        transitions = np.ones((num_event_types, num_event_types), dtype=np.float64)
        delta_sum = np.zeros(num_event_types, dtype=np.float64)
        delta_count = np.zeros(num_event_types, dtype=np.float64)

        for sequence in load_jsonl_sequences(context.train_path):
            for previous, following in zip(sequence.events[:-1], sequence.events[1:]):
                source = _mark_index(context.tokenizer, previous.event_type)
                target = _mark_index(context.tokenizer, following.event_type)
                if not (0 <= source < num_event_types and 0 <= target < num_event_types):
                    continue
                transitions[source, target] += 1.0
                delta_sum[source] += np.log1p(max(0.0, following.timestamp - previous.timestamp))
                delta_count[source] += 1.0

        probabilities = transitions / transitions.sum(axis=1, keepdims=True)
        fallback = float(delta_sum.sum() / max(1.0, delta_count.sum()))
        mean_delta = np.where(delta_count > 0, delta_sum / np.maximum(1.0, delta_count), fallback)

        predicted_marks: List[int] = []
        true_marks: List[int] = []
        squared_errors: List[float] = []
        absolute_errors: List[float] = []
        for sequence in load_jsonl_sequences(context.test_path):
            for previous, following in zip(sequence.events[:-1], sequence.events[1:]):
                source = _mark_index(context.tokenizer, previous.event_type)
                target = _mark_index(context.tokenizer, following.event_type)
                if not (0 <= source < num_event_types and 0 <= target < num_event_types):
                    continue
                predicted_delta = float(mean_delta[source])
                observed = float(np.log1p(max(0.0, following.timestamp - previous.timestamp)))
                predicted_marks.append(int(probabilities[source].argmax()))
                true_marks.append(target)
                squared_errors.append((predicted_delta - observed) ** 2)
                absolute_errors.append(abs(predicted_delta - observed))

        predictions = np.asarray(predicted_marks, dtype=np.int64)
        labels = np.asarray(true_marks, dtype=np.int64)
        return {
            "next_type_accuracy": (
                float((predictions == labels).mean()) if labels.size else float("nan")
            ),
            "next_type_macro_f1": (
                macro_f1(labels, predictions, num_event_types) if labels.size else float("nan")
            ),
            "delta_log_rmse": (
                float(np.sqrt(np.mean(squared_errors))) if squared_errors else float("nan")
            ),
            "delta_log_mae": float(np.mean(absolute_errors)) if absolute_errors else float("nan"),
            "num_parameters": float(num_event_types * num_event_types + num_event_types),
        }


register_method(
    MethodSpec(
        name="count-logistic",
        status=STATUS_IMPLEMENTED,
        divergence="no paper to be faithful to; this is an engineered control",
        display_name="Count + logistic regression",
        reference="non-neural control",
        family="classic",
        factory=CountLogistic,
        supports=("classification",),
        notes="Mark-share features plus inter-arrival summaries.",
    )
)

register_method(
    MethodSpec(
        name="markov",
        status=STATUS_IMPLEMENTED,
        divergence="no paper to be faithful to; this is a marked temporal control",
        display_name="Marked Markov chain",
        reference="non-neural control",
        family="classic",
        factory=MarkovChain,
        supports=("tpp",),
        notes="First-order transition matrix with per-mark mean log inter-arrival time.",
    )
)
