"""Lightweight JSONL IO that does not depend on PyTorch."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from eventfm.data.schema import EventSequence, sequence_from_json, sequence_to_json


def write_jsonl(path: str, sequences: Iterable[EventSequence]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence_to_json(sequence), sort_keys=True) + "\n")


def load_jsonl_sequences(path: str) -> List[EventSequence]:
    sequences: List[EventSequence] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sequences.append(sequence_from_json(json.loads(line)))
            except Exception as exc:
                raise ValueError(
                    "Failed to parse {} line {}: {}".format(path, line_idx, exc)
                ) from exc
    return sequences


def dataset_summary(path: str) -> Dict[str, Any]:
    sequences = load_jsonl_sequences(path)
    lengths = [len(sequence.events) for sequence in sequences]
    labels = [sequence.label for sequence in sequences if sequence.label is not None]
    return {
        "path": path,
        "num_sequences": len(sequences),
        "min_events": min(lengths) if lengths else 0,
        "max_events": max(lengths) if lengths else 0,
        "mean_events": sum(lengths) / float(len(lengths)) if lengths else 0.0,
        "positive_rate": sum(labels) / float(len(labels)) if labels else None,
    }
