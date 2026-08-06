"""PyTorch datasets for JSONL event histories."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from eventfm.data.io import load_jsonl_sequences
from eventfm.data.schema import EventSequence


class JsonlEventDataset(Dataset):
    """Return full sequences for MLM pretraining or sequence classification."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.sequences = load_jsonl_sequences(path)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> EventSequence:
        return self.sequences[index]


@dataclass
class NextEventExample:
    sequence: EventSequence
    next_event_type: str
    next_delta_seconds: float
    next_timestamp: Optional[float] = None
    next_features: Optional[Dict[str, Any]] = None


class NextEventJsonlDataset(Dataset):
    """Prefix-to-next-event examples for simple TPP benchmarking."""

    def __init__(
        self,
        path: str,
        min_prefix_events: int = 4,
        max_prefixes_per_user: Optional[int] = 16,
        seed: int = 13,
    ) -> None:
        self.path = path
        self.examples: List[NextEventExample] = []
        rng = random.Random(seed)
        for sequence in load_jsonl_sequences(path):
            if len(sequence.events) <= min_prefix_events:
                continue
            candidate_indices = list(range(min_prefix_events, len(sequence.events)))
            if max_prefixes_per_user is not None and len(candidate_indices) > max_prefixes_per_user:
                candidate_indices = sorted(rng.sample(candidate_indices, max_prefixes_per_user))
            for next_idx in candidate_indices:
                prefix_events = sequence.events[:next_idx]
                next_event = sequence.events[next_idx]
                previous_event = sequence.events[next_idx - 1]
                delta = max(0.0, next_event.timestamp - previous_event.timestamp)
                self.examples.append(
                    NextEventExample(
                        sequence=EventSequence(
                            user_id=sequence.user_id,
                            events=list(prefix_events),
                            label=sequence.label,
                            profile=dict(sequence.profile),
                            metadata=dict(sequence.metadata),
                        ),
                        next_event_type=next_event.event_type,
                        next_delta_seconds=delta,
                        next_timestamp=float(next_event.timestamp),
                        next_features=dict(next_event.features),
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> NextEventExample:
        return self.examples[index]
