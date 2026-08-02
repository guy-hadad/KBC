"""Flat (non-hierarchical) tensor contract shared by most benchmarked models.

PRAGMA encodes each event as a bag of key/value tokens and needs a 3-D batch.
Every other architecture in the zoo — neural TPPs, TabFormer, CoLES, Mambular,
the NVIDIA blueprint — consumes one embedding per event, so they share this
2-D contract instead:

``event_type_ids``     (B, L)     dense mark index
``feature_value_ids``  (B, L, F)  categorical feature-value tokens
``delta_log``          (B, L)     log1p seconds since the previous event
``time_since_start``   (B, L)     log1p seconds since the first event
``calendar_features``  (B, L, 6)  sin/cos hour, weekday, day-of-month
``attention_mask``     (B, L)
"""

from typing import Dict, List, Sequence, Union

import numpy as np
import torch

from eventfm.data.dataset import NextEventExample
from eventfm.data.schema import EventSequence
from eventfm.data.tokenizer import EventTokenizer, _calendar_features

FlatExample = Union[EventSequence, NextEventExample]


class FlatEventCollator:
    """Pad event histories into the 2-D contract described in the module docstring."""

    def __init__(
        self,
        tokenizer: EventTokenizer,
        task: str,
        max_events: int = 64,
        feature_fields: Sequence[str] = (),
    ) -> None:
        self.tokenizer = tokenizer
        self.task = task
        self.max_events = int(max_events)
        self.feature_fields = list(feature_fields)

    @property
    def num_feature_fields(self) -> int:
        return max(1, len(self.feature_fields))

    def _encode_one(self, sequence: EventSequence) -> Dict[str, np.ndarray]:
        events = sequence.events[-self.max_events :]
        length = len(events)
        timestamps = np.asarray([event.timestamp for event in events], dtype=np.float64)
        deltas = np.zeros(length, dtype=np.float32)
        if length > 1:
            deltas[1:] = np.maximum(0.0, np.diff(timestamps))

        type_ids = np.asarray(
            [self.tokenizer.event_type_index(event.event_type) for event in events], dtype=np.int64
        )
        width = self.num_feature_fields
        feature_ids = np.full((length, width), self.tokenizer.vocab.pad_token_id, dtype=np.int64)
        for row, event in enumerate(events):
            for column, field_name in enumerate(self.feature_fields[:width]):
                token = "value:{}:{}".format(field_name, event.features.get(field_name, "na"))
                feature_ids[row, column] = self.tokenizer.vocab.encode(token)

        return {
            "event_type_ids": type_ids,
            "feature_value_ids": feature_ids,
            "delta_log": np.log1p(deltas).astype(np.float32),
            "time_since_start": np.log1p(
                np.maximum(0.0, timestamps - timestamps[0]).astype(np.float32)
            ),
            "calendar_features": np.stack(
                [_calendar_features(value) for value in timestamps], axis=0
            ).astype(np.float32),
        }

    def __call__(self, examples: Sequence[FlatExample]) -> Dict[str, torch.Tensor]:
        sequences: List[EventSequence] = []
        next_examples: List[NextEventExample] = []
        for example in examples:
            if isinstance(example, NextEventExample):
                sequences.append(example.sequence)
                next_examples.append(example)
            else:
                sequences.append(example)

        encoded = [self._encode_one(sequence) for sequence in sequences]
        batch_size = len(encoded)
        length = max(1, max(item["event_type_ids"].shape[0] for item in encoded))
        width = self.num_feature_fields

        event_type_ids = np.zeros((batch_size, length), dtype=np.int64)
        feature_value_ids = np.full(
            (batch_size, length, width), self.tokenizer.vocab.pad_token_id, dtype=np.int64
        )
        delta_log = np.zeros((batch_size, length), dtype=np.float32)
        time_since_start = np.zeros((batch_size, length), dtype=np.float32)
        calendar = np.zeros((batch_size, length, 6), dtype=np.float32)
        attention_mask = np.zeros((batch_size, length), dtype=np.int64)

        for row, item in enumerate(encoded):
            size = item["event_type_ids"].shape[0]
            event_type_ids[row, :size] = item["event_type_ids"]
            feature_value_ids[row, :size, :] = item["feature_value_ids"]
            delta_log[row, :size] = item["delta_log"]
            time_since_start[row, :size] = item["time_since_start"]
            calendar[row, :size, :] = item["calendar_features"]
            attention_mask[row, :size] = 1

        batch = {
            "event_type_ids": torch.as_tensor(event_type_ids),
            "feature_value_ids": torch.as_tensor(feature_value_ids),
            "delta_log": torch.as_tensor(delta_log),
            "time_since_start": torch.as_tensor(time_since_start),
            "calendar_features": torch.as_tensor(calendar),
            "attention_mask": torch.as_tensor(attention_mask),
        }

        if self.task == "classification":
            batch["labels"] = torch.as_tensor(
                [0 if sequence.label is None else int(sequence.label) for sequence in sequences],
                dtype=torch.long,
            )
        elif self.task == "tpp":
            if len(next_examples) != len(examples):
                raise ValueError("TPP batches must contain `NextEventExample` items.")
            batch["next_event_type_labels"] = torch.as_tensor(
                [self.tokenizer.event_type_index(item.next_event_type) for item in next_examples],
                dtype=torch.long,
            )
            batch["next_delta_log"] = torch.as_tensor(
                [float(np.log1p(item.next_delta_seconds)) for item in next_examples],
                dtype=torch.float,
            )
        elif self.task != "pretrain":
            raise ValueError("Unsupported task: {}".format(self.task))
        return batch
