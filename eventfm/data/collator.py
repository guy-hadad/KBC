"""Task-aware data collators for PRAGMA-style models."""

from typing import List, Sequence, Union

import numpy as np
import torch

from eventfm.data.dataset import NextEventExample
from eventfm.data.schema import EventSequence
from eventfm.data.tokenizer import EventTokenizer

BatchExample = Union[EventSequence, NextEventExample]


class PragmaDataCollator:
    """Pad sequences and apply paper-inspired dynamic masking.

    Mask sources:
    - random individual feature values,
    - whole events,
    - all occurrences of randomly selected semantic keys.
    """

    def __init__(
        self,
        tokenizer: EventTokenizer,
        task: str,
        max_events: int = 128,
        mlm_probability: float = 0.15,
        event_mask_probability: float = 0.10,
        key_mask_probability: float = 0.10,
        unk_probability: float = 0.10,
    ) -> None:
        self.tokenizer = tokenizer
        self.task = task
        self.max_events = int(max_events)
        self.mlm_probability = float(mlm_probability)
        self.event_mask_probability = float(event_mask_probability)
        self.key_mask_probability = float(key_mask_probability)
        self.unk_probability = float(unk_probability)

    def __call__(self, examples: Sequence[BatchExample]) -> dict:
        sequences: List[EventSequence] = []
        next_examples: List[NextEventExample] = []
        for example in examples:
            if isinstance(example, NextEventExample):
                sequences.append(example.sequence)
                next_examples.append(example)
            else:
                sequences.append(example)

        encoded = [
            self.tokenizer.encode_sequence(sequence, max_events=self.max_events)
            for sequence in sequences
        ]
        batch_size = len(encoded)
        max_events = min(self.max_events, max(item.key_ids.shape[0] for item in encoded))
        width = self.tokenizer.max_features_per_event

        key_ids = np.full(
            (batch_size, max_events, width), self.tokenizer.vocab.pad_token_id, dtype=np.int64
        )
        value_ids = np.full_like(key_ids, self.tokenizer.vocab.pad_token_id)
        feature_position_ids = np.zeros_like(key_ids)
        feature_attention_mask = np.zeros_like(key_ids)
        event_attention_mask = np.zeros((batch_size, max_events), dtype=np.int64)
        timestamps = np.zeros((batch_size, max_events), dtype=np.float32)
        delta_seconds = np.zeros((batch_size, max_events), dtype=np.float32)
        event_age_seconds = np.zeros((batch_size, max_events), dtype=np.float32)
        calendar_features = np.zeros((batch_size, max_events, 6), dtype=np.float32)

        for row, item in enumerate(encoded):
            length = min(max_events, item.key_ids.shape[0])
            key_ids[row, :length, :] = item.key_ids[-length:]
            value_ids[row, :length, :] = item.value_ids[-length:]
            feature_position_ids[row, :length, :] = item.feature_position_ids[-length:]
            feature_attention_mask[row, :length, :] = item.feature_attention_mask[-length:]
            event_attention_mask[row, :length] = item.event_attention_mask[-length:]
            timestamps[row, :length] = item.timestamps[-length:]
            delta_seconds[row, :length] = item.delta_seconds[-length:]
            event_age_seconds[row, :length] = item.event_age_seconds[-length:]
            calendar_features[row, :length, :] = item.calendar_features[-length:]

        batch = {
            "key_ids": torch.as_tensor(key_ids, dtype=torch.long),
            "value_ids": torch.as_tensor(value_ids, dtype=torch.long),
            "feature_position_ids": torch.as_tensor(feature_position_ids, dtype=torch.long),
            "feature_attention_mask": torch.as_tensor(feature_attention_mask, dtype=torch.long),
            "event_attention_mask": torch.as_tensor(event_attention_mask, dtype=torch.long),
            "timestamps": torch.as_tensor(timestamps, dtype=torch.float),
            "delta_seconds": torch.as_tensor(delta_seconds, dtype=torch.float),
            "event_age_seconds": torch.as_tensor(event_age_seconds, dtype=torch.float),
            "calendar_features": torch.as_tensor(calendar_features, dtype=torch.float),
        }

        if self.task == "pretrain":
            masked_values, labels = self._mask_values(
                batch["value_ids"], batch["key_ids"], batch["feature_attention_mask"]
            )
            batch["value_ids"] = masked_values
            batch["labels"] = labels
        elif self.task == "classification":
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
                [np.log1p(item.next_delta_seconds) for item in next_examples],
                dtype=torch.float,
            )
        else:
            raise ValueError("Unsupported task: {}".format(self.task))

        return batch

    def _mask_values(
        self,
        value_ids: torch.Tensor,
        key_ids: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> tuple:
        labels = torch.full_like(value_ids, fill_value=-100)
        candidate = feature_attention_mask.bool()

        random_token_mask = (
            torch.rand(value_ids.shape, device=value_ids.device) < self.mlm_probability
        )
        event_mask = (
            torch.rand(value_ids.shape[:2], device=value_ids.device) < self.event_mask_probability
        ).unsqueeze(-1)

        key_mask = torch.zeros_like(candidate)
        for row in range(value_ids.shape[0]):
            active_keys = key_ids[row][candidate[row]].unique()
            for key in active_keys:
                if int(key.item()) == self.tokenizer.vocab.pad_token_id:
                    continue
                if torch.rand((), device=value_ids.device) < self.key_mask_probability:
                    key_mask[row] |= key_ids[row].eq(key)

        selected = candidate & (random_token_mask | event_mask | key_mask)
        if not bool(selected.any()) and bool(candidate.any()):
            candidate_indices = candidate.nonzero(as_tuple=False)
            choice = candidate_indices[
                torch.randint(candidate_indices.shape[0], (1,), device=value_ids.device).item()
            ]
            selected[choice[0], choice[1], choice[2]] = True
        labels[selected] = value_ids[selected]

        unk_selected = selected & (
            torch.rand(value_ids.shape, device=value_ids.device) < self.unk_probability
        )
        mask_selected = selected & ~unk_selected
        labels[unk_selected] = -100

        masked_values = value_ids.clone()
        masked_values[mask_selected] = self.tokenizer.vocab.mask_token_id
        masked_values[unk_selected] = self.tokenizer.vocab.unk_token_id
        return masked_values, labels
