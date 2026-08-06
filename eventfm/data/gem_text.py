"""Prompt construction and target parsing for pure-token GEM experiments."""

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from eventfm.data.dataset import NextEventExample
from eventfm.data.schema import EventSequence
from eventfm.data.temporal_tokenizers import TemporalTokenizer

BEGIN_EVENT = "<|begin_of_event|>"
TYPE_PREFIX = "<|type_prefix|>"
TIME_PREFIX = "<|time_prefix|>"
END_EVENT = "<|end_of_event|>"
STRUCTURAL_TOKENS = [BEGIN_EVENT, TYPE_PREFIX, TIME_PREFIX, END_EVENT]


def _normalise_type(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


class GemTemporalSerialiser:
    """Serialize history and continuation using the GEM event template."""

    def __init__(
        self,
        temporal_tokenizer: TemporalTokenizer,
        feature_fields: Sequence[str] = (),
        max_events: int = 32,
        template_order: str = "type-time",
        event_type_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        if template_order not in {"type-time", "time-type"}:
            raise ValueError("template_order must be `type-time` or `time-type`")
        self.temporal_tokenizer = temporal_tokenizer
        self.feature_fields = list(feature_fields)
        self.max_events = int(max_events)
        self.template_order = template_order
        self.event_type_aliases = dict(event_type_aliases or {})

    def _event_type_text(self, event) -> str:
        fields = []
        for name in self.feature_fields[:2]:
            if name in event.features:
                fields.append("{}={}".format(name, event.features[name]))
        event_type = self.event_type_aliases.get(str(event.event_type), str(event.event_type))
        return " ".join([event_type] + fields)

    def _time_value(self, timestamp: float, previous_timestamp: float) -> float:
        if self.temporal_tokenizer.uses_absolute_time:
            return float(timestamp)
        return max(0.0, float(timestamp) - float(previous_timestamp))

    def _format(self, event_type_text: str, time_value: float) -> str:
        time_text = " ".join(self.temporal_tokenizer.encode(time_value))
        if self.template_order == "type-time":
            body = "{} {} {} {}".format(TYPE_PREFIX, event_type_text, TIME_PREFIX, time_text)
        else:
            body = "{} {} {} {}".format(TIME_PREFIX, time_text, TYPE_PREFIX, event_type_text)
        return "{} {} {}".format(BEGIN_EVENT, body, END_EVENT)

    def history(self, sequence: EventSequence) -> str:
        events = sequence.events[-self.max_events :]
        chunks: List[str] = []
        for index, event in enumerate(events):
            previous = event.timestamp if index == 0 else events[index - 1].timestamp
            chunks.append(
                self._format(
                    self._event_type_text(event),
                    self._time_value(event.timestamp, previous),
                )
            )
        return " ".join(chunks)

    def prompt(self, example: NextEventExample) -> str:
        history = self.history(example.sequence)
        first_prefix = TYPE_PREFIX if self.template_order == "type-time" else TIME_PREFIX
        return "{} {} {}".format(history, BEGIN_EVENT, first_prefix).strip()

    def completion(self, example: NextEventExample) -> str:
        if self.temporal_tokenizer.uses_absolute_time:
            previous = example.sequence.events[-1].timestamp
            time_value = float(
                example.next_timestamp
                if example.next_timestamp is not None
                else previous + example.next_delta_seconds
            )
        else:
            time_value = float(example.next_delta_seconds)
        time_text = " ".join(self.temporal_tokenizer.encode(time_value))
        event_type = self.event_type_aliases.get(
            str(example.next_event_type), str(example.next_event_type)
        )
        if self.template_order == "type-time":
            return " {} {} {} {}".format(
                event_type, TIME_PREFIX, time_text, END_EVENT
            )
        return " {} {} {} {}".format(
            time_text, TYPE_PREFIX, event_type, END_EVENT
        )

    def true_time_value(self, example: NextEventExample) -> float:
        if self.temporal_tokenizer.uses_absolute_time:
            return float(
                example.next_timestamp
                if example.next_timestamp is not None
                else example.sequence.events[-1].timestamp + example.next_delta_seconds
            )
        return float(example.next_delta_seconds)


class GemCausalCollator:
    """Create causal-LM labels only for the held-out next event."""

    def __init__(self, hf_tokenizer, serialiser: GemTemporalSerialiser, max_length: int = 768):
        self.hf_tokenizer = hf_tokenizer
        self.serialiser = serialiser
        self.max_length = int(max_length)

    def __call__(self, examples: Sequence[NextEventExample]) -> Dict[str, torch.Tensor]:
        encoded_rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        eos_id = self.hf_tokenizer.eos_token_id
        for example in examples:
            prompt_ids = self.hf_tokenizer(
                self.serialiser.prompt(example), add_special_tokens=False
            )["input_ids"]
            target_ids = self.hf_tokenizer(
                self.serialiser.completion(example), add_special_tokens=False
            )["input_ids"]
            if eos_id is not None:
                target_ids = list(target_ids) + [int(eos_id)]
            ids = list(prompt_ids) + list(target_ids)
            labels = [-100] * len(prompt_ids) + list(target_ids)
            # Left truncation preserves the supervised continuation.
            ids = ids[-self.max_length :]
            labels = labels[-self.max_length :]
            encoded_rows.append(ids)
            label_rows.append(labels)

        width = max(len(row) for row in encoded_rows)
        pad_id = int(self.hf_tokenizer.pad_token_id)
        input_ids = torch.full((len(encoded_rows), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(encoded_rows), width), dtype=torch.long)
        labels = torch.full((len(encoded_rows), width), -100, dtype=torch.long)
        for index, (ids, targets) in enumerate(zip(encoded_rows, label_rows)):
            size = len(ids)
            input_ids[index, :size] = torch.as_tensor(ids, dtype=torch.long)
            attention[index, :size] = 1
            labels[index, :size] = torch.as_tensor(targets, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def fit_temporal_tokenizer(
    temporal_tokenizer: TemporalTokenizer, sequences: Sequence[EventSequence]
) -> TemporalTokenizer:
    values: List[float] = []
    for sequence in sequences:
        previous = None
        for event in sequence.events:
            if temporal_tokenizer.uses_absolute_time:
                values.append(float(event.timestamp))
            elif previous is not None:
                values.append(max(0.0, float(event.timestamp) - previous))
            previous = float(event.timestamp)
    return temporal_tokenizer.fit(values)


def parse_generated_event(
    generated_ids: Sequence[int],
    hf_tokenizer,
    temporal_tokenizer: TemporalTokenizer,
    template_order: str,
    event_type_to_index: Dict[str, int],
    previous_timestamp: float,
) -> Dict[str, Any]:
    """Parse one generated continuation into mark index and log interval."""

    type_prefix_id = int(hf_tokenizer.convert_tokens_to_ids(TYPE_PREFIX))
    time_prefix_id = int(hf_tokenizer.convert_tokens_to_ids(TIME_PREFIX))
    end_id = int(hf_tokenizer.convert_tokens_to_ids(END_EVENT))
    ids = [int(value) for value in generated_ids]
    if end_id in ids:
        ids = ids[: ids.index(end_id)]

    try:
        if template_order == "type-time":
            split = ids.index(time_prefix_id)
            type_ids, time_ids = ids[:split], ids[split + 1 :]
        else:
            split = ids.index(type_prefix_id)
            time_ids, type_ids = ids[:split], ids[split + 1 :]

        type_text = _normalise_type(
            hf_tokenizer.decode(type_ids, skip_special_tokens=True).strip()
        )
        type_lookup = {_normalise_type(name): index for name, index in event_type_to_index.items()}
        if type_text not in type_lookup:
            raise ValueError("Unknown generated event type")

        if isinstance(temporal_tokenizer, object) and temporal_tokenizer.key == "time-numeric-p6":
            time_tokens = [hf_tokenizer.decode(time_ids, skip_special_tokens=True)]
        else:
            time_tokens = hf_tokenizer.convert_ids_to_tokens(time_ids)
        decoded_time = float(temporal_tokenizer.decode(time_tokens))
        delta = (
            max(0.0, decoded_time - float(previous_timestamp))
            if temporal_tokenizer.uses_absolute_time
            else max(0.0, decoded_time)
        )
        return {
            "type_index": int(type_lookup[type_text]),
            "delta_log": float(np.log1p(delta)),
            "invalid": False,
        }
    except (ValueError, OverflowError, IndexError):
        return {"type_index": 0, "delta_log": 0.0, "invalid": True}
