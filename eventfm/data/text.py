"""Text serialisation of event histories for the language-model methods.

The project note's central question is *how should event content and time be
tokenised*. The three LM-based entries answer it differently, and this module
implements each answer literally so the comparison isolates the tokenisation
choice rather than the backbone:

``type_text``
    TPP-LLM. The prompt carries only the textual event descriptions; timing is
    injected as a continuous temporal embedding added to the token embeddings
    at event-boundary positions.

``byte_time``
    Language-TPP. Each inter-arrival interval is serialised into byte tokens
    that live in the text stream alongside the event description, so a plain
    autoregressive interface can model both *what* and *when*.

``byte_time_compressed``
    MM-TPP. Byte-token times plus adaptive compression: runs of consecutive
    events with near-identical inter-arrival intervals collapse into a single
    ``[CMP]`` token carrying a repeat count, which buys a longer effective
    history inside a fixed context window.
"""

import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np
import torch

from eventfm.data.dataset import NextEventExample
from eventfm.data.schema import EventSequence

TextExample = Union[EventSequence, NextEventExample]

BYTE_TOKENS = ["<byte_{}>".format(index) for index in range(256)]
SPECIAL_EVENT_TOKENS = ["<evt>", "<time>", "<cmp>"]


def float_to_byte_tokens(value: float) -> List[str]:
    """Language-TPP byte encoding: a float32 becomes four byte tokens."""

    packed = struct.pack(">f", float(np.float32(value)))
    return ["<byte_{}>".format(byte) for byte in packed]


def humanise_delta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90.0:
        return "{:.0f}s".format(seconds)
    if seconds < 5400.0:
        return "{:.0f}m".format(seconds / 60.0)
    if seconds < 172800.0:
        return "{:.1f}h".format(seconds / 3600.0)
    return "{:.1f}d".format(seconds / 86400.0)


@dataclass
class SerialisedHistory:
    text: str
    event_token_positions: List[int]
    event_log_deltas: List[float]
    num_events_kept: int
    num_events_compressed: int


def _event_phrase(event, feature_fields: Sequence[str], max_features: int) -> str:
    parts = [str(event.event_type)]
    for field_name in list(feature_fields)[:max_features]:
        value = event.features.get(field_name)
        if value is not None:
            parts.append("{}={}".format(field_name, value))
    return " ".join(parts)


def compress_by_temporal_similarity(
    log_deltas: Sequence[float],
    threshold: float,
) -> List[int]:
    """MM-TPP compression: keep an event only when its interval changes.

    Returns the number of events each kept position absorbs (1 = not merged).
    Consecutive events whose log inter-arrival times differ by less than
    ``threshold`` are folded into the preceding kept event.
    """

    if not log_deltas:
        return []
    runs = [1]
    previous = log_deltas[0]
    for value in log_deltas[1:]:
        if abs(value - previous) < threshold:
            runs[-1] += 1
        else:
            runs.append(1)
        previous = value
    return runs


class EventTextSerialiser:
    """Turn an event history into the prompt format of one LM-based method."""

    def __init__(
        self,
        style: str,
        feature_fields: Sequence[str] = (),
        max_features_per_event: int = 2,
        max_events: int = 32,
        compression_threshold: float = 0.25,
    ) -> None:
        if style not in {"type_text", "byte_time", "byte_time_compressed"}:
            raise ValueError("Unknown serialisation style: {}".format(style))
        self.style = style
        self.feature_fields = list(feature_fields)
        self.max_features_per_event = int(max_features_per_event)
        self.max_events = int(max_events)
        self.compression_threshold = float(compression_threshold)

    def __call__(self, sequence: EventSequence) -> SerialisedHistory:
        events = sequence.events[-self.max_events :]
        if not events:
            return SerialisedHistory("history: (empty)", [], [], 0, 0)

        timestamps = np.asarray([event.timestamp for event in events], dtype=np.float64)
        deltas = np.zeros(len(events), dtype=np.float64)
        if len(events) > 1:
            deltas[1:] = np.maximum(0.0, np.diff(timestamps))
        log_deltas = np.log1p(deltas)

        runs = (
            compress_by_temporal_similarity(log_deltas.tolist(), self.compression_threshold)
            if self.style == "byte_time_compressed"
            else [1] * len(events)
        )

        chunks: List[str] = ["history:"]
        kept = 0
        compressed = 0
        cursor = 0
        for run_length in runs:
            event = events[cursor]
            phrase = _event_phrase(event, self.feature_fields, self.max_features_per_event)
            if self.style == "type_text":
                chunks.append("<evt> {}".format(phrase))
            else:
                time_tokens = " ".join(float_to_byte_tokens(log_deltas[cursor]))
                chunks.append("<evt> {} <time> {}".format(phrase, time_tokens))
            if run_length > 1:
                chunks.append("<cmp> x{}".format(run_length))
                compressed += run_length - 1
            kept += 1
            cursor += run_length

        chunks.append("next:")
        return SerialisedHistory(
            text=" ".join(chunks),
            event_token_positions=[],
            event_log_deltas=log_deltas.tolist(),
            num_events_kept=kept,
            num_events_compressed=compressed,
        )


class TextEventCollator:
    """Tokenise serialised histories and attach the task targets."""

    def __init__(
        self,
        hf_tokenizer,
        serialiser: EventTextSerialiser,
        task: str,
        event_type_to_index: Dict[str, int],
        max_length: int = 512,
    ) -> None:
        self.hf_tokenizer = hf_tokenizer
        self.serialiser = serialiser
        self.task = task
        self.event_type_to_index = dict(event_type_to_index)
        self.max_length = int(max_length)

    def __call__(self, examples: Sequence[TextExample]) -> Dict[str, torch.Tensor]:
        sequences: List[EventSequence] = []
        next_examples: List[NextEventExample] = []
        for example in examples:
            if isinstance(example, NextEventExample):
                sequences.append(example.sequence)
                next_examples.append(example)
            else:
                sequences.append(example)

        serialised = [self.serialiser(sequence) for sequence in sequences]
        encoded = self.hf_tokenizer(
            [item.text for item in serialised],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            # TPP-LLM injects timing as a continuous embedding rather than text.
            "recent_log_delta": torch.as_tensor(
                [
                    float(item.event_log_deltas[-1]) if item.event_log_deltas else 0.0
                    for item in serialised
                ],
                dtype=torch.float,
            ),
            "history_log_span": torch.as_tensor(
                [float(np.sum(item.event_log_deltas)) for item in serialised],
                dtype=torch.float,
            ),
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
                [
                    self.event_type_to_index.get(item.next_event_type, 0)
                    for item in next_examples
                ],
                dtype=torch.long,
            )
            batch["next_delta_log"] = torch.as_tensor(
                [float(np.log1p(item.next_delta_seconds)) for item in next_examples],
                dtype=torch.float,
            )
        return batch


def extend_tokenizer(hf_tokenizer, style: str) -> int:
    """Add the byte / structural tokens the LM styles rely on."""

    additional = list(SPECIAL_EVENT_TOKENS)
    if style in {"byte_time", "byte_time_compressed"}:
        additional += BYTE_TOKENS
    added = hf_tokenizer.add_tokens(additional)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    return int(added)
