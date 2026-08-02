"""Vocabulary and tokenisation for typed event sequences."""

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from eventfm.data.schema import Event, EventSequence

SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]", "[USR]", "[EVT]"]


@dataclass
class EncodedSequence:
    key_ids: np.ndarray
    value_ids: np.ndarray
    feature_position_ids: np.ndarray
    feature_attention_mask: np.ndarray
    event_attention_mask: np.ndarray
    timestamps: np.ndarray
    delta_seconds: np.ndarray
    event_age_seconds: np.ndarray
    calendar_features: np.ndarray


class EventVocabulary:
    """Shared key/value vocabulary used by the PRAGMA-style encoder.

    Keys and values live in one namespace, matching the shared lookup table used
    by the paper baseline. Feature values are represented as
    `value:<field_name>:<field_value>`.
    """

    def __init__(self, token_to_id: Optional[Dict[str, int]] = None) -> None:
        self.token_to_id = dict(token_to_id or {})
        self.id_to_token = [None] * len(self.token_to_id)
        for token, idx in self.token_to_id.items():
            self.id_to_token[idx] = token
        for token in SPECIAL_TOKENS:
            self.add(token)

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["[PAD]"]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id["[MASK]"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["[UNK]"]

    @property
    def usr_token_id(self) -> int:
        return self.token_to_id["[USR]"]

    @property
    def evt_token_id(self) -> int:
        return self.token_to_id["[EVT]"]

    def add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_token_id)

    def decode(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self.id_to_token):
            return "[UNK]"
        token = self.id_to_token[token_id]
        return token if token is not None else "[UNK]"

    def event_type_token(self, event_type: str) -> str:
        return "value:event_type:{}".format(event_type)

    def event_type_id(self, event_type: str) -> int:
        return self.encode(self.event_type_token(event_type))

    def key_id(self, field_name: str) -> int:
        return self.encode("key:{}".format(field_name))

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump({"token_to_id": self.token_to_id}, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "EventVocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(token_to_id={str(k): int(v) for k, v in payload["token_to_id"].items()})

    @classmethod
    def from_event_types(
        cls,
        event_types: Sequence[str],
        feature_values: Optional[Mapping[str, Iterable[Any]]] = None,
    ) -> "EventVocabulary":
        vocab = cls()
        vocab.add("key:event_type")
        for event_type in event_types:
            vocab.add(vocab.event_type_token(str(event_type)))

        for feature_name, values in (feature_values or {}).items():
            vocab.add("key:{}".format(feature_name))
            for value in values:
                vocab.add("value:{}:{}".format(feature_name, _normalise_value(value)))
        return vocab

    @classmethod
    def synthetic(cls, num_event_types: int = 50) -> "EventVocabulary":
        return cls.from_event_types(["type_{}".format(i) for i in range(num_event_types)])

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable["EventSequence"],
        event_types: Optional[Sequence[str]] = None,
    ) -> "EventVocabulary":
        """Build a vocabulary by scanning observed keys, values and profiles.

        `event_types` pins the order of the mark vocabulary so the next-event
        head keeps a stable label space across splits and sample sizes.
        """

        vocab = cls()
        vocab.add("key:event_type")
        for event_type in event_types or []:
            vocab.add(vocab.event_type_token(str(event_type)))

        for sequence in sequences:
            for field_name, value in sequence.profile.items():
                vocab.add("key:{}".format(field_name))
                vocab.add("value:{}:{}".format(field_name, _normalise_value(value)))
            for event in sequence.events:
                vocab.add(vocab.event_type_token(event.event_type))
                for field_name, value in event.features.items():
                    vocab.add("key:{}".format(field_name))
                    vocab.add("value:{}:{}".format(field_name, _normalise_value(value)))
        return vocab


def _normalise_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return "{:.6g}".format(value)
    return str(value)


def _calendar_features(timestamp: float) -> np.ndarray:
    dt = datetime.utcfromtimestamp(float(timestamp))
    hour = dt.hour + dt.minute / 60.0
    dow = float(dt.weekday())
    dom = float(dt.day - 1)
    return np.asarray(
        [
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
            math.sin(2.0 * math.pi * dow / 7.0),
            math.cos(2.0 * math.pi * dow / 7.0),
            math.sin(2.0 * math.pi * dom / 31.0),
            math.cos(2.0 * math.pi * dom / 31.0),
        ],
        dtype=np.float32,
    )


class EventTokenizer:
    """Turn event histories into dense arrays consumed by collators/models."""

    def __init__(
        self,
        vocab: EventVocabulary,
        max_features_per_event: int = 8,
        include_unknown_feature_values: bool = True,
        event_type_to_index: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.vocab = vocab
        self.max_features_per_event = int(max_features_per_event)
        self.include_unknown_feature_values = bool(include_unknown_feature_values)
        self.event_type_to_index = (
            {str(key): int(value) for key, value in event_type_to_index.items()}
            if event_type_to_index is not None
            else None
        )

    @property
    def num_event_types(self) -> int:
        if self.event_type_to_index is None:
            raise ValueError("Tokenizer has no explicit event-type index map.")
        return len(self.event_type_to_index)

    def save_pretrained(self, save_directory: str) -> tuple:
        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        vocab_path = target / "vocab.json"
        config_path = target / "event_tokenizer_config.json"
        self.vocab.save(str(vocab_path))
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "max_features_per_event": self.max_features_per_event,
                    "include_unknown_feature_values": self.include_unknown_feature_values,
                    "event_type_to_index": self.event_type_to_index,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        return (str(vocab_path), str(config_path))

    @classmethod
    def from_pretrained(cls, path: str) -> "EventTokenizer":
        source = Path(path)
        vocab = EventVocabulary.load(str(source / "vocab.json"))
        config_path = source / "event_tokenizer_config.json"
        config = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        return cls(vocab=vocab, **config)

    def event_type_index(self, event_type: str) -> int:
        """Dense mark index used by every next-event-prediction head.

        Datasets supply their own ordered event-type list; the legacy
        `type_<idx>` convention of the synthetic generator is the fallback.
        """

        if self.event_type_to_index is not None:
            return self.event_type_to_index.get(str(event_type), 0)
        if not str(event_type).startswith("type_"):
            raise ValueError(
                "No event-type index map was provided and `{}` does not follow the "
                "synthetic `type_<idx>` convention.".format(event_type)
            )
        return int(str(event_type).split("_", 1)[1])

    def encode_event(self, event: Event) -> Tuple[List[int], List[int], List[int]]:
        pairs = [("event_type", event.event_type)]
        pairs.extend(sorted(event.features.items(), key=lambda item: item[0]))
        pairs = pairs[: self.max_features_per_event]

        key_ids: List[int] = []
        value_ids: List[int] = []
        position_ids: List[int] = []
        for pos, (field_name, value) in enumerate(pairs):
            key_token = "key:{}".format(field_name)
            value_token = "value:{}:{}".format(field_name, _normalise_value(value))
            key_ids.append(self.vocab.encode(key_token))
            value_ids.append(self.vocab.encode(value_token))
            position_ids.append(pos)
        return key_ids, value_ids, position_ids

    def encode_sequence(
        self,
        sequence: EventSequence,
        max_events: Optional[int] = None,
    ) -> EncodedSequence:
        if not sequence.events:
            raise ValueError("Cannot encode an empty event sequence.")

        events = sorted(sequence.events, key=lambda event: event.timestamp)
        if max_events is not None and len(events) > max_events:
            events = events[-int(max_events) :]

        num_events = len(events)
        width = self.max_features_per_event
        key_ids = np.full((num_events, width), self.vocab.pad_token_id, dtype=np.int64)
        value_ids = np.full((num_events, width), self.vocab.pad_token_id, dtype=np.int64)
        position_ids = np.zeros((num_events, width), dtype=np.int64)
        feature_mask = np.zeros((num_events, width), dtype=np.int64)
        timestamps = np.asarray([event.timestamp for event in events], dtype=np.float32)
        deltas = np.zeros(num_events, dtype=np.float32)
        if num_events > 1:
            deltas[1:] = np.maximum(0.0, np.diff(timestamps))
        ages = np.maximum(0.0, timestamps[-1] - timestamps).astype(np.float32)
        calendar = np.stack([_calendar_features(ts) for ts in timestamps], axis=0)

        for event_idx, event in enumerate(events):
            event_key_ids, event_value_ids, event_position_ids = self.encode_event(event)
            feature_count = len(event_key_ids)
            if feature_count == 0:
                continue
            key_ids[event_idx, :feature_count] = np.asarray(event_key_ids, dtype=np.int64)
            value_ids[event_idx, :feature_count] = np.asarray(event_value_ids, dtype=np.int64)
            position_ids[event_idx, :feature_count] = np.asarray(event_position_ids, dtype=np.int64)
            feature_mask[event_idx, :feature_count] = 1

        return EncodedSequence(
            key_ids=key_ids,
            value_ids=value_ids,
            feature_position_ids=position_ids,
            feature_attention_mask=feature_mask,
            event_attention_mask=np.ones(num_events, dtype=np.int64),
            timestamps=timestamps,
            delta_seconds=deltas,
            event_age_seconds=ages,
            calendar_features=calendar,
        )
