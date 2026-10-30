"""Columnar event store: the whole split as a handful of flat numpy arrays.

The benchmark's JSONL contract materialises every event as a Python object,
which is fine for 30k histories and impossible for the ~529k-client MBD
pretraining pool inside a 48 GB job. ETAM therefore reads a split once and
keeps it as

``offsets``     (N+1,)   int64    sequence ``i`` owns events ``offsets[i]:offsets[i+1]``
``types``       (E,)     int16    index into ``EtamVocab.event_types``
``features``    (E, F)   int16    per-field value index (0 = pad, 1 = unknown)
``timestamps``  (E,)     float64  seconds, stably sorted inside each sequence
``labels``      (N,)     int8     -1 when the sequence carries no label
``user_ids``    (N,)     str

Event order and the event-type index are exactly the benchmark's, so every
example the shared evaluators build from the JSONL can be rebuilt here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

PAD_VALUE = 0
UNK_VALUE = 1
_ARRAYS = ("offsets", "types", "features", "timestamps", "labels")


@dataclass
class EtamVocab:
    """Event-type list plus one value table per feature field.

    Built from the prepared dataset's ``meta.json`` rather than from a training
    subset, so the store of the unlabeled pool and the store of the labeled
    subset share one index space by construction.
    """

    event_types: List[str]
    feature_fields: List[str]
    field_values: Dict[str, List[str]]

    def __post_init__(self) -> None:
        self.type_index = {name: index for index, name in enumerate(self.event_types)}
        self.value_index = {
            field_name: {value: index + 2 for index, value in enumerate(values)}
            for field_name, values in self.field_values.items()
        }

    @property
    def num_event_types(self) -> int:
        return len(self.event_types)

    @property
    def field_sizes(self) -> List[int]:
        """Vocabulary size per field, counting the pad and unknown slots."""

        return [len(self.field_values[name]) + 2 for name in self.feature_fields]

    @classmethod
    def from_meta(cls, meta: Dict[str, object]) -> "EtamVocab":
        fields = [str(name) for name in meta["feature_fields"]]
        categorical = dict(meta.get("categorical_levels", {}))
        edges = dict(meta.get("numeric_bucket_edges", {}))
        values: Dict[str, List[str]] = {}
        for name in fields:
            if name in edges:
                count = len(edges[name]) + 1
                values[name] = ["b_{}".format(index) for index in range(count)] + ["na"]
            else:
                values[name] = [str(level) for level in categorical.get(name, [])]
                values[name] += [token for token in ("other", "na") if token not in values[name]]
        return cls(
            event_types=[str(name) for name in meta["event_types"]],
            feature_fields=fields,
            field_values=values,
        )

    @classmethod
    def from_meta_path(cls, path: str) -> "EtamVocab":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_meta(json.load(handle))

    def to_json(self) -> Dict[str, object]:
        return {
            "event_types": self.event_types,
            "feature_fields": self.feature_fields,
            "field_values": self.field_values,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "EtamVocab":
        return cls(
            event_types=list(payload["event_types"]),
            feature_fields=list(payload["feature_fields"]),
            field_values={key: list(value) for key, value in payload["field_values"].items()},
        )

    def encode_type(self, event_type: str) -> int:
        if event_type in self.type_index:
            return self.type_index[event_type]
        if "other" in self.type_index:
            return self.type_index["other"]
        raise KeyError("Event type `{}` is not in the vocabulary.".format(event_type))

    def encode_value(self, field_name: str, value: object) -> int:
        return self.value_index[field_name].get(str(value), UNK_VALUE)


@dataclass
class EventStore:
    offsets: np.ndarray
    types: np.ndarray
    features: np.ndarray
    timestamps: np.ndarray
    labels: np.ndarray
    user_ids: np.ndarray
    vocab: EtamVocab
    #: Last instant at which a history is complete, e.g. the benchmark's label
    #: cutoff. ``None`` means "the sequence's own last event".
    observation_end: Optional[float] = None

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def num_events(self) -> int:
        return int(self.offsets[-1])

    def sequence_slice(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))

    def subset(self, indices: Sequence[int]) -> "EventStore":
        """Copy a selection of sequences into a new, contiguous store."""

        indices = np.asarray(indices, dtype=np.int64)
        lengths = self.lengths[indices]
        offsets = np.zeros(indices.shape[0] + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        rows = (
            np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in indices])
            if indices.size
            else np.zeros(0, dtype=np.int64)
        )
        return EventStore(
            offsets=offsets,
            types=self.types[rows],
            features=self.features[rows],
            timestamps=self.timestamps[rows],
            labels=self.labels[indices],
            user_ids=self.user_ids[indices],
            vocab=self.vocab,
            observation_end=self.observation_end,
        )

    def save(self, directory: str) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for name in _ARRAYS:
            np.save(target / "{}.npy".format(name), getattr(self, name))
        np.save(target / "user_ids.npy", self.user_ids.astype(np.str_))
        payload = {"vocab": self.vocab.to_json(), "observation_end": self.observation_end}
        with (target / "store.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return target

    @classmethod
    def load(cls, directory: str, mmap: bool = False) -> "EventStore":
        source = Path(directory)
        mode = "r" if mmap else None
        arrays = {name: np.load(source / "{}.npy".format(name), mmap_mode=mode) for name in _ARRAYS}
        with (source / "store.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            user_ids=np.load(source / "user_ids.npy"),
            vocab=EtamVocab.from_json(payload["vocab"]),
            observation_end=payload.get("observation_end"),
            **arrays,
        )


class StoreBuilder:
    """Append sequences one at a time without keeping Python event objects."""

    def __init__(self, vocab: EtamVocab, observation_end: Optional[float] = None) -> None:
        self.vocab = vocab
        self.observation_end = observation_end
        self._lengths: List[int] = []
        self._types: List[np.ndarray] = []
        self._features: List[np.ndarray] = []
        self._timestamps: List[np.ndarray] = []
        self._labels: List[int] = []
        self._user_ids: List[str] = []

    def add_arrays(
        self,
        user_id: str,
        types: np.ndarray,
        features: np.ndarray,
        timestamps: np.ndarray,
        label: Optional[int],
    ) -> None:
        # Stable, like `sequence_from_json`'s `list.sort`, so tied timestamps
        # keep file order and the evaluators' prefixes line up exactly.
        order = np.argsort(timestamps, kind="stable")
        self._lengths.append(int(types.shape[0]))
        self._types.append(np.asarray(types, dtype=np.int16)[order])
        self._features.append(np.asarray(features, dtype=np.int16)[order])
        self._timestamps.append(np.asarray(timestamps, dtype=np.float64)[order])
        self._labels.append(-1 if label is None else int(label))
        self._user_ids.append(str(user_id))

    def add_record(self, record: Dict[str, object]) -> None:
        """Add one JSONL record in the benchmark's data contract."""

        events = record.get("events", [])
        fields = self.vocab.feature_fields
        types = np.empty(len(events), dtype=np.int16)
        features = np.empty((len(events), len(fields)), dtype=np.int16)
        timestamps = np.empty(len(events), dtype=np.float64)
        for row, event in enumerate(events):
            types[row] = self.vocab.encode_type(str(event.get("event_type", event.get("type"))))
            values = event.get("features", {})
            for column, field_name in enumerate(fields):
                value = values.get(field_name, event.get(field_name, "na"))
                features[row, column] = self.vocab.encode_value(field_name, value)
            timestamps[row] = float(event["timestamp"])
        label = record.get("label")
        self.add_arrays(str(record["user_id"]), types, features, timestamps, label)

    def build(self) -> EventStore:
        offsets = np.zeros(len(self._lengths) + 1, dtype=np.int64)
        np.cumsum(np.asarray(self._lengths, dtype=np.int64), out=offsets[1:])
        width = len(self.vocab.feature_fields)
        return EventStore(
            offsets=offsets,
            types=np.concatenate(self._types) if self._types else np.zeros(0, np.int16),
            features=(
                np.concatenate(self._features) if self._features else np.zeros((0, width), np.int16)
            ),
            timestamps=(np.concatenate(self._timestamps) if self._timestamps else np.zeros(0)),
            labels=np.asarray(self._labels, dtype=np.int8),
            user_ids=np.asarray(self._user_ids, dtype=np.str_),
            vocab=self.vocab,
            observation_end=self.observation_end,
        )


def iter_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def store_from_jsonl(
    path: str,
    vocab: EtamVocab,
    observation_end: Optional[float] = None,
    max_sequences: Optional[int] = None,
) -> EventStore:
    builder = StoreBuilder(vocab, observation_end=observation_end)
    for index, record in enumerate(iter_jsonl(path)):
        if max_sequences is not None and index >= max_sequences:
            break
        builder.add_record(record)
    return builder.build()
