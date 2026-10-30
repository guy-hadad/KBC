"""PyTorch datasets for JSONL event histories."""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from eventfm.data.io import load_jsonl_sequences
from eventfm.data.schema import EventSequence, sequence_from_json


class JsonlEventDataset(Dataset):
    """Return full sequences for MLM pretraining or sequence classification."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.sequences = load_jsonl_sequences(path)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> EventSequence:
        return self.sequences[index]


class LazyJsonlEventDataset(Dataset):
    """Same records as :class:`JsonlEventDataset`, parsed one line per access.

    Unlabeled pretraining corpora are an order of magnitude larger than the
    labeled splits (all ~529k MBD train-split clients is ~9 GB of JSONL), and
    holding them as ``EventSequence`` objects would exceed a 48 GB job. Only the
    byte offset of every non-blank line is kept in memory.

    The file is opened on first access in each process rather than in
    ``__init__``: a handle inherited by forked DataLoader workers would share
    one file position across processes, so concurrent seeks would interleave.
    """

    def __init__(self, path: str, max_sequences: Optional[int] = None) -> None:
        self.path = str(path)
        offsets: List[int] = []
        position = 0
        with Path(self.path).open("rb") as handle:
            for line in handle:
                # The first N lines, not a sample, so a capped smoke run is
                # deterministic and reads only the head of the file.
                if max_sequences is not None and len(offsets) >= max_sequences:
                    break
                if not line.isspace():
                    offsets.append(position)
                position += len(line)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self._handle: Optional[BinaryIO] = None
        self._handle_pid: Optional[int] = None

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def __getitem__(self, index: int) -> EventSequence:
        offset = int(self.offsets[index])
        handle = self._file()
        handle.seek(offset)
        line = handle.readline()
        try:
            return sequence_from_json(json.loads(line))
        except Exception as exc:
            raise ValueError(
                "Failed to parse {} at byte offset {}: {}".format(self.path, offset, exc)
            ) from exc

    def _file(self) -> BinaryIO:
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            self._handle = Path(self.path).open("rb")
            self._handle_pid = pid
        return self._handle

    def __getstate__(self) -> Dict[str, Any]:
        # Spawned workers receive a pickled copy, and an open file cannot be
        # pickled; each copy reopens the file on its first access instead.
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_handle_pid"] = None
        return state


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
