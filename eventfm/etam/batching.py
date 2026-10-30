"""Views over the event store and the collator that turns them into tensors.

A *view* is one model input: the events of sequence ``seq`` strictly before
index ``end`` (the last ``max_events`` of them), followed by a query slot at
time ``query_time >= t[end - 1]``. The query slot is where the user embedding
lives; its time token carries "now".

Every slot (``max_events`` event slots, left-padded, then the query slot) gets
one time-feature row and one event-feature row, so the model can either build
two tokens per slot (ETAM) or add the time embedding onto the event embedding
(the single-token control) from the same batch.

Targets are read from the *full* sequence, not from the window, so the last
event of a cut view is still supervised by the event that follows it.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from eventfm.etam.store import EventStore

#: Absolute-time origin, 2020-01-01T00:00:00Z. Only used to keep day counts
#: small enough for float32 features; the calendar is computed exactly.
ORIGIN_SECONDS = 1_577_836_800.0
SECONDS_PER_DAY = 86_400.0
IGNORE_INDEX = -100
NUM_CALENDAR_CONT = 8


@dataclass(frozen=True)
class View:
    seq: int
    end: int
    query_time: float


def calendar_arrays(seconds: np.ndarray):
    """UTC calendar ids ``(hour, weekday, day-of-month, month)`` plus cyclic coordinates."""

    seconds = np.asarray(seconds, dtype=np.float64)
    whole = np.floor(seconds).astype("int64")
    stamp = whole.astype("datetime64[s]")
    day = stamp.astype("datetime64[D]")
    hour = ((whole - day.astype("int64") * 86_400) // 3_600).astype(np.int64)
    weekday = ((day.astype("int64") + 3) % 7).astype(np.int64)  # 1970-01-01 was a Thursday
    month_start = stamp.astype("datetime64[M]")
    month = (month_start.astype("int64") % 12).astype(np.int64)
    dom = (day - month_start.astype("datetime64[D]")).astype(np.int64)
    doy = (day - stamp.astype("datetime64[Y]").astype("datetime64[D]")).astype(np.int64)
    hour_frac = (seconds - day.astype("int64") * 86_400.0) / 86_400.0
    angles = np.stack([hour_frac, weekday / 7.0, dom / 31.0, doy / 366.0], axis=-1) * (
        2.0 * math.pi
    )
    continuous = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)
    ids = np.stack([hour, weekday, dom, month], axis=-1)
    return ids, continuous


class ViewCollator:
    """Assemble a padded batch from views; all heavy lifting is vectorised numpy."""

    def __init__(
        self,
        store: EventStore,
        max_events: int = 128,
        future_window_days: float = 30.0,
        window_time: bool = False,
    ) -> None:
        self.store = store
        self.max_events = int(max_events)
        # ``window_time`` measures age and the first gap inside the window, as
        # the baselines' collator does; the default reads them off the stored
        # history, which only differs when the window is shorter than it.
        self.window_time = bool(window_time)
        self.future_window = float(future_window_days) * SECONDS_PER_DAY
        self.num_fields = int(store.features.shape[1])
        self.num_types = int(store.vocab.num_event_types)

    def __call__(self, views: Sequence[View]) -> Dict[str, torch.Tensor]:
        store = self.store
        offsets, times, types, features = (
            store.offsets,
            store.timestamps,
            store.types,
            store.features,
        )
        batch = len(views)
        windows = []
        for view in views:
            start = int(offsets[view.seq])
            stop = int(offsets[view.seq + 1])
            end_abs = start + int(view.end)
            if not start < end_abs <= stop:
                raise ValueError("View end {} outside sequence {}.".format(view.end, view.seq))
            first = max(start, end_abs - self.max_events)
            windows.append((start, stop, first, end_abs))
        width = max(end_abs - first for _, _, first, end_abs in windows)
        slots = width + 1

        slot_valid = np.zeros((batch, slots), dtype=bool)
        is_query = np.zeros((batch, slots), dtype=bool)
        slot_time = np.zeros((batch, slots), dtype=np.float64)
        prev_time = np.zeros((batch, slots), dtype=np.float64)
        has_prev = np.zeros((batch, slots), dtype=np.float32)
        first_time = np.zeros((batch, 1), dtype=np.float64)
        position = np.zeros((batch, slots), dtype=np.int64)
        type_ids = np.zeros((batch, slots), dtype=np.int64)
        feature_ids = np.zeros((batch, slots, self.num_fields), dtype=np.int64)
        cur_type = np.full((batch, slots), IGNORE_INDEX, dtype=np.int64)
        cur_features = np.full((batch, slots, self.num_fields), IGNORE_INDEX, dtype=np.int64)
        next_type = np.full((batch, slots), IGNORE_INDEX, dtype=np.int64)
        next_log_dt = np.zeros((batch, slots), dtype=np.float32)
        future_types = np.zeros((batch, self.num_types), dtype=np.float32)
        future_log_count = np.zeros(batch, dtype=np.float32)
        future_mask = np.zeros(batch, dtype=np.float32)
        labels = np.full(batch, -1, dtype=np.int64)

        for row, (view, (start, stop, first, end_abs)) in enumerate(zip(views, windows)):
            count = end_abs - first
            offset = width - count
            rows = np.arange(first, end_abs)
            event_slots = slice(offset, width)
            query_time = float(view.query_time)
            last_time = float(times[end_abs - 1])
            if query_time < last_time:
                raise ValueError("Query time precedes the last observed event.")

            slot_valid[row, offset:] = True
            is_query[row, width] = True
            slot_time[row, event_slots] = times[rows]
            slot_time[row, width] = query_time
            slot_time[row, :offset] = times[first]
            previous = rows - 1
            known = previous >= (first if self.window_time else start)
            prev_time[row, event_slots] = np.where(known, times[np.maximum(previous, start)], 0.0)
            has_prev[row, event_slots] = known
            prev_time[row, width] = last_time
            has_prev[row, width] = 1.0
            first_time[row, 0] = times[first if self.window_time else start]
            position[row, offset:] = np.arange(count + 1)
            type_ids[row, event_slots] = types[rows]
            feature_ids[row, event_slots] = features[rows]
            cur_type[row, event_slots] = types[rows]
            cur_features[row, event_slots] = features[rows]

            following = rows + 1
            has_next = following < stop
            next_rows = following[has_next]
            target_slots = np.arange(offset, width)[has_next]
            next_type[row, target_slots] = types[next_rows]
            next_log_dt[row, target_slots] = np.log1p(
                np.maximum(0.0, times[next_rows] - times[next_rows - 1])
            )
            if end_abs < stop:
                next_type[row, width] = types[end_abs]
                next_log_dt[row, width] = math.log1p(max(0.0, float(times[end_abs]) - query_time))

            # Without a known observation cutoff the window could only be judged
            # complete when a later event exists, which selects on the future,
            # so the target is then left out entirely.
            horizon = query_time + self.future_window
            if store.observation_end is not None and horizon <= float(store.observation_end):
                future_mask[row] = 1.0
                upcoming = np.arange(end_abs, stop)
                inside = upcoming[(times[upcoming] > query_time) & (times[upcoming] <= horizon)]
                if inside.size:
                    future_types[row, types[inside]] = 1.0
                future_log_count[row] = math.log1p(float(inside.size))
            # The sequence label describes the future after the *whole* history,
            # so a cut view must not inherit it.
            if end_abs == stop:
                labels[row] = int(store.labels[view.seq])

        gap = np.where(has_prev > 0, np.maximum(0.0, slot_time - prev_time), 0.0)
        age = np.maximum(0.0, slot_time - first_time)
        calendar_ids, calendar_cont = calendar_arrays(slot_time)
        days = (slot_time - ORIGIN_SECONDS) / SECONDS_PER_DAY
        return {
            "slot_valid": torch.from_numpy(slot_valid),
            "is_query": torch.from_numpy(is_query),
            "position": torch.from_numpy(position),
            "log_gap": torch.from_numpy(np.log1p(gap).astype(np.float32)),
            "log_age": torch.from_numpy(np.log1p(age).astype(np.float32)),
            "has_prev": torch.from_numpy(has_prev),
            "calendar_ids": torch.from_numpy(calendar_ids),
            "calendar_cont": torch.from_numpy(calendar_cont),
            # float64 on purpose: the time-rotary angles need sub-hour precision
            # over multi-year spans, which float32 day counts do not have.
            "abs_days": torch.from_numpy(days),
            "type_ids": torch.from_numpy(type_ids),
            "feature_ids": torch.from_numpy(feature_ids),
            "cur_type": torch.from_numpy(cur_type),
            "cur_features": torch.from_numpy(cur_features),
            "next_type": torch.from_numpy(next_type),
            "next_log_dt": torch.from_numpy(next_log_dt),
            "future_types": torch.from_numpy(future_types),
            "future_log_count": torch.from_numpy(future_log_count),
            "future_mask": torch.from_numpy(future_mask),
            "labels": torch.from_numpy(labels),
        }


# ---------------------------------------------------------------------------
# View generators
# ---------------------------------------------------------------------------


def full_views(store: EventStore, query: str = "last", indices: Optional[Sequence[int]] = None):
    """One view per sequence over its whole history (classification inputs).

    ``query="last"`` puts "now" at the last event; ``"observation_end"`` puts it
    at the store's observation cutoff, which exposes recency explicitly.
    """

    chosen = range(len(store)) if indices is None else indices
    views = []
    for seq in chosen:
        length = int(store.offsets[seq + 1] - store.offsets[seq])
        last = float(store.timestamps[int(store.offsets[seq + 1]) - 1])
        now = last
        if query == "observation_end" and store.observation_end is not None:
            now = max(last, float(store.observation_end))
        views.append(View(int(seq), length, now))
    return views


def benchmark_tpp_views(
    store: EventStore,
    seed: int,
    min_prefix_events: int = 4,
    max_prefixes_per_user: Optional[int] = 8,
    max_users: Optional[int] = None,
) -> List[View]:
    """Replay `NextEventJsonlDataset`'s prefix sampling exactly.

    Same RNG, same call sequence, same file order: the prefix ending before
    event ``next_idx`` becomes a view with ``end = next_idx`` and "now" at the
    prefix's last event, so ETAM is scored on the identical TPP examples as
    every baseline in the benchmark (seed ``cell.seed + 1`` for test).
    """

    rng = random.Random(seed)
    views: List[View] = []
    limit = len(store) if max_users is None else min(len(store), int(max_users))
    for seq in range(limit):
        start = int(store.offsets[seq])
        length = int(store.offsets[seq + 1]) - start
        if length <= min_prefix_events:
            continue
        candidates = list(range(min_prefix_events, length))
        if max_prefixes_per_user is not None and len(candidates) > max_prefixes_per_user:
            candidates = sorted(rng.sample(candidates, max_prefixes_per_user))
        for next_idx in candidates:
            views.append(View(seq, next_idx, float(store.timestamps[start + next_idx - 1])))
    return views


class RandomCutViews(torch.utils.data.Dataset):
    """Pretraining / TPP fine-tuning views: a random cut per sequence per draw.

    With probability ``p_full`` the view covers the whole history (densest
    supervision, and the classification-time input distribution). Otherwise the
    cut is uniform over ``[min_prefix, length - 1]`` and, with probability
    ``p_between``, "now" falls uniformly between the last kept event and the
    next one, so the query token learns time-to-next under elapsed silence.
    """

    def __init__(
        self,
        store: EventStore,
        min_prefix: int = 2,
        p_full: float = 0.5,
        p_between: float = 0.5,
        seed: int = 0,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.store = store
        self.indices = np.arange(len(store)) if indices is None else np.asarray(indices)
        self.min_prefix = int(min_prefix)
        self.p_full = float(p_full)
        self.p_between = float(p_between)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __getitem__(self, item: int) -> View:
        rng = self._rng
        seq = int(self.indices[item])
        start = int(self.store.offsets[seq])
        length = int(self.store.offsets[seq + 1]) - start
        times = self.store.timestamps
        if length <= self.min_prefix or rng.random() < self.p_full:
            return View(seq, length, float(times[start + length - 1]))
        end = int(rng.integers(self.min_prefix, length))
        last = float(times[start + end - 1])
        now = last
        if rng.random() < self.p_between:
            following = float(times[start + end])
            now = last + float(rng.random()) * max(0.0, following - last)
        return View(seq, end, now)


def worker_init(worker_id: int) -> None:
    """Give every DataLoader worker its own view RNG stream."""

    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if isinstance(dataset, RandomCutViews):
        dataset.reseed(dataset.seed * 1_000 + info.seed % 1_000_003 + worker_id)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items: Sequence[View]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> View:
        return self.items[item]
