"""Stage 2 collator: the Stage 1 batch plus the targets the new objectives need.

Added per view (all read from the *full* sequence, never fed as inputs):

``next_features``     attributes of the next event (feature decoder)
``mtp_types/_log_dt`` type of event i+l and log1p(t_{i+l} - t_i), l = 1..L
``censor_*``          full histories: log1p(T_obs - t_last) and log1p(T_obs - now)
``between_mask``      "now" strictly inside the gap before the next event, with
``total_log_dt``      log1p(t_next - t_last) and ``elapsed_log_dt`` log1p(now - t_last)
``counts``            per-mark and total counts in (now, now + h], h = 1/7/30 d,
                      masked unless now + h <= T_obs (never censored)
``log_amount``        signed log1p(|amount|) when the store carries raw amounts
``repr.*``            the last ``repr_block`` events of each window as a
                      stand-alone sub-batch (history dropped), for the
                      predictive-information objective
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from dataclasses import dataclass

from eventfm.etam.batching import IGNORE_INDEX, SECONDS_PER_DAY, RandomCutViews, View, ViewCollator
from eventfm.etam.store import EventStore


#: View kinds. Count targets need a "now" that does not depend on the future, so they
#: are only built for calendar-time queries.
FULL, EVENT_CUT, CALENDAR = 0, 1, 2


@dataclass(frozen=True)
class View2(View):
    kind: int = EVENT_CUT


class Views2(RandomCutViews):
    """Stage 2 training views: full histories, event cuts and calendar-time queries.

    * full (``p_full``): the whole history, "now" at its last event;
    * otherwise, with probability ``p_between``, a *calendar-time query*: "now" is
      uniform in calendar time over [t_{min_prefix}, T_end] (T_end = the observation
      cutoff when known, else the last event) and the history is every event at or
      before it. A gap then contains "now" with probability proportional to its
      length, so p(gap | elapsed s) = p(gap) / S(s) holds exactly: the survival-
      conditioned likelihood is proper, and queries after the last event are
      right-censored at T_end;
    * else an *event cut*: the history ends at a uniformly chosen event, "now" at it.

    (Stage 1's RandomCutViews put "now" at a uniform fraction of the next gap, which
    makes the gap given the elapsed time length-biased the wrong way, p(gap)/gap.)
    """

    def __getitem__(self, item: int) -> View2:
        rng = self._rng
        seq = int(self.indices[item])
        start = int(self.store.offsets[seq])
        length = int(self.store.offsets[seq + 1]) - start
        times = self.store.timestamps
        last = float(times[start + length - 1])
        if length <= self.min_prefix or rng.random() < self.p_full:
            return View2(seq, length, last, FULL)
        if rng.random() < self.p_between:
            low = float(times[start + self.min_prefix - 1])
            end_time = self.store.observation_end
            high = max(last, float(end_time)) if end_time is not None else last
            now = low + float(rng.random()) * max(0.0, high - low)
            end = int(np.searchsorted(times[start : start + length], now, side="right"))
            return View2(seq, max(end, self.min_prefix), max(now, float(times[start + max(end, self.min_prefix) - 1])), CALENDAR)
        end = int(rng.integers(self.min_prefix, length))
        return View2(seq, end, float(times[start + end - 1]), EVENT_CUT)


def load_extras(store: EventStore, directory: str) -> EventStore:
    """Attach optional Stage 2 arrays (raw amounts, all MBD targets) to a store."""

    path = Path(directory)
    for name in ("amount", "targets", "targets_oct"):
        file = path / "{}.npy".format(name)
        setattr(store, name, np.load(file) if file.exists() else None)
    return store


class Collator2(ViewCollator):
    def __init__(
        self,
        store: EventStore,
        max_events: int = 128,
        future_window_days: float = 30.0,
        window_time: bool = False,
        mtp_steps: int = 1,
        count_horizons_days: Sequence[float] = (1.0, 7.0, 30.0),
        repr_block: int = 0,
    ) -> None:
        super().__init__(store, max_events, future_window_days, window_time)
        self.mtp_steps = max(1, int(mtp_steps))
        self.horizons = [float(days) * SECONDS_PER_DAY for days in count_horizons_days]
        self.repr_block = int(repr_block)
        self.sub_collator = (
            Collator2(store, max_events=self.repr_block, window_time=True, mtp_steps=1, count_horizons_days=())
            if self.repr_block
            else None
        )
        self.amount = getattr(store, "amount", None)

    def __call__(self, views: Sequence[View]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(views)
        store = self.store
        offsets, times, types, features = store.offsets, store.timestamps, store.types, store.features
        size, slots = batch["slot_valid"].shape
        width = slots - 1
        steps = self.mtp_steps
        horizons = len(self.horizons)
        marks = self.num_types
        observation_end = store.observation_end

        next_features = np.full((size, slots, self.num_fields), IGNORE_INDEX, dtype=np.int64)
        mtp_types = np.full((size, slots, steps), IGNORE_INDEX, dtype=np.int64)
        mtp_log_dt = np.zeros((size, slots, steps), dtype=np.float32)
        censor_mask = np.zeros(size, dtype=bool)
        censor_log_dt = np.zeros(size, dtype=np.float32)
        censor_q_log_dt = np.zeros(size, dtype=np.float32)
        between_mask = np.zeros(size, dtype=bool)
        total_log_dt = np.zeros(size, dtype=np.float32)
        elapsed_log_dt = np.zeros(size, dtype=np.float32)
        counts = np.zeros((size, horizons, marks + 1), dtype=np.float32)
        count_mask = np.zeros((size, horizons), dtype=np.float32)
        log_amount = np.zeros((size, slots), dtype=np.float32)
        has_amount = np.zeros((size, slots), dtype=np.float32)
        window_counts = np.zeros(size, dtype=np.int64)

        for row, view in enumerate(views):
            start = int(offsets[view.seq])
            stop = int(offsets[view.seq + 1])
            end_abs = start + int(view.end)
            first = max(start, end_abs - self.max_events)
            count = end_abs - first
            window_counts[row] = count
            offset = width - count
            rows = np.arange(first, end_abs)
            now = float(view.query_time)
            last = float(times[end_abs - 1])

            following = rows + 1
            has_next = following < stop
            slots_next = np.arange(offset, width)[has_next]
            next_features[row, slots_next] = features[following[has_next]]
            if end_abs < stop:
                next_features[row, width] = features[end_abs]

            for step in range(1, steps + 1):
                target = rows + step
                inside = target < stop
                event_slots = np.arange(offset, width)[inside]
                mtp_types[row, event_slots, step - 1] = types[target[inside]]
                mtp_log_dt[row, event_slots, step - 1] = np.log1p(
                    np.maximum(0.0, times[target[inside]] - times[rows[inside]])
                )
                query_target = end_abs + step - 1
                if query_target < stop:
                    mtp_types[row, width, step - 1] = types[query_target]
                    mtp_log_dt[row, width, step - 1] = math.log1p(max(0.0, float(times[query_target]) - now))

            if end_abs == stop and observation_end is not None:
                censor_mask[row] = True
                censor_log_dt[row] = math.log1p(max(0.0, float(observation_end) - last))
                censor_q_log_dt[row] = math.log1p(max(0.0, float(observation_end) - now))
            if end_abs < stop and now > last:
                between_mask[row] = True
                total_log_dt[row] = math.log1p(max(0.0, float(times[end_abs]) - last))
                elapsed_log_dt[row] = math.log1p(now - last)

            if horizons and observation_end is not None and getattr(view, "kind", EVENT_CUT) == CALENDAR:
                upcoming = np.arange(end_abs, stop)
                upcoming_times = times[upcoming]
                for index, horizon in enumerate(self.horizons):
                    if now + horizon > float(observation_end):
                        continue
                    count_mask[row, index] = 1.0
                    inside = upcoming[(upcoming_times > now) & (upcoming_times <= now + horizon)]
                    if inside.size:
                        np.add.at(counts[row, index], types[inside].astype(np.int64), 1.0)
                        counts[row, index, marks] = float(inside.size)

            if self.amount is not None:
                values = self.amount[rows].astype(np.float64)
                finite = np.isfinite(values)
                log_amount[row, offset:width] = np.where(
                    finite, np.sign(values) * np.log1p(np.abs(np.where(finite, values, 0.0))), 0.0
                )
                has_amount[row, offset:width] = finite

        batch.update(
            next_features=torch.from_numpy(next_features),
            mtp_types=torch.from_numpy(mtp_types),
            mtp_log_dt=torch.from_numpy(mtp_log_dt),
            censor_mask=torch.from_numpy(censor_mask),
            censor_log_dt=torch.from_numpy(censor_log_dt),
            censor_q_log_dt=torch.from_numpy(censor_q_log_dt),
            between_mask=torch.from_numpy(between_mask),
            total_log_dt=torch.from_numpy(total_log_dt),
            elapsed_log_dt=torch.from_numpy(elapsed_log_dt),
            counts=torch.from_numpy(counts),
            count_mask=torch.from_numpy(count_mask),
            window_counts=torch.from_numpy(window_counts),
        )
        if self.amount is not None:
            batch.update(log_amount=torch.from_numpy(log_amount), has_amount=torch.from_numpy(has_amount))
        if self.sub_collator is not None:
            keep = [row for row in range(size) if window_counts[row] >= self.repr_block + 1]
            batch["repr_rows"] = torch.tensor(keep, dtype=torch.long)
            if keep:
                sub_views = [views[row] for row in keep]
                sub = self.sub_collator(sub_views)
                for key, value in sub.items():
                    batch["repr." + key] = value
        return batch


def sub_batch(batch: Dict[str, torch.Tensor], prefix: str = "repr.") -> Dict[str, torch.Tensor]:
    return {key[len(prefix):]: value for key, value in batch.items() if key.startswith(prefix)}


def fixed_cut_views(store: EventStore, cut_time: float, indices: Optional[Sequence[int]] = None) -> List[View]:
    """One view per sequence: history strictly before ``cut_time``, "now" = cut_time.

    Used for count forecasting and long-horizon simulation: the future window
    after the cut is fully observed when cut_time + horizon <= T_obs.
    """

    chosen = range(len(store)) if indices is None else indices
    views = []
    for seq in chosen:
        start, stop = int(store.offsets[seq]), int(store.offsets[seq + 1])
        before = int(np.searchsorted(store.timestamps[start:stop], cut_time, side="left"))
        if before >= 1:
            views.append(View2(int(seq), before, float(cut_time), CALENDAR))
    return views


def between_views(store: EventStore, seed: int, max_users: Optional[int] = None, min_prefix: int = 4) -> List[View]:
    """One seeded calendar-time query per user, strictly inside an observed gap.

    "now" is uniform in calendar time over [t_{min_prefix}, t_last), so the gap that
    contains it is length-biased exactly as for a forecast requested at a random
    time, and p(gap | elapsed s) = p(gap) / S(s): direct and survival-conditioned
    forecasts are scored on the distribution both are meant to model.
    """

    rng = np.random.default_rng(seed)
    limit = len(store) if max_users is None else min(len(store), int(max_users))
    views = []
    for seq in range(limit):
        start, stop = int(store.offsets[seq]), int(store.offsets[seq + 1])
        length = stop - start
        if length <= min_prefix:
            continue
        times = store.timestamps[start:stop]
        low, high = float(times[min_prefix - 1]), float(times[-1])
        if high <= low:
            continue
        now = low + float(rng.random()) * (high - low)
        end = int(np.searchsorted(times, now, side="right"))
        if end >= length or now <= float(times[end - 1]):
            continue
        views.append(View2(seq, end, now, CALENDAR))
    return views
