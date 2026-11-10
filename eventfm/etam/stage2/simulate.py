"""Long-horizon simulation (E 6.3): roll the joint decoder forward and score sequences.

From a fixed cut ``T0 = T_obs - H`` each test history is continued S times by
sampling (y, k) from the query-slot distribution (the first step conditions on
the elapsed silence since the last real event, later steps start at the
previous generated event), attributes from the feature decoder (or, for models
without one, from the empirical attributes of that mark), until the next event
falls beyond the horizon. Scored on sequence-level statistics: count
distributions (CRPS, coverage, P(no event)), pooled mark mix (JS) and pooled
log-gap distribution (W1, KS), against a history-rate Poisson baseline.
"""

import math
import time
from typing import Dict, List

import numpy as np
import torch

from eventfm.etam.batching import View
from eventfm.etam.store import EventStore
from eventfm.etam.stage2 import evaluate as E
from eventfm.etam.stage2 import metrics as M
from eventfm.etam.stage2.batching import Collator2, fixed_cut_views

SECONDS_PER_DAY = 86_400.0
MAX_GENERATED = 200


class _Trajectories:
    """Mutable event lists, materialised as an EventStore-like object per step."""

    def __init__(self, store: EventStore, views: List[View], samples: int, amount_centres=None, amount_field: int = -1) -> None:
        self.vocab = store.vocab
        self.observation_end = None
        self.types, self.features, self.times, self.amounts = [], [], [], []
        self.origin = []
        source_amounts = getattr(store, "amount", None)
        # A generated event gets the mean amount of its sampled amount bucket (residual 0),
        # so the amount input stays in the range the model saw in training.
        self.centres = None if amount_centres is None or len(amount_centres) == 0 else np.asarray(amount_centres)
        self.amount_field = amount_field
        for view in views:
            start = int(store.offsets[view.seq])
            stop = start + int(view.end)
            for _ in range(samples):
                self.types.append(store.types[start:stop].astype(np.int16).copy())
                self.features.append(store.features[start:stop].astype(np.int16).copy())
                self.times.append(store.timestamps[start:stop].astype(np.float64).copy())
                if source_amounts is not None:
                    self.amounts.append(source_amounts[start:stop].astype(np.float32).copy())
                else:
                    self.amounts.append(np.full(stop - start, np.nan, dtype=np.float32))
                self.origin.append(stop - start)
        self.labels = np.full(len(self.types), -1, dtype=np.int8)
        self.user_ids = np.asarray([str(i) for i in range(len(self.types))])

    def materialise(self, active: np.ndarray) -> EventStore:
        lengths = np.asarray([self.types[i].shape[0] for i in active], dtype=np.int64)
        offsets = np.zeros(lengths.shape[0] + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        store = EventStore(
            offsets=offsets,
            types=np.concatenate([self.types[i] for i in active]),
            features=np.concatenate([self.features[i] for i in active]),
            timestamps=np.concatenate([self.times[i] for i in active]),
            labels=self.labels[active],
            user_ids=self.user_ids[active],
            vocab=self.vocab,
            observation_end=None,
        )
        store.amount = np.concatenate([self.amounts[i] for i in active])
        if not np.isfinite(store.amount).any():
            store.amount = None
        return store

    def append(self, index: int, mark: int, features: np.ndarray, stamp: float) -> None:
        self.types[index] = np.append(self.types[index], np.int16(mark))
        self.features[index] = np.vstack([self.features[index], features.astype(np.int16)[None]])
        self.times[index] = np.append(self.times[index], stamp)
        amount = np.nan
        if self.centres is not None and 0 <= self.amount_field < features.shape[0]:
            code = int(features[self.amount_field])
            if 0 <= code < self.centres.shape[0]:
                centre = float(self.centres[code])
                amount = float(np.sign(centre) * np.expm1(abs(centre)))
        self.amounts[index] = np.append(self.amounts[index], np.float32(amount))


def _empirical_features(store: EventStore, seed: int):
    rng = np.random.default_rng(seed)
    by_type = {}
    types = store.types.astype(np.int64)
    for mark in np.unique(types):
        rows = np.flatnonzero(types == mark)
        by_type[int(mark)] = rows[rng.choice(rows.shape[0], size=min(rows.shape[0], 5000), replace=False)]

    def sample(marks: np.ndarray) -> np.ndarray:
        out = np.zeros((marks.shape[0], store.features.shape[1]), dtype=np.int64)
        for row, mark in enumerate(marks):
            pool = by_type.get(int(mark))
            if pool is not None:
                out[row] = store.features[pool[rng.integers(pool.shape[0])]]
        return out

    return sample


@torch.no_grad()
def simulate_and_score(model, config) -> Dict[str, object]:
    from eventfm.etam.stage2.train import load_store2

    started = time.time()
    dev = next(model.parameters()).device
    store = load_store2(config, "test")
    train = load_store2(config, "train")
    horizon_days = float(config.simulate_days)
    cut = float(store.observation_end) - horizon_days * SECONDS_PER_DAY
    end = cut + horizon_days * SECONDS_PER_DAY
    views = fixed_cut_views(store, cut)[: config.simulate_users]
    samples = int(config.simulate_samples)
    model_config = getattr(model, "config", None)
    trajectories = _Trajectories(
        store, views, samples,
        amount_centres=getattr(model_config, "amount_bucket_centers", None),
        amount_field=getattr(model_config, "amount_field", -1),
    )
    total = len(trajectories.types)
    generator = torch.Generator(device=dev).manual_seed(config.seed)
    feature_sampler = _empirical_features(train, config.seed) if getattr(model, "feature_head", None) is None else None
    now = np.full(total, cut, dtype=np.float64)
    active = np.arange(total)
    done_by_cap = 0
    steps = 0
    model.eval()
    while active.size:
        steps += 1
        temp = trajectories.materialise(active)
        collator = Collator2(temp, max_events=config.max_events, count_horizons_days=())
        new_active = []
        for chunk_start in range(0, active.size, config.eval_batch_size):
            chunk = active[chunk_start : chunk_start + config.eval_batch_size]
            local = np.arange(chunk_start, chunk_start + chunk.size)
            chunk_views = [
                View(int(position), int(temp.offsets[position + 1] - temp.offsets[position]), float(now[index]))
                for position, index in zip(local, chunk)
            ]
            batch = E.to_device(collator(chunk_views), dev)
            with E.autocast(dev, config.bf16):
                out = model(batch)
            hidden = out["h"][:, -1].float()
            dist = model.decoder(hidden)
            y, marks = dist.sample(generator)
            gaps = torch.expm1(y.clamp(min=0.0, max=20.0)).double().cpu().numpy()
            if feature_sampler is None:
                logits = model.feature_head(hidden, marks, y)
                feats = torch.stack(
                    [torch.multinomial(torch.softmax(head.float(), -1), 1, generator=generator).squeeze(-1) for head in logits],
                    dim=-1,
                ).cpu().numpy()
            else:
                feats = feature_sampler(marks.cpu().numpy())
            marks_np = marks.cpu().numpy()
            for row, index in enumerate(chunk):
                stamp = now[index] + gaps[row]
                if stamp > end:
                    continue
                trajectories.append(int(index), int(marks_np[row]), feats[row], float(stamp))
                now[index] = stamp
                if trajectories.types[index].shape[0] - trajectories.origin[index] >= MAX_GENERATED:
                    done_by_cap += 1
                    continue
                new_active.append(int(index))
        active = np.asarray(new_active, dtype=np.int64)

    return _score(store, views, trajectories, samples, cut, horizon_days, done_by_cap, steps, started)


def _score(store, views, trajectories, samples, cut, horizon_days, capped, steps, started):
    num_marks = store.vocab.num_event_types
    horizons = sorted({7.0, horizon_days})
    results: Dict[str, object] = {"users": len(views), "samples": samples, "steps": steps, "capped": capped}
    real_types, real_gaps, gen_types, gen_gaps = [], [], [], []
    for horizon in horizons:
        limit = cut + horizon * SECONDS_PER_DAY
        actual = np.zeros(len(views))
        predicted = np.zeros((len(views), samples))
        recent = np.zeros(len(views))
        for row, view in enumerate(views):
            start, stop = int(store.offsets[view.seq]), int(store.offsets[view.seq + 1])
            times = store.timestamps[start:stop]
            future = (times > cut) & (times <= limit)
            actual[row] = future.sum()
            history = times[:view.end]
            recent[row] = ((history > cut - 30 * SECONDS_PER_DAY) & (history <= cut)).sum() / 30.0
            for sample in range(samples):
                index = row * samples + sample
                generated = trajectories.times[index][trajectories.origin[index]:]
                predicted[row, sample] = (generated <= limit).sum()
                if horizon == horizons[-1] and sample == 0:
                    gen_types.append(trajectories.types[index][trajectories.origin[index]:])
                    all_times = trajectories.times[index]
                    gen_gaps.append(np.diff(all_times[trajectories.origin[index] - 1:]))
            if horizon == horizons[-1]:
                real_types.append(store.types[start:stop][future])
                real_window = times[(times <= limit)]
                real_after = real_window[real_window > cut]
                if real_after.size:
                    previous = times[times <= cut]
                    anchor = previous[-1:] if previous.size else real_after[:1]
                    real_gaps.append(np.diff(np.concatenate([anchor, real_after])))
        name = "{}d".format(int(horizon))
        results["sim_" + name] = M.count_forecast_metrics(pred_samples=predicted, actual=actual)
        support = np.arange(0, 501)
        from scipy.special import gammaln

        rate = recent * horizon
        pmf = np.exp(support[None] * np.log(rate[:, None] + 1e-9) - rate[:, None] - gammaln(support[None] + 1))
        results["history_poisson_" + name] = M.count_forecast_metrics(pred_pmf=pmf, actual=actual)
        results["mean_actual_" + name] = float(actual.mean())
        results["mean_simulated_" + name] = float(predicted.mean())
    real_types = np.concatenate(real_types) if real_types else np.zeros(0, np.int64)
    gen_types = np.concatenate(gen_types) if gen_types else np.zeros(0, np.int64)
    results["type_js"] = M.categorical_js(
        np.bincount(real_types.astype(np.int64), minlength=num_marks),
        np.bincount(gen_types.astype(np.int64), minlength=num_marks),
    )
    real_gaps = np.log1p(np.maximum(np.concatenate(real_gaps), 0.0)) if real_gaps else np.zeros(0)
    gen_gaps = np.log1p(np.maximum(np.concatenate(gen_gaps), 0.0)) if gen_gaps else np.zeros(0)
    if real_gaps.size and gen_gaps.size:
        results["log_gap_distance"] = M.distribution_distance_1d(real_gaps, gen_gaps)
    results["simulate_minutes"] = (time.time() - started) / 60.0
    return results
