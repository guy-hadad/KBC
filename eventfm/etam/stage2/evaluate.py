"""Stage 2 evaluation suite (works for ETAM-2 and, through an adapter, Stage 1 checkpoints).

All TPP numbers are on the benchmark's own replayed prefixes (seed + 1), "now"
at the prefix's last event, read at the query (USR) slot, so the headline
metrics stay comparable with Stage 1 and every baseline. Likelihoods are in
log1p-second units; ``mean_target_y`` converts a time NLL to seconds units
(NLL_seconds = NLL_y + mean_target_y).
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eventfm.etam.batching import ListDataset, View, worker_init
from eventfm.etam.stage2 import metrics as M
from eventfm.etam.stage2.batching import Collator2, between_views, fixed_cut_views
from eventfm.etam.stage2.distributions import IndependentJoint, LogNormalMixture
from eventfm.training.metrics import macro_f1

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
SURVIVAL_HORIZONS = {"1h": math.log1p(3600.0), "1d": math.log1p(86400.0), "7d": math.log1p(604800.0)}


def loader(store, views, collator, batch_size: int, workers: int = 4):
    return torch.utils.data.DataLoader(
        ListDataset(views),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
        worker_init_fn=worker_init,
        pin_memory=torch.cuda.is_available(),
    )


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast(device, enabled=True):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


class Stage1Adapter(nn.Module):
    """Expose a Stage 1 EtamModel through the Stage 2 interface (independent decoder at USR)."""

    def __init__(self, model) -> None:
        super().__init__()
        self.inner = model
        self.config = model.config
        self.has_unimodal = False
        self.mtp = None
        self.counts = None
        self.feature_head = None
        self.decoder = _Stage1Decoder(model)

    def forward(self, batch, record_gates: bool = False):
        out = self.inner(batch)
        return {"h": out["event_out"], "user": out["user"], "user_joint": out["user"], "aux": {}, "gates": []}

    def classify(self, user):
        return self.inner.classify(user)


class _Stage1Decoder:
    kind = "independent"

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, states):
        logits, means, log_scales = self.model.time_head(states.float())
        return IndependentJoint(LogNormalMixture(logits, means, log_scales), self.model.type_head(states.float()))

    def cheap_mark_logits(self, states):
        return self.model.type_head(states.float())


@torch.no_grad()
def collect_next_event(model, store, views, collator, batch_size, device, type_frequency=None, bf16=True):
    """One pass over fixed views; per-view arrays for every next-event metric."""

    model.eval()
    parts: Dict[str, List[np.ndarray]] = {}

    def push(name, value):
        parts.setdefault(name, []).append(value.detach().float().cpu().numpy())

    quantiles = torch.tensor(QUANTILES)
    for batch in loader(store, views, collator, batch_size):
        batch = to_device(batch, device)
        with autocast(device, bf16):
            out = model(batch)
        hidden = out["h"].float()
        y = batch["next_log_dt"][:, -1].float()
        k = batch["next_type"][:, -1]
        dist = model.decoder(hidden[:, -1])
        marks = dist.mark_log_probs()
        joint = dist.log_prob(y, k)
        time = dist.time_log_prob(y)
        mark_true = marks.gather(-1, k.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        push("mark_log_probs", marks)
        push("target_type", k)
        push("target_y", y)
        push("time_mean", dist.time_mean())
        push("joint_lp", joint)
        push("time_lp", time)
        push("type_lp", mark_true)
        push("pmi", joint - time - mark_true)
        push("pit", dist.time_cdf(y))
        push(
            "quantiles",
            torch.stack([dist.time_quantile(float(level)) for level in quantiles], dim=-1),
        )
        push(
            "survival",
            torch.stack(
                [torch.exp(dist.time_log_survival(torch.full_like(y, value))) for value in SURVIVAL_HORIZONS.values()],
                dim=-1,
            ),
        )
        push("oracle_pred", dist.mark_log_probs_given_time(y).argmax(-1))
        last = model.decoder(hidden[:, -2])
        push("last_joint_lp", last.log_prob(y, k))
        push("last_pred", last.mark_log_probs().argmax(-1))
        push("cheap_pred", model.decoder.cheap_mark_logits(hidden[:, -1]).argmax(-1))
        push("cheap_type_lp", torch.log_softmax(model.decoder.cheap_mark_logits(hidden[:, -1]), -1).gather(-1, k.clamp(min=0).unsqueeze(-1)).squeeze(-1))
        if getattr(model, "has_unimodal", False):
            uni_logits = model.unimodal_event(out["pre_event"][:, -2].float())
            push("uni_type_lp", torch.log_softmax(uni_logits, -1).gather(-1, k.clamp(min=0).unsqueeze(-1)).squeeze(-1))
            push("uni_pred", uni_logits.argmax(-1))
            push("uni_time_lp", model.unimodal_time(out["pre_time"][:, -2].float()).log_prob(y))
        if getattr(model, "mtp", None) is not None:
            drafts = model.mtp.states(hidden[:, -2])
            for step, state in enumerate(drafts):
                lag = step + 2
                push("mtp_pred_{}".format(lag), model.decoder.mark_proj(state).argmax(-1))
                push("mtp_target_{}".format(lag), batch["mtp_types"][:, -2, lag - 1])
                if model.mtp.with_time:
                    target = batch["mtp_log_dt"][:, -2, lag - 1]
                    push("mtp_time_lp_{}".format(lag), model.mtp.time_dist(hidden[:, -2], step).log_prob(target))
        span_days = (batch["abs_days"][:, -1] - _first_valid_days(batch)).float().clamp(min=1.0)
        push("activity", batch["window_counts"].float() / span_days)
        push("target_month", _target_month(batch))
        if out.get("gates"):
            pass
    arrays = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    return arrays


def _first_valid_days(batch):
    valid = batch["slot_valid"]
    first = valid.float().argmax(dim=1)
    return batch["abs_days"].gather(1, first.unsqueeze(-1)).squeeze(-1)


def _target_month(batch):
    # Month (0-11) of the next event: now + gap.
    from eventfm.etam.batching import ORIGIN_SECONDS

    seconds = batch["abs_days"][:, -1].double() * 86400.0 + ORIGIN_SECONDS
    seconds = seconds + torch.expm1(batch["next_log_dt"][:, -1].double())
    stamp = seconds.cpu().numpy().astype("int64").astype("datetime64[s]")
    months = (stamp.astype("datetime64[M]").astype("int64") % 12).astype(np.int64)
    return torch.from_numpy(months)


def next_event_metrics(arrays: Dict[str, np.ndarray], type_frequency: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Benchmark metrics plus likelihood, calibration and dependency diagnostics."""

    target = arrays["target_type"].astype(np.int64)
    y = arrays["target_y"].astype(np.float64)
    marks = arrays["mark_log_probs"].astype(np.float64)
    predicted = marks.argmax(-1)
    num_marks = marks.shape[-1]
    residual = arrays["time_mean"] - y
    results: Dict[str, object] = {
        "n": int(target.shape[0]),
        "next_type_accuracy": float((predicted == target).mean()),
        "next_type_macro_f1": macro_f1(target, predicted, num_marks),
        "delta_log_rmse": float(np.sqrt(np.mean(residual**2))),
        "delta_log_mae": float(np.mean(np.abs(residual))),
        "joint_nll": float(-arrays["joint_lp"].mean()),
        "time_nll": float(-arrays["time_lp"].mean()),
        "type_nll": float(-arrays["type_lp"].mean()),
        "mean_target_y": float(y.mean()),
        "pmi_mean": float(arrays["pmi"].mean()),
        "oracle_time_accuracy": float((arrays["oracle_pred"] == target).mean()),
        "last_event_joint_nll": float(-arrays["last_joint_lp"].mean()),
        "last_event_accuracy": float((arrays["last_pred"] == target).mean()),
        "cheap_accuracy": float((arrays["cheap_pred"] == target).mean()),
        "cheap_type_nll": float(-arrays["cheap_type_lp"].mean()),
    }
    median = arrays["quantiles"][:, QUANTILES.index(0.5)]
    results["delta_log_mae_median"] = float(np.mean(np.abs(median - y)))
    results.update(M.pit_calibration(arrays["pit"]))
    for low, high, nominal in ((0.25, 0.75, 0.5), (0.1, 0.9, 0.8), (0.05, 0.95, 0.9)):
        cover = M.interval_coverage(
            arrays["quantiles"][:, QUANTILES.index(low)], arrays["quantiles"][:, QUANTILES.index(high)], y, nominal
        )
        results["coverage{}".format(int(nominal * 100))] = cover["coverage"]
        results["interval_width{}".format(int(nominal * 100))] = cover["mean_width"]
    survival = M.survival_calibration(arrays["survival"], y, list(SURVIVAL_HORIZONS.values()))
    results["survival_ece"] = survival.get("mean_ece")
    results["survival_brier"] = survival.get("mean_brier")
    for index, name in enumerate(SURVIVAL_HORIZONS):
        entry = survival.get(index) or survival.get(str(index)) or {}
        results["survival_ece_" + name] = entry.get("ece")
    probabilities = np.exp(marks - marks.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    mark_cal = M.mark_calibration(probabilities, target)
    results["mark_ece"] = mark_cal["ece"]
    results["mark_brier"] = mark_cal["brier"]
    # PMI per mark: E[log p(k|y,H) - log p(k|H)] over events of that mark (top marks by count).
    counts = np.bincount(target, minlength=num_marks)
    top = np.argsort(-counts)[:10]
    results["pmi_by_type"] = {
        int(mark): {"n": int(counts[mark]), "pmi": float(arrays["pmi"][target == mark].mean())}
        for mark in top
        if counts[mark] > 0
    }
    if "uni_type_lp" in arrays:
        results["unimodal_type_nll"] = float(-arrays["uni_type_lp"].mean())
        results["unimodal_accuracy"] = float((arrays["uni_pred"] == target).mean())
        results["unimodal_time_nll"] = float(-arrays["uni_time_lp"].mean())
    for key in sorted(arrays):
        if key.startswith("mtp_pred_"):
            lag = key.split("_")[-1]
            goal = arrays["mtp_target_" + lag]
            valid = goal >= 0
            results["mtp_accuracy_{}".format(lag)] = float((arrays[key][valid] == goal[valid]).mean()) if valid.any() else None
            if "mtp_time_lp_" + lag in arrays:
                results["mtp_time_nll_{}".format(lag)] = float(-arrays["mtp_time_lp_" + lag][valid].mean())
    # Strata: activity terciles, target month, target-type frequency terciles.
    strata = {}
    activity = M.quantile_bins(arrays["activity"], 3)
    strata["activity"] = _strata_table(arrays, activity, predicted, target)
    strata["month"] = _strata_table(arrays, arrays["target_month"].astype(np.int64), predicted, target)
    if type_frequency is not None:
        frequency = type_frequency[target]
        strata["type_frequency"] = _strata_table(arrays, M.quantile_bins(frequency, 3), predicted, target)
    results["strata"] = strata
    return results


def _strata_table(arrays, strata, predicted, target):
    table = {}
    for value in np.unique(strata):
        rows = strata == value
        table[int(value)] = {
            "n": int(rows.sum()),
            "joint_nll": float(-arrays["joint_lp"][rows].mean()),
            "time_nll": float(-arrays["time_lp"][rows].mean()),
            "type_nll": float(-arrays["type_lp"][rows].mean()),
            "accuracy": float((predicted[rows] == target[rows]).mean()),
        }
    return table


@torch.no_grad()
def elapsed_silence_metrics(model, store, collator, batch_size, device, seed, max_users=None, bf16=True):
    """"Now" strictly inside the gap: direct query-slot forecast vs survival-conditioned
    forecast from the last event's distribution, both as residual densities in y units."""

    views = between_views(store, seed=seed, max_users=max_users)
    model.eval()
    direct, survival, direct_time, survival_time = [], [], [], []
    for batch in loader(store, views, collator, batch_size):
        batch = to_device(batch, device)
        with autocast(device, bf16):
            out = model(batch)
        hidden = out["h"].float()
        k = batch["next_type"][:, -1]
        y_residual = batch["next_log_dt"][:, -1].float()
        total, elapsed = batch["total_log_dt"].float(), batch["elapsed_log_dt"].float()
        query = model.decoder(hidden[:, -1])
        last = model.decoder(hidden[:, -2])
        direct.append(query.log_prob(y_residual, k).cpu())
        direct_time.append(query.time_log_prob(y_residual).cpu())
        survival.append(last.residual_log_prob(total, elapsed, k, y_residual).cpu())
        conditioned = last.time_log_prob(total) - last.time_log_survival(elapsed) - total + y_residual
        survival_time.append(conditioned.cpu())
    cat = lambda parts: torch.cat(parts).numpy().astype(np.float64)  # noqa: E731
    direct, survival = cat(direct), cat(survival)
    return {
        "n": int(direct.shape[0]),
        "direct_joint_nll": float(-direct.mean()),
        "survival_joint_nll": float(-survival.mean()),
        "direct_time_nll": float(-cat(direct_time).mean()),
        "survival_time_nll": float(-cat(survival_time).mean()),
        "best_of_two_joint_nll": float(-np.logaddexp(direct, survival).mean() + math.log(2.0)),
    }


@torch.no_grad()
def count_metrics(model, store, collator, batch_size, device, cut_time, horizons_days, max_users=None, bf16=True):
    """Count forecasts at a fixed cut (history < cut, window after it fully observed)."""

    from eventfm.etam.stage2.heads import count_log_prob

    indices = range(min(len(store), max_users)) if max_users else None
    views = fixed_cut_views(store, cut_time, indices)
    model.eval()
    results: Dict[str, object] = {"n": len(views)}
    actual, pmfs, means, rates = [], [], [], []
    support = torch.arange(0, 301, dtype=torch.float32, device=device)
    for batch in loader(store, views, collator, batch_size):
        batch = to_device(batch, device)
        with autocast(device, bf16):
            out = model(batch)
        counts = batch["counts"]  # (B, H, K + 1)
        actual.append(counts[..., -1].cpu())
        # History-rate Poisson baseline: events in the last 30 days of history / 30 per day.
        days = batch["abs_days"]
        recent = ((days[:, -1:] - days) <= 30.0) & batch["slot_valid"] & ~batch["is_query"]
        rates.append((recent.sum(-1).float() / 30.0).cpu())
        if model.counts is not None and model.counts.kind in ("poisson", "negbin"):
            params = model.counts(out["user"])
            total = {key: value[..., -1] for key, value in params.items()}  # (B, H)
            expanded = {key: value.unsqueeze(-1) for key, value in total.items()}
            log_pmf = count_log_prob(expanded, support.view(1, 1, -1))
            pmfs.append(torch.exp(log_pmf).cpu())
            means.append(total["mean"].cpu())
    actual = torch.cat(actual).numpy()
    rates = torch.cat(rates).numpy()
    for index, days in enumerate(horizons_days):
        name = "{}d".format(int(days))
        target = actual[:, index]
        baseline = rates * float(days)
        # Poisson pmf of the history-rate baseline on the same support.
        support_np = np.arange(0, 301)
        from scipy.special import gammaln  # noqa: WPS433

        base_pmf = np.exp(support_np[None, :] * np.log(baseline[:, None] + 1e-9) - baseline[:, None] - gammaln(support_np[None, :] + 1))
        results["history_poisson_" + name] = M.count_forecast_metrics(pred_pmf=base_pmf, actual=target)
        if pmfs:
            pmf = torch.cat(pmfs).numpy()[:, index]
            results["head_" + name] = M.count_forecast_metrics(pred_pmf=pmf, actual=target)
        results["mean_actual_" + name] = float(target.mean())
    return results


@torch.no_grad()
def embeddings(model, store, views, collator, batch_size, device, bf16=True) -> Dict[str, np.ndarray]:
    model.eval()
    parts: Dict[str, List[np.ndarray]] = {}
    for batch in loader(store, views, collator, batch_size):
        batch = to_device(batch, device)
        with autocast(device, bf16):
            out = model(batch)
        parts.setdefault("user", []).append(out["user"].float().cpu().numpy().astype(np.float16))
        parts.setdefault("joint", []).append(out["user_joint"].float().cpu().numpy().astype(np.float16))
        if "h_event" in out:
            parts.setdefault("event", []).append(out["h_event"][:, -1].float().cpu().numpy().astype(np.float16))
            parts.setdefault("time", []).append(out["h_time"][:, -1].float().cpu().numpy().astype(np.float16))
        parts.setdefault("labels", []).append(batch["labels"].cpu().numpy())
    return {name: np.concatenate(values) for name, values in parts.items()}


def efficiency(model, batch, device, repeats: int = 20) -> Dict[str, float]:
    """Parameters, forward FLOPs per sequence and median forward latency (ms per batch)."""

    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    batch = to_device(batch, device)
    result = {"parameters": float(sum(p.numel() for p in model.parameters())), "batch": float(batch["slot_valid"].shape[0])}
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(batch)
        result["gflops_per_sequence"] = counter.get_total_flops() / 1e9 / batch["slot_valid"].shape[0]
    except Exception as error:  # noqa: BLE001
        result["flops_error"] = str(error)[:200]
    if device.type == "cuda":
        times = []
        with torch.no_grad(), autocast(device):
            for _ in range(5):
                model(batch)
            for _ in range(repeats):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(batch)
                stop.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(stop))
        result["latency_ms"] = float(np.median(times))
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def type_frequency(store) -> np.ndarray:
    counts = np.bincount(store.types.astype(np.int64), minlength=store.vocab.num_event_types).astype(np.float64)
    return counts / max(1.0, counts.sum())
