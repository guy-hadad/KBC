"""Numpy metrics for evaluating marked temporal point process models on bank sequences.

All functions accept numpy arrays and return plain dicts of Python floats/lists.
Empty or degenerate inputs return NaN values and never raise.
Time variable convention: y = log1p(gap in seconds).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from eventfm.training.metrics import average_precision, binary_auc_score

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    """Logit of probabilities, clipped away from 0/1."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# 1. PIT calibration
# ---------------------------------------------------------------------------


def pit_calibration(
    pit: np.ndarray,
    num_bins: int = 10,
    levels: int = 50,
) -> dict:
    """Assess calibration of probability integral transform (PIT) values in [0,1].

    pit_ks      : Kolmogorov-Smirnov statistic vs Uniform, max |ECDF − u|.
    pit_pce     : probabilistic calibration error (Bösser & Ben Taieb 2023),
                  mean_{m=1..levels} |mean(pit ≤ m/levels) − m/levels|.
    pit_hist    : list of relative frequencies in num_bins equal-width bins.
    pit_hist_l1 : sum |freq − 1/num_bins| (L1 deviation from uniform histogram).
    """
    _nan: dict = {
        "pit_ks": float("nan"),
        "pit_pce": float("nan"),
        "pit_hist": [float("nan")] * num_bins,
        "pit_hist_l1": float("nan"),
    }
    pit = np.asarray(pit, dtype=np.float64).ravel()
    if pit.size == 0:
        return _nan

    n = pit.size
    sorted_pit = np.sort(pit)

    # KS: empirical CDF at sorted values = i/n; Uniform CDF = value itself
    ecdf = np.arange(1, n + 1) / n
    pit_ks = float(np.max(np.abs(ecdf - sorted_pit)))

    # PCE
    p_m = np.arange(1, levels + 1) / levels  # (levels,)
    frac_below = np.mean(pit[:, None] <= p_m[None, :], axis=0)  # (levels,)
    pit_pce = float(np.mean(np.abs(frac_below - p_m)))

    # Histogram (num_bins equal-width bins over [0, 1])
    counts, _ = np.histogram(pit, bins=num_bins, range=(0.0, 1.0))
    freqs = counts / n
    pit_hist = freqs.tolist()
    pit_hist_l1 = float(np.sum(np.abs(freqs - 1.0 / num_bins)))

    return {
        "pit_ks": pit_ks,
        "pit_pce": pit_pce,
        "pit_hist": pit_hist,
        "pit_hist_l1": pit_hist_l1,
    }


# ---------------------------------------------------------------------------
# 2. Interval coverage
# ---------------------------------------------------------------------------


def interval_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    nominal: float,
) -> dict:
    """Fraction of target values inside the [lower, upper] prediction interval.

    Returns coverage, nominal, abs_error = |coverage − nominal|, mean_width.
    """
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if target.size == 0:
        return {
            "coverage": float("nan"),
            "nominal": float(nominal),
            "abs_error": float("nan"),
            "mean_width": float("nan"),
        }
    coverage = float(np.mean((target >= lower) & (target <= upper)))
    return {
        "coverage": coverage,
        "nominal": float(nominal),
        "abs_error": abs(coverage - float(nominal)),
        "mean_width": float(np.mean(upper - lower)),
    }


# ---------------------------------------------------------------------------
# 3. Binary calibration
# ---------------------------------------------------------------------------


def binary_calibration(
    prob: np.ndarray,
    outcome: np.ndarray,
    num_bins: int = 15,
    strategy: str = "quantile",
) -> dict:
    """Reliability diagram metrics for binary probabilistic predictions.

    ece      : expected calibration error, weighted mean |mean_prob − mean_outcome|.
    brier    : mean squared error (Brier score).
    bins     : list of [mean_prob, mean_outcome, count] per non-empty bin.
    strategy : 'quantile' (equal-mass bins, default) or 'uniform' (equal-width).
    """
    _nan: dict = {
        "ece": float("nan"),
        "brier": float("nan"),
        "mean_prob": float("nan"),
        "mean_outcome": float("nan"),
        "bins": [],
    }
    prob = np.asarray(prob, dtype=np.float64).ravel()
    outcome = np.asarray(outcome, dtype=np.float64).ravel()
    if prob.size == 0:
        return _nan

    n = prob.size
    brier = float(np.mean((prob - outcome) ** 2))
    mean_prob = float(np.mean(prob))
    mean_outcome = float(np.mean(outcome))

    if strategy == "quantile":
        order = np.argsort(prob, kind="stable")
        rank_bin = np.floor(np.arange(n) * num_bins / n).astype(int)
        bin_assignments = np.empty(n, dtype=int)
        bin_assignments[order] = rank_bin
    else:  # uniform
        edges = np.linspace(0.0, 1.0, num_bins + 1)
        bin_assignments = np.searchsorted(edges[1:-1], prob, side="right")

    bins = []
    ece_num = 0.0
    total = 0.0
    for b in range(num_bins):
        mask = bin_assignments == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mp = float(prob[mask].mean())
        mo = float(outcome[mask].mean())
        bins.append([mp, mo, cnt])
        ece_num += cnt * abs(mp - mo)
        total += cnt

    ece = float(ece_num / total) if total > 0 else float("nan")
    return {
        "ece": ece,
        "brier": brier,
        "mean_prob": mean_prob,
        "mean_outcome": mean_outcome,
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# 4. Survival calibration
# ---------------------------------------------------------------------------


def survival_calibration(
    survival_prob_by_horizon: np.ndarray,
    gaps_y: np.ndarray,
    horizons_y: np.ndarray,
) -> dict:
    """Calibration of predicted survival probabilities across multiple time horizons.

    survival_prob_by_horizon : (N, H) predicted P(gap > h | history).
    gaps_y                   : (N,) actual inter-event times in log1p(seconds).
    horizons_y               : (H,) evaluation horizons in the same scale.
    Returns per-horizon binary_calibration dicts keyed by int index, plus
    'mean_ece' and 'mean_brier' averaged across horizons.
    """
    surv = np.asarray(survival_prob_by_horizon, dtype=np.float64)
    gaps_y = np.asarray(gaps_y, dtype=np.float64).ravel()
    horizons_y = np.asarray(horizons_y, dtype=np.float64).ravel()

    result: dict = {}
    eces, briers = [], []
    for h, hz in enumerate(horizons_y):
        prob_h = surv[:, h] if surv.ndim == 2 else surv
        # Outcome = 1 if gap > horizon (event not yet seen at this horizon)
        outcome_h = (gaps_y > hz).astype(np.float64)
        cal = binary_calibration(prob_h, outcome_h)
        result[h] = cal
        if not np.isnan(cal["ece"]):
            eces.append(cal["ece"])
        if not np.isnan(cal["brier"]):
            briers.append(cal["brier"])

    result["mean_ece"] = float(np.mean(eces)) if eces else float("nan")
    result["mean_brier"] = float(np.mean(briers)) if briers else float("nan")
    return result


# ---------------------------------------------------------------------------
# 5. Mark calibration
# ---------------------------------------------------------------------------


def mark_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> dict:
    """Top-1 confidence ECE and NLL for multiclass categorical mark predictions.

    probs    : (N, K) predicted class probabilities (rows should sum to 1).
    labels   : (N,) integer class indices.
    ece      : top-1 confidence ECE, binned by max predicted probability.
    nll      : mean −log p[true_label], clipped at 1e-12.
    accuracy : fraction where argmax(probs) == label.
    brier    : multiclass Brier score, mean over N of sum_k (p_k − onehot_k)^2.
    """
    _nan: dict = {
        "ece": float("nan"),
        "nll": float("nan"),
        "accuracy": float("nan"),
        "brier": float("nan"),
    }
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if probs.ndim != 2 or probs.shape[0] == 0:
        return _nan

    n, k = probs.shape
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (preds == labels).astype(np.float64)
    accuracy = float(correct.mean())

    true_probs = probs[np.arange(n), labels]
    nll = float(np.mean(-np.log(np.maximum(true_probs, 1e-12))))

    # Multiclass Brier: mean sum_k (p_k - onehot_k)^2
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    # ECE: bin by top-1 confidence (uniform bins over [0, 1])
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.searchsorted(edges[1:-1], confidence, side="right")
    ece_num = 0.0
    total = 0.0
    for b in range(num_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece_num += cnt * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
        total += cnt
    ece = float(ece_num / total) if total > 0 else float("nan")

    return {"ece": ece, "nll": nll, "accuracy": accuracy, "brier": brier}


# ---------------------------------------------------------------------------
# 6. Ranking metrics
# ---------------------------------------------------------------------------


def ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    ks: Tuple[int, ...] = (100, 500, 1000, 2000),
) -> dict:
    """AUC, average precision, and precision/recall@k for binary ranking tasks.

    Uses eventfm.training.metrics helpers for AUC and average precision.
    Returns auc, average_precision, precision@k, recall@k (per k), positives, n.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    n = len(labels)
    pos = int(labels.sum())

    result: dict = {
        "auc": binary_auc_score(labels, scores),
        "average_precision": average_precision(labels, scores),
        "positives": pos,
        "n": n,
    }

    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cum_pos = np.cumsum(sorted_labels)

    for k in ks:
        if k > n:
            result[f"precision@{k}"] = float("nan")
            result[f"recall@{k}"] = float("nan")
            continue
        tp = int(cum_pos[k - 1])
        result[f"precision@{k}"] = float(tp / k)
        result[f"recall@{k}"] = float(tp / pos) if pos > 0 else float("nan")

    return result


# ---------------------------------------------------------------------------
# 7. Prior-shift logit adjustment
# ---------------------------------------------------------------------------


def prior_shift_logit(
    scores_logit: np.ndarray,
    train_prior: float,
    eval_prior: float,
) -> np.ndarray:
    """Adjust logit scores to account for a change in label base rate.

    Returns scores_logit + logit(eval_prior) − logit(train_prior).
    """
    scores_logit = np.asarray(scores_logit, dtype=np.float64)
    offset = np.log(eval_prior / (1.0 - eval_prior)) - np.log(
        train_prior / (1.0 - train_prior)
    )
    return scores_logit + offset


# ---------------------------------------------------------------------------
# 8. Classification report
# ---------------------------------------------------------------------------


def classification_report(
    logits_or_prob: np.ndarray,
    labels: np.ndarray,
    train_prior: Optional[float] = None,
    eval_prior: Optional[float] = None,
    ks: Tuple[int, ...] = (100, 500, 1000, 2000),
) -> dict:
    """Full binary classification report with optional prior-shift calibration.

    logits_or_prob : 1-D logit or probability vector (auto-detected by range).
    train_prior    : prevalence used during training (e.g. 0.20 after rebalancing).
    eval_prior     : true prevalence at evaluation (e.g. 0.0114 for natural rate).
    Returns auc, average_precision, precision@k, recall@k, ece, brier, ece_raw.
    """
    scores = np.asarray(logits_or_prob, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()

    # Auto-detect: if all values in [0, 1], treat as probabilities
    is_prob = bool(np.all((scores >= 0.0) & (scores <= 1.0)))
    raw_logits = _logit(scores) if is_prob else scores

    ranking = ranking_metrics(raw_logits, labels, ks=ks)

    prob_raw = _sigmoid(raw_logits)
    cal_raw = binary_calibration(prob_raw, labels.astype(np.float64))

    if train_prior is not None and eval_prior is not None:
        corrected_logits = prior_shift_logit(raw_logits, train_prior, eval_prior)
    else:
        corrected_logits = raw_logits

    prob_corrected = _sigmoid(corrected_logits)
    cal = binary_calibration(prob_corrected, labels.astype(np.float64))

    result = dict(ranking)
    result.update({"ece": cal["ece"], "brier": cal["brier"], "ece_raw": cal_raw["ece"]})
    return result


# ---------------------------------------------------------------------------
# 9. Count forecast metrics
# ---------------------------------------------------------------------------


def count_forecast_metrics(
    pred_samples: Optional[np.ndarray] = None,
    pred_mean: Optional[np.ndarray] = None,
    actual: Optional[np.ndarray] = None,
    pred_pmf: Optional[np.ndarray] = None,
    max_count: Optional[int] = None,
) -> dict:
    """Evaluate predictive distributions of non-negative integer counts per item.

    pred_samples : (N, S) integer samples from the predictive distribution.
    pred_pmf     : (N, C) probability mass over counts 0..C−1.
    pred_mean    : (N,) point-estimate mean (may be supplied independently).
    actual       : (N,) observed integer counts.
    Returns mae_mean, rmse_mean, crps, coverage80, p_zero_brier, log_score (pmf only).
    CRPS for samples uses the energy form E|X−y| − 0.5·E|X−X'| computed via sorting.
    CRPS for pmf uses sum_c (F(c) − 1[actual≤c])^2 over c = 0..C−1.
    """
    _nan: dict = {
        "mae_mean": float("nan"),
        "rmse_mean": float("nan"),
        "crps": float("nan"),
        "coverage80": float("nan"),
        "p_zero_brier": float("nan"),
        "log_score": float("nan"),
    }
    if actual is None:
        return _nan
    actual = np.asarray(actual, dtype=np.float64).ravel()
    n = actual.size
    if n == 0:
        return _nan

    result: dict = {"log_score": float("nan")}

    # --- Mean-based metrics ---
    if pred_samples is not None:
        pred_samples = np.asarray(pred_samples, dtype=np.float64)
        if pred_samples.ndim == 1:
            pred_samples = pred_samples.reshape(n, -1)
        mean_pred = pred_samples.mean(axis=1)
    elif pred_pmf is not None:
        pmf_arr = np.asarray(pred_pmf, dtype=np.float64)
        c_vals = np.arange(pmf_arr.shape[1], dtype=np.float64)
        mean_pred = pmf_arr @ c_vals
    elif pred_mean is not None:
        mean_pred = np.asarray(pred_mean, dtype=np.float64).ravel()
    else:
        mean_pred = None

    if mean_pred is not None:
        err = mean_pred - actual
        result["mae_mean"] = float(np.mean(np.abs(err)))
        result["rmse_mean"] = float(np.sqrt(np.mean(err ** 2)))
    else:
        result["mae_mean"] = float("nan")
        result["rmse_mean"] = float("nan")

    # --- Distribution-based metrics ---
    if pred_samples is not None:
        S = pred_samples.shape[1]
        # Energy CRPS: E|X−y| − 0.5·E|X−X'|
        # E|X−X'| = (2/S^2)·sum_i x_{(i)}·(2i − S + 1) for 0-indexed sorted x_{(i)}
        e_abs_diff = np.mean(np.abs(pred_samples - actual[:, None]), axis=1)  # (N,)
        sorted_s = np.sort(pred_samples, axis=1)  # (N, S)
        w = 2.0 * np.arange(S) - S + 1  # (S,), so that 0.5·E|X−X'| = dot(sorted,w)/S^2
        e_abs_pair_half = (sorted_s @ w) / (S * S)  # (N,)  equals 0.5·E|X−X'|
        result["crps"] = float(np.mean(e_abs_diff - e_abs_pair_half))

        # Coverage of central 80% interval [q10, q90] using inclusive integer quantiles
        p10 = np.percentile(pred_samples, 10, axis=1, method="lower")
        p90 = np.percentile(pred_samples, 90, axis=1, method="higher")
        result["coverage80"] = float(np.mean((actual >= p10) & (actual <= p90)))

        # Brier for P(N=0)
        p_zero = np.mean(pred_samples == 0, axis=1)
        result["p_zero_brier"] = float(np.mean((p_zero - (actual == 0).astype(float)) ** 2))

    elif pred_pmf is not None:
        C = pmf_arr.shape[1]
        row_sums = pmf_arr.sum(axis=1, keepdims=True)
        pmf_norm = pmf_arr / np.maximum(row_sums, 1e-12)
        F = np.cumsum(pmf_norm, axis=1)  # (N, C), CDF

        # CRPS: sum_c (F(c) − 1[actual ≤ c])^2 for c = 0..C−1
        c_idx = np.arange(C)
        indicator = (actual[:, None] <= c_idx[None, :]).astype(np.float64)  # (N, C)
        result["crps"] = float(np.mean(np.sum((F - indicator) ** 2, axis=1)))

        # Coverage80: smallest c with F[i,c] ≥ threshold
        q10 = np.argmax(F >= 0.1, axis=1).astype(np.float64)
        q90 = np.argmax(F >= 0.9, axis=1).astype(np.float64)
        last = F[:, -1]
        q10 = np.where(last < 0.1, C - 1, q10)
        q90 = np.where(last < 0.9, C - 1, q90)
        result["coverage80"] = float(np.mean((actual >= q10) & (actual <= q90)))

        # Brier for P(N=0)
        p_zero = pmf_norm[:, 0]
        result["p_zero_brier"] = float(np.mean((p_zero - (actual == 0).astype(float)) ** 2))

        # Log score
        actual_int = actual.astype(np.int64)
        valid = (actual_int >= 0) & (actual_int < C)
        idx_clipped = np.clip(actual_int, 0, C - 1)
        log_probs = np.where(
            valid,
            -np.log(np.maximum(pmf_norm[np.arange(n), idx_clipped], 1e-12)),
            -np.log(1e-12),
        )
        result["log_score"] = float(np.mean(log_probs))
    else:
        result["crps"] = float("nan")
        result["coverage80"] = float("nan")
        result["p_zero_brier"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# 10. 1-D distribution distances
# ---------------------------------------------------------------------------


def distribution_distance_1d(a: np.ndarray, b: np.ndarray) -> dict:
    """Wasserstein-1 distance and two-sample KS statistic between empirical samples.

    w1 : integral |F_a(x) − F_b(x)| dx, computed exactly on the merged sorted grid.
    ks : max |F_a(x) − F_b(x)| over all x.
    """
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        return {"w1": float("nan"), "ks": float("nan")}

    n, m = a.size, b.size
    all_vals = np.sort(np.concatenate([a, b]))

    # Right-continuous ECDFs at each merged value: F(x) = P(X ≤ x)
    fa = np.searchsorted(a, all_vals, side="right") / n
    fb = np.searchsorted(b, all_vals, side="right") / m

    # W1 = sum of gap * |F_a − F_b| over consecutive intervals
    # Between all_vals[i] and all_vals[i+1] both CDFs are constant at fa[i], fb[i]
    gaps = np.diff(all_vals)
    w1 = float(np.dot(gaps, np.abs(fa[:-1] - fb[:-1])))
    ks = float(np.max(np.abs(fa - fb)))

    return {"w1": w1, "ks": ks}


# ---------------------------------------------------------------------------
# 11. Jensen–Shannon divergence
# ---------------------------------------------------------------------------


def categorical_js(p_counts: np.ndarray, q_counts: np.ndarray) -> dict:
    """Jensen–Shannon divergence (natural log, in nats) between two discrete distributions.

    Both vectors are normalized internally; zeros are allowed; pads shorter to equal length.
    """
    p = np.asarray(p_counts, dtype=np.float64).ravel()
    q = np.asarray(q_counts, dtype=np.float64).ravel()
    # Pad to equal length
    ln = max(len(p), len(q))
    p = np.pad(p, (0, ln - len(p)))
    q = np.pad(q, (0, ln - len(q)))

    ps, qs = p.sum(), q.sum()
    if ps == 0.0 or qs == 0.0:
        return {"js": float("nan")}

    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        # KL(a || b); convention 0·log(0/0) = 0
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return {"js": float(js)}


# ---------------------------------------------------------------------------
# 12. Stratified mean
# ---------------------------------------------------------------------------


def stratified_mean(
    values: np.ndarray,
    strata: np.ndarray,
    names: Optional[Sequence] = None,
) -> dict:
    """Mean of a per-example metric within each stratum.

    Returns dict mapping stratum label (or names[i]) → {'mean': float, 'n': int}.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    strata = np.asarray(strata).ravel()
    unique_strata = np.unique(strata)

    name_map = None
    if names is not None:
        name_map = {s: n for s, n in zip(unique_strata, names)}

    result: dict = {}
    for s in unique_strata:
        mask = strata == s
        key = name_map[s] if name_map is not None else str(s)
        result[key] = {"mean": float(values[mask].mean()), "n": int(mask.sum())}
    return result


# ---------------------------------------------------------------------------
# 13. Quantile bins
# ---------------------------------------------------------------------------


def quantile_bins(x: np.ndarray, num_bins: int) -> np.ndarray:
    """Assign integer stratum ids in [0, num_bins) by quantile (equal-mass bins).

    Ties are handled via searchsorted on unique quantile edges; bins may differ
    in size when there are repeated values.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.array([], dtype=np.int64)

    # Compute num_bins-1 interior quantile thresholds
    quantile_probs = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    if quantile_probs.size == 0:
        return np.zeros(x.size, dtype=np.int64)

    edges = np.quantile(x, quantile_probs)
    unique_edges = np.unique(edges)
    # bin id = number of unique edges that are strictly less than x
    bin_ids = np.searchsorted(unique_edges, x, side="right").astype(np.int64)
    return bin_ids
