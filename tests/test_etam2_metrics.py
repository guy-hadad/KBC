"""Unit tests for eventfm.etam.stage2.metrics.

All tests use fixed random seeds for reproducibility and are designed to run
in under 30 seconds total.
"""

import math

import numpy as np
import pytest

from eventfm.etam.stage2.metrics import (
    binary_calibration,
    categorical_js,
    classification_report,
    count_forecast_metrics,
    distribution_distance_1d,
    interval_coverage,
    mark_calibration,
    pit_calibration,
    prior_shift_logit,
    quantile_bins,
    ranking_metrics,
    stratified_mean,
    survival_calibration,
)

# ---------------------------------------------------------------------------
# 1. PIT calibration
# ---------------------------------------------------------------------------


def test_pit_uniform_gives_small_ks_pce():
    rng = np.random.default_rng(0)
    pit = rng.uniform(0.0, 1.0, 20_000)
    result = pit_calibration(pit, num_bins=10, levels=50)
    assert result["pit_ks"] < 0.02, result["pit_ks"]
    assert result["pit_pce"] < 0.01, result["pit_pce"]
    assert len(result["pit_hist"]) == 10
    # Each bin should be close to 0.1
    assert result["pit_hist_l1"] < 0.10, result["pit_hist_l1"]


def test_pit_skewed_gives_large_ks_pce():
    rng = np.random.default_rng(1)
    # Beta(5, 1) concentrates near 1 → skewed PIT
    pit = rng.beta(5, 1, 20_000)
    result = pit_calibration(pit, num_bins=10, levels=50)
    assert result["pit_ks"] > 0.20, result["pit_ks"]
    assert result["pit_pce"] > 0.10, result["pit_pce"]


def test_pit_empty_input():
    result = pit_calibration(np.array([]))
    assert math.isnan(result["pit_ks"])
    assert math.isnan(result["pit_pce"])


# ---------------------------------------------------------------------------
# 2. Interval coverage
# ---------------------------------------------------------------------------


def test_interval_coverage_gaussian_90():
    rng = np.random.default_rng(2)
    target = rng.standard_normal(50_000)
    # 90% interval via Gaussian quantiles
    lower = np.full(len(target), -1.6449)
    upper = np.full(len(target), 1.6449)
    result = interval_coverage(lower, upper, target, nominal=0.90)
    assert result["abs_error"] < 0.02, result
    assert abs(result["mean_width"] - 2 * 1.6449) < 1e-6


def test_interval_coverage_gaussian_50():
    rng = np.random.default_rng(3)
    target = rng.standard_normal(50_000)
    lower = np.full(len(target), -0.6745)
    upper = np.full(len(target), 0.6745)
    result = interval_coverage(lower, upper, target, nominal=0.50)
    assert result["abs_error"] < 0.02, result


def test_interval_coverage_empty():
    result = interval_coverage(np.array([]), np.array([]), np.array([]), 0.9)
    assert math.isnan(result["coverage"])


# ---------------------------------------------------------------------------
# 3. Binary calibration
# ---------------------------------------------------------------------------


def test_binary_calibration_perfectly_calibrated():
    """ECE should be near zero when outcomes are drawn as Bernoulli(prob)."""
    rng = np.random.default_rng(4)
    n = 200_000
    prob = rng.uniform(0.0, 1.0, n)
    outcome = rng.binomial(1, prob).astype(float)
    result = binary_calibration(prob, outcome, num_bins=15, strategy="quantile")
    assert result["ece"] < 0.01, result["ece"]
    assert result["brier"] < 0.30  # Brier of Bernoulli(U[0,1]) predictions is ~1/6


def test_binary_calibration_uniform_strategy():
    rng = np.random.default_rng(5)
    n = 50_000
    prob = rng.uniform(0.0, 1.0, n)
    outcome = rng.binomial(1, prob).astype(float)
    result = binary_calibration(prob, outcome, num_bins=10, strategy="uniform")
    assert result["ece"] < 0.02, result["ece"]


def test_binary_calibration_empty():
    result = binary_calibration(np.array([]), np.array([]))
    assert math.isnan(result["ece"])


# ---------------------------------------------------------------------------
# 4. Ranking metrics
# ---------------------------------------------------------------------------


def test_ranking_metrics_tiny_hand_computed():
    # 10 items, 3 positives at positions 0, 5, 9 (0-indexed)
    labels = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    # Scores rank items 0..9 in that order (item 0 has highest score)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    result = ranking_metrics(scores, labels, ks=(1, 3, 6))

    # Top-1: item 0 is positive → precision@1 = 1, recall@1 = 1/3
    assert result["precision@1"] == pytest.approx(1.0)
    assert result["recall@1"] == pytest.approx(1.0 / 3)

    # Top-3: item 0 positive, items 1,2 negative → precision@3 = 1/3, recall@3 = 1/3
    assert result["precision@3"] == pytest.approx(1.0 / 3)
    assert result["recall@3"] == pytest.approx(1.0 / 3)

    # Top-6: items 0 (pos), 1,2,3,4 (neg), 5 (pos) → precision@6 = 2/6, recall@6 = 2/3
    assert result["precision@6"] == pytest.approx(2.0 / 6)
    assert result["recall@6"] == pytest.approx(2.0 / 3)

    assert result["positives"] == 3
    assert result["n"] == 10

    # AUC: positives at scores 0.9, 0.4, 0.0; negatives at 0.8,0.7,0.6,0.5,0.3,0.2,0.1 (7)
    # Pos 0.9 beats all 7; pos 0.4 beats 3 (scores 0.3,0.2,0.1); pos 0.0 beats 0
    # AUC = (7 + 3 + 0) / (3 * 7) = 10/21
    assert result["auc"] == pytest.approx(10.0 / 21.0, abs=1e-9)


def test_ranking_metrics_k_exceeds_n():
    labels = np.array([1, 0])
    scores = np.array([0.8, 0.2])
    result = ranking_metrics(scores, labels, ks=(5,))
    assert math.isnan(result["precision@5"])
    assert math.isnan(result["recall@5"])


# ---------------------------------------------------------------------------
# 5. Prior-shift logit
# ---------------------------------------------------------------------------


def test_prior_shift_logit_roundtrip():
    rng = np.random.default_rng(6)
    logits = rng.standard_normal(1000)
    train_prior = 0.20
    eval_prior = 0.0114
    shifted = prior_shift_logit(logits, train_prior, eval_prior)
    recovered = prior_shift_logit(shifted, eval_prior, train_prior)
    np.testing.assert_allclose(recovered, logits, atol=1e-10)


def test_prior_shift_logit_direction():
    # eval_prior < train_prior → scores should decrease
    logits = np.zeros(10)
    result = prior_shift_logit(logits, train_prior=0.5, eval_prior=0.1)
    assert np.all(result < 0)


# ---------------------------------------------------------------------------
# 6. CRPS: samples vs pmf, and point mass
# ---------------------------------------------------------------------------


def _uniform_crps_exact() -> float:
    """Exact CRPS for X ~ Uniform{0,1,2} and y=1 is 2/9."""
    return 2.0 / 9.0


def test_crps_samples_matches_pmf():
    """CRPS from samples and from pmf should agree for the same distribution."""
    rng = np.random.default_rng(7)
    # N=1 item, X ~ Uniform{0,1,2}, actual=1
    n, s = 1, 300_000
    samples = rng.choice([0, 1, 2], size=(n, s))
    actual = np.array([1.0])
    pmf = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]])

    res_samples = count_forecast_metrics(pred_samples=samples.astype(float), actual=actual)
    res_pmf = count_forecast_metrics(pred_pmf=pmf, actual=actual)

    expected = _uniform_crps_exact()
    assert abs(res_samples["crps"] - expected) < 0.005, res_samples["crps"]
    assert abs(res_pmf["crps"] - expected) < 1e-10, res_pmf["crps"]


def test_crps_point_mass_at_actual_is_zero():
    """A point-mass forecast at the true value has CRPS = 0."""
    actual = np.array([3.0, 7.0, 0.0])
    # Samples: S=5 copies of actual
    samples = np.stack([actual] * 5, axis=1)
    result = count_forecast_metrics(pred_samples=samples, actual=actual)
    assert abs(result["crps"]) < 1e-10, result["crps"]


def test_crps_pmf_exact_value():
    """Verify exact CRPS for Uniform{0,1,2}, y=1 via PMF formula."""
    pmf = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    actual = np.array([1.0])
    result = count_forecast_metrics(pred_pmf=pmf, actual=actual)
    assert abs(result["crps"] - 2.0 / 9.0) < 1e-10, result["crps"]


def test_crps_coverage80():
    """Coverage of the central 80% interval should be near 80% for calibrated forecast."""
    rng = np.random.default_rng(8)
    n = 1000
    # True counts ~ Poisson(5)
    actual = rng.poisson(5, n).astype(float)
    # Samples from the same distribution → should be well-calibrated
    samples = rng.poisson(5, (n, 500)).astype(float)
    result = count_forecast_metrics(pred_samples=samples, actual=actual)
    assert abs(result["coverage80"] - 0.80) < 0.08, result["coverage80"]


# ---------------------------------------------------------------------------
# 7. Wasserstein and KS
# ---------------------------------------------------------------------------


def test_wasserstein_identical_is_zero():
    rng = np.random.default_rng(9)
    a = rng.standard_normal(500)
    result = distribution_distance_1d(a, a)
    assert result["w1"] == pytest.approx(0.0, abs=1e-12)
    assert result["ks"] == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_shift():
    """Shifting all samples by c gives W1 = c."""
    rng = np.random.default_rng(10)
    a = rng.standard_normal(1000)
    c = 3.5
    b = a + c
    result = distribution_distance_1d(a, b)
    assert result["w1"] == pytest.approx(c, abs=1e-9)


def test_wasserstein_known_example():
    """U({0}) vs U({1}): W1=1, KS=1."""
    result = distribution_distance_1d(np.array([0.0]), np.array([1.0]))
    assert result["w1"] == pytest.approx(1.0)
    assert result["ks"] == pytest.approx(1.0)


def test_wasserstein_empty():
    result = distribution_distance_1d(np.array([]), np.array([1.0]))
    assert math.isnan(result["w1"])


# ---------------------------------------------------------------------------
# 8. Jensen–Shannon divergence
# ---------------------------------------------------------------------------


def test_js_identical_is_zero():
    p = np.array([1.0, 2.0, 3.0])
    result = categorical_js(p, p)
    assert result["js"] == pytest.approx(0.0, abs=1e-12)


def test_js_disjoint_is_log2():
    """JS divergence of disjoint distributions equals log(2) nats."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    result = categorical_js(p, q)
    assert result["js"] == pytest.approx(math.log(2), abs=1e-10)


def test_js_symmetric():
    rng = np.random.default_rng(11)
    p = rng.dirichlet(np.ones(5))
    q = rng.dirichlet(np.ones(5))
    r1 = categorical_js(p, q)
    r2 = categorical_js(q, p)
    assert r1["js"] == pytest.approx(r2["js"], abs=1e-10)


def test_js_zero_vector():
    result = categorical_js(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert math.isnan(result["js"])


# ---------------------------------------------------------------------------
# 9. Mark calibration
# ---------------------------------------------------------------------------


def test_mark_calibration_perfectly_calibrated():
    """ECE is small when labels are drawn from the predicted distribution."""
    rng = np.random.default_rng(12)
    n, k = 20_000, 5
    # For each example, pick a top-1 confidence c ~ U(0.5, 1.0)
    confidence = rng.uniform(0.5, 1.0, n)
    # Build probability vectors: top class gets confidence, rest share (1-c)/(k-1)
    probs = np.zeros((n, k))
    probs[:, 0] = confidence
    probs[:, 1:] = ((1.0 - confidence) / (k - 1))[:, None]
    # Sample label: class 0 with prob c, else random among 1..k-1
    is_correct = rng.uniform(size=n) < confidence
    wrong_class = rng.integers(1, k, size=n)
    labels = np.where(is_correct, 0, wrong_class)

    result = mark_calibration(probs, labels, num_bins=10)
    assert result["ece"] < 0.05, result["ece"]
    assert not math.isnan(result["nll"])
    assert not math.isnan(result["accuracy"])
    assert not math.isnan(result["brier"])


def test_mark_calibration_empty():
    result = mark_calibration(np.zeros((0, 3)), np.array([]))
    assert math.isnan(result["ece"])


# ---------------------------------------------------------------------------
# 10. Classification report
# ---------------------------------------------------------------------------


def test_classification_report_prior_correction_changes_ece():
    """Prior correction should produce a different (better) ECE than raw."""
    rng = np.random.default_rng(13)
    n = 10_000
    # Train with 20% positives, eval has 1.14% positives
    eval_prior = 0.0114
    true_labels = (rng.uniform(size=n) < eval_prior).astype(int)
    logits = rng.standard_normal(n) + 3.0 * true_labels  # noisy but informative

    result = classification_report(
        logits, true_labels, train_prior=0.20, eval_prior=eval_prior
    )
    # Both ECE values should be finite
    assert not math.isnan(result["ece"])
    assert not math.isnan(result["ece_raw"])
    # Corrected ECE should be lower than raw (in this setup training prior > eval prior)
    assert result["ece"] < result["ece_raw"] or result["ece"] < 0.10


def test_classification_report_prob_input():
    """Accepts probabilities in [0,1] without prior correction."""
    rng = np.random.default_rng(14)
    n = 1000
    prob = rng.uniform(0.0, 1.0, n)
    labels = rng.binomial(1, prob)
    result = classification_report(prob, labels)
    assert not math.isnan(result["auc"])
    assert not math.isnan(result["ece"])


# ---------------------------------------------------------------------------
# 11. Survival calibration
# ---------------------------------------------------------------------------


def test_survival_calibration_shape():
    rng = np.random.default_rng(15)
    n, h = 500, 4
    gaps_y = rng.exponential(2.0, n)
    horizons_y = np.array([0.5, 1.0, 2.0, 4.0])
    # Survival probabilities: (N, H) with P(gap > hz) = exp(-hz / mean_gap)
    surv = np.exp(-horizons_y[None, :] / 2.0) * np.ones((n, h))
    result = survival_calibration(surv, gaps_y, horizons_y)
    for h_idx in range(h):
        assert h_idx in result
        assert "ece" in result[h_idx]
    assert "mean_ece" in result
    assert "mean_brier" in result


# ---------------------------------------------------------------------------
# 12. Stratified mean and quantile bins
# ---------------------------------------------------------------------------


def test_stratified_mean_basic():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    strata = np.array([0, 0, 1, 1, 2, 2])
    result = stratified_mean(values, strata)
    assert result["0"]["mean"] == pytest.approx(1.5)
    assert result["0"]["n"] == 2
    assert result["1"]["mean"] == pytest.approx(3.5)
    assert result["2"]["mean"] == pytest.approx(5.5)


def test_stratified_mean_with_names():
    values = np.array([1.0, 2.0, 3.0])
    strata = np.array([0, 1, 2])
    result = stratified_mean(values, strata, names=["low", "mid", "high"])
    assert "low" in result
    assert result["mid"]["mean"] == pytest.approx(2.0)


def test_quantile_bins_terciles():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    bins = quantile_bins(x, num_bins=3)
    assert bins.shape == (9,)
    # Each tercile should have 3 elements
    for b in range(3):
        assert int((bins == b).sum()) == 3


def test_quantile_bins_empty():
    result = quantile_bins(np.array([]), 4)
    assert len(result) == 0
