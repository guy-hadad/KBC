"""ETAM-2: proper densities, coherent joints, causality of every mixer/memory, collator targets."""

import math

import numpy as np
import pytest
import torch

from eventfm.etam.batching import RandomCutViews, View, full_views
from eventfm.etam.stage2.batching import Collator2
from eventfm.etam.stage2.config import Etam2Config, Etam2Weights
from eventfm.etam.stage2.distributions import (
    DiscreteBins,
    EventFirstJoint,
    IndependentJoint,
    LogNormalMixture,
    MixtureJoint,
    PiecewiseHazard,
    SplineFlow,
    TimeFirstJoint,
    default_grid,
)
from eventfm.etam.stage2.losses import compute_losses2
from eventfm.etam.stage2.model import Etam2Model
from eventfm.etam.store import EtamVocab, StoreBuilder

torch.manual_seed(0)
GRID = torch.linspace(-12.0, 34.0, 20001)


def _integrate(values: torch.Tensor) -> torch.Tensor:
    return torch.trapz(values, GRID, dim=-1)


def _dists():
    generator = torch.Generator().manual_seed(1)
    batch = 3
    return {
        "lognormal": LogNormalMixture(
            torch.randn(batch, 4, generator=generator),
            10 + 2 * torch.randn(batch, 4, generator=generator),
            0.3 * torch.randn(batch, 4, generator=generator),
        ),
        "spline": SplineFlow(
            torch.randn(batch, 8, generator=generator),
            torch.randn(batch, 8, generator=generator),
            torch.randn(batch, 7, generator=generator),
        ),
        "hazard": PiecewiseHazard(
            torch.randn(batch, 9, generator=generator) - 1.0, torch.tensor([0.0, 2, 4, 6, 8, 10, 12, 14, 16])
        ),
        "discrete": DiscreteBins(torch.randn(batch, 6, generator=generator), torch.linspace(0.0, 18.0, 7)),
    }


@pytest.mark.parametrize("family", ["lognormal", "spline", "hazard", "discrete"])
def test_time_distributions_are_proper(family):
    dist = _dists()[family]
    grid = GRID.expand(3, -1)
    density = torch.exp(dist.log_prob(grid))
    assert torch.allclose(_integrate(density), torch.ones(3), atol=2e-3)
    # The CDF is the integral of the density, and the survival is its complement.
    point = torch.tensor([9.0, 11.0, 13.0])
    upto = GRID <= 11.0
    partial = torch.trapz(density[:, upto], GRID[upto], dim=-1)
    assert torch.allclose(dist.cdf(torch.full((3,), 11.0)), partial, atol=3e-3)
    assert torch.allclose(torch.exp(dist.log_survival(point)), 1 - dist.cdf(point), atol=1e-5)
    # Quantiles invert the CDF.
    levels = torch.tensor([0.1, 0.5, 0.9])
    assert torch.allclose(dist.cdf(dist.quantile(levels)), levels, atol=2e-3)
    # The mean matches the integral of y f(y).
    mean = _integrate(GRID * density)
    assert torch.allclose(dist.mean(), mean, atol=0.05)
    samples = torch.stack([dist.sample() for _ in range(4000)], dim=-1)
    assert torch.allclose(samples.mean(-1), mean, atol=0.25)


def _joints(kind):
    generator = torch.Generator().manual_seed(2)
    batch, marks = 2, 5
    hidden = torch.randn(batch, 8, generator=generator)
    project = torch.randn(8, marks, generator=generator)
    time_weight = torch.randn(1, marks, generator=generator)

    def time_dist(shift):
        return LogNormalMixture(torch.zeros(*shift.shape, 2), torch.stack([shift + 9, shift + 12], -1), torch.zeros(*shift.shape, 2))

    base = time_dist(torch.zeros(batch))
    mark_logits = hidden @ project
    if kind == "independent":
        return IndependentJoint(base, mark_logits)
    first = TimeFirstJoint(
        base,
        lambda y: (mark_logits.unsqueeze(-2) if y.dim() == 2 else mark_logits) + (y.unsqueeze(-1) / 10.0) * time_weight,
        marks,
        default_grid(400, -2.0, 24.0),
    )
    offsets = torch.linspace(-1.0, 1.0, marks)

    def time_fn(k):
        if k is None:
            return time_dist(offsets.expand(batch, marks))
        return time_dist(offsets[k])

    second = EventFirstJoint(mark_logits, time_fn)
    return {"time_first": first, "event_first": second, "dual": MixtureJoint(first, second)}[kind]


@pytest.mark.parametrize("kind", ["independent", "time_first", "event_first", "dual"])
def test_joint_distributions_are_coherent(kind):
    dist = _joints(kind)
    marks = dist.num_marks
    grid = GRID[(GRID > -3) & (GRID < 26)]
    total = 0.0
    per_mark = []
    for mark in range(marks):
        values = torch.stack(
            [dist.log_prob(y.expand(2), torch.full((2,), mark)) for y in grid], dim=-1
        ).exp()
        mass = torch.trapz(values, grid, dim=-1)
        per_mark.append(mass)
        total = total + mass
    assert torch.allclose(total, torch.ones(2), atol=3e-3)
    # The time-marginalised mark distribution equals the integral of the joint.
    assert torch.allclose(dist.mark_log_probs().exp(), torch.stack(per_mark, -1), atol=5e-3)
    # The mark-marginalised time density equals the sum of the joint.
    y = torch.tensor([10.0, 12.5])
    joint_sum = torch.logsumexp(torch.stack([dist.log_prob(y, torch.full((2,), m)) for m in range(marks)], -1), -1)
    assert torch.allclose(dist.time_log_prob(y), joint_sum, atol=1e-4)
    table = dist.discretized(default_grid(48))
    assert torch.allclose(table.sum(dim=(-1, -2)), torch.ones(2), atol=1e-4)
    # PMI identity: log p(k|y) - log p(k) = log p(y,k) - log p(y) - log p(k).
    given = dist.mark_log_probs_given_time(y)
    assert torch.allclose(given.exp().sum(-1), torch.ones(2), atol=1e-4)


def test_elapsed_silence_residual_is_a_density():
    dist = _joints("dual")
    elapsed = torch.tensor([math.log1p(3600.0), math.log1p(86400.0)])
    grid = torch.linspace(0.0, 22.0, 6001)
    total = torch.zeros(2)
    for mark in range(dist.num_marks):
        values = []
        for y_residual in grid:
            seconds = torch.expm1(y_residual)
            total_y = torch.log1p(torch.expm1(elapsed) + seconds)
            values.append(dist.residual_log_prob(total_y, elapsed, torch.full((2,), mark), y_residual.expand(2)).exp())
        total = total + torch.trapz(torch.stack(values, -1), grid, dim=-1)
    assert torch.allclose(total, torch.ones(2), atol=5e-3)


# ---------------------------------------------------------------------------
# Model-level tests on a synthetic store
# ---------------------------------------------------------------------------


def _store(users=12, seed=0):
    rng = np.random.default_rng(seed)
    vocab = EtamVocab(event_types=["a", "b", "c", "d"], feature_fields=["amount", "currency"],
                      field_values={"amount": ["b_0", "b_1", "na"], "currency": ["1", "2", "na"]})
    builder = StoreBuilder(vocab, observation_end=1_700_000_000.0)
    for user in range(users):
        length = int(rng.integers(6, 20))
        times = 1_690_000_000.0 + np.cumsum(rng.exponential(40_000.0, size=length))
        types = rng.integers(0, 4, size=length)
        features = rng.integers(2, 4, size=(length, 2))
        builder.add_arrays("u{}".format(user), types, features, times, int(user % 2))
    store = builder.build()
    store.amount = np.exp(rng.normal(5, 1, store.num_events)).astype(np.float32)
    return store


def _model(store, **overrides):
    base = dict(
        num_event_types=store.vocab.num_event_types,
        field_sizes=store.vocab.field_sizes,
        amount_field=0,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        bridge_layers=[1, 2],
        dropout=0.0,
        mtp_horizon=3,
        mtp_time=True,
        repr_dim=8,
        repr_block=3,
    )
    base.update(overrides)
    model = Etam2Model(Etam2Config(**base))
    model.eval()
    return model


VARIANTS = [
    {},
    {"mixer": "shared"},
    {"mixer": "stage1"},
    {"mixer": "private", "bridge_layers": []},
    {"mixer": "private_shared"},
    {"memory": "ssm"},
    {"memory": "ode"},
    {"memory": "count"},
    {"memory": "exp_learned", "relative": "stage1", "calendar": "fixed"},
    {"bridge_bias": "lag", "bridge_gate": False},
    {"time_vq": 16},
    {"inputs": "event_only"},
    {"inputs": "time_only"},
]


@pytest.mark.parametrize("overrides", VARIANTS)
def test_states_are_causal(overrides):
    store = _store()
    model = _model(store, **overrides)
    collator = Collator2(store, max_events=32, mtp_steps=3)
    seq = 3
    start = int(store.offsets[seq])
    end = min(12, int(store.offsets[seq + 1]) - start)
    view = View(seq, end, float(store.timestamps[start + end - 1]))
    with torch.no_grad():
        reference = model(collator([view]))["h"]
    # Change the mark, attributes and time of event 5: slots 0..4 must not move.
    event = 5
    row = start + event
    changed = _store()
    changed.types[row] = (store.types[row] + 1) % 4
    changed.features[row] = 2
    changed.timestamps[row] = 0.5 * (store.timestamps[row - 1] + store.timestamps[row])
    with torch.no_grad():
        perturbed = model(Collator2(changed, max_events=32, mtp_steps=3)([view]))["h"]
    assert torch.allclose(reference[:, :event], perturbed[:, :event], atol=1e-5)
    assert not torch.allclose(reference[:, event:], perturbed[:, event:], atol=1e-5)


def test_targets_never_leak_into_states():
    store = _store()
    model = _model(store)
    collator = Collator2(store, max_events=16, mtp_steps=3)
    seq = 2
    start = int(store.offsets[seq])
    view = View(seq, 6, float(store.timestamps[start + 5]))
    with torch.no_grad():
        reference = model(collator([view]))["h"]
    changed = _store()
    changed.types[start + 6:] = (changed.types[start + 6:] + 1) % 4
    with torch.no_grad():
        perturbed = model(Collator2(changed, max_events=16, mtp_steps=3)([view]))["h"]
    assert torch.allclose(reference, perturbed, atol=1e-6)


def test_collator_targets():
    store = _store()
    collator = Collator2(store, max_events=8, mtp_steps=3, count_horizons_days=(1.0, 30.0))
    seq = 1
    start, stop = int(store.offsets[seq]), int(store.offsets[seq + 1])
    end = 5
    now = float(store.timestamps[start + end - 1])
    from eventfm.etam.stage2.batching import CALENDAR, View2

    batch = collator([View2(seq, end, now, CALENDAR)])
    width = batch["slot_valid"].shape[1] - 1
    offset = width - end
    for step in (1, 2, 3):
        slot = offset + 1  # window event index 1
        assert batch["mtp_types"][0, slot, step - 1] == store.types[start + 1 + step]
        expected = math.log1p(store.timestamps[start + 1 + step] - store.timestamps[start + 1])
        assert abs(float(batch["mtp_log_dt"][0, slot, step - 1]) - expected) < 1e-4
    future = store.timestamps[start + end:stop]
    inside = (future > now) & (future <= now + 86400.0)
    assert batch["counts"][0, 0, -1] == inside.sum()
    full = collator([View(seq, stop - start, float(store.timestamps[stop - 1]))])
    assert bool(full["censor_mask"][0])
    assert abs(float(full["censor_log_dt"][0]) - math.log1p(1_700_000_000.0 - store.timestamps[stop - 1])) < 1e-4


def test_every_objective_trains_on_cpu():
    store = _store(users=16)
    model = _model(store, mtp_target="soft")
    model.train()
    collator = Collator2(store, max_events=16, mtp_steps=3, repr_block=3)
    from eventfm.etam.stage2.batching import Views2

    cuts = Views2(store, p_full=0.3, p_between=0.7, seed=0)
    views = [cuts[i] for i in range(16)]
    batch = collator(views)
    weights = Etam2Weights(mtp=1.0, horizon=1.0, repr=0.1, cls=1.0)
    terms = compute_losses2(model, model(batch), batch, weights)
    for name in ("joint", "dual", "cons", "feature", "query", "count", "unimodal", "mtp", "horizon", "repr", "cls"):
        assert name in terms and torch.isfinite(terms[name]), name
    terms["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("decoder", ["independent", "time_first", "event_first"])
@pytest.mark.parametrize("time_dist", ["lognormal", "spline", "hazard"])
def test_decoders_and_families_train(decoder, time_dist):
    store = _store(users=10)
    model = _model(store, decoder=decoder, time_dist=time_dist, mtp_horizon=0, repr_dim=0,
                   hazard_edges=[0.0, 4.0, 8.0, 10.0, 12.0, 14.0])
    model.train()
    batch = Collator2(store, max_events=16)(full_views(store))
    terms = compute_losses2(model, model(batch), batch, Etam2Weights())
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()


# ---------------------------------------------------------------------------
# Regression tests for the 2026-11-08 review
# ---------------------------------------------------------------------------


def _view_batch(store, seq, end):
    start = int(store.offsets[seq])
    end = min(end, int(store.offsets[seq + 1]) - start)
    return Collator2(store, max_events=32, mtp_steps=3)([View(seq, end, float(store.timestamps[start + end - 1]))])


def test_event_only_inputs_ignore_time():
    store = _store()
    model = _model(store, inputs="event_only")
    seq, end = 3, 10
    with torch.no_grad():
        reference = model(_view_batch(store, seq, end))["h"]
    stretched = _store()
    start = int(stretched.offsets[seq])
    stop = int(stretched.offsets[seq + 1])
    times = stretched.timestamps[start:stop].copy()
    stretched.timestamps[start:stop] = times[0] + 3.7 * (times - times[0])
    with torch.no_grad():
        moved = model(_view_batch(stretched, seq, end))["h"]
    assert torch.allclose(reference, moved, atol=1e-5)


def test_time_only_inputs_ignore_marks():
    store = _store()
    model = _model(store, inputs="time_only")
    seq, end = 3, 10
    with torch.no_grad():
        reference = model(_view_batch(store, seq, end))["h"]
    relabelled = _store()
    start = int(relabelled.offsets[seq])
    stop = int(relabelled.offsets[seq + 1])
    relabelled.types[start:stop] = (relabelled.types[start:stop] + 1) % 4
    relabelled.features[start:stop] = 2
    with torch.no_grad():
        moved = model(_view_batch(relabelled, seq, end))["h"]
    assert torch.allclose(reference, moved, atol=1e-5)


def test_calendar_views_and_count_targets():
    from eventfm.etam.stage2.batching import CALENDAR, FULL, Views2

    store = _store(users=30)
    views = Views2(store, p_full=0.3, p_between=0.6, seed=1)
    seen = {FULL: 0, CALENDAR: 0}
    collator = Collator2(store, max_events=32, mtp_steps=2, count_horizons_days=(1.0,))
    for index in range(30):
        view = views[index]
        start, stop = int(store.offsets[view.seq]), int(store.offsets[view.seq + 1])
        times = store.timestamps[start:stop]
        assert view.query_time >= times[view.end - 1]
        if view.end < stop - start:
            assert view.query_time < times[view.end] or view.query_time == times[view.end - 1]
        if view.kind in seen:
            seen[view.kind] += 1
        batch = collator([view])
        if view.kind != CALENDAR:
            assert float(batch["count_mask"].sum()) == 0.0
    assert seen[FULL] and seen[CALENDAR]


def test_censoring_and_survival_terms_active():
    store = _store(users=24)
    model = _model(store, mtp_horizon=0, repr_dim=0)
    model.train()
    cuts = __import__("eventfm.etam.stage2.batching", fromlist=["Views2"]).Views2(store, p_full=0.5, p_between=0.9, seed=3)
    batch = Collator2(store, max_events=32)([cuts[i] for i in range(24)])
    terms = compute_losses2(model, model(batch), batch, Etam2Weights())
    assert "censor_nll" in terms and torch.isfinite(terms["censor_nll"])
    assert "survival" in terms and torch.isfinite(terms["survival"])
