"""External TPP baselines in ETAM-2: S2P2 / ITPP backbones, FlexTPP / Kairos interleaved models.

Causality, one CPU training step, proper joints, KV-cache exactness and the
``Trainer2`` / evaluation integration of the interleaved models.
"""

import math

import numpy as np
import pytest
import torch

from eventfm.etam.batching import RandomCutViews, View
from eventfm.etam.stage2 import evaluate as E
from eventfm.etam.stage2.baselines import InterleavedTPP
from eventfm.etam.stage2.batching import Collator2
from eventfm.etam.stage2.config import Etam2Config, Etam2Weights
from eventfm.etam.stage2.distributions import default_grid
from eventfm.etam.stage2.losses import compute_losses2
from eventfm.etam.stage2.model import Etam2Model
from eventfm.etam.store import EtamVocab, StoreBuilder

MIXERS = ["s2p2", "itpp", "flextpp", "kairos"]
INTERLEAVED = ["flextpp", "kairos"]


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


def _model(store, mixer, **overrides):
    base = dict(
        num_event_types=store.vocab.num_event_types,
        field_sizes=store.vocab.field_sizes,
        amount_field=0,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        bridge_layers=[1, 2],
        dropout=0.0,
        mixer=mixer,
        s2p2_states=8,
        s2p2_layers=2,
        itpp_dim=8,
    )
    base.update(overrides)
    config = Etam2Config(**base)
    torch.manual_seed(0)
    model = InterleavedTPP.from_store(config, store, max_events=32) if mixer in INTERLEAVED else Etam2Model(config)
    model.eval()
    return model


@pytest.mark.parametrize("mixer", MIXERS)
def test_states_are_causal(mixer):
    store = _store()
    model = _model(store, mixer)
    seq = 3
    start = int(store.offsets[seq])
    end = min(12, int(store.offsets[seq + 1]) - start)
    view = View(seq, end, float(store.timestamps[start + end - 1]))
    with torch.no_grad():
        reference = model(Collator2(store, max_events=32)([view]))["h"]
    # Change the mark, attributes and time of window event 5: slots 0..4 must not move.
    event = 5
    row = start + event
    changed = _store()
    changed.types[row] = (store.types[row] + 1) % 4
    changed.features[row] = 2
    changed.timestamps[row] = 0.5 * (store.timestamps[row - 1] + store.timestamps[row])
    with torch.no_grad():
        perturbed = model(Collator2(changed, max_events=32)([view]))["h"]
    assert torch.allclose(reference[:, :event], perturbed[:, :event], atol=1e-5)
    assert not torch.allclose(reference[:, event:], perturbed[:, event:], atol=1e-5)


@pytest.mark.parametrize("mixer", MIXERS)
def test_events_after_the_window_change_nothing(mixer):
    store = _store()
    model = _model(store, mixer)
    seq = 2
    start = int(store.offsets[seq])
    stop = int(store.offsets[seq + 1])
    view = View(seq, 6, float(store.timestamps[start + 5]))
    changed = _store()
    changed.types[start + 6 : stop] = (changed.types[start + 6 : stop] + 1) % 4
    changed.features[start + 6 : stop] = 2
    changed.timestamps[start + 6 : stop] += 5_000.0
    results = []
    for current in (store, changed):
        with torch.no_grad():
            out = model(Collator2(current, max_events=16)([view]))
            results.append((out["h"], model.decoder(out["h"][:, -1]).mark_log_probs()))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-6)
    assert torch.allclose(results[0][1], results[1][1], atol=1e-6)


@pytest.mark.parametrize("mixer", MIXERS)
def test_left_padding_is_a_no_op(mixer):
    store = _store()
    model = _model(store, mixer)
    short_start = int(store.offsets[4])
    short = View(4, 3, float(store.timestamps[short_start + 2]) + 1234.0)
    seq = int(np.flatnonzero(np.diff(store.offsets) >= 14)[0])
    long = View(seq, 13, float(store.timestamps[int(store.offsets[seq]) + 12]))
    collator = Collator2(store, max_events=32)
    with torch.no_grad():
        alone = model(collator([short]))["h"][0]
        padded = model(collator([long, short]))["h"][1]
    assert torch.allclose(alone, padded[-alone.shape[0]:], atol=1e-5)


@pytest.mark.parametrize("mixer", MIXERS)
def test_one_training_step_on_cpu(mixer):
    store = _store(users=16)
    model = _model(store, mixer).to(torch.device("cpu"))  # .to() walks every submodule's _apply
    model.train()
    cuts = RandomCutViews(store, p_full=0.5, p_between=0.5, seed=0)
    batch = Collator2(store, max_events=16)([cuts[i] for i in range(16)])
    weights = Etam2Weights(cls=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    outputs = model(batch)
    if mixer in INTERLEAVED:
        terms = model.compute_losses(outputs, batch, weights)
        expected = ("joint", "query", "aux_mark", "cls")
        trained = list(model.named_parameters())
    else:
        terms = compute_losses2(model, outputs, batch, weights)
        expected = ("joint", "query", "cls")
        trained = list(model.backbone.named_parameters())
    for name in expected + ("loss",):
        assert name in terms and torch.isfinite(terms[name]), name
    optimizer.zero_grad()
    terms["loss"].backward()
    missing = [name for name, parameter in trained if parameter.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(parameter.grad).all() for _, parameter in trained)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def _views(store):
    """Windows of different lengths (left padding), one with "now" inside the next gap."""

    views = []
    for seq, end in ((0, 5), (1, 9), (4, 3), (5, 12)):
        start = int(store.offsets[seq])
        end = min(end, int(store.offsets[seq + 1]) - start - 1)
        views.append(View(seq, end, float(store.timestamps[start + end - 1])))
    seq, end = views[1].seq, views[1].end
    start = int(store.offsets[seq])
    last, following = store.timestamps[start + end - 1], store.timestamps[start + end]
    views.append(View(seq, end, float(last + 0.4 * (following - last))))
    return views


@pytest.mark.parametrize("mixer", INTERLEAVED)
def test_interleaved_joint_is_proper_and_the_cache_is_exact(mixer):
    store = _store()
    model = _model(store, mixer)
    batch = Collator2(store, max_events=32)(_views(store))
    rows, marks = batch["slot_valid"].shape[0], store.vocab.num_event_types
    with torch.no_grad():
        out = model(batch)
        dist = model.decoder(out["h"][:, -1])
        table = dist.discretized(default_grid(48))
        assert torch.allclose(table.sum(dim=(-1, -2)), torch.ones(rows), atol=1e-4)
        assert torch.allclose(dist.mark_log_probs().exp().sum(-1), torch.ones(rows), atol=1e-4)
        if mixer == "kairos":
            # Piecewise constant in y: trapezoid with two points around every bin edge is exact.
            edges = model.decoder.time_edges
            grid = torch.sort(torch.cat([edges - 1e-4, edges + 1e-4])).values
        else:
            grid = torch.linspace(-12.0, 34.0, 20001)
        ys = grid.expand(rows, -1)
        total = torch.zeros(rows)
        for mark in range(marks):
            density = dist.log_prob(ys, torch.full(ys.shape, mark, dtype=torch.long)).exp()
            total = total + torch.trapz(density, grid, dim=-1)
        assert torch.allclose(total, torch.ones(rows), atol=3e-3)

        # mark_fn through the KV cache == the full forward with the extra time token appended.
        y = torch.linspace(2.0, 15.0, rows)
        reference = model.decoder.mark_logits(model.encode(batch, extra_y=y)[0][:, -1])
        assert torch.allclose(dist.mark_fn(y), reference, atol=1e-4)
        grid_y = torch.stack([y, y + 1.5, torch.full_like(y, 7.3)], dim=-1)
        reference = torch.stack(
            [model.decoder.mark_logits(model.encode(batch, extra_y=grid_y[:, q])[0][:, -1]) for q in range(3)], dim=1
        )
        assert torch.allclose(dist.mark_fn(grid_y), reference, atol=1e-4)
        # The query slot repeats the last event's state, so both decoders agree.
        assert torch.equal(out["h"][:, -1], out["h"][:, -2])


@pytest.mark.parametrize("mixer", INTERLEAVED)
def test_teacher_forced_terms_match_inference(mixer):
    """The dense term at slot m of a longer window equals the cached joint of the window ending at m."""

    store = _store()
    model = _model(store, mixer)
    m = 6
    seq = int(np.flatnonzero(np.diff(store.offsets) >= m + 5)[0])
    start = int(store.offsets[seq])
    long_view = View(seq, m + 4, float(store.timestamps[start + m + 3]))
    short_view = View(seq, m + 1, float(store.timestamps[start + m]))
    collator = Collator2(store, max_events=32)
    with torch.no_grad():
        long_batch = collator([long_view])
        long_out = model(long_batch)
        y, k = long_batch["next_log_dt"][:, m], long_batch["next_type"][:, m]
        teacher = model._joint_log_prob(long_out["h"][:, m], long_out["time_states"][:, m + 1], y, k)
        short_out = model(collator([short_view]))
        inference = model.decoder(short_out["h"][:, -1]).log_prob(y, k)
    assert torch.allclose(teacher, inference, atol=1e-4)


@pytest.mark.parametrize("mixer", INTERLEAVED)
def test_trainer_and_evaluation_run_interleaved_models(mixer, tmp_path):
    from eventfm.etam.stage2.train import RunConfig2, Trainer2, build_model2, make_collator, quick_tpp

    store = _store(users=16)
    config = RunConfig2(
        model=dict(mixer=mixer, hidden_size=32, num_heads=4, num_layers=2, bridge_layers=[1, 2], dropout=0.0),
        max_events=16, batch_size=8, max_steps=2, eval_every=2, warmup_steps=1, num_workers=0, bf16=False,
        eval_batch_size=8, output_dir=str(tmp_path), run_name="fit",
    )
    model = build_model2(config, store)
    assert isinstance(model, InterleavedTPP) and model.config.mixer == mixer
    collator = make_collator(store, config, model.config, train=True)
    views = _views(store)

    def evaluate(current):
        metrics = quick_tpp(current, store, views, collator, config)
        return metrics["joint_nll"], metrics

    dataset = RandomCutViews(store, p_full=0.5, p_between=0.0, seed=0)
    summary = Trainer2(config).fit(model, dataset, collator, evaluate, higher_is_better=False)
    assert summary["final_step"] == 2 and math.isfinite(summary["best_score"])
    model.to(torch.device("cpu"))
    arrays = E.collect_next_event(model, store, views, collator, 8, torch.device("cpu"), bf16=False)
    metrics = E.next_event_metrics(arrays)
    for name in ("joint_nll", "time_nll", "type_nll", "next_type_accuracy", "coverage80", "last_event_joint_nll"):
        assert math.isfinite(metrics[name]), name
