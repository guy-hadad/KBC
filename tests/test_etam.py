"""ETAM: store parity with the benchmark, leakage-free tokens, routing, pipeline."""

import json
import random
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from eventfm.data.dataset import NextEventJsonlDataset
from eventfm.etam.batching import (
    RandomCutViews,
    View,
    ViewCollator,
    benchmark_tpp_views,
    calendar_arrays,
    full_views,
)
from eventfm.etam.model import EtamConfig, EtamModel, LossWeights, Routed, compute_losses
from eventfm.etam.store import EtamVocab, EventStore, store_from_jsonl

FIELDS = ["amount", "currency"]


def _vocab() -> EtamVocab:
    return EtamVocab(
        event_types=["a", "b", "c", "other"],
        feature_fields=FIELDS,
        field_values={"amount": ["b_0", "b_1", "b_2", "na"], "currency": ["1", "2", "other", "na"]},
    )


def _write_fixture(path, num_users=40, seed=0):
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as handle:
        for user in range(num_users):
            length = rng.randint(2, 30)
            stamp = 1_640_000_000.0 + rng.random() * 1e6
            events = []
            for _ in range(length):
                # Repeated timestamps on purpose: ordering of ties must match.
                stamp += rng.choice([0.0, 60.0, 3600.0 * rng.random(), 86400.0 * rng.random()])
                events.append(
                    {
                        "event_type": rng.choice(["a", "b", "c", "zzz"]),
                        "timestamp": stamp,
                        "features": {
                            "amount": rng.choice(["b_0", "b_1", "b_2", "b_9"]),
                            "currency": rng.choice(["1", "2", "other"]),
                        },
                    }
                )
            record = {"user_id": "u{}".format(user), "events": events, "label": user % 3 == 0}
            handle.write(json.dumps(record) + "\n")


@pytest.fixture()
def jsonl(tmp_path):
    path = tmp_path / "split.jsonl"
    _write_fixture(path)
    return str(path)


@pytest.fixture()
def store(jsonl) -> EventStore:
    return store_from_jsonl(jsonl, _vocab(), observation_end=1_700_000_000.0)


def _tiny_model(store, **overrides) -> EtamModel:
    config = dict(hidden_size=32, num_layers=2, num_heads=4, dropout=0.0)
    config.update(overrides)
    return EtamModel(
        EtamConfig(
            num_event_types=store.vocab.num_event_types,
            field_sizes=store.vocab.field_sizes,
            **config,
        )
    ).eval()


def test_calendar_matches_datetime():
    rng = np.random.default_rng(0)
    stamps = rng.uniform(1.5e9, 1.8e9, 500)
    ids, continuous = calendar_arrays(stamps)
    for stamp, row in zip(stamps, ids):
        moment = datetime.fromtimestamp(float(stamp), tz=timezone.utc)
        assert list(row) == [moment.hour, moment.weekday(), moment.day - 1, moment.month - 1]
    assert continuous.shape == (500, 8)
    assert np.all(np.abs(continuous) <= 1.0)


def test_store_roundtrip_and_unknowns(store, tmp_path):
    assert store.types.dtype == np.int16
    assert store.vocab.type_index["other"] == 3
    # `zzz` is not a level and maps to `other`; `b_9` is an unknown amount.
    assert (store.types == 3).any()
    assert (store.features[:, 0] == 1).any()
    store.save(str(tmp_path / "s"))
    loaded = EventStore.load(str(tmp_path / "s"))
    assert np.array_equal(loaded.offsets, store.offsets)
    assert np.array_equal(loaded.features, store.features)
    assert loaded.observation_end == store.observation_end
    assert list(loaded.user_ids) == list(store.user_ids)


@pytest.mark.parametrize("seed", [14, 30])
def test_tpp_views_replay_benchmark_prefixes(jsonl, store, seed):
    reference = NextEventJsonlDataset(
        jsonl, min_prefix_events=4, max_prefixes_per_user=8, seed=seed
    )
    views = benchmark_tpp_views(store, seed=seed)
    assert len(views) == len(reference)
    collator = ViewCollator(store, max_events=64)
    batch = collator(views)
    index = store.vocab.type_index
    for row, (view, example) in enumerate(zip(views, reference.examples)):
        assert store.user_ids[view.seq] == example.sequence.user_id
        assert view.end == len(example.sequence.events)
        expected_type = index.get(example.next_event_type, index["other"])
        assert int(batch["next_type"][row, -1]) == expected_type
        assert float(batch["next_log_dt"][row, -1]) == pytest.approx(
            np.log1p(example.next_delta_seconds), rel=1e-6, abs=1e-6
        )


def test_collator_targets_and_labels(store):
    collator = ViewCollator(store, max_events=5, future_window_days=1.0)
    seq = next(i for i in range(len(store)) if store.lengths[i] >= 8)
    start = int(store.offsets[seq])
    times = store.timestamps
    between = float(times[start + 5] + 0.5 * (times[start + 6] - times[start + 5]))
    views = [
        View(seq, 6, between),
        View(seq, int(store.lengths[seq]), float(times[int(store.offsets[seq + 1]) - 1])),
    ]
    batch = collator(views)
    # Window keeps the last five events; slot -1 is the query.
    assert batch["slot_valid"][0].sum() == 6
    assert int(batch["type_ids"][0, -2]) == int(store.types[start + 5])
    assert int(batch["next_type"][0, -1]) == int(store.types[start + 6])
    assert float(batch["next_log_dt"][0, -1]) == pytest.approx(
        np.log1p(times[start + 6] - between), rel=1e-5
    )
    # Elapsed silence is the query token's gap.
    assert float(batch["log_gap"][0, -1]) == pytest.approx(
        np.log1p(between - times[start + 5]), rel=1e-5
    )
    # Only the full-history view carries the sequence label.
    assert int(batch["labels"][0]) == -1
    assert int(batch["labels"][1]) == int(store.labels[seq])
    # The first window event still sees its true predecessor outside the window.
    assert float(batch["has_prev"][0, -6]) == 1.0


@pytest.mark.parametrize("layout", ["token", "additive"])
@pytest.mark.parametrize("rope", ["index", "index+time"])
def test_no_future_leakage(store, layout, rope):
    """Outputs at a slot must not depend on anything that happens later."""

    torch.manual_seed(0)
    model = _tiny_model(store, layout=layout, rope=rope)
    seq = next(i for i in range(len(store)) if store.lengths[i] >= 10)
    start = int(store.offsets[seq])
    view = View(seq, 8, float(store.timestamps[start + 7]))
    collator = ViewCollator(store, max_events=16)
    base = collator([view])

    altered = {key: value.clone() for key, value in base.items()}
    # Change the mark and features of the last history event.
    altered["type_ids"][0, -2] = (altered["type_ids"][0, -2] + 1) % store.vocab.num_event_types
    altered["feature_ids"][0, -2] = 2
    with torch.no_grad():
        first = model(base)
        second = model(altered)
    # Every earlier event slot is unchanged; the last event and the user change.
    assert torch.allclose(first["event_out"][0, :-2], second["event_out"][0, :-2], atol=1e-5)
    assert not torch.allclose(first["user"], second["user"])
    if layout == "token":
        # The time token of the changed event has not seen its mark.
        assert torch.allclose(first["time_out"][0, :-1], second["time_out"][0, :-1], atol=1e-5)

        # Changing only that event's *time* changes its own time token but not
        # the previous event token, which is what predicts that time.
        timed = {key: value.clone() for key, value in base.items()}
        timed["log_gap"][0, -2] += 1.0
        timed["abs_days"][0, -2] += 0.3
        with torch.no_grad():
            third = model(timed)
        assert torch.allclose(first["event_out"][0, :-2], third["event_out"][0, :-2], atol=1e-5)
        assert not torch.allclose(first["time_out"][0, -2], third["time_out"][0, -2])


def test_routed_uses_one_expert_per_token():
    module = Routed(lambda: torch.nn.Linear(3, 2, bias=False), 2)
    with torch.no_grad():
        module.experts[0].weight.fill_(1.0)
        module.experts[1].weight.fill_(-1.0)
    hidden = torch.ones(1, 4, 3)
    route = torch.tensor([[0, 1, 0, 1]])
    output = module(hidden, route)
    assert torch.equal(output[0, :, 0], torch.tensor([3.0, -3.0, 3.0, -3.0]))


def test_placements_scale_parameters(store):
    counts = {
        placement: sum(p.numel() for p in _tiny_model(store, placement=placement).parameters())
        for placement in ("shared", "ffn", "attn_ffn", "deep")
    }
    assert counts["shared"] < counts["ffn"] < counts["attn_ffn"] < counts["deep"]
    additive = sum(
        p.numel() for p in _tiny_model(store, layout="additive", placement="deep").parameters()
    )
    assert additive == counts["shared"]


def test_losses_finite_and_backward(store):
    model = _tiny_model(store).train()
    views = [RandomCutViews(store, seed=3)[i] for i in range(16)]
    batch = ViewCollator(store, max_events=12)(views)
    terms = compute_losses(
        model, model(batch), batch, LossWeights(), ["tpp", "cond", "user", "future"]
    )
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_pipeline_end_to_end(tmp_path, jsonl, monkeypatch):
    from eventfm.etam import train as etam_train

    monkeypatch.setenv("KBC_OUTPUT_ROOT", str(tmp_path / "out"))
    vocab = _vocab()
    root = tmp_path / "stores" / "toy"
    for split, seed in (("train", 1), ("validation", 2), ("test", 3), ("pool", 4)):
        path = tmp_path / "{}.jsonl".format(split)
        _write_fixture(path, num_users=30, seed=seed)
        store_from_jsonl(str(path), vocab, observation_end=1_700_000_000.0).save(str(root / split))
    common = dict(
        dataset="toy",
        store_root=str(tmp_path / "stores"),
        output_dir=str(tmp_path / "runs"),
        batch_size=8,
        eval_batch_size=16,
        max_steps=4,
        eval_every=2,
        num_workers=0,
        bf16=False,
        max_events=16,
    )
    architecture = {"hidden_size": 32, "num_layers": 1, "num_heads": 4}
    etam_train.run(
        etam_train.RunConfig(stage="pretrain", run_name="pre", model=architecture, **common)
    )
    checkpoint = str(tmp_path / "runs" / "pre" / "best.pt")
    results = etam_train.run(
        etam_train.RunConfig(
            stage="finetune", task="multi", run_name="ft", init_checkpoint=checkpoint, **common
        )
    )
    assert {"classification", "tpp", "valid", "final_step"} <= set(results)
    assert set(results["final_step"]) == {"classification", "tpp"}
    assert "score" in results["valid"]
    assert 0.0 <= results["tpp"]["next_type_accuracy"] <= 1.0
    probe = etam_train.run(
        etam_train.RunConfig(stage="probe", run_name="probe", init_checkpoint=checkpoint, **common)
    )
    assert "auc" in probe["classification"] and "score" in probe["valid"]
    # An HPO trial: validation only, no test numbers and no result files.
    trial = etam_train.run(
        etam_train.RunConfig(
            stage="finetune",
            task="multi",
            run_name="trial",
            init_checkpoint=checkpoint,
            test_eval=False,
            model={"dropout": 0.2},
            pretrain_fraction=0.5,
            **common,
        )
    )
    assert "classification" not in trial and trial["valid"]["score"] > 0
    written = list((tmp_path / "out" / "paper_runs" / "etam" / "results").glob("*.json"))
    assert len(written) == 4
    payload = json.loads(written[0].read_text())
    assert payload["status"] == "ok" and payload["cell"]["dataset"] == "toy"
    full = full_views(EventStore.load(str(root / "test")))
    assert len(full) == 30


def test_window_time_matches_baseline_collator(store):
    """With ``window_time`` the first window event has no gap and age starts at it."""

    seq = next(i for i in range(len(store)) if store.lengths[i] >= 12)
    start = int(store.offsets[seq])
    view = View(seq, 12, float(store.timestamps[start + 11]))
    default = ViewCollator(store, max_events=5)([view])
    windowed = ViewCollator(store, max_events=5, window_time=True)([view])
    first_slot = -6
    assert float(default["has_prev"][0, first_slot]) == 1.0
    assert float(windowed["has_prev"][0, first_slot]) == 0.0
    assert float(windowed["log_gap"][0, first_slot]) == 0.0
    assert float(windowed["log_age"][0, first_slot]) == 0.0
    assert float(default["log_age"][0, first_slot]) > 0.0
    # Later slots are identical: only the window boundary changes.
    assert torch.equal(
        default["log_gap"][0, first_slot + 1 :], windowed["log_gap"][0, first_slot + 1 :]
    )


def test_future_target_needs_an_observation_cutoff(store):
    views = full_views(store)[:8]
    unknown = EventStore(**{**store.__dict__, "observation_end": None})
    assert float(ViewCollator(unknown, max_events=8)(views)["future_mask"].sum()) == 0.0
