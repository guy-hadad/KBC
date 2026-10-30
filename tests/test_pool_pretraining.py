"""Self-supervised pretraining on an unlabeled corpus larger than the labeled split."""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="pool pretraining tests require torch",
)

from torch.utils.data import DataLoader  # noqa: E402

import eventfm.benchmark.experiments as experiments  # noqa: E402
from eventfm.benchmark.experiments import EXPERIMENT_SUITES, resolve_sample_sizes  # noqa: E402
from eventfm.benchmark.runner import (  # noqa: E402
    BenchmarkCell,
    build_context,
    enumerate_cells,
    run_cell,
)
from eventfm.data.dataset import JsonlEventDataset, LazyJsonlEventDataset  # noqa: E402
from eventfm.data.io import write_jsonl  # noqa: E402
from eventfm.data.schema import Event, EventSequence, sequence_to_json  # noqa: E402
from eventfm.data.synthetic import generate_synthetic_sequences  # noqa: E402
from eventfm.datasets.registry import DatasetMeta  # noqa: E402
from eventfm.methods import build_method, coles, pragma, representation_learning  # noqa: E402
from eventfm.methods.base import TrainingSpec  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
EVENT_TYPES = ["purchase", "transfer", "withdrawal"]
NUM_TRAIN = 6
NUM_CORPUS = 10


def _sequences(prefix, count, event_types, amounts):
    return [
        EventSequence(
            user_id="{}{}".format(prefix, row),
            events=[
                Event(
                    event_type=event_types[(row + index) % len(event_types)],
                    timestamp=1_700_000_000.0 + index * (900 + row * 60),
                    features={"amount": amounts[(row + index) % len(amounts)]},
                )
                for index in range(8)
            ],
            label=row % 2,
        )
        for row in range(count)
    ]


def _user_ids(batch):
    return [sequence.user_id for sequence in batch]


@pytest.fixture
def mbd_like(tmp_path, monkeypatch):
    """A tiny prepared `mbd` with a pool corpus, under a throwaway data root.

    The corpus holds a mark and a feature value that never occur in the labeled
    training split, which is how the tests tell the two files apart.
    """

    monkeypatch.setenv("KBC_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("KBC_OUTPUT_ROOT", str(tmp_path / "outputs"))
    directory = tmp_path / "data" / "processed" / "mbd"
    splits = {}
    for split in ("train", "validation", "test"):
        path = directory / "{}.jsonl".format(split)
        write_jsonl(str(path), _sequences(split, NUM_TRAIN, EVENT_TYPES[:2], ["b_0", "b_1"]))
        splits[split] = str(path)
    write_jsonl(
        str(directory / "pool_train.jsonl"),
        _sequences("pool", NUM_CORPUS, EVENT_TYPES[:2] + ["refund"], ["b_0", "b_9"]),
    )
    meta = DatasetMeta(
        name="mbd",
        display_name="MBD (test)",
        splits=splits,
        event_types=list(EVENT_TYPES),
        feature_fields=["amount"],
        profile_fields=[],
        stats={"train": {"num_sequences": NUM_TRAIN}},
    )
    (directory / "meta.json").write_text(json.dumps(meta.to_json()), encoding="utf-8")
    return directory


def _training(tmp_path, workers: int = 0) -> TrainingSpec:
    return TrainingSpec(
        num_train_epochs=1,
        max_steps=1,
        batch_size=4,
        eval_batch_size=8,
        max_events=12,
        use_cpu=True,
        dataloader_num_workers=workers,
        output_dir=str(tmp_path / "runs"),
    )


def _context(tmp_path, method, extra, task="classification"):
    return build_context(
        BenchmarkCell("mbd", task, method, NUM_TRAIN, 13),
        training=_training(tmp_path),
        hidden_size=16,
        num_hidden_layers=1,
        extra=extra,
    )


def _record_trainer(monkeypatch, module, train=True):
    """Swap a method module's `Trainer` for one that records what it was given."""

    seen = []
    original = module.Trainer

    class RecordingTrainer(original):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs)
            super().__init__(*args, **kwargs)

        def train(self, *args, **kwargs):
            if train:
                return super().train(*args, **kwargs)
            return None

    monkeypatch.setattr(module, "Trainer", RecordingTrainer)
    return seen


def _record_lazy_reads(monkeypatch):
    fetched = []
    original = LazyJsonlEventDataset.__getitem__

    def getitem(self, index):
        sequence = original(self, index)
        fetched.append(sequence.user_id)
        return sequence

    monkeypatch.setattr(LazyJsonlEventDataset, "__getitem__", getitem)
    return fetched


def test_lazy_dataset_matches_the_eager_reader(tmp_path):
    sequences = generate_synthetic_sequences(num_users=7, num_event_types=5, max_events=20)
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(json.dumps(sequence_to_json(sequence), sort_keys=True) + "\n")
            if index % 3 == 0:
                handle.write("\n   \n")

    eager = JsonlEventDataset(str(path))
    lazy = LazyJsonlEventDataset(str(path))

    assert len(lazy) == len(eager) == len(sequences)
    assert lazy.offsets.dtype.name == "int64"
    for index in range(len(eager)):
        assert sequence_to_json(lazy[index]) == sequence_to_json(eager[index])
    with pytest.raises(IndexError):
        lazy[len(eager)]


def test_lazy_dataset_cap_keeps_the_first_lines(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_jsonl(str(path), _sequences("u", 5, EVENT_TYPES, ["b_0"]))

    capped = LazyJsonlEventDataset(str(path), max_sequences=3)

    assert [capped[index].user_id for index in range(len(capped))] == ["u0", "u1", "u2"]


def test_lazy_dataset_survives_dataloader_workers_and_pickling(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_jsonl(str(path), _sequences("u", 9, EVENT_TYPES, ["b_0"]))
    dataset = LazyJsonlEventDataset(str(path))
    # Open the parent's handle first: forked workers must not reuse its offset.
    assert dataset[0].user_id == "u0"

    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=_user_ids)
    seen = [user_id for batch in loader for user_id in batch]
    restored = pickle.loads(pickle.dumps(dataset))

    assert seen == ["u{}".format(index) for index in range(9)]
    assert restored[8].user_id == "u8"


def test_pool_corpus_resolves_next_to_the_splits_and_leaves_the_vocab_alone(
    tmp_path, mbd_like
):
    baseline = _context(tmp_path, "transaction-mlm", {})
    context = _context(tmp_path, "transaction-mlm", {"pretrain_corpus": "pool"})

    assert context.extra["pretrain_corpus_path"] == str(mbd_like / "pool_train.jsonl")
    assert context.vocab.token_to_id == baseline.vocab.token_to_id
    assert "value:amount:b_9" not in context.vocab.token_to_id
    assert "value:event_type:refund" not in context.vocab.token_to_id


def test_missing_pool_corpus_is_a_clear_error(tmp_path, mbd_like):
    (mbd_like / "pool_train.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="pool_train.jsonl"):
        _context(tmp_path, "transaction-mlm", {"pretrain_corpus": "pool"})


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"pretrain_corpus": "everything"}, "Unknown `pretrain_corpus`"),
        (
            {"pretrain_corpus": "pool", "pretrain_datasets": "target"},
            "set only one",
        ),
    ],
)
def test_ambiguous_corpus_settings_are_rejected(tmp_path, mbd_like, extra, message):
    with pytest.raises(ValueError, match=message):
        _context(tmp_path, "transaction-mlm", extra)


@pytest.mark.parametrize(
    "method, module",
    [
        ("transaction-mlm", representation_learning),
        ("autoregressive-transformer", representation_learning),
        ("pragma-mlm", pragma),
    ],
)
def test_objective_pretraining_reads_the_corpus(tmp_path, mbd_like, monkeypatch, method, module):
    seen = _record_trainer(monkeypatch, module)
    fetched = _record_lazy_reads(monkeypatch)
    context = _context(
        tmp_path,
        method,
        {"pretrain_corpus": "pool", "pretrain_epochs": 1, "pretrain_max_steps": 1},
    )
    runner = build_method(method, context)

    runner.pretrain(runner.build_model())

    assert len(seen) == 1
    dataset = seen[0]["train_dataset"]
    assert isinstance(dataset, LazyJsonlEventDataset)
    assert len(dataset) == NUM_CORPUS
    assert context.extra["pretrain_corpus_num_sequences"] == NUM_CORPUS
    assert seen[0]["args"].num_train_epochs == 1.0
    assert seen[0]["args"].max_steps == 1
    assert fetched and all(user_id.startswith("pool") for user_id in fetched)


def test_pragma_mlm_without_a_corpus_keeps_its_default_stage(tmp_path, mbd_like, monkeypatch):
    seen = _record_trainer(monkeypatch, pragma, train=False)
    context = _context(tmp_path, "pragma-mlm", {})
    runner = build_method("pragma-mlm", context)

    runner.pretrain(runner.build_model())

    dataset = seen[0]["train_dataset"]
    assert type(dataset) is JsonlEventDataset
    assert dataset.path == context.train_path
    assert seen[0]["args"].num_train_epochs == 3.0
    assert seen[0]["args"].max_steps == -1


def test_supervised_pragma_ignores_a_shared_pretraining_budget(tmp_path, mbd_like, monkeypatch):
    seen = _record_trainer(monkeypatch, pragma, train=False)
    context = _context(tmp_path, "pragma", {"pretrain_corpus": "pool", "pretrain_epochs": 1})
    runner = build_method("pragma", context)

    runner.pretrain(runner.build_model())

    assert seen == []


@pytest.mark.parametrize(
    "budget, expected_reads",
    [
        # One epoch over the corpus, not the class default of three.
        ({"pretrain_epochs": 1}, NUM_CORPUS),
        # A single optimiser step reads a single batch of 4 clients.
        ({"pretrain_epochs": 1, "pretrain_max_steps": 1}, 4),
    ],
)
def test_coles_contrastive_stage_reads_the_corpus(
    tmp_path, mbd_like, monkeypatch, budget, expected_reads
):
    views = []
    original = coles._SubSequenceView

    class RecordingView(original):
        def __init__(self, sequences, **kwargs):
            views.append(sequences)
            super().__init__(sequences, **kwargs)

    monkeypatch.setattr(coles, "_SubSequenceView", RecordingView)
    fetched = _record_lazy_reads(monkeypatch)
    context = _context(tmp_path, "coles", dict(budget, pretrain_corpus="pool"))
    runner = build_method("coles", context)

    runner.pretrain(runner.build_model())

    assert isinstance(views[0], LazyJsonlEventDataset)
    assert len(views[0]) == NUM_CORPUS
    assert len(fetched) == expected_reads
    assert all(user_id.startswith("pool") for user_id in fetched)


def test_pool_cell_runs_end_to_end_and_records_the_corpus(tmp_path, mbd_like):
    cell = BenchmarkCell("mbd", "classification", "transaction-mlm", NUM_TRAIN, 13, "full", "pool")

    result = run_cell(
        cell,
        training=_training(tmp_path, workers=2),
        hidden_size=16,
        num_hidden_layers=1,
        extra={"pretrain_corpus": "pool", "pretrain_epochs": 1, "pretrain_max_steps": 1},
        results_dir=tmp_path / "results",
    )

    assert result.status == "ok", result.error
    assert cell.key.endswith("__vpool")
    assert result.num_train_sequences == NUM_TRAIN
    assert result.metadata["extra"]["pretrain_corpus_path"] == str(mbd_like / "pool_train.jsonl")
    assert result.metadata["extra"]["pretrain_corpus_num_sequences"] == NUM_CORPUS


def test_pool_suite_targets_the_whole_labeled_pool(monkeypatch):
    class _Meta:
        stats = {"train": {"num_sequences": 30665}}

    monkeypatch.setattr(experiments, "load_dataset_meta", lambda name: _Meta())
    suite = EXPERIMENT_SUITES["mbd-pool-baselines"]

    cells = enumerate_cells(
        suite.datasets,
        suite.tasks,
        suite.methods,
        resolve_sample_sizes(suite),
        suite.seeds,
        suite.regimes,
        suite.variants,
    )

    assert resolve_sample_sizes(suite) == {"mbd": (30665,)}
    assert len(cells) == 2 * 4 * len(suite.seeds)
    assert all(cell.key.endswith("__n30665__s{}__vpool".format(cell.seed)) for cell in cells)
    assert suite.variant_extra["pool"] == {"pretrain_corpus": "pool", "pretrain_epochs": 1}


def test_pool_suite_lists_before_the_dataset_is_prepared(tmp_path):
    environment = dict(
        os.environ,
        KBC_DATA_ROOT=str(tmp_path / "data"),
        KBC_OUTPUT_ROOT=str(tmp_path / "outputs"),
    )

    listing = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "run_experiments.py"),
            "--suite",
            "mbd-pool-baselines",
            "--list",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout

    assert "24 cells" in listing
    assert "mbd__tpp__coles__" in listing
