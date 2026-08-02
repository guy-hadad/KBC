"""Tests for the grid expansion and the scaling-point subsampling."""

import pytest

from eventfm.benchmark.runner import BenchmarkCell, _stratified_subsample, enumerate_cells
from eventfm.data.schema import Event, EventSequence
from eventfm.methods import METHOD_REGISTRY


def _sequences(num_positive: int, num_negative: int):
    def make(index: int, label: int) -> EventSequence:
        return EventSequence(
            user_id="{}_{}".format(label, index),
            events=[Event(event_type="a", timestamp=float(step)) for step in range(4)],
            label=label,
        )

    return [make(index, 1) for index in range(num_positive)] + [
        make(index, 0) for index in range(num_negative)
    ]


def test_grid_skips_unsupported_method_task_pairs():
    cells = enumerate_cells(
        datasets=["banksim"],
        tasks=["classification", "tpp"],
        methods=["markov", "count-logistic", "pragma"],
        sample_sizes=[64],
    )
    pairs = {(cell.method, cell.task) for cell in cells}

    # The controls each support exactly one task.
    assert ("markov", "tpp") in pairs
    assert ("markov", "classification") not in pairs
    assert ("count-logistic", "classification") in pairs
    assert ("count-logistic", "tpp") not in pairs
    assert {("pragma", "classification"), ("pragma", "tpp")} <= pairs


def test_grid_size_is_the_product_of_supported_combinations():
    methods = sorted(METHOD_REGISTRY)
    cells = enumerate_cells(
        datasets=["banksim", "paysim"],
        tasks=["classification", "tpp"],
        methods=methods,
        sample_sizes=[64, 128, 256],
    )
    supported = sum(
        1
        for method in methods
        for task in ("classification", "tpp")
        if task in METHOD_REGISTRY[method].supports
    )

    assert len(cells) == 2 * 3 * supported
    assert len({cell.key for cell in cells}) == len(cells)


def test_cell_key_is_unique_per_coordinate():
    left = BenchmarkCell("banksim", "tpp", "thp", 64, 13)
    right = BenchmarkCell("banksim", "tpp", "thp", 128, 13)

    assert left.key != right.key
    assert left.key == "banksim__tpp__thp__n64__s13"


def test_subsample_preserves_the_label_balance():
    sequences = _sequences(num_positive=200, num_negative=800)

    subset = _stratified_subsample(sequences, sample_size=100, seed=3)

    positives = sum(1 for item in subset if item.label == 1)
    assert len(subset) == 100
    assert positives == pytest.approx(20, abs=2)


def test_subsample_keeps_a_positive_even_when_tiny():
    """The smallest scaling point must not collapse to a single class."""

    sequences = _sequences(num_positive=5, num_negative=495)

    subset = _stratified_subsample(sequences, sample_size=16, seed=1)

    assert len(subset) == 16
    assert any(item.label == 1 for item in subset)


def test_subsample_is_deterministic_for_a_seed():
    sequences = _sequences(num_positive=50, num_negative=150)

    first = _stratified_subsample(sequences, 40, seed=7)
    second = _stratified_subsample(sequences, 40, seed=7)

    assert [item.user_id for item in first] == [item.user_id for item in second]


def test_subsample_returns_everything_when_asked_for_more_than_exists():
    sequences = _sequences(num_positive=10, num_negative=10)

    assert len(_stratified_subsample(sequences, 500, seed=0)) == 20


def test_every_method_declares_its_fidelity_and_divergence():
    """A registry entry named after a paper must disclose how it differs.

    `MethodSpec.__post_init__` enforces this, so a new method cannot be added
    without saying whether it is a control or an approximation.
    """

    for name, spec in METHOD_REGISTRY.items():
        assert spec.status in ("Implemented", "Approximation"), name
        assert spec.divergence, "{} must disclose a divergence".format(name)
        assert spec.reference, "{} must cite something".format(name)


def test_only_the_controls_claim_full_fidelity():
    implemented = {
        name for name, spec in METHOD_REGISTRY.items() if spec.status == "Implemented"
    }

    # Everything else approximates a published recipe at benchmark scale.
    assert implemented == {"markov", "count-logistic"}


def test_registering_an_approximation_without_disclosure_is_rejected():
    from eventfm.methods.base import MethodSpec

    with pytest.raises(ValueError, match="must disclose"):
        MethodSpec(
            name="undisclosed",
            display_name="Undisclosed",
            reference="Somebody et al.",
            family="neural-tpp",
            factory=lambda context: None,
        )
