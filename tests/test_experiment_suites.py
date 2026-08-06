from eventfm.benchmark.experiments import EXPERIMENT_SUITES
from eventfm.benchmark.runner import enumerate_cells
from eventfm.datasets.registry import CHRONOLOGICAL_DATASETS, DATASET_REGISTRY, GEM_DATASETS
from eventfm.methods import METHOD_REGISTRY


def test_every_experiment_suite_references_registered_components():
    for suite in EXPERIMENT_SUITES.values():
        assert set(suite.datasets) <= set(DATASET_REGISTRY), suite.name
        assert set(suite.methods) <= set(METHOD_REGISTRY), suite.name
        cells = enumerate_cells(
            suite.datasets,
            suite.tasks,
            suite.methods,
            suite.sample_sizes,
            suite.seeds,
            suite.regimes,
            suite.variants,
        )
        assert cells, suite.name
        assert len({cell.key for cell in cells}) == len(cells), suite.name
        assert set(suite.variant_extra) <= set(suite.variants), suite.name


def test_gem_datasets_are_tpp_only():
    for dataset in GEM_DATASETS:
        assert DATASET_REGISTRY[dataset].supports == ("tpp",)


def test_chronological_datasets_reuse_the_primary_adapters():
    for dataset in CHRONOLOGICAL_DATASETS:
        spec = DATASET_REGISTRY[dataset]
        assert spec.adapter in {"banksim", "paysim", "ibm_aml", "mbd_mini"}
        assert spec.config.split_strategy == "chronological"


def test_gem_full_suite_contains_every_registered_gem_method():
    suite_methods = set(EXPERIMENT_SUITES["gem-full-banking"].methods)
    gem_methods = {name for name in METHOD_REGISTRY if name.startswith("gem-time-")}

    assert gem_methods
    assert gem_methods <= suite_methods


def test_adaptation_suite_covers_all_three_regimes():
    suite = EXPERIMENT_SUITES["adaptation-regimes"]

    assert suite.regimes == ("frozen", "peft", "full")


def test_factorial_suite_has_distinct_result_variants():
    suite = EXPERIMENT_SUITES["objective-factorial"]

    assert "masked" in suite.variants
    assert "full" in suite.variants
    assert set(suite.variants) == set(suite.variant_extra)
