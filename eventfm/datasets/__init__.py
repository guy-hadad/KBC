"""Open event-sequence benchmarks referenced by the KBC/TPP project note.

Every adapter downloads the public source, converts it to the shared
``EventSequence`` JSONL contract used by :mod:`eventfm.data`, and writes
``train/validation/test`` splits plus a ``meta.json`` describing the event
vocabulary and the classification target.

The primary banking datasets are MBD-mini, PaySim, BankSim and IBM AML. The five
TPP-LLM/GEM temporal-tokenization datasets are also available for external
reproduction experiments.
"""

from eventfm.datasets.registry import (
    CHRONOLOGICAL_DATASETS,
    DATASET_REGISTRY,
    GEM_DATASETS,
    PRIMARY_DATASETS,
    DatasetMeta,
    build_dataset,
    dataset_names,
    load_dataset_meta,
    processed_dir,
)

__all__ = [
    "DATASET_REGISTRY",
    "CHRONOLOGICAL_DATASETS",
    "GEM_DATASETS",
    "PRIMARY_DATASETS",
    "DatasetMeta",
    "build_dataset",
    "dataset_names",
    "load_dataset_meta",
    "processed_dir",
]
