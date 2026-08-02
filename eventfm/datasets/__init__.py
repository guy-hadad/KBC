"""Open event-sequence benchmarks referenced by the KBC/TPP project note.

Every adapter downloads the public source, converts it to the shared
``EventSequence`` JSONL contract used by :mod:`eventfm.data`, and writes
``train/validation/test`` splits plus a ``meta.json`` describing the event
vocabulary and the classification target.

The four datasets are exactly the open benchmarks listed in section 3.1.5 of
the project note: MBD, PaySim, BankSim and the IBM AML generator.
"""

from eventfm.datasets.registry import (
    DATASET_REGISTRY,
    DatasetMeta,
    build_dataset,
    dataset_names,
    load_dataset_meta,
    processed_dir,
)

__all__ = [
    "DATASET_REGISTRY",
    "DatasetMeta",
    "build_dataset",
    "dataset_names",
    "load_dataset_meta",
    "processed_dir",
]
