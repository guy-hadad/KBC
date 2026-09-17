"""Dataset registry: names, build entrypoints and cached metadata."""

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from eventfm.datasets.paths import processed_root
from eventfm.datasets.tabular import (
    ConversionConfig,
    build_sequences,
    summarise_splits,
    write_meta,
    write_splits,
)


@dataclass
class DatasetSpec:
    name: str
    display_name: str
    source: str
    citation: str
    entity: str
    mark: str
    label: str
    config: ConversionConfig = field(default_factory=ConversionConfig)
    supports: Tuple[str, ...] = ("classification", "tpp")
    adapter: Optional[str] = None


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "banksim": DatasetSpec(
        name="banksim",
        display_name="BankSim",
        source="github:atavci/fraud-detection-on-banksim-data",
        citation="Lopez-Rojas & Axelsson, BankSim payment simulator",
        entity="customer",
        mark="merchant category (15 types)",
        label="fraud occurs in the held-out tail of the history",
        config=ConversionConfig(
            min_events_per_entity=16,
            max_events_per_entity=128,
            label_horizon_fraction=0.30,
            target_positive_rate=None,
            max_entities=None,
        ),
    ),
    "paysim": DatasetSpec(
        name="paysim",
        display_name="PaySim",
        source="hf:theman10/paysim",
        citation="Lopez-Rojas et al., PaySim mobile-money simulator",
        entity="destination account",
        mark="transfer type (5 types)",
        label="fraud occurs in the held-out tail of the history",
        config=ConversionConfig(
            min_events_per_entity=8,
            max_events_per_entity=64,
            label_horizon_fraction=0.30,
            target_positive_rate=0.20,
            max_entities=40000,
        ),
    ),
    "ibm_aml": DatasetSpec(
        name="ibm_aml",
        display_name="IBM AML (HI-Small)",
        source="hf:OsamaMIT/IBM-AML-HI-Small",
        citation="Altman et al., IBM AMLSim transactions for anti-money laundering",
        entity="receiving account",
        mark="payment format (7 types)",
        label="laundering occurs in the held-out tail of the history",
        config=ConversionConfig(
            min_events_per_entity=8,
            max_events_per_entity=64,
            label_horizon_fraction=0.30,
            target_positive_rate=0.20,
            max_entities=40000,
        ),
    ),
    "mbd_mini": DatasetSpec(
        name="mbd_mini",
        display_name="MBD-mini",
        source="hf:ai-lab/MBD-mini",
        citation="Multimodal Banking Dataset (MBD), 10% client subsample",
        entity="bank client",
        mark="transaction event type",
        label="product-propensity target shipped with the benchmark",
        config=ConversionConfig(
            min_events_per_entity=16,
            max_events_per_entity=128,
            max_entities=40000,
            target_positive_rate=0.20,
        ),
    ),
    "mbd": DatasetSpec(
        name="mbd",
        display_name="MBD (full)",
        source="hf:ai-lab/MBD",
        citation="Multimodal Banking Dataset (MBD), full release",
        entity="bank client",
        mark="transaction event type",
        label="product-propensity target shipped with the benchmark",
        config=ConversionConfig(
            min_events_per_entity=16,
            max_events_per_entity=128,
            max_entities=None,
            max_eval_entities=40000,
            target_positive_rate=0.20,
        ),
    ),
    "synthea": DatasetSpec(
        name="synthea",
        display_name="Synthea EHR",
        source="hf:richardyoung/synthea-575k-patients",
        citation="Walonoski et al., Synthea synthetic patient generator (575k release)",
        entity="patient",
        mark="SNOMED condition description",
        label="major adverse cardiovascular or renal event in the held-out tail",
        config=ConversionConfig(
            min_events_per_entity=16,
            max_events_per_entity=128,
            label_horizon_fraction=0.30,
            max_entities=None,
            target_positive_rate=0.20,
        ),
    ),
    "amazon_beauty": DatasetSpec(
        name="amazon_beauty",
        display_name="Amazon Beauty 2014",
        source="hf:milistu/Amazon_Beauty_2014",
        citation="McAuley et al., Amazon product data (2014), Beauty category",
        entity="reviewer",
        mark="item level-2 product category",
        label="negative review (rating <= 2) in the held-out tail",
        config=ConversionConfig(
            min_events_per_entity=5,
            max_events_per_entity=64,
            label_horizon_fraction=0.30,
            max_entities=None,
            target_positive_rate=0.20,
        ),
    ),
    "synthetic": DatasetSpec(
        name="synthetic",
        display_name="Synthetic",
        source="eventfm.data.synthetic",
        citation="in-repo generator, used for smoke tests only",
        entity="synthetic user",
        mark="type_<idx> (50 types)",
        label="generator-assigned binary label",
    ),
    "stackoverflow": DatasetSpec(
        name="stackoverflow",
        display_name="Stack Overflow",
        source="hf:tppllm/stack-overflow-description",
        citation="TPP-LLM public event-sequence benchmark",
        entity="user sequence",
        mark="badge text (25 types)",
        label="next event type and time only",
        supports=("tpp",),
    ),
    "chicago_crime": DatasetSpec(
        name="chicago_crime",
        display_name="Chicago Crime",
        source="hf:tppllm/chicago-crime-description",
        citation="TPP-LLM public event-sequence benchmark",
        entity="spatial sequence",
        mark="crime event text (20 types)",
        label="next event type and time only",
        supports=("tpp",),
    ),
    "nyc_taxi": DatasetSpec(
        name="nyc_taxi",
        display_name="NYC Taxi",
        source="hf:tppllm/nyc-taxi-description",
        citation="TPP-LLM public event-sequence benchmark",
        entity="taxi trajectory sequence",
        mark="taxi event text (8 types)",
        label="next event type and time only",
        supports=("tpp",),
    ),
    "us_earthquake": DatasetSpec(
        name="us_earthquake",
        display_name="US Earthquake",
        source="hf:tppllm/us-earthquake-description",
        citation="TPP-LLM public event-sequence benchmark",
        entity="earthquake sequence",
        mark="earthquake event text (3 types)",
        label="next event type and time only",
        supports=("tpp",),
    ),
    "amazon_review": DatasetSpec(
        name="amazon_review",
        display_name="Amazon Review",
        source="hf:tppllm/amazon-review-description",
        citation="TPP-LLM public event-sequence benchmark",
        entity="reviewer sequence",
        mark="review category text (18 types)",
        label="next event type and time only",
        supports=("tpp",),
    ),
}

PRIMARY_DATASETS = ("banksim", "paysim", "ibm_aml", "mbd_mini")
# The three large-scale additions. These carry the sample-scaling programme past
# the ~3k-sequence ceiling of the primary benchmarks and add a medical and a
# recommendation domain to what was a banking-only screen.
SCALE_DATASETS = ("mbd", "synthea", "amazon_beauty")
CHRONOLOGICAL_DATASETS = tuple("{}_chrono".format(name) for name in PRIMARY_DATASETS)
GEM_DATASETS = (
    "stackoverflow",
    "chicago_crime",
    "nyc_taxi",
    "us_earthquake",
    "amazon_review",
)

for _base_name, _chronological_name in zip(PRIMARY_DATASETS, CHRONOLOGICAL_DATASETS):
    _base_spec = DATASET_REGISTRY[_base_name]
    DATASET_REGISTRY[_chronological_name] = DatasetSpec(
        name=_chronological_name,
        display_name="{} (chronological)".format(_base_spec.display_name),
        source=_base_spec.source,
        citation=_base_spec.citation,
        entity=_base_spec.entity,
        mark=_base_spec.mark,
        label=_base_spec.label,
        config=replace(_base_spec.config, split_strategy="chronological"),
        supports=_base_spec.supports,
        adapter=_base_name,
    )


@dataclass
class DatasetMeta:
    name: str
    display_name: str
    splits: Dict[str, str]
    event_types: List[str]
    feature_fields: List[str]
    profile_fields: List[str]
    num_labels: int = 2
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def num_event_types(self) -> int:
        return len(self.event_types)

    def event_type_to_index(self) -> Dict[str, int]:
        return {event_type: index for index, event_type in enumerate(self.event_types)}

    def to_json(self) -> Dict[str, object]:
        payload = {
            "name": self.name,
            "display_name": self.display_name,
            "splits": self.splits,
            "event_types": self.event_types,
            "feature_fields": self.feature_fields,
            "profile_fields": self.profile_fields,
            "num_labels": self.num_labels,
            "num_event_types": self.num_event_types,
            "stats": self.stats,
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "DatasetMeta":
        known = {
            "name",
            "display_name",
            "splits",
            "event_types",
            "feature_fields",
            "profile_fields",
            "num_labels",
            "num_event_types",
            "stats",
        }
        return cls(
            name=str(payload["name"]),
            display_name=str(payload.get("display_name", payload["name"])),
            splits={str(k): str(v) for k, v in dict(payload["splits"]).items()},
            event_types=[str(value) for value in payload["event_types"]],
            feature_fields=[str(value) for value in payload.get("feature_fields", [])],
            profile_fields=[str(value) for value in payload.get("profile_fields", [])],
            num_labels=int(payload.get("num_labels", 2)),
            stats=dict(payload.get("stats", {})),  # type: ignore[arg-type]
            extra={k: v for k, v in payload.items() if k not in known},
        )


def dataset_names() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def resolve_prepare_names(
    datasets: Optional[Sequence[str]] = None,
    paper: bool = False,
    scale: bool = False,
) -> List[str]:
    """Which datasets `scripts/prepare_data.py` should build.

    Explicit names win. ``--scale`` on its own means the three large datasets
    only, not the primaries as well, so preparing them does not walk the whole
    primary set first.
    """

    if datasets:
        return list(datasets)
    if paper and scale:
        return list(PRIMARY_DATASETS + CHRONOLOGICAL_DATASETS + GEM_DATASETS + SCALE_DATASETS)
    if paper:
        return list(PRIMARY_DATASETS + CHRONOLOGICAL_DATASETS + GEM_DATASETS)
    if scale:
        return list(SCALE_DATASETS)
    return list(PRIMARY_DATASETS)


def processed_dir(name: str) -> Path:
    return processed_root() / name


def load_dataset_meta(name: str) -> DatasetMeta:
    path = processed_dir(name) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            "Dataset `{}` is not prepared yet. "
            "Run scripts/prepare_data.py {}".format(name, name)
        )
    with path.open("r", encoding="utf-8") as handle:
        return DatasetMeta.from_json(json.load(handle))


def build_dataset(
    name: str,
    config: Optional[ConversionConfig] = None,
    force: bool = False,
) -> DatasetMeta:
    """Download, convert and cache one benchmark; returns its metadata."""

    if name not in DATASET_REGISTRY:
        raise KeyError("Unknown dataset `{}`. Available: {}".format(name, dataset_names()))
    output_dir = processed_dir(name)
    if not force and (output_dir / "meta.json").exists():
        return load_dataset_meta(name)

    spec = DATASET_REGISTRY[name]
    conversion = config or spec.config

    if name == "synthetic":
        splits, raw_meta = _build_synthetic(conversion)
    elif name in GEM_DATASETS:
        splits, raw_meta = _build_tppllm(name)
    else:
        from eventfm.datasets.adapters import ADAPTERS

        table = ADAPTERS[spec.adapter or name](conversion)
        splits, raw_meta = build_sequences(table, conversion)

    paths = write_splits(splits, output_dir)
    split_sha256 = {}
    for split, path in paths.items():
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        split_sha256[split] = digest.hexdigest()
    reserved = {"event_types", "feature_fields", "profile_fields"}
    conversion_metadata = {key: value for key, value in raw_meta.items() if key not in reserved}
    meta = DatasetMeta(
        name=name,
        display_name=spec.display_name,
        splits=paths,
        event_types=[str(value) for value in raw_meta["event_types"]],
        feature_fields=[str(value) for value in raw_meta["feature_fields"]],
        profile_fields=[str(value) for value in raw_meta.get("profile_fields", [])],
        num_labels=2,
        stats=summarise_splits(splits),
        extra={
            **conversion_metadata,
            "source": spec.source,
            "citation": spec.citation,
            "entity": spec.entity,
            "mark": spec.mark,
            "label": spec.label,
            "supported_tasks": list(spec.supports),
            "split_sha256": split_sha256,
        },
    )
    write_meta(output_dir, meta.to_json())
    return meta


def _build_tppllm(name: str):
    """Convert the public TPP-LLM sequence datasets without regrouping rows."""

    from datasets import load_dataset

    from eventfm.data.schema import Event, EventSequence

    repository = str(DATASET_REGISTRY[name].source).replace("hf:", "", 1)
    dataset = load_dataset(repository)
    unit_seconds = {
        "stackoverflow": 30.0 * 86400.0,
        "chicago_crime": 30.0 * 86400.0,
        "nyc_taxi": 3600.0,
        "us_earthquake": 86400.0,
        "amazon_review": 7.0 * 86400.0,
    }[name]
    origin = 1_640_995_200.0
    splits = {"train": [], "validation": [], "test": []}
    event_types = set()
    for split_name in splits:
        if split_name not in dataset:
            continue
        for row_index, row in enumerate(dataset[split_name]):
            times = list(row.get("time_since_start") or [])
            types = list(row.get("type_text") or row.get("type_event") or [])
            size = min(len(times), len(types))
            events = []
            for index in range(size):
                event_type = str(types[index])
                event_types.add(event_type)
                events.append(
                    Event(
                        event_type=event_type,
                        timestamp=origin + float(times[index]) * unit_seconds,
                    )
                )
            if not events:
                continue
            sequence_id = row.get("seq_idx", row_index)
            splits[split_name].append(
                EventSequence(
                    user_id="{}:{}:{}".format(name, split_name, sequence_id),
                    events=events,
                    label=None,
                    metadata={
                        "source_split": split_name,
                        "description": row.get("description"),
                        "time_unit_seconds": unit_seconds,
                    },
                )
            )
    return splits, {
        "event_types": sorted(event_types),
        "feature_fields": [],
        "profile_fields": [],
        "natural_positive_rate": None,
        "used_positive_rate": None,
        "num_entities_total": sum(len(values) for values in splits.values()),
        "num_rows": sum(
            len(sequence.events) for values in splits.values() for sequence in values
        ),
    }


def _build_synthetic(config: ConversionConfig):
    from eventfm.data.synthetic import generate_synthetic_sequences

    num_event_types = 50
    sequences = generate_synthetic_sequences(
        num_users=6000,
        num_event_types=num_event_types,
        min_events=16,
        max_events=64,
        seed=config.seed,
    )
    splits = {"train": [], "validation": [], "test": []}
    for index, sequence in enumerate(sequences):
        bucket = index % 20
        split = "test" if bucket < 4 else ("validation" if bucket < 7 else "train")
        splits[split].append(sequence)
    meta = {
        "event_types": ["type_{}".format(index) for index in range(num_event_types)],
        "feature_fields": [],
        "profile_fields": [],
        "natural_positive_rate": None,
        "used_positive_rate": None,
        "num_entities_total": len(sequences),
        "num_rows": sum(len(sequence.events) for sequence in sequences),
    }
    return splits, meta
