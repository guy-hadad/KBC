"""Dataset registry: names, build entrypoints and cached metadata."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
    "synthetic": DatasetSpec(
        name="synthetic",
        display_name="Synthetic",
        source="eventfm.data.synthetic",
        citation="in-repo generator, used for smoke tests only",
        entity="synthetic user",
        mark="type_<idx> (50 types)",
        label="generator-assigned binary label",
    ),
}


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
    else:
        from eventfm.datasets.adapters import ADAPTERS

        table = ADAPTERS[name](conversion)
        splits, raw_meta = build_sequences(table, conversion)

    paths = write_splits(splits, output_dir)
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
            "source": spec.source,
            "citation": spec.citation,
            "entity": spec.entity,
            "mark": spec.mark,
            "label": spec.label,
            "natural_positive_rate": raw_meta.get("natural_positive_rate"),
            "used_positive_rate": raw_meta.get("used_positive_rate"),
            "num_entities_total": raw_meta.get("num_entities_total"),
            "num_rows": raw_meta.get("num_rows"),
        },
    )
    write_meta(output_dir, meta.to_json())
    return meta


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
