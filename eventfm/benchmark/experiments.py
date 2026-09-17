"""Named, version-controlled experiment suites for the research programme."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from eventfm.data.temporal_tokenizers import temporal_tokenizer_names
from eventfm.datasets.registry import (
    CHRONOLOGICAL_DATASETS,
    GEM_DATASETS,
    PRIMARY_DATASETS,
    SCALE_DATASETS,
    load_dataset_meta,
)
from eventfm.methods.gem_tokenization import method_key_for_tokenizer

DEFAULT_SIZES = (64, 128, 256, 512, 1024, 2048)

# The scaling programme on the large datasets continues in octaves past the
# 2048-sequence ceiling of the primary benchmarks. The grid is clamped per
# dataset at enumeration time, so the last point is always the dataset's whole
# training pool rather than a repeat of the previous point.
SCALE_GRID = (
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)
AUTO_SCALING = "auto-scaling"
SCREENING_SEEDS = (13, 29, 43)
HEADLINE_SEEDS = (13, 29, 43, 71, 101)


def scaling_sample_sizes(dataset: str, grid: Tuple[int, ...] = SCALE_GRID) -> Tuple[int, ...]:
    """Octave grid for one dataset, ending at the size of its training pool.

    Falls back to :data:`DEFAULT_SIZES` when the dataset has not been prepared
    yet, so `--list` works before `scripts/prepare_data.py` has run.
    """

    try:
        meta = load_dataset_meta(dataset)
    except FileNotFoundError:
        return DEFAULT_SIZES
    pool = int(meta.stats.get("train", {}).get("num_sequences", 0) or 0)
    if pool <= 0:
        return DEFAULT_SIZES
    # A grid point within 25 % of the pool would be a near-duplicate of the
    # final point and would weight the scaling fit at one end of the range.
    sizes = [size for size in grid if size * 1.25 <= pool]
    sizes.append(pool)
    return tuple(sizes)


@dataclass(frozen=True)
class ExperimentSuite:
    name: str
    description: str
    datasets: Tuple[str, ...]
    tasks: Tuple[str, ...]
    methods: Tuple[str, ...]
    sample_sizes: Tuple[int, ...] = DEFAULT_SIZES
    seeds: Tuple[int, ...] = SCREENING_SEEDS
    regimes: Tuple[str, ...] = ("full",)
    extra: Dict[str, object] = field(default_factory=dict)
    variants: Tuple[str, ...] = ("default",)
    variant_extra: Dict[str, Dict[str, object]] = field(default_factory=dict)


def resolve_sample_sizes(suite: "ExperimentSuite"):
    """Sample sizes for a suite: one shared grid, or one grid per dataset.

    A suite that declares ``sample_size_mode: auto-scaling`` gets a per-dataset
    mapping, because its datasets have training pools that differ by an order of
    magnitude and a shared grid would either truncate or repeat points.
    """

    if suite.extra.get("sample_size_mode") != AUTO_SCALING:
        return suite.sample_sizes
    return {dataset: scaling_sample_sizes(dataset) for dataset in suite.datasets}


CORE_METHODS = (
    "count-logistic",
    "markov",
    "ntpp-gru",
    "thp",
    "pragma",
    "pragma-mlm",
    "tabbert",
    "tabgpt",
    "nvidia-tfm",
    "coles",
    "mambular",
    "tpp-llm",
    "language-tpp",
    "mm-tpp",
)

STRONG_BASELINES = (
    "count-logistic",
    "engineered-gbdt",
    "markov",
    "supervised-gru",
    "autoencoder",
    "transaction-mlm",
    "autoregressive-transformer",
    "coles",
)

OBJECTIVE_METHODS = (
    "supervised-gru",
    "autoencoder",
    "transaction-mlm",
    "autoregressive-transformer",
    "coles",
    "nppr",
    "mlem",
    "cmlm-coles",
    "motor",
    "ora",
    "eventfm-joint",
)

TEMPORAL_METHODS = (
    "markov",
    "ntpp-gru",
    "nhp",
    "thp",
    "sahp",
    "attnhp",
    "iftpp",
    "cotic",
    "tpp-llm",
    "language-tpp",
)

TIME_ABLATIONS = (
    "time-no-time",
    "time-ordinal",
    "time-raw-gap",
    "time-log-gap",
    "time-gap-age",
    "time-calendar-fourier",
    "time-gap-age-calendar",
    "time-time2vec",
    "time-functional",
    "time-continuous-rope",
    "time-discrete-buckets",
)

GEM_MAIN_TOKENIZERS = (
    "time-numeric-p6",
    "time-byte-f32",
    "time-calendar-abs-day",
    "time-calendar-abs-second",
    "time-calendar-rel-day",
    "time-calendar-rel-second",
    "time-bin-linear-k256",
    "time-bin-log-k256",
    "time-rsq-linear-256",
    "time-rsq-linear-64-64-64-64",
    "time-rsq-log-256",
    "time-rsq-log-64-64-64-64",
)


def _gem_methods(names) -> Tuple[str, ...]:
    return tuple(method_key_for_tokenizer(name) for name in names)


EXPERIMENT_SUITES: Dict[str, ExperimentSuite] = {
    "core-reproduction": ExperimentSuite(
        "core-reproduction",
        "Original 624-cell architecture-screening grid, repeated with three seeds.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        CORE_METHODS,
    ),
    "strong-baselines": ExperimentSuite(
        "strong-baselines",
        "Non-neural, supervised, reconstructive, autoregressive, and contrastive controls.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        STRONG_BASELINES,
    ),
    "objective-ablation": ExperimentSuite(
        "objective-ablation",
        "Single and hybrid representation-learning objectives under shared budgets.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        OBJECTIVE_METHODS,
    ),
    "objective-factorial": ExperimentSuite(
        "objective-factorial",
        "Single, pairwise, and full EventFM objectives under the same encoder and budget.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        ("eventfm-joint",),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        variants=(
            "masked",
            "next",
            "marked-tte",
            "contrastive",
            "masked-next",
            "next-tte",
            "masked-contrastive",
            "next-contrastive",
            "full",
        ),
        variant_extra={
            "masked": {"pretraining_objectives": ["masked"], "use_contrastive": False},
            "next": {
                "pretraining_objectives": ["next", "next-feature", "next-time"],
                "use_contrastive": False,
            },
            "marked-tte": {
                "pretraining_objectives": ["marked-tte"],
                "use_contrastive": False,
            },
            "contrastive": {"pretraining_objectives": [], "use_contrastive": True},
            "masked-next": {
                "pretraining_objectives": ["masked", "next", "next-feature", "next-time"],
                "use_contrastive": False,
            },
            "next-tte": {
                "pretraining_objectives": ["next", "next-feature", "marked-tte"],
                "use_contrastive": False,
            },
            "masked-contrastive": {
                "pretraining_objectives": ["masked"],
                "use_contrastive": True,
            },
            "next-contrastive": {
                "pretraining_objectives": ["next", "next-feature", "next-time"],
                "use_contrastive": True,
            },
            "full": {
                "pretraining_objectives": [
                    "masked",
                    "next",
                    "next-feature",
                    "marked-tte",
                ],
                "use_contrastive": True,
            },
        },
    ),
    "temporal-models": ExperimentSuite(
        "temporal-models",
        "Recurrent, attention, intensity-free, convolutional, and LLM temporal models.",
        PRIMARY_DATASETS,
        ("tpp",),
        TEMPORAL_METHODS,
    ),
    "time-representation-ablation": ExperimentSuite(
        "time-representation-ablation",
        "All continuous, calendar, positional, bucket, and rotary time conditions.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        TIME_ABLATIONS,
    ),
    "pooling-ablation": ExperimentSuite(
        "pooling-ablation",
        "Last-event, mean, and max customer readouts under otherwise fixed objectives.",
        PRIMARY_DATASETS,
        ("classification",),
        ("autoencoder", "coles", "eventfm-joint"),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        variants=("last", "mean", "max"),
        variant_extra={
            "last": {"pooling_override": "last"},
            "mean": {"pooling_override": "mean"},
            "max": {"pooling_override": "max"},
        },
    ),
    "context-length-ablation": ExperimentSuite(
        "context-length-ablation",
        "Short, medium, and long event contexts with truncation and efficiency metrics.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        ("supervised-gru", "attnhp", "cotic", "eventfm-joint"),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        variants=("events-32", "events-64", "events-128"),
        variant_extra={
            "events-32": {"max_events": 32},
            "events-64": {"max_events": 64},
            "events-128": {"max_events": 128},
        },
    ),
    "model-scale-ablation": ExperimentSuite(
        "model-scale-ablation",
        "Small, base, and larger shared backbones under fixed data and tasks.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        ("supervised-gru", "attnhp", "eventfm-joint"),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        variants=("small", "base", "large"),
        variant_extra={
            "small": {"hidden_size": 64, "num_hidden_layers": 1},
            "base": {"hidden_size": 128, "num_hidden_layers": 2},
            "large": {"hidden_size": 256, "num_hidden_layers": 4},
        },
    ),
    "adaptation-regimes": ExperimentSuite(
        "adaptation-regimes",
        "Frozen probes, bottleneck-adapter PEFT, and full tuning after pretraining.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        (
            "autoencoder",
            "transaction-mlm",
            "autoregressive-transformer",
            "coles",
            "nppr",
            "cmlm-coles",
            "motor",
            "ora",
            "eventfm-joint",
        ),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        regimes=("frozen", "peft", "full"),
    ),
    "gem-main-banking": ExperimentSuite(
        "gem-main-banking",
        "The GEM 2026 main tokenizer table on the four banking datasets.",
        PRIMARY_DATASETS,
        ("tpp",),
        _gem_methods(GEM_MAIN_TOKENIZERS) + ("tpp-llm",),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
    ),
    "gem-full-banking": ExperimentSuite(
        "gem-full-banking",
        "All calendar resolutions and RSQ level allocations on banking datasets.",
        PRIMARY_DATASETS,
        ("tpp",),
        _gem_methods(temporal_tokenizer_names()) + ("tpp-llm",),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
    ),
    "gem-reference-datasets": ExperimentSuite(
        "gem-reference-datasets",
        "GEM tokenizer reproduction on Stack Overflow, crime, taxi, earthquake, and reviews.",
        GEM_DATASETS,
        ("tpp",),
        _gem_methods(GEM_MAIN_TOKENIZERS) + ("tpp-llm",),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
    ),
    "gem-template-time-type": ExperimentSuite(
        "gem-template-time-type",
        "Prompt-order ablation with time tokens before event type tokens.",
        ("stackoverflow", "nyc_taxi"),
        ("tpp",),
        _gem_methods(GEM_MAIN_TOKENIZERS),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        extra={"gem_template_order": "time-type"},
    ),
    "gem-context-budget": ExperimentSuite(
        "gem-context-budget",
        "Equal-event versus equal-token prompt budgets for the main GEM tokenizers.",
        PRIMARY_DATASETS,
        ("tpp",),
        _gem_methods(GEM_MAIN_TOKENIZERS),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        variants=("equal-events", "equal-tokens"),
        variant_extra={
            "equal-events": {"gem_max_events": 32, "gem_max_text_length": 2048},
            "equal-tokens": {"gem_max_events": 128, "gem_max_text_length": 768},
        },
    ),
    "gem-model-scale": ExperimentSuite(
        "gem-model-scale",
        "Tokenizer ranking on controlled small and larger causal-LM backbones.",
        PRIMARY_DATASETS,
        ("tpp",),
        _gem_methods(GEM_MAIN_TOKENIZERS),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        variants=("smollm-135m", "smollm-360m"),
        variant_extra={
            "smollm-135m": {"llm_base_model": "HuggingFaceTB/SmolLM2-135M"},
            "smollm-360m": {"llm_base_model": "HuggingFaceTB/SmolLM2-360M"},
        },
    ),
    "gem-event-text": ExperimentSuite(
        "gem-event-text",
        "Descriptive event names versus anonymized type IDs for language-prior analysis.",
        PRIMARY_DATASETS,
        ("tpp",),
        _gem_methods(GEM_MAIN_TOKENIZERS),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
        variants=("descriptive", "anonymous"),
        variant_extra={
            "descriptive": {"anonymize_event_types": False},
            "anonymous": {"anonymize_event_types": True},
        },
    ),
    "paper-headline": ExperimentSuite(
        "paper-headline",
        "Compute-manageable five-seed suite spanning every central method family.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        (
            "engineered-gbdt",
            "supervised-gru",
            "coles",
            "transaction-mlm",
            "autoregressive-transformer",
            "nppr",
            "cmlm-coles",
            "pragma-mlm",
            "sohet",
            "nhp",
            "attnhp",
            "iftpp",
            "cotic",
            "ora",
            "eventfm-joint",
        ),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
    ),
    "chronological-robustness": ExperimentSuite(
        "chronological-robustness",
        "Headline methods on whole-entity splits ordered by final observed timestamp.",
        CHRONOLOGICAL_DATASETS,
        ("classification", "tpp"),
        (
            "engineered-gbdt",
            "supervised-gru",
            "coles",
            "transaction-mlm",
            "attnhp",
            "cotic",
            "ora",
            "eventfm-joint",
        ),
        sample_sizes=(2048,),
        seeds=HEADLINE_SEEDS,
    ),
    "cross-schema-transfer": ExperimentSuite(
        "cross-schema-transfer",
        "In-domain versus leave-one-banking-dataset-out pretraining at fixed target labels.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        ("transaction-mlm", "coles", "cmlm-coles", "eventfm-joint"),
        sample_sizes=(64, 256, 2048),
        seeds=HEADLINE_SEEDS,
        regimes=("frozen", "peft", "full"),
        variants=("in-domain", "leave-one-out"),
        variant_extra={
            "in-domain": {},
            "leave-one-out": {"pretrain_datasets": "leave-one-out-primary"},
        },
    ),
    "scale-datasets": ExperimentSuite(
        "scale-datasets",
        "Sample-scaling curves from 64 sequences to the whole training pool on "
        "full MBD, Synthea EHR and Amazon Beauty 2014.",
        SCALE_DATASETS,
        ("classification", "tpp"),
        CORE_METHODS,
        seeds=SCREENING_SEEDS,
        extra={"sample_size_mode": AUTO_SCALING},
    ),
    "scale-datasets-baselines": ExperimentSuite(
        "scale-datasets-baselines",
        "The strong supervised baselines that the core zoo does not already "
        "cover, over the same large-dataset scaling grid.",
        SCALE_DATASETS,
        ("classification", "tpp"),
        # Disjoint from `scale-datasets` on purpose: the controls and CoLES
        # appear in both method lists, and running them twice would put the same
        # cell in two result directories.
        tuple(name for name in STRONG_BASELINES if name not in CORE_METHODS),
        seeds=SCREENING_SEEDS,
        extra={"sample_size_mode": AUTO_SCALING},
    ),
    "labeled-adaptation-scaling": ExperimentSuite(
        "labeled-adaptation-scaling",
        "Nested target-label sizes after a fixed-size in-domain self-supervised stage.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        ("transaction-mlm", "coles", "cmlm-coles", "eventfm-joint"),
        sample_sizes=DEFAULT_SIZES,
        seeds=SCREENING_SEEDS,
        extra={
            "pretrain_datasets": "target",
            "pretrain_sample_size_per_dataset": 2048,
        },
    ),
    "unlabeled-pretraining-scaling": ExperimentSuite(
        "unlabeled-pretraining-scaling",
        "Source-pretraining scale at a fixed target adaptation size.",
        PRIMARY_DATASETS,
        ("classification", "tpp"),
        ("transaction-mlm", "coles", "cmlm-coles", "eventfm-joint"),
        sample_sizes=(256,),
        seeds=SCREENING_SEEDS,
        variants=("pretrain-64", "pretrain-256", "pretrain-1024", "pretrain-2048"),
        variant_extra={
            "pretrain-64": {
                "pretrain_datasets": "leave-one-out-primary",
                "pretrain_sample_size_per_dataset": 64,
            },
            "pretrain-256": {
                "pretrain_datasets": "leave-one-out-primary",
                "pretrain_sample_size_per_dataset": 256,
            },
            "pretrain-1024": {
                "pretrain_datasets": "leave-one-out-primary",
                "pretrain_sample_size_per_dataset": 1024,
            },
            "pretrain-2048": {
                "pretrain_datasets": "leave-one-out-primary",
                "pretrain_sample_size_per_dataset": 2048,
            },
        },
    ),
}


def experiment_names() -> List[str]:
    return sorted(EXPERIMENT_SUITES)


def get_experiment(name: str) -> ExperimentSuite:
    if name not in EXPERIMENT_SUITES:
        raise KeyError("Unknown experiment `{}`. Available: {}".format(name, experiment_names()))
    return EXPERIMENT_SUITES[name]
