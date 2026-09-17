#!/usr/bin/env python
"""Regenerate the paper-facing implementation manifest from live registries."""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eventfm.benchmark.experiments import (  # noqa: E402
    EXPERIMENT_SUITES,
    resolve_sample_sizes,
)
from eventfm.benchmark.runner import enumerate_cells  # noqa: E402
from eventfm.data.temporal_tokenizers import temporal_tokenizer_names  # noqa: E402
from eventfm.datasets.registry import DATASET_REGISTRY  # noqa: E402
from eventfm.methods import METHOD_REGISTRY  # noqa: E402


def _escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render() -> str:
    statuses = Counter(spec.status for spec in METHOD_REGISTRY.values())
    lines = [
        "# Runnable research manifest",
        "",
        "This file is generated from the method, dataset, tokenizer, and experiment",
        "registries. Regenerate it with `python scripts/export_research_manifest.py`.",
        "It inventories runnable code, not completed results or paper-faithful",
        "reproductions.",
        "",
        "## Coverage summary",
        "",
        "- {} registered methods: {} implemented controls and {} disclosed approximations;".format(
            len(METHOD_REGISTRY),
            statuses.get("Implemented", 0),
            statuses.get("Approximation", 0),
        ),
        "- {} prepared-dataset definitions;".format(len(DATASET_REGISTRY)),
        "- {} GEM temporal tokenizer configurations;".format(len(temporal_tokenizer_names())),
        "- {} named experiment suites.".format(len(EXPERIMENT_SUITES)),
        "",
        "`Implemented` means an in-repository control with a runnable contract and",
        "smoke coverage. Paper-named systems are deliberately marked `Approximation`",
        "when data, scale, likelihood, architecture, or training recipe differs.",
        "",
        "## Methods",
        "",
        (
            "| Key | Display name | Family | Tasks | Fidelity | GPU | Reference | "
            "Disclosed divergence |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, spec in sorted(METHOD_REGISTRY.items()):
        lines.append(
            "| `{}` | {} | `{}` | {} | {} | {} | {} | {} |".format(
                key,
                _escape(spec.display_name),
                _escape(spec.family),
                ", ".join(spec.supports),
                spec.status,
                "yes" if spec.requires_gpu else "no",
                _escape(spec.reference),
                _escape(spec.divergence),
            )
        )

    lines.extend(
        [
            "",
            "## Datasets",
            "",
            "| Key | Dataset | Supported tasks | Split | Source | Entity | Mark | Target |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for key, spec in DATASET_REGISTRY.items():
        lines.append(
            "| `{}` | {} | {} | `{}` | {} | {} | {} | {} |".format(
                key,
                _escape(spec.display_name),
                ", ".join(spec.supports),
                spec.config.split_strategy,
                _escape(spec.source),
                _escape(spec.entity),
                _escape(spec.mark),
                _escape(spec.label),
            )
        )

    lines.extend(
        [
            "",
            "## Experiment suites",
            "",
            "Cell counts include task-support filtering and all seeds, regimes, and variants.",
            "They describe scheduled runs; they do not imply that the cells have completed.",
            "",
            (
                "| Suite | Cells | Datasets | Tasks | Methods | Sizes | Seeds | Regimes | "
                "Variants | Purpose |"
            ),
            "| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for name, suite in sorted(EXPERIMENT_SUITES.items()):
        sizes = resolve_sample_sizes(suite)
        cells = enumerate_cells(
            suite.datasets,
            suite.tasks,
            suite.methods,
            sizes,
            suite.seeds,
            suite.regimes,
            suite.variants,
        )
        if isinstance(sizes, dict):
            size_label = "; ".join(
                "{}: {}".format(dataset, ", ".join(str(value) for value in grid))
                for dataset, grid in sizes.items()
            )
        else:
            size_label = ", ".join(str(value) for value in sizes)
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                len(cells),
                len(suite.datasets),
                ", ".join(suite.tasks),
                len(suite.methods),
                size_label,
                ", ".join(str(value) for value in suite.seeds),
                ", ".join(suite.regimes),
                ", ".join(suite.variants),
                _escape(suite.description),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The repository now covers every central comparison in the research plan",
            "with a runnable control or a disclosed approximation, including all GEM",
            "tokenizers. It does not make every external system a faithful reproduction.",
            "Faithful status requires reference-recipe parity, reference-data parity where",
            "available, validation against published numbers, and completed multi-seed runs.",
            "Related-work-only industrial systems remain outside the runnable manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs/research/implementation_manifest.md",
        help="manifest destination",
    )
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
