"""Collect finished benchmark cells into tables and scaling figures.

    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results-dir <dir> --publish docs/benchmark
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from eventfm.benchmark.report import (  # noqa: E402
    load_results,
    write_csv,
    write_dataset_comparison_figures,
    write_method_scaling_figures,
    write_result_tables,
)
from eventfm.datasets.paths import output_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--publish", type=str, default="docs/benchmark")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else (output_root() / "results")
    report_dir = Path(args.report_dir) if args.report_dir else (output_root() / "report")
    report_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(results_dir)
    if not records:
        raise SystemExit("No result files found under {}".format(results_dir))

    statuses = Counter(record.status for record in records)
    print("loaded {} cells: {}".format(len(records), dict(statuses)))
    failures = [record for record in records if record.status != "ok"]
    if failures:
        print("\nfailed cells ({}):".format(len(failures)))
        for record in failures[:20]:
            print(
                "  {} / {} / {} / n={}".format(
                    record.dataset, record.task, record.method, record.sample_size
                )
            )

    csv_path = write_csv(records, report_dir / "results.csv")
    table_path = write_result_tables(records, report_dir / "results.md")
    print("\ntables: {}\ncsv:    {}".format(table_path, csv_path))

    figures = []
    if not args.no_figures:
        figures += write_method_scaling_figures(records, report_dir / "figures")
        figures += write_dataset_comparison_figures(records, report_dir / "figures")
        print("figures: {} files under {}".format(len(figures), report_dir / "figures"))

    if args.publish:
        target = Path(args.publish)
        if not target.is_absolute():
            target = Path(__file__).resolve().parents[1] / target
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, target / csv_path.name)
        shutil.copy2(table_path, target / table_path.name)
        for figure in figures:
            destination = target / "figures" / figure.relative_to(report_dir / "figures")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(figure, destination)
        print("published to {}".format(target))


if __name__ == "__main__":
    main()
