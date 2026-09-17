"""Download and convert the open benchmarks into the shared JSONL contract.

    python scripts/prepare_data.py                # all datasets
    python scripts/prepare_data.py banksim paysim # a subset
    python scripts/prepare_data.py --force        # rebuild from raw
    python scripts/prepare_data.py --scale        # only the three large datasets
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eventfm.datasets.paths import configure_hf_cache, processed_root  # noqa: E402
from eventfm.datasets.registry import build_dataset, resolve_prepare_names  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=None, help="dataset names (default: all)")
    parser.add_argument("--force", action="store_true", help="rebuild even if meta.json exists")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="prepare primary, chronological, and external GEM datasets",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="prepare the three large-scale datasets (full MBD, Synthea, Amazon Beauty)",
    )
    args = parser.parse_args()

    configure_hf_cache()
    names = resolve_prepare_names(args.datasets, paper=args.paper, scale=args.scale)
    summary = {}
    for name in names:
        print("[prepare] {} ...".format(name), flush=True)
        meta = build_dataset(name, force=args.force)
        summary[name] = {
            "num_event_types": meta.num_event_types,
            "stats": meta.stats,
            "natural_positive_rate": meta.extra.get("natural_positive_rate"),
            "used_positive_rate": meta.extra.get("used_positive_rate"),
        }
        print(json.dumps({name: summary[name]}, indent=2, default=str), flush=True)

    print("\nprocessed root: {}".format(processed_root()))


if __name__ == "__main__":
    main()
