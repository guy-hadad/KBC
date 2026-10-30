"""Turn one benchmark cell's RunResult JSON into a campaign ``result.json``.

`scripts/run_experiments.py` writes ``<results-dir>/<cell key>.json`` and exits 0
even when the cell failed (a failed cell must not kill an array). The campaign
driver needs the opposite: no ``result.json`` unless the cell succeeded.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    files = sorted(
        path for path in Path(args.results_dir).glob("*.json") if path.name != "result.json"
    )
    if len(files) != 1:
        sys.exit(
            "Expected exactly one cell result in {}, found {}.".format(args.results_dir, len(files))
        )
    payload = json.loads(files[0].read_text())
    if payload.get("status") != "ok":
        sys.exit("Cell {} failed:\n{}".format(files[0].name, payload.get("error")))
    result = dict(payload["metrics"])
    result["wall_minutes"] = payload.get("wall_seconds", 0.0) / 60.0
    result["cell_key"] = files[0].stem
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
