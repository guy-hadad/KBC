"""ETAM-2 (Stage 2) command line: any stage from dotted ``key=value`` overrides.

Examples::

    # pretrain the reference ETAM-2 model on the MBD pool (10 pool epochs)
    python scripts/etam2.py run stage=pretrain run_name=ref output_dir=/tmp/x

    # one ablation = one override
    python scripts/etam2.py run stage=pretrain model.decoder=time_first ...

    # fine-tune, evaluate or simulate a checkpoint (Stage 1 checkpoints work for
    # evaluate/simulate through an adapter)
    python scripts/etam2.py run stage=finetune task=multi init_checkpoint=.../best.pt
    python scripts/etam2.py run stage=evaluate init_checkpoint=.../best.pt
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(payload: dict, overrides) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError("Override `{}` must look like key=value.".format(item))
        key, raw = item.split("=", 1)
        target = payload
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _parse_value(raw)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("run")
    stage.add_argument("--config", default="")
    stage.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    from eventfm.etam.stage2.train import RunConfig2, run

    payload = {}
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    payload = apply_overrides(payload, args.overrides)
    results = run(RunConfig2.from_dict(payload))
    print(json.dumps({key: value for key, value in results.items() if not isinstance(value, dict)}, default=float))


if __name__ == "__main__":
    main()
