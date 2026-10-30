"""ETAM command line: build stores, then pretrain / fine-tune / probe.

Examples::

    # columnar stores for the benchmark splits (the pool comes from
    # scripts/build_mbd_pool.py)
    python scripts/etam.py build-store --dataset mbd

    # any stage from a JSON config plus dotted overrides
    python scripts/etam.py run --config configs/etam/pretrain_pool.json \\
        run_name=pool_base seed=13 model.placement=ffn weights.future=0.5
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eventfm.datasets.paths import output_root  # noqa: E402
from eventfm.datasets.registry import processed_dir  # noqa: E402
from eventfm.etam.store import EtamVocab, store_from_jsonl  # noqa: E402

# Copied from `eventfm.datasets.adapters.MBD_HISTORY_CUTOFF` so training jobs do
# not import polars (which dies with SIGILL on the cluster's older CPU nodes).
MBD_HISTORY_CUTOFF = 1_664_496_000.0
OBSERVATION_END = {"mbd": MBD_HISTORY_CUTOFF, "mbd_mini": MBD_HISTORY_CUTOFF}


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


def build_store(args) -> None:
    meta_path = processed_dir(args.dataset) / "meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    vocab = EtamVocab.from_meta(meta)
    root = Path(args.store_root) if args.store_root else output_root() / "etam" / "stores"
    for split in args.splits:
        store = store_from_jsonl(
            meta["splits"][split],
            vocab,
            observation_end=OBSERVATION_END.get(args.dataset),
            max_sequences=args.max_sequences or None,
        )
        target = store.save(str(root / args.dataset / split))
        print(
            "{}: {} sequences, {} events -> {}".format(split, len(store), store.num_events, target),
            flush=True,
        )


def run(args) -> None:
    from eventfm.etam.train import RunConfig
    from eventfm.etam.train import run as run_stage

    payload = {}
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    payload = apply_overrides(payload, args.overrides)
    results = run_stage(RunConfig.from_dict(payload))
    print(json.dumps(results, indent=2, default=float), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    store = commands.add_parser("build-store")
    store.add_argument("--dataset", default="mbd")
    store.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    store.add_argument("--max-sequences", type=int, default=0)
    store.add_argument("--store-root", default="")
    store.set_defaults(handler=build_store)

    stage = commands.add_parser("run")
    stage.add_argument("--config", default="")
    stage.add_argument("overrides", nargs="*")
    stage.set_defaults(handler=run)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
