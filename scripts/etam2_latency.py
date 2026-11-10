"""Parameters, forward FLOPs and inference latency of many checkpoints on ONE GPU.

Latency is only comparable on the same device, so every checkpoint of the
campaign is timed in a single job: forward pass over a fixed batch of 256
benchmark test prefixes (bf16), median of 20 repeats after 5 warm-ups.

    python scripts/etam2_latency.py --glob '<root>/runs/*/*/best.pt' --out result.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eventfm.etam.batching import benchmark_tpp_views  # noqa: E402
from eventfm.etam.stage2 import evaluate as E  # noqa: E402
from eventfm.etam.stage2.train import RunConfig2, load_store2, make_collator, model_from_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RunConfig2()
    store = load_store2(config, "test")
    views = benchmark_tpp_views(store, seed=14)[:256]
    results = {}
    paths = sorted({path for pattern in args.glob for path in glob.glob(pattern)})
    for path in paths:
        name = "/".join(Path(path).parts[-4:-1])
        model = None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            saved = dict(payload.get("run_config", {}))
            local = RunConfig2(
                max_events=int(saved.get("max_events", 128)), window_time=bool(saved.get("window_time", False))
            )
            del payload
            model = model_from_checkpoint(path).to(device)
            batch = make_collator(store, local, model.config, train=False)(views)
            results[name] = E.efficiency(model, batch, device)
            results[name]["max_events"] = local.max_events
        except Exception as error:  # noqa: BLE001
            results[name] = {"error": str(error)[:300]}
        print(json.dumps({name: results[name]}), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        Path(args.out).write_text(json.dumps(results, indent=2))
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
