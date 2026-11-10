"""Time ETAM-2 / Stage 1 training steps on CPU (to size CPU-side campaigns)."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eventfm.etam.batching import RandomCutViews  # noqa: E402
from eventfm.etam.model import EtamConfig, EtamModel, LossWeights, compute_losses  # noqa: E402
from eventfm.etam.stage2.batching import Collator2  # noqa: E402
from eventfm.etam.stage2.config import Etam2Weights  # noqa: E402
from eventfm.etam.stage2.losses import compute_losses2  # noqa: E402
from eventfm.etam.stage2.train import RunConfig2, build_model2, load_store2  # noqa: E402

SIZES = {"XS": (128, 2, 4), "S": (192, 4, 6), "M": (256, 6, 8)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    config = RunConfig2()
    store = load_store2(config, "train")
    views = RandomCutViews(store, seed=0)
    results = {"threads": torch.get_num_threads()}
    for size, (width, layers, heads) in SIZES.items():
        arch = {"hidden_size": width, "num_layers": layers, "num_heads": heads,
                "bridge_layers": [1, 2] if layers == 2 else ([2, 4] if layers >= 4 else [1])}
        for name in ("etam2", "stage1"):
            torch.manual_seed(0)
            if name == "etam2":
                config.model = dict(arch)
                model = build_model2(config, store)
                collate = Collator2(store, 128)
            else:
                model = EtamModel(EtamConfig(num_event_types=53, field_sizes=store.vocab.field_sizes,
                                             hidden_size=width, num_layers=layers, num_heads=heads))
                collate = Collator2(store, 128)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            times = []
            for step in range(args.steps + 1):
                batch = collate([views[i] for i in range(step * args.batch, (step + 1) * args.batch)])
                start = time.time()
                outputs = model(batch)
                if name == "etam2":
                    terms = compute_losses2(model, outputs, batch, Etam2Weights())
                else:
                    terms = compute_losses(model, outputs, batch, LossWeights(), ["tpp", "cond", "user", "future"])
                optimizer.zero_grad()
                terms["loss"].backward()
                optimizer.step()
                if step:
                    times.append(time.time() - start)
            results["{}_{}".format(name, size)] = sum(times) / len(times)
            print(json.dumps(results), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
