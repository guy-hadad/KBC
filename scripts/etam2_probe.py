"""Frozen-embedding probes for ETAM-2 (Section H), on CPU from dumped embeddings.

For each representation (``user`` = W_u[u_E;u_T;u_J], ``joint`` = u_J, the private
``event``/``time`` states, their concatenation, or a concatenation across runs)
and each history protocol (standard; ``_strict`` = histories cut before the
label window), fits

* a logistic probe (C chosen on validation AUC) and a small MLP probe (early
  stopping on validation AUC) for the benchmark label at label fractions
  1 / 5 / 10 / 25 / 100 % (stratified, 3 subsample seeds), and
* logistic probes for the three other MBD product targets (frozen cross-task
  transfer), at the full label budget,

and reports AUC, AP, P@K, R@K and prior-corrected calibration on test.

    python scripts/etam2_probe.py --emb RUN/embeddings --out result.json
    python scripts/etam2_probe.py --emb A/embeddings --concat B/embeddings --reps joint ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eventfm.etam.stage2 import metrics as M  # noqa: E402

FRACTIONS = (0.01, 0.05, 0.1, 0.25, 1.0)
TARGET_NAMES = ("bcard", "cred", "zp", "acquiring")


def load(directory: Path, suffix: str):
    data = {}
    for split in ("train", "validation", "test"):
        path = directory / "{}{}.npz".format(split, suffix)
        if not path.exists():
            return None
        with np.load(path) as payload:
            data[split] = {key: payload[key] for key in payload.files}
    return data


def representation(data, name, extra=None):
    out = {}
    for split, payload in data.items():
        if name == "concat":
            parts = [payload[key] for key in ("event", "time", "joint") if key in payload]
            value = np.concatenate(parts, axis=1)
        else:
            value = payload[name]
        if extra is not None:
            value = np.concatenate([value, extra[split]["joint"]], axis=1)
        out[split] = value.astype(np.float32)
    return out


def subsample(labels, fraction, seed):
    if fraction >= 1.0:
        return np.arange(labels.shape[0])
    rng = np.random.default_rng(1000 + seed)
    keep = []
    for value in np.unique(labels):
        rows = np.flatnonzero(labels == value)
        keep.append(rng.choice(rows, size=max(2, int(round(fraction * rows.shape[0]))), replace=False))
    return np.sort(np.concatenate(keep))


def logistic(x, y, xv, yv):
    from sklearn.linear_model import LogisticRegression

    best = None
    for strength in (0.001, 0.01, 0.1, 1.0):
        model = LogisticRegression(C=strength, max_iter=3000)
        model.fit(x, y)
        auc = M.ranking_metrics(model.decision_function(xv), yv)["auc"]
        if best is None or auc > best[0]:
            best = (auc, model)
    return best[1].decision_function, best[0]


def mlp(x, y, xv, yv, seed):
    import torch

    torch.manual_seed(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    xvt = torch.from_numpy(xv)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state, stale = None, None, 0
    batch = min(256, x.shape[0])
    for epoch in range(200):
        net.train()
        order = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], batch):
            rows = order[start : start + batch]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[rows]).squeeze(-1), yt[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            auc = M.ranking_metrics(net(xvt).squeeze(-1).numpy(), yv)["auc"]
        if best is None or auc > best:
            best, stale = auc, 0
            state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break
    net.load_state_dict(state)

    def score(z):
        with torch.no_grad():
            return net(torch.from_numpy(z)).squeeze(-1).numpy()

    return score, best


def report(score_fn, x_test, y_test, train_prior, eval_prior):
    result = M.classification_report(score_fn(x_test), y_test, train_prior=train_prior, eval_prior=eval_prior)
    return {k: v for k, v in result.items() if not isinstance(v, (list, dict))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb", required=True)
    parser.add_argument("--concat", default="", help="second run's embeddings: concatenate its `joint`")
    parser.add_argument("--reps", default="user,joint,event,time,concat")
    parser.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from sklearn.preprocessing import StandardScaler

    results = {}
    fractions = [float(value) for value in args.fractions.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    for suffix in ("", "_strict"):
        data = load(Path(args.emb), suffix)
        if data is None:
            continue
        extra = load(Path(args.concat), suffix) if args.concat else None
        protocol = suffix.strip("_") or "standard"
        labels = {split: data[split]["labels"].astype(int) for split in data}
        train_prior = float(labels["train"].mean())
        eval_prior = float(labels["validation"].mean())
        for name in args.reps.split(","):
            if name not in ("concat",) and name not in data["train"]:
                continue
            reps = representation(data, name, extra)
            scaler = StandardScaler().fit(reps["train"])
            reps = {split: scaler.transform(value).astype(np.float32) for split, value in reps.items()}
            key = "{}.{}".format(protocol, name if not args.concat else name + "+concat")
            entry = {}
            for fraction in fractions:
                for kind in (("logistic",) if args.no_mlp else ("logistic", "mlp")):
                    rows_by_seed = []
                    for seed in (seeds if fraction < 1.0 else seeds[:1]) if kind == "logistic" else seeds[:1]:
                        rows = subsample(labels["train"], fraction, seed)
                        x, y = reps["train"][rows], labels["train"][rows]
                        if kind == "logistic":
                            score_fn, valid_auc = logistic(x, y, reps["validation"], labels["validation"])
                        else:
                            score_fn, valid_auc = mlp(x, y, reps["validation"], labels["validation"], seed)
                        row = report(score_fn, reps["test"], labels["test"], float(y.mean()), eval_prior)
                        row["valid_auc"] = valid_auc
                        rows_by_seed.append(row)
                    merged = {}
                    for metric in rows_by_seed[0]:
                        values = [row[metric] for row in rows_by_seed if row.get(metric) is not None]
                        values = [v for v in values if v == v]
                        if values:
                            merged[metric] = float(np.mean(values))
                            if len(values) > 1:
                                merged[metric + "_sd"] = float(np.std(values, ddof=1))
                    merged["n_seeds"] = len(rows_by_seed)
                    entry["{}@{:g}".format(kind, fraction)] = merged
            if "targets" in data["train"]:
                for column, target in enumerate(TARGET_NAMES[1:], start=1):
                    y_train = data["train"]["targets"][:, column].astype(int)
                    y_valid = data["validation"]["targets"][:, column].astype(int)
                    y_test = data["test"]["targets"][:, column].astype(int)
                    if y_train.min() < 0 or len(np.unique(y_train)) < 2:
                        continue
                    score_fn, valid_auc = logistic(reps["train"], y_train, reps["validation"], y_valid)
                    row = report(score_fn, reps["test"], y_test, float(y_train.mean()), float(y_valid.mean()))
                    row["valid_auc"] = valid_auc
                    entry["target_" + target] = row
            results[key] = entry
            print(json.dumps({key: {k: v.get("auc") for k, v in entry.items()}}), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
