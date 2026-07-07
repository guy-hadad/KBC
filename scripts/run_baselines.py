"""Run lightweight classic baselines on the synthetic splits."""

import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from eventfm.baselines.classic import run_classification_baseline, run_tpp_baseline
from eventfm.data.synthetic import write_synthetic_dataset


def _ensure_data(cfg: DictConfig) -> None:
    expected = [
        to_absolute_path(cfg.data.train_path),
        to_absolute_path(cfg.data.validation_path),
        to_absolute_path(cfg.data.test_path),
    ]
    if bool(cfg.data.auto_generate) and not all(Path(path).exists() for path in expected):
        write_synthetic_dataset(
            output_dir=to_absolute_path(cfg.data.data_dir),
            num_users=int(cfg.data.num_users),
            num_event_types=int(cfg.data.num_event_types),
            min_events=int(cfg.data.min_events),
            max_events=int(cfg.data.max_events),
            seed=int(cfg.data.seed),
        )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _ensure_data(cfg)
    train_path = to_absolute_path(cfg.data.train_path)
    validation_path = to_absolute_path(cfg.data.validation_path)
    test_path = to_absolute_path(cfg.data.test_path)
    results = {
        "classification_count_logistic": run_classification_baseline(
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            num_event_types=int(cfg.data.num_event_types),
        ),
        "tpp_markov": run_tpp_baseline(
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            num_event_types=int(cfg.data.num_event_types),
        ),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
