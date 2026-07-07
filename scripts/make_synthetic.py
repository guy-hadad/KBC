"""Generate the toy synthetic event dataset."""

import json

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from eventfm.data.io import dataset_summary
from eventfm.data.synthetic import write_synthetic_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    paths, vocab_path = write_synthetic_dataset(
        output_dir=to_absolute_path(cfg.data.data_dir),
        num_users=int(cfg.data.num_users),
        num_event_types=int(cfg.data.num_event_types),
        min_events=int(cfg.data.min_events),
        max_events=int(cfg.data.max_events),
        seed=int(cfg.data.seed),
    )
    summary = {split: dataset_summary(path) for split, path in paths.items()}
    print(json.dumps({"paths": paths, "vocab_path": vocab_path, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
