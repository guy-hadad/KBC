"""Small training utilities shared by scripts."""

import os
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from transformers import set_seed


def cfg_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def prepare_wandb(cfg: DictConfig) -> None:
    if not bool(cfg.logging.wandb.enabled):
        os.environ.setdefault("WANDB_MODE", "disabled")
        return
    os.environ.setdefault("WANDB_PROJECT", str(cfg.logging.wandb.project))
    if cfg.logging.wandb.entity:
        os.environ.setdefault("WANDB_ENTITY", str(cfg.logging.wandb.entity))


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    set_seed(int(seed))
