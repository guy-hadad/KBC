"""Filesystem roots for raw downloads, processed splits and run outputs.

Large artefacts live on group storage rather than in the git checkout. Every
path can be overridden with an environment variable so SLURM jobs and local
smoke tests can point at different disks.
"""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = "/groups/bshapira_group/guyhada/KBC_data"
DEFAULT_OUTPUT_ROOT = "/groups/bshapira_group/guyhada/KBC_outputs"
DEFAULT_HF_CACHE = "/groups/bshapira_group/guyhada/hf_cache"


def data_root() -> Path:
    return Path(os.environ.get("KBC_DATA_ROOT", DEFAULT_DATA_ROOT))


def raw_root() -> Path:
    path = data_root() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_root() -> Path:
    path = data_root() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_root() -> Path:
    path = Path(os.environ.get("KBC_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", DEFAULT_HF_CACHE))


def configure_hf_cache() -> None:
    """Point `huggingface_hub` at group storage before any download starts."""

    cache = hf_cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
