"""Fetch the raw public files for each open benchmark.

Downloads are idempotent: an existing file of non-trivial size is reused, so a
failed SLURM job can be resubmitted without re-pulling gigabytes.
"""

import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict

from eventfm.datasets.paths import configure_hf_cache, raw_root

BANKSIM_URL = (
    "https://raw.githubusercontent.com/atavci/fraud-detection-on-banksim-data/"
    "master/Data/synthetic-data-from-a-financial-payment-system/bs140513_032310.csv"
)

HF_FILES: Dict[str, Dict[str, str]] = {
    "paysim": {"repo_id": "theman10/paysim", "filename": "paysim.csv"},
    "ibm_aml": {"repo_id": "OsamaMIT/IBM-AML-HI-Small", "filename": "HI-Small_Trans.csv"},
}

MBD_MINI_REPO = "ai-lab/MBD-mini"
MBD_MINI_ARCHIVES = ["ptls.tar.gz", "targets.tar.gz", "client_split.tar.gz"]


def _is_present(path: Path, min_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def _hf_download(repo_id: str, filename: str) -> Path:
    configure_hf_cache()
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))


def download_banksim() -> Path:
    target = raw_root() / "banksim" / "bs140513_032310.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_present(target, min_bytes=1_000_000):
        return target
    with urllib.request.urlopen(BANKSIM_URL, timeout=300) as response:
        target.write_bytes(response.read())
    return target


def download_hf_csv(name: str) -> Path:
    spec = HF_FILES[name]
    target = raw_root() / name / spec["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_present(target, min_bytes=1_000_000):
        return target
    cached = _hf_download(spec["repo_id"], spec["filename"])
    shutil.copy2(cached, target)
    return target


def download_mbd_mini() -> Path:
    """Pull and unpack the MBD-mini archives into ``raw/mbd_mini``."""

    target_dir = raw_root() / "mbd_mini"
    target_dir.mkdir(parents=True, exist_ok=True)
    for archive_name in MBD_MINI_ARCHIVES:
        stem = archive_name.replace(".tar.gz", "")
        marker = target_dir / "{}.done".format(stem)
        if marker.exists():
            continue
        cached = _hf_download(MBD_MINI_REPO, archive_name)
        with tarfile.open(cached, "r:gz") as archive:
            archive.extractall(target_dir)
        marker.write_text("ok\n", encoding="utf-8")
    return target_dir


DOWNLOADERS = {
    "banksim": download_banksim,
    "paysim": lambda: download_hf_csv("paysim"),
    "ibm_aml": lambda: download_hf_csv("ibm_aml"),
    "mbd_mini": download_mbd_mini,
}


def download(name: str) -> Path:
    if name not in DOWNLOADERS:
        raise KeyError("Unknown dataset: {}".format(name))
    return DOWNLOADERS[name]()
