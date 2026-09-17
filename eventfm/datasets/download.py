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

SYNTHEA_REPO = "richardyoung/synthea-575k-patients"
# The event stream is built from the two code tables that carry a start date and
# a clinical description; `procedures` (8.3 GB) and `observations` (10.4 GB) add
# no new mark structure for this benchmark and are not fetched.
SYNTHEA_FILES = ("data/patients.parquet", "data/conditions.parquet", "data/medications.parquet")

AMAZON_BEAUTY_REPO = "milistu/Amazon_Beauty_2014"
# `reviews/` is already one row per reviewer with parallel list columns, which is
# the event-stream shape this benchmark wants; `metadata/` supplies item
# categories that become the event marks.
AMAZON_BEAUTY_FILES = (
    "reviews/train-00000-of-00002.parquet",
    "reviews/train-00001-of-00002.parquet",
    "metadata/train-00000-of-00001.parquet",
)

MBD_MINI_REPO = "ai-lab/MBD-mini"
MBD_FULL_REPO = "ai-lab/MBD"
MBD_ARCHIVES = ["ptls.tar.gz", "targets.tar.gz", "client_split.tar.gz"]

# `ptls.tar.gz` also carries dialog embeddings and geo streams, which this
# benchmark does not read. Full MBD ships 30.7 GB there against ~11 GB of
# transactions, so only the transaction members are unpacked.
# Member paths inside the archive are `ptls/<source>/fold=N/part-*.parquet`.
MBD_PTLS_KEEP_PREFIXES = ("ptls/trx", "trx")


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


def _extract_stream(archive_path: Path, target_dir: Path, prefixes: tuple = ()) -> None:
    """Unpack a tar.gz in one pass, keeping only members under ``prefixes``.

    Read in stream mode (``r|gz``) on purpose. ``getmembers()`` has to inflate
    the whole stream to collect the headers and ``extractall`` then inflates it
    again, which is two full passes over 30 GB for full MBD; iterating the
    stream extracts as it goes.
    """

    with tarfile.open(archive_path, "r|gz") as archive:
        extracted = 0
        for member in archive:
            if prefixes and not member.name.lstrip("./").startswith(prefixes):
                continue
            archive.extract(member, target_dir)
            extracted += 1
    if prefixes and not extracted:
        raise ValueError(
            "No members of {} matched {}; the archive layout has changed.".format(
                archive_path.name, prefixes
            )
        )


def _download_mbd(repo_id: str, name: str) -> Path:
    """Pull and unpack the MBD archives for one repository into ``raw/<name>``."""

    target_dir = raw_root() / name
    target_dir.mkdir(parents=True, exist_ok=True)
    for archive_name in MBD_ARCHIVES:
        stem = archive_name.replace(".tar.gz", "")
        marker = target_dir / "{}.done".format(stem)
        if marker.exists():
            continue
        cached = _hf_download(repo_id, archive_name)
        prefixes = MBD_PTLS_KEEP_PREFIXES if stem == "ptls" else ()
        _extract_stream(cached, target_dir, prefixes)
        marker.write_text("ok\n", encoding="utf-8")
    return target_dir


def download_mbd_mini() -> Path:
    """Pull and unpack the MBD-mini archives into ``raw/mbd_mini``."""

    return _download_mbd(MBD_MINI_REPO, "mbd_mini")


def download_mbd() -> Path:
    """Pull and unpack the full 69 GB MBD archives into ``raw/mbd``.

    Only `ptls`, `targets` and `client_split` are fetched; `detail.tar.gz`
    (38 GB of row-per-event duplicates of `ptls`) is never needed here.
    """

    return _download_mbd(MBD_FULL_REPO, "mbd")


def _download_flat(repo_id: str, name: str, filenames) -> Path:
    """Copy a fixed set of repository files into ``raw/<name>``, flattening paths."""

    target_dir = raw_root() / name
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        target = target_dir / Path(filename).name
        if _is_present(target, min_bytes=1_000_000):
            continue
        cached = _hf_download(repo_id, filename)
        shutil.copy2(cached, target)
    return target_dir


def download_synthea() -> Path:
    """Pull the Synthea 575k patient tables this benchmark reads."""

    return _download_flat(SYNTHEA_REPO, "synthea", SYNTHEA_FILES)


def download_amazon_beauty() -> Path:
    """Pull the Amazon Beauty 2014 reviewer streams and item metadata."""

    target_dir = raw_root() / "amazon_beauty"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in AMAZON_BEAUTY_FILES:
        kind = Path(filename).parent.name
        target = target_dir / "{}__{}".format(kind, Path(filename).name)
        if _is_present(target, min_bytes=1_000_000):
            continue
        cached = _hf_download(AMAZON_BEAUTY_REPO, filename)
        shutil.copy2(cached, target)
    return target_dir


DOWNLOADERS = {
    "banksim": download_banksim,
    "paysim": lambda: download_hf_csv("paysim"),
    "ibm_aml": lambda: download_hf_csv("ibm_aml"),
    "mbd_mini": download_mbd_mini,
    "mbd": download_mbd,
    "synthea": download_synthea,
    "amazon_beauty": download_amazon_beauty,
}


def download(name: str) -> Path:
    if name not in DOWNLOADERS:
        raise KeyError("Unknown dataset: {}".format(name))
    return DOWNLOADERS[name]()
