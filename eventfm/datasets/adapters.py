"""Per-benchmark readers that produce a :class:`TransactionTable`.

Each function only knows about column names and timestamp conventions; the
shared machinery in :mod:`eventfm.datasets.tabular` does the bucketing,
grouping, labelling and splitting.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from eventfm.datasets.download import download
from eventfm.datasets.tabular import ConversionConfig, TransactionTable

# All benchmarks are re-based onto a common origin so time features (calendar
# encodings, log inter-arrival) behave the same way across datasets.
EPOCH_2022 = 1_640_995_200.0  # 2022-01-01T00:00:00Z


def _within_period_offset(period_index: np.ndarray, period_seconds: float) -> np.ndarray:
    """Spread rows that share a coarse time bucket evenly inside that bucket.

    BankSim and PaySim only record the day / hour of a transaction. Without this
    every event inside a bucket would have a zero inter-arrival time, which makes
    the TPP time target degenerate.
    """

    order = np.argsort(period_index, kind="stable")
    ranks = np.empty(order.shape[0], dtype=np.int64)
    ranks[order] = np.arange(order.shape[0])
    _, starts, counts = np.unique(period_index[order], return_index=True, return_counts=True)
    rank_within = ranks - np.repeat(starts, counts)[ranks]
    size = np.repeat(counts, counts)[ranks]
    return rank_within / np.maximum(1.0, size) * period_seconds


def load_banksim(config: ConversionConfig) -> TransactionTable:
    """BankSim: Spanish bank payment simulator, merchant category as the mark."""

    path = download("banksim")
    frame = pl.read_csv(path, infer_schema_length=10000)
    # Every string column ships quoted as `'C1093826151'`; strip the quotes.
    frame = frame.with_columns(
        [
            pl.col(name).cast(pl.Utf8).str.strip_chars("'").alias(name)
            for name, dtype in zip(frame.columns, frame.dtypes)
            if dtype == pl.Utf8
        ]
    )
    # `step` counts days from the start of the simulation. The source has no
    # intra-day ordering, so rows are spread evenly across each day to keep
    # inter-arrival times strictly positive for the TPP task.
    days = frame["step"].cast(pl.Float64).to_numpy()
    timestamps = EPOCH_2022 + days * 86400.0 + _within_period_offset(days, 86400.0)

    return TransactionTable(
        entity_ids=frame["customer"].to_numpy(),
        timestamps=timestamps,
        event_types=frame["category"].to_numpy(),
        categorical={"merchant": frame["merchant"].to_numpy()},
        numeric={"amount": frame["amount"].cast(pl.Float64).to_numpy()},
        flags=frame["fraud"].cast(pl.Int64).to_numpy(),
        profile={
            "age": frame["age"].to_numpy(),
            "gender": frame["gender"].to_numpy(),
        },
    )


def load_paysim(config: ConversionConfig) -> TransactionTable:
    """PaySim: mobile-money simulator, transfer type as the mark.

    Originating accounts appear roughly once each, so histories are built on the
    destination account, which is the entity that actually accumulates a stream.
    """

    path = download("paysim")
    frame = pl.read_csv(path, infer_schema_length=10000)
    frame = frame.filter(pl.col("nameDest").is_not_null())

    counts = frame.group_by("nameDest").len()
    frequent = counts.filter(pl.col("len") >= config.min_events_per_entity)["nameDest"]
    frame = frame.filter(pl.col("nameDest").is_in(frequent))

    # `step` is hours since the start of the 30-day simulation window.
    hours = frame["step"].cast(pl.Float64).to_numpy()
    timestamps = EPOCH_2022 + hours * 3600.0 + _within_period_offset(hours, 3600.0)

    balance_delta = (
        frame["newbalanceDest"].cast(pl.Float64).to_numpy()
        - frame["oldbalanceDest"].cast(pl.Float64).to_numpy()
    )
    return TransactionTable(
        entity_ids=frame["nameDest"].to_numpy(),
        timestamps=timestamps,
        event_types=frame["type"].to_numpy(),
        categorical={
            "counterparty_kind": np.asarray(
                [
                    "merchant" if str(value).startswith("M") else "customer"
                    for value in frame["nameOrig"].to_numpy()
                ],
                dtype=object,
            )
        },
        numeric={
            "amount": frame["amount"].cast(pl.Float64).to_numpy(),
            "dest_balance": frame["oldbalanceDest"].cast(pl.Float64).to_numpy(),
            "balance_delta": balance_delta,
        },
        flags=frame["isFraud"].cast(pl.Int64).to_numpy(),
    )


def load_ibm_aml(config: ConversionConfig) -> TransactionTable:
    """IBM AML (HI-Small): AMLSim transactions, payment format as the mark."""

    path = download("ibm_aml")
    frame = pl.read_csv(path, infer_schema_length=10000)
    rename = {
        "From Bank": "from_bank",
        "To Bank": "to_bank",
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",
    }
    columns = frame.columns
    # The two account columns are both literally named `Account`; polars
    # disambiguates the second as `Account_duplicated_0` or `Account.1`.
    account_columns = [name for name in columns if name.lower().startswith("account")]
    frame = frame.rename({key: value for key, value in rename.items() if key in columns})
    frame = frame.rename({account_columns[0]: "src_account", account_columns[1]: "dst_account"})

    timestamps = (
        frame["Timestamp"]
        .str.to_datetime(format="%Y/%m/%d %H:%M", strict=False)
        .dt.epoch(time_unit="s")
        .cast(pl.Float64)
        .to_numpy()
    )

    counts = frame.group_by("dst_account").len()
    frequent = counts.filter(pl.col("len") >= config.min_events_per_entity)["dst_account"]
    mask = frame["dst_account"].is_in(frequent).to_numpy()

    frame = frame.filter(pl.Series(mask))
    timestamps = timestamps[mask]

    return TransactionTable(
        entity_ids=frame["dst_account"].to_numpy(),
        timestamps=timestamps,
        event_types=frame["payment_format"].to_numpy(),
        categorical={
            "payment_currency": frame["payment_currency"].to_numpy(),
            "same_bank": np.where(
                frame["from_bank"].to_numpy() == frame["to_bank"].to_numpy(), "yes", "no"
            ),
        },
        numeric={
            "amount_paid": frame["amount_paid"].cast(pl.Float64).to_numpy(),
            "fx_ratio": (
                frame["amount_received"].cast(pl.Float64).to_numpy()
                / np.maximum(1e-6, frame["amount_paid"].cast(pl.Float64).to_numpy())
            ),
        },
        flags=frame["is_laundering"].cast(pl.Int64).to_numpy(),
    )


def _mbd_root() -> Path:
    root = download("mbd_mini")
    for candidate in [root, root / "mbd_mini", root / "MBD-mini"]:
        if (candidate / "ptls").exists() or (candidate / "targets").exists():
            return candidate
    return root


def _read_parquet_dir(path: Path, columns: Optional[list] = None) -> pl.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No parquet files under {}".format(path))
    frames = [pl.read_parquet(file, columns=columns) for file in files]
    return pl.concat(frames, how="vertical_relaxed")


# MBD-mini transactions run to Nov 2022 while the monthly targets run to Jan
# 2023. History is cut at the end of Sep 2022 and the label is taken from the
# reporting months that follow, so the classification task is genuinely
# forward-looking rather than a re-read of the observed window.
MBD_HISTORY_CUTOFF = 1_664_496_000.0  # 2022-09-30T00:00:00Z
MBD_LABEL_MONTHS = ("2022-09-30", "2022-10-31", "2022-11-30", "2022-12-31", "2023-01-31")

MBD_EVENT_COLUMNS = [
    "event_time",
    "event_type",
    "event_subtype",
    "amount",
    "currency",
    "src_type11",
    "src_type12",
    "dst_type11",
    "dst_type12",
]


def load_mbd_mini(config: ConversionConfig, target_index: int = 1) -> TransactionTable:
    """MBD-mini: real anonymised multi-source bank data from the MBD benchmark.

    Events come from the transaction stream; the label is one of the four
    product-propensity targets shipped with the benchmark, so unlike the three
    simulators the classification target is supplied rather than derived.
    """

    root = _mbd_root()
    trx_dir = next(
        (path for path in [root / "ptls" / "trx", root / "trx"] if path.exists()),
        None,
    )
    if trx_dir is None:
        raise FileNotFoundError("MBD-mini transaction directory not found under {}".format(root))

    frame = _read_parquet_dir(trx_dir, columns=["client_id"] + MBD_EVENT_COLUMNS)
    frame = frame.explode([name for name in MBD_EVENT_COLUMNS if name in frame.columns])
    frame = frame.filter(pl.col("event_time").is_not_null())

    targets_dir = root / "targets"
    if not targets_dir.exists():
        raise FileNotFoundError("MBD-mini targets directory not found under {}".format(root))
    targets = _read_parquet_dir(targets_dir)
    label_column = "target_{}".format(target_index)
    if label_column not in targets.columns:
        label_column = next(name for name in targets.columns if name.startswith("target"))
    future = targets.filter(pl.col("mon").is_in(list(MBD_LABEL_MONTHS)))
    aggregated = future.group_by("client_id").agg(pl.col(label_column).max().alias("label"))
    entity_labels = {
        str(client): int(value or 0)
        for client, value in zip(aggregated["client_id"].to_list(), aggregated["label"].to_list())
    }

    categorical = {
        name: frame[name].cast(pl.Utf8).to_numpy()
        for name in ["event_subtype", "currency", "src_type11", "dst_type11"]
        if name in frame.columns
    }
    return TransactionTable(
        entity_ids=frame["client_id"].cast(pl.Utf8).to_numpy(),
        timestamps=frame["event_time"].cast(pl.Float64).to_numpy(),
        event_types=frame["event_type"].cast(pl.Utf8).to_numpy(),
        categorical=categorical,
        numeric={"amount": frame["amount"].cast(pl.Float64).to_numpy()},
        flags=None,
        entity_labels=entity_labels,
        time_cutoff=MBD_HISTORY_CUTOFF,
    )


ADAPTERS = {
    "banksim": load_banksim,
    "paysim": load_paysim,
    "ibm_aml": load_ibm_aml,
    "mbd_mini": load_mbd_mini,
}
