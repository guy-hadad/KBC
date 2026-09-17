"""Per-benchmark readers that produce a :class:`TransactionTable`.

Each function only knows about column names and timestamp conventions; the
shared machinery in :mod:`eventfm.datasets.tabular` does the bucketing,
grouping, labelling and splitting.
"""

import os
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


def _interned(series: pl.Series, null_token: str = "na") -> np.ndarray:
    """Object array whose duplicate values share a single Python string.

    ``Series.to_numpy()`` on a text column allocates a fresh ``str`` per row,
    which is what makes a full-scale conversion run out of memory: 64M events
    across six text columns is ~20 GB of string objects. Mapping through the
    distinct values instead stores pointers, so a column with 54 marks costs
    8 bytes per row rather than ~55, and the downstream ``str(value)`` calls in
    :mod:`eventfm.datasets.tabular` return the same objects.
    """

    values = series.cast(pl.Utf8).fill_null(null_token)
    levels = values.unique().sort().to_list()
    codes = values.cast(pl.Enum(levels)).to_physical().to_numpy().astype(np.intp, copy=False)
    return np.asarray(levels, dtype=object)[codes]


def _mbd_full_root() -> Path:
    return download("mbd")


# Uniform client subsample used to keep a full-MBD conversion inside a 48 GB
# job. 500k clients truncated to `max_events_per_entity` events is ~64M rows,
# which is the largest table that fits alongside the grouping index.
MBD_FULL_DEFAULT_MAX_CLIENTS = 1_000_000
MBD_FILE_CHUNK = 200


# MBD-mini anonymises its four product-propensity targets as `target_1..4`;
# the full release names them. The order is the same, confirmed by matching
# per-client prevalence in the label months: mini target_1/2/3/4 measure
# 1.40/0.22/1.40/1.07 % against 1.42/0.22/1.38/1.05 % for the columns below.
MBD_FULL_TARGET_COLUMNS = (
    "bcard_target",
    "cred_target",
    "zp_target",
    "acquiring_target",
)


def _mbd_label_column(columns, target_index: int) -> str:
    """Resolve the requested product target across both MBD releases."""

    anonymised = "target_{}".format(target_index)
    if anonymised in columns:
        return anonymised
    if 1 <= target_index <= len(MBD_FULL_TARGET_COLUMNS):
        named = MBD_FULL_TARGET_COLUMNS[target_index - 1]
        if named in columns:
            return named
    raise ValueError(
        "No MBD target column for index {} in {}".format(target_index, sorted(columns))
    )


def _mbd_client_labels(root: Path, target_index: int) -> tuple:
    """Return ``(entity_labels, label_column)`` from the small targets table."""

    targets_dir = root / "targets"
    if not targets_dir.exists():
        raise FileNotFoundError("MBD targets directory not found under {}".format(root))
    targets = _read_parquet_dir(targets_dir)
    label_column = _mbd_label_column(targets.columns, target_index)
    months = targets["mon"].cast(pl.Utf8).str.slice(0, 10)
    future = targets.filter(months.is_in(list(MBD_LABEL_MONTHS)))
    aggregated = future.group_by("client_id").agg(pl.col(label_column).max().alias("label"))
    entity_labels = {
        str(client): int(value or 0)
        for client, value in zip(aggregated["client_id"].to_list(), aggregated["label"].to_list())
    }
    return entity_labels, label_column


def _select_mbd_clients(entity_labels: dict, max_clients: Optional[int], seed: int) -> set:
    """Uniformly subsample clients, which leaves the label prevalence intact.

    Sampling *before* the transaction scan is what bounds memory. It is
    deliberately label-blind: keeping all positives and downsampling negatives
    here would alter validation and test prevalence, which is exactly the flaw
    the legacy results carry.
    """

    clients = sorted(entity_labels)
    if max_clients is None or len(clients) <= max_clients:
        return set(clients)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(clients, dtype=object), size=max_clients, replace=False)
    return {str(client) for client in chosen}


def load_mbd(
    config: ConversionConfig,
    target_index: int = 1,
    max_clients: Optional[int] = None,
) -> TransactionTable:
    """Full MBD: the same schema as MBD-mini over roughly ten times the clients.

    MBD-mini is the official 10 % client subsample, so the only differences here
    are scale and the memory strategy: transactions are scanned in file chunks,
    truncated to the tail of each client's history inside the scan, and stored
    as interned categories.
    """

    root = _mbd_full_root()
    trx_dir = next(
        (path for path in [root / "ptls" / "trx", root / "trx"] if path.exists()),
        None,
    )
    if trx_dir is None:
        raise FileNotFoundError("MBD transaction directory not found under {}".format(root))

    entity_labels, _ = _mbd_client_labels(root, target_index)
    if max_clients is None:
        max_clients = int(os.environ.get("KBC_MBD_MAX_CLIENTS", MBD_FULL_DEFAULT_MAX_CLIENTS))
    if max_clients <= 0:
        max_clients = None
    keep_clients = _select_mbd_clients(entity_labels, max_clients, config.seed)
    entity_labels = {
        client: label for client, label in entity_labels.items() if client in keep_clients
    }

    columns = [name for name in MBD_EVENT_COLUMNS]
    files = sorted(trx_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No parquet files under {}".format(trx_dir))

    keep_series = pl.Series("keep", sorted(keep_clients), dtype=pl.Utf8)
    chunks = []
    for start in range(0, len(files), MBD_FILE_CHUNK):
        batch = files[start : start + MBD_FILE_CHUNK]
        frame = (
            pl.scan_parquet(batch)
            .select(["client_id"] + columns)
            .filter(pl.col("client_id").is_in(keep_series))
            .explode(columns)
            .filter(pl.col("event_time").is_not_null())
            .filter(pl.col("event_time").cast(pl.Float64) <= MBD_HISTORY_CUTOFF)
            .sort("event_time")
            .group_by("client_id", maintain_order=True)
            .tail(config.max_events_per_entity)
            .collect(engine="streaming")
        )
        if frame.height:
            chunks.append(frame)
    if not chunks:
        raise ValueError("No MBD transactions survived the history cutoff.")
    frame = pl.concat(chunks, how="vertical_relaxed")
    del chunks

    categorical = {
        name: _interned(frame[name])
        for name in ["event_subtype", "currency", "src_type11", "dst_type11"]
        if name in frame.columns
    }
    return TransactionTable(
        entity_ids=_interned(frame["client_id"]),
        timestamps=frame["event_time"].cast(pl.Float64).to_numpy(),
        event_types=_interned(frame["event_type"]),
        categorical=categorical,
        numeric={"amount": frame["amount"].cast(pl.Float64).to_numpy()},
        flags=None,
        entity_labels=entity_labels,
        time_cutoff=MBD_HISTORY_CUTOFF,
    )


AMAZON_REVIEW_COLUMNS = ["overall", "reviewTime", "asin"]
AMAZON_NEGATIVE_RATING = 2


def _epoch_seconds(series: pl.Series) -> np.ndarray:
    """Seconds since the Unix epoch from either a timestamp or a text column."""

    if series.dtype in (pl.Datetime, pl.Date):
        return series.cast(pl.Datetime("us")).dt.epoch("s").cast(pl.Float64).to_numpy()
    parsed = series.cast(pl.Utf8).str.to_datetime(strict=False)
    return parsed.dt.epoch("s").cast(pl.Float64).to_numpy()


def load_amazon_beauty(config: ConversionConfig) -> TransactionTable:
    """Amazon Reviews 2014, Beauty category: one sequence per reviewer.

    The mark is the reviewed item's second-level product category, which is the
    informative level once every path starts at ``Beauty``. The forward-looking
    label is whether the reviewer posts a negative review (rating <=
    ``AMAZON_NEGATIVE_RATING``) in the held-out tail of their history, which is
    the recommendation analogue of the fraud/churn labels on the banking
    benchmarks.
    """

    root = download("amazon_beauty")
    review_files = sorted(root.glob("reviews__*.parquet"))
    metadata_files = sorted(root.glob("metadata__*.parquet"))
    if not review_files:
        raise FileNotFoundError("No Amazon Beauty review parquet files under {}".format(root))

    frame = (
        pl.scan_parquet(review_files)
        .select(["reviewerID"] + AMAZON_REVIEW_COLUMNS)
        .explode(AMAZON_REVIEW_COLUMNS)
        .filter(pl.col("reviewTime").is_not_null() & pl.col("asin").is_not_null())
        .collect(engine="streaming")
    )

    # Item metadata supplies the mark. `categories` is a list of category paths;
    # every Beauty path starts at "Beauty", so level 2 is what separates items.
    category_by_asin = None
    price_by_asin = None
    if metadata_files:
        metadata = (
            pl.scan_parquet(metadata_files)
            .select(
                pl.col("asin"),
                pl.col("categories")
                .list.first()
                .list.get(1, null_on_oob=True)
                .alias("category"),
                pl.col("price"),
                pl.col("brand"),
            )
            .unique(subset=["asin"])
            .collect(engine="streaming")
        )
        frame = frame.join(metadata, on="asin", how="left")
        category_by_asin = "category"
        price_by_asin = "price"

    seconds = _epoch_seconds(frame["reviewTime"])
    # The 2014 dump records only the review date, so events inside one day are
    # spread across it to keep TPP inter-arrival times strictly positive.
    days = np.floor(seconds / 86400.0)
    timestamps = days * 86400.0 + _within_period_offset(days, 86400.0)

    if category_by_asin is not None:
        event_types = _interned(frame[category_by_asin], null_token="uncategorised")
    else:
        event_types = _interned(frame["asin"])

    ratings = frame["overall"].cast(pl.Float64).to_numpy()
    categorical = {"rating": _interned(frame["overall"].cast(pl.Int64))}
    if "brand" in frame.columns:
        categorical["brand"] = _interned(frame["brand"], null_token="unbranded")
    numeric = {"rating_value": ratings}
    if price_by_asin is not None:
        numeric["price"] = frame[price_by_asin].cast(pl.Float64).to_numpy()

    return TransactionTable(
        entity_ids=_interned(frame["reviewerID"]),
        timestamps=timestamps,
        event_types=event_types,
        categorical=categorical,
        numeric=numeric,
        flags=(ratings <= AMAZON_NEGATIVE_RATING).astype(np.int64),
        entity_labels=None,
        time_cutoff=None,
    )


# A patient's label is the onset of a major adverse cardiovascular or renal
# event in the held-out tail of their condition history. The group is fixed here
# rather than learned so the label is a clinical definition, not a threshold.
SYNTHEA_TARGET_CONDITIONS = (
    "Myocardial infarction (disorder)",
    "Acute ST segment elevation myocardial infarction (disorder)",
    "Acute non-ST segment elevation myocardial infarction (disorder)",
    "Chronic congestive heart failure (disorder)",
    "Chronic kidney disease stage 4 (disorder)",
    "End-stage renal disease (disorder)",
)

SYNTHEA_PROFILE_FIELDS = ("GENDER", "RACE", "MARITAL")
SYNTHEA_DEFAULT_MAX_PATIENTS = 500_000


def load_synthea(config: ConversionConfig, max_patients: Optional[int] = None) -> TransactionTable:
    """Synthea 575k synthetic EHR: the diagnosis stream, one sequence per patient.

    The mark is the SNOMED condition description (323 distinct, collapsed to the
    shared cardinality cap), and the flagged event is a major adverse
    cardiovascular or renal diagnosis, so the classification label is
    "does this patient suffer a MACE/ESRD event in the unobserved tail?".
    `medications` and `procedures` ship in the same repository and are not read
    here: the diagnosis stream alone already carries the largest mark vocabulary
    in the benchmark.
    """

    root = download("synthea")
    conditions_path = root / "conditions.parquet"
    if not conditions_path.exists():
        raise FileNotFoundError("Synthea conditions table not found at {}".format(conditions_path))

    if max_patients is None:
        max_patients = int(os.environ.get("KBC_SYNTHEA_MAX_PATIENTS", SYNTHEA_DEFAULT_MAX_PATIENTS))

    lazy = pl.scan_parquet(conditions_path).select(
        pl.col("PATIENT"),
        pl.col("START").str.to_date(strict=False).alias("start_date"),
        pl.col("STOP").str.to_date(strict=False).alias("stop_date"),
        pl.col("DESCRIPTION"),
    ).filter(pl.col("start_date").is_not_null())

    if max_patients > 0:
        # Uniform patient subsample, drawn before the 56M-row scan so the table
        # stays inside the job's memory budget. Label prevalence is untouched.
        patients = (
            pl.scan_parquet(conditions_path)
            .select(pl.col("PATIENT").unique())
            .collect(engine="streaming")["PATIENT"]
        )
        if patients.len() > max_patients:
            rng = np.random.default_rng(config.seed)
            chosen = rng.choice(
                np.asarray(sorted(patients.to_list()), dtype=object),
                size=max_patients,
                replace=False,
            )
            lazy = lazy.filter(
                pl.col("PATIENT").is_in(pl.Series("keep", sorted(str(x) for x in chosen)))
            )

    frame = (
        lazy.sort("start_date")
        .group_by("PATIENT", maintain_order=True)
        .tail(config.max_events_per_entity)
        .collect(engine="streaming")
    )

    days = frame["start_date"].cast(pl.Date).to_numpy().astype("datetime64[D]").astype(np.float64)
    # Synthea records diagnoses at day resolution; spread events inside one day
    # so inter-arrival times stay strictly positive for the TPP task.
    timestamps = days * 86400.0 + _within_period_offset(days, 86400.0)

    duration = (
        (pl.Series(frame["stop_date"]).cast(pl.Date).cast(pl.Int32)
         - pl.Series(frame["start_date"]).cast(pl.Date).cast(pl.Int32))
        .fill_null(0)
        .cast(pl.Float64)
        .to_numpy()
    )

    descriptions = frame["DESCRIPTION"]
    flags = descriptions.is_in(list(SYNTHEA_TARGET_CONDITIONS)).cast(pl.Int64).to_numpy()

    profile = {}
    patients_path = root / "patients.parquet"
    if patients_path.exists():
        demographics = (
            pl.scan_parquet(patients_path)
            .select(["Id"] + list(SYNTHEA_PROFILE_FIELDS))
            .collect(engine="streaming")
        )
        joined = frame.select("PATIENT").join(
            demographics, left_on="PATIENT", right_on="Id", how="left"
        )
        profile = {
            name.lower(): _interned(joined[name], null_token="unknown")
            for name in SYNTHEA_PROFILE_FIELDS
            if name in joined.columns
        }

    return TransactionTable(
        entity_ids=_interned(frame["PATIENT"]),
        timestamps=timestamps,
        event_types=_interned(descriptions),
        categorical={},
        numeric={"duration_days": duration},
        flags=flags,
        entity_labels=None,
        profile=profile,
        time_cutoff=None,
    )


ADAPTERS["mbd"] = load_mbd
ADAPTERS["synthea"] = load_synthea
ADAPTERS["amazon_beauty"] = load_amazon_beauty
