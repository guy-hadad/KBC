"""Train-only temporal tokenizers for the GEM 2026 comparison.

The tokenizers in this module are deliberately independent of Hugging Face.
They turn one scalar timestamp or interval into reversible string tokens; the
LLM adapter decides how those strings enter a model vocabulary. Data-dependent
tokenizers expose ``fit`` and must be fitted on the training split only.
"""

import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _finite_nonnegative(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    return np.maximum(0.0, array)


class TemporalTokenizer:
    """Minimal scalar-to-token contract shared by every time representation."""

    key = "time"
    uses_absolute_time = False
    data_dependent = False

    def fit(self, values: Iterable[float]) -> "TemporalTokenizer":
        del values
        return self

    def encode(self, value: float) -> List[str]:
        raise NotImplementedError

    def decode(self, tokens: Sequence[str]) -> float:
        raise NotImplementedError

    def vocabulary(self) -> List[str]:
        return []


class NumericStringTokenizer(TemporalTokenizer):
    key = "time-numeric-p6"

    def __init__(self, precision: int = 6) -> None:
        self.precision = int(precision)

    def encode(self, value: float) -> List[str]:
        return [("{:.%df}" % self.precision).format(max(0.0, float(value)))]

    def decode(self, tokens: Sequence[str]) -> float:
        text = "".join(tokens).strip().replace(" ", "")
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        if match is None:
            raise ValueError("No numeric time value in {!r}".format(text))
        return max(0.0, float(match.group(0)))


class FloatByteTokenizer(TemporalTokenizer):
    key = "time-byte-f32"

    def __init__(self, byte_order: str = ">") -> None:
        if byte_order not in {">", "<"}:
            raise ValueError("byte_order must be `>` or `<`")
        self.byte_order = byte_order

    def encode(self, value: float) -> List[str]:
        packed = struct.pack(self.byte_order + "f", float(np.float32(max(0.0, value))))
        return ["<|byte_{:03d}|>".format(byte) for byte in packed]

    def decode(self, tokens: Sequence[str]) -> float:
        values = []
        for token in tokens:
            match = re.fullmatch(r"<\|byte_(\d{3})\|>", token.strip())
            if match is not None:
                values.append(int(match.group(1)))
        if len(values) != 4 or any(value > 255 for value in values):
            raise ValueError("A float32 time requires exactly four byte tokens")
        return max(0.0, float(struct.unpack(self.byte_order + "f", bytes(values))[0]))

    def vocabulary(self) -> List[str]:
        return ["<|byte_{:03d}|>".format(index) for index in range(256)]


_RESOLUTION_COMPONENTS = {
    "day": ("year", "month", "day"),
    "hour": ("year", "month", "day", "hour"),
    "minute": ("year", "month", "day", "hour", "minute"),
    "second": ("year", "month", "day", "hour", "minute", "second"),
}


class CalendarTokenizer(TemporalTokenizer):
    """Gregorian absolute time or hierarchical duration components."""

    def __init__(self, absolute: bool, resolution: str = "second") -> None:
        if resolution not in _RESOLUTION_COMPONENTS:
            raise ValueError("Unknown calendar resolution: {}".format(resolution))
        self.uses_absolute_time = bool(absolute)
        self.resolution = resolution
        self.key = "time-calendar-{}-{}".format("abs" if absolute else "rel", resolution)
        self._min_year = 1970 if absolute else 0
        self._max_year = 2100 if absolute else 999

    def fit(self, values: Iterable[float]) -> "CalendarTokenizer":
        array = _finite_nonnegative(values)
        if not array.size:
            return self
        if self.uses_absolute_time:
            years = [datetime.fromtimestamp(float(value), tz=timezone.utc).year for value in array]
        else:
            years = np.floor(array / (365 * 86400)).astype(np.int64).tolist()
        self._min_year = int(min(years))
        self._max_year = int(max(years))
        return self

    @staticmethod
    def _token(name: str, value: int) -> str:
        width = 4 if name == "year" else 2
        return "<|{}_{:0{}d}|>".format(name, value, width)

    def _components(self, value: float) -> Dict[str, int]:
        value = max(0.0, float(value))
        if self.uses_absolute_time:
            timestamp = datetime.fromtimestamp(value, tz=timezone.utc)
            return {
                "year": min(self._max_year, max(self._min_year, timestamp.year)),
                "month": timestamp.month,
                "day": timestamp.day,
                "hour": timestamp.hour,
                "minute": timestamp.minute,
                "second": timestamp.second,
            }
        remaining = int(round(value))
        year, remaining = divmod(remaining, 365 * 86400)
        month, remaining = divmod(remaining, 30 * 86400)
        day, remaining = divmod(remaining, 86400)
        hour, remaining = divmod(remaining, 3600)
        minute, second = divmod(remaining, 60)
        return {
            "year": min(self._max_year, max(self._min_year, year)),
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }

    def encode(self, value: float) -> List[str]:
        components = self._components(value)
        return [
            self._token(name, components[name])
            for name in _RESOLUTION_COMPONENTS[self.resolution]
        ]

    def decode(self, tokens: Sequence[str]) -> float:
        values: Dict[str, int] = {}
        for token in tokens:
            match = re.fullmatch(r"<\|(year|month|day|hour|minute|second)_(\d+)\|>", token.strip())
            if match is not None:
                values[match.group(1)] = int(match.group(2))
        required = _RESOLUTION_COMPONENTS[self.resolution]
        if any(name not in values for name in required):
            raise ValueError("Incomplete calendar token sequence")
        if self.uses_absolute_time:
            timestamp = datetime(
                values["year"],
                values["month"],
                values["day"],
                values.get("hour", 0),
                values.get("minute", 0),
                values.get("second", 0),
                tzinfo=timezone.utc,
            )
            return timestamp.timestamp()
        return float(
            values["year"] * 365 * 86400
            + values["month"] * 30 * 86400
            + values["day"] * 86400
            + values.get("hour", 0) * 3600
            + values.get("minute", 0) * 60
            + values.get("second", 0)
        )

    def vocabulary(self) -> List[str]:
        tokens = [self._token("year", value) for value in range(self._min_year, self._max_year + 1)]
        first = 1 if self.uses_absolute_time else 0
        tokens += [self._token("month", value) for value in range(first, 13)]
        tokens += [self._token("day", value) for value in range(first, 32)]
        if self.resolution in {"hour", "minute", "second"}:
            tokens += [self._token("hour", value) for value in range(24)]
        if self.resolution in {"minute", "second"}:
            tokens += [self._token("minute", value) for value in range(60)]
        if self.resolution == "second":
            tokens += [self._token("second", value) for value in range(60)]
        return tokens


class _TransformedTokenizer(TemporalTokenizer):
    data_dependent = True

    def __init__(self, scale: str) -> None:
        if scale not in {"linear", "log"}:
            raise ValueError("scale must be `linear` or `log`")
        self.scale = scale

    def _transform(self, values: np.ndarray) -> np.ndarray:
        return np.log10(1.0 + values) if self.scale == "log" else values

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.power(10.0, values) - 1.0) if self.scale == "log" else values


class ScaleBinTokenizer(_TransformedTokenizer):
    def __init__(self, scale: str, num_bins: int = 256) -> None:
        super().__init__(scale)
        self.num_bins = int(num_bins)
        self.key = "time-bin-{}-k{}".format(scale, self.num_bins)
        self.edges = np.linspace(0.0, 1.0, self.num_bins + 1)

    def fit(self, values: Iterable[float]) -> "ScaleBinTokenizer":
        array = self._transform(_finite_nonnegative(values))
        if array.size:
            low, high = float(array.min()), float(array.max())
            if not high > low:
                high = low + 1.0
            self.edges = np.linspace(low, high, self.num_bins + 1)
        return self

    def encode(self, value: float) -> List[str]:
        transformed = float(self._transform(np.asarray([max(0.0, value)]))[0])
        index = int(np.searchsorted(self.edges, transformed, side="right") - 1)
        index = min(self.num_bins - 1, max(0, index))
        return ["<|bin_{:03d}|>".format(index)]

    def decode(self, tokens: Sequence[str]) -> float:
        matches = [re.fullmatch(r"<\|bin_(\d{3})\|>", token.strip()) for token in tokens]
        indices = [int(match.group(1)) for match in matches if match is not None]
        if len(indices) != 1 or indices[0] >= self.num_bins:
            raise ValueError("Expected exactly one valid bin token")
        index = indices[0]
        center = 0.5 * (self.edges[index] + self.edges[index + 1])
        return float(self._inverse(np.asarray([center]))[0])

    def vocabulary(self) -> List[str]:
        return ["<|bin_{:03d}|>".format(index) for index in range(self.num_bins)]


def _one_dimensional_kmeans(values: np.ndarray, clusters: int, seed: int) -> np.ndarray:
    """Deterministic, dependency-free Lloyd updates for one-dimensional data."""

    if values.size == 0:
        return np.zeros(clusters, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if values.size > 50000:
        values = rng.choice(values, size=50000, replace=False)
    quantiles = (np.arange(clusters, dtype=np.float64) + 0.5) / clusters
    centroids = np.quantile(values, quantiles).astype(np.float64)
    for _ in range(50):
        counts = np.zeros(clusters, dtype=np.float64)
        sums = np.zeros(clusters, dtype=np.float64)
        for start in range(0, values.size, 4096):
            chunk = values[start : start + 4096]
            assignments = np.abs(chunk[:, None] - centroids[None, :]).argmin(axis=1)
            counts += np.bincount(assignments, minlength=clusters)
            sums += np.bincount(assignments, weights=chunk, minlength=clusters)
        updated = np.where(counts > 0, sums / np.maximum(counts, 1.0), centroids)
        if np.max(np.abs(updated - centroids)) < 1e-7:
            centroids = updated
            break
        centroids = updated
    return np.sort(centroids)


class ResidualScalarQuantizer(_TransformedTokenizer):
    def __init__(self, scale: str, level_sizes: Sequence[int], seed: int = 13) -> None:
        super().__init__(scale)
        self.level_sizes = tuple(int(value) for value in level_sizes)
        self.seed = int(seed)
        self.codebooks = [np.zeros(size, dtype=np.float64) for size in self.level_sizes]
        self.key = "time-rsq-{}-{}".format(scale, "-".join(str(v) for v in self.level_sizes))

    def fit(self, values: Iterable[float]) -> "ResidualScalarQuantizer":
        residual = self._transform(_finite_nonnegative(values))
        for level, size in enumerate(self.level_sizes):
            centroids = _one_dimensional_kmeans(residual, size, self.seed + level)
            self.codebooks[level] = centroids
            assignments = np.abs(residual[:, None] - centroids[None, :]).argmin(axis=1)
            residual = residual - centroids[assignments]
        return self

    def encode(self, value: float) -> List[str]:
        residual = float(self._transform(np.asarray([max(0.0, value)]))[0])
        tokens = []
        for level, centroids in enumerate(self.codebooks):
            index = int(np.abs(centroids - residual).argmin())
            tokens.append("<|rsq_l{}_{:03d}|>".format(level, index))
            residual -= float(centroids[index])
        return tokens

    def decode(self, tokens: Sequence[str]) -> float:
        by_level: Dict[int, int] = {}
        for token in tokens:
            match = re.fullmatch(r"<\|rsq_l(\d+)_(\d{3})\|>", token.strip())
            if match is not None:
                by_level[int(match.group(1))] = int(match.group(2))
        if len(by_level) != len(self.codebooks):
            raise ValueError("Incomplete RSQ token sequence")
        reconstructed = 0.0
        for level, centroids in enumerate(self.codebooks):
            index = by_level[level]
            if index >= len(centroids):
                raise ValueError("RSQ code index outside codebook")
            reconstructed += float(centroids[index])
        return float(self._inverse(np.asarray([reconstructed]))[0])

    def vocabulary(self) -> List[str]:
        return [
            "<|rsq_l{}_{:03d}|>".format(level, index)
            for level, size in enumerate(self.level_sizes)
            for index in range(size)
        ]


@dataclass(frozen=True)
class TemporalTokenizerSpec:
    key: str
    family: str
    factory: object


def _calendar_factory(absolute: bool, resolution: str):
    return lambda: CalendarTokenizer(absolute=absolute, resolution=resolution)


def _rsq_factory(scale: str, sizes: Sequence[int]):
    return lambda: ResidualScalarQuantizer(scale=scale, level_sizes=sizes)


TEMPORAL_TOKENIZER_SPECS: Dict[str, TemporalTokenizerSpec] = {}


def _register(key: str, family: str, factory) -> None:
    TEMPORAL_TOKENIZER_SPECS[key] = TemporalTokenizerSpec(key, family, factory)


_register("time-numeric-p6", "numeric", lambda: NumericStringTokenizer(6))
_register("time-byte-f32", "byte", FloatByteTokenizer)
for _absolute in (True, False):
    for _resolution in ("day", "hour", "minute", "second"):
        _key = "time-calendar-{}-{}".format("abs" if _absolute else "rel", _resolution)
        _register(_key, "calendar", _calendar_factory(_absolute, _resolution))
for _scale in ("linear", "log"):
    _register(
        "time-bin-{}-k256".format(_scale),
        "scale-bin",
        lambda scale=_scale: ScaleBinTokenizer(scale, 256),
    )
    for _sizes in ((256,), (128, 128), (85, 85, 86), (64, 64, 64, 64)):
        _key = "time-rsq-{}-{}".format(_scale, "-".join(str(value) for value in _sizes))
        _register(_key, "rsq", _rsq_factory(_scale, _sizes))


def build_temporal_tokenizer(key: str) -> TemporalTokenizer:
    if key not in TEMPORAL_TOKENIZER_SPECS:
        raise KeyError(
            "Unknown temporal tokenizer `{}`. Available: {}".format(
                key, sorted(TEMPORAL_TOKENIZER_SPECS)
            )
        )
    return TEMPORAL_TOKENIZER_SPECS[key].factory()  # type: ignore[operator]


def temporal_tokenizer_names() -> List[str]:
    return sorted(TEMPORAL_TOKENIZER_SPECS)
