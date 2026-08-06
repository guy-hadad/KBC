"""Empirical data-scaling laws: how fast does each method improve with data?

The per-method curves in :mod:`eventfm.benchmark.report` show *what* each model
scores at each training-set size. This module answers the sharper question the
project note actually cares about — **how steeply does each architecture convert
extra labelled sequences into accuracy** — by fitting the standard empirical
form to the error of each curve:

.. math::  E(N) = a \\cdot N^{-b}

so that ``log E`` is linear in ``log N`` and the fitted slope ``-b`` is the
data-scaling exponent. A larger ``b`` means the method is still hungry for
data; a ``b`` near zero means it has flattened out and more sequences will not
help.

Two honesty notes, because six points is not many:

* the two-parameter form is fitted, not the three-parameter
  ``E_inf + a N^-b``. Estimating an irreducible-error term from six points is
  not defensible, and leaving it out makes ``b`` a *local* slope over the range
  actually measured rather than an asymptotic claim;
* every fit reports :math:`R^2`, and anything below ``MIN_R_SQUARED`` is
  reported but flagged, because a power law is a model of the data, not a fact
  about it.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Below this, the power law does not describe the curve and the exponent should
# not be read as a scaling rate.
MIN_R_SQUARED = 0.70

# Metric -> (display label, how to turn it into an error that decreases with N).
# "complement" means error = 1 - metric; "direct" means the metric already is an
# error and is used as-is.
ERROR_METRICS: Dict[str, List[Tuple[str, str, str]]] = {
    "classification": [
        ("auc", "1 − ROC-AUC", "complement"),
        ("average_precision", "1 − avg. precision", "complement"),
    ],
    "tpp": [
        ("next_type_accuracy", "1 − next-type acc.", "complement"),
        ("delta_log_rmse", "Time RMSE (log1p s)", "direct"),
    ],
}


@dataclass
class ScalingFit:
    """A fitted power law for one (dataset, task, method, metric) curve."""

    exponent: float
    coefficient: float
    r_squared: float
    num_points: int
    sizes: List[int]
    errors: List[float]

    @property
    def is_reliable(self) -> bool:
        return self.num_points >= 4 and self.r_squared >= MIN_R_SQUARED

    def predict(self, sizes: Sequence[float]) -> np.ndarray:
        return self.coefficient * np.power(np.asarray(sizes, dtype=np.float64), -self.exponent)


def to_error(value: float, mode: str) -> Optional[float]:
    """Convert a reported metric into an error that should shrink with data."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if mode == "complement":
        return float(max(1e-6, 1.0 - value))
    if mode == "direct":
        return float(value) if value > 0 else None
    raise ValueError("Unknown error mode: {}".format(mode))


def fit_scaling_law(points: Sequence[Tuple[int, float]]) -> Optional[ScalingFit]:
    """Least-squares fit of ``log(error) = log(a) - b log(N)``.

    Returns ``None`` when there are too few usable points to fit a line, or when
    the sizes do not actually vary.
    """

    usable = [
        (int(size), float(error))
        for size, error in points
        if size > 0 and error is not None and error > 0 and math.isfinite(error)
    ]
    if len(usable) < 3:
        return None

    sizes = np.asarray([size for size, _ in usable], dtype=np.float64)
    errors = np.asarray([error for _, error in usable], dtype=np.float64)
    if np.unique(sizes).size < 2:
        return None

    log_sizes = np.log(sizes)
    log_errors = np.log(errors)
    slope, intercept = np.polyfit(log_sizes, log_errors, 1)

    predicted = slope * log_sizes + intercept
    residual = float(np.sum((log_errors - predicted) ** 2))
    total = float(np.sum((log_errors - log_errors.mean()) ** 2))
    r_squared = 1.0 if total <= 1e-12 else float(1.0 - residual / total)

    return ScalingFit(
        exponent=float(-slope),
        coefficient=float(np.exp(intercept)),
        r_squared=r_squared,
        num_points=len(usable),
        sizes=[int(size) for size in sizes],
        errors=[float(error) for error in errors],
    )


def fit_all(
    records,
    datasets: Sequence[str],
    tasks: Sequence[str],
    methods: Sequence[str],
) -> Dict[Tuple[str, str, str, str], ScalingFit]:
    """Fit every (dataset, task, method, metric) curve that has enough points."""

    from eventfm.benchmark.report import series_for

    fits: Dict[Tuple[str, str, str, str], ScalingFit] = {}
    for task in tasks:
        for metric, _, mode in ERROR_METRICS.get(task, []):
            for dataset in datasets:
                for method in methods:
                    points = series_for(records, dataset, task, method, metric)
                    errors = [
                        (size, to_error(value, mode))
                        for size, value in points
                    ]
                    fit = fit_scaling_law(
                        [(size, error) for size, error in errors if error is not None]
                    )
                    if fit is not None:
                        fits[(dataset, task, method, metric)] = fit
    return fits
