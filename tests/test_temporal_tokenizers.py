import math

import numpy as np
import pytest

from eventfm.data.temporal_tokenizers import (
    CalendarTokenizer,
    FloatByteTokenizer,
    ResidualScalarQuantizer,
    ScaleBinTokenizer,
    build_temporal_tokenizer,
    temporal_tokenizer_names,
)


def test_complete_gem_tokenizer_registry():
    names = temporal_tokenizer_names()

    assert len(names) == 20
    assert "time-numeric-p6" in names
    assert "time-byte-f32" in names
    assert "time-calendar-abs-minute" in names
    assert "time-calendar-rel-hour" in names
    assert "time-bin-log-k256" in names
    assert "time-rsq-linear-85-85-86" in names
    assert "time-rsq-log-64-64-64-64" in names


def test_float_byte_tokenizer_is_bit_exact():
    tokenizer = FloatByteTokenizer()

    tokens = tokenizer.encode(12345.125)

    assert len(tokens) == 4
    assert tokenizer.decode(tokens) == pytest.approx(np.float32(12345.125))
    assert len(tokenizer.vocabulary()) == 256


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize(
    "resolution,token_count,tolerance",
    [("day", 3, 86400.0), ("hour", 4, 3600.0), ("minute", 5, 60.0), ("second", 6, 1.0)],
)
def test_calendar_resolutions_round_trip(absolute, resolution, token_count, tolerance):
    value = 1_700_123_456.0 if absolute else 400 * 86400.0 + 3723.0
    tokenizer = CalendarTokenizer(absolute=absolute, resolution=resolution).fit([value])

    tokens = tokenizer.encode(value)
    decoded = tokenizer.decode(tokens)

    assert len(tokens) == token_count
    assert abs(decoded - value) <= tolerance


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_scale_bins_fit_and_clip(scale):
    tokenizer = ScaleBinTokenizer(scale, num_bins=16).fit([0, 1, 10, 100, 1000])

    decoded = tokenizer.decode(tokenizer.encode(50.0))
    clipped = tokenizer.decode(tokenizer.encode(1e12))

    assert decoded >= 0
    assert math.isfinite(clipped)
    assert len(tokenizer.vocabulary()) == 16


@pytest.mark.parametrize("scale", ["linear", "log"])
@pytest.mark.parametrize("levels", [(32,), (16, 16), (8, 8, 8, 8)])
def test_residual_scalar_quantizers_are_compositional(scale, levels):
    values = np.geomspace(0.01, 10000, 500)
    tokenizer = ResidualScalarQuantizer(scale, levels, seed=7).fit(values)

    tokens = tokenizer.encode(123.0)
    decoded = tokenizer.decode(tokens)

    assert len(tokens) == len(levels)
    assert len(tokenizer.vocabulary()) == sum(levels)
    assert decoded >= 0
    assert math.isfinite(decoded)


def test_unknown_temporal_tokenizer_is_rejected():
    with pytest.raises(KeyError, match="Unknown temporal tokenizer"):
        build_temporal_tokenizer("not-a-tokenizer")

