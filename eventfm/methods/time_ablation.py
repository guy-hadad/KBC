"""Controlled time-representation ablations under one causal Transformer."""

from eventfm.methods.base import STATUS_IMPLEMENTED, MethodSpec, register_method
from eventfm.methods.flat_method import FlatMethod
from eventfm.models.configuration_flat import FlatEventConfig


class _TimeAblation(FlatMethod):
    backbone = "bert-causal"
    pooling = "last"


class NoTime(_TimeAblation):
    name = "time-no-time"
    use_time_features = False
    use_calendar_features = False
    use_position_embeddings = False
    pooling = "mean"


class OrdinalTime(_TimeAblation):
    name = "time-ordinal"
    use_time_features = False
    use_calendar_features = False


class RawGap(_TimeAblation):
    name = "time-raw-gap"
    time_feature_mode = "raw-gap"
    use_calendar_features = False


class LogGap(_TimeAblation):
    name = "time-log-gap"
    time_feature_mode = "log-gap"
    use_calendar_features = False


class GapAndAge(_TimeAblation):
    name = "time-gap-age"
    time_feature_mode = "fourier"
    use_calendar_features = False


class CalendarOnly(_TimeAblation):
    name = "time-calendar-fourier"
    use_time_features = False
    use_calendar_features = True


class GapAgeCalendar(_TimeAblation):
    name = "time-gap-age-calendar"
    time_feature_mode = "fourier"


class Time2Vec(_TimeAblation):
    name = "time-time2vec"
    time_feature_mode = "time2vec"
    use_calendar_features = False


class FunctionalTime(_TimeAblation):
    name = "time-functional"
    time_feature_mode = "functional"
    use_calendar_features = False


class ContinuousRoPE(_TimeAblation):
    name = "time-continuous-rope"
    time_feature_mode = "continuous-rope"
    use_calendar_features = False


class DiscreteTimeBuckets(_TimeAblation):
    name = "time-discrete-buckets"
    time_feature_mode = "bucket"
    use_calendar_features = False

    def configure(self, config: FlatEventConfig) -> FlatEventConfig:
        config.num_time_buckets = int(self.context.extra.get("num_time_buckets", 128))
        return config


_CONDITIONS = [
    (NoTime, "No time", "event attributes only; even ordinal positions are disabled"),
    (OrdinalTime, "Ordinal position", "sequence order without elapsed wall-clock time"),
    (RawGap, "Raw inter-event gap", "raw gaps in days through a shared nonlinear encoder"),
    (LogGap, "Log inter-event gap", "log1p gaps only"),
    (GapAndAge, "Gap + cumulative time", "log gap and time since sequence start"),
    (CalendarOnly, "Calendar Fourier", "hour/day/week periodic features only"),
    (GapAgeCalendar, "Gap + age + calendar", "local, cumulative, and periodic time"),
    (Time2Vec, "Time2Vec", "learned linear and periodic time coordinates"),
    (FunctionalTime, "Functional time", "learned continuous radial time kernels"),
    (ContinuousRoPE, "Continuous-time RoPE", "elapsed time as rotary event coordinates"),
    (DiscreteTimeBuckets, "Discrete time buckets", "fixed log-gap bucket embeddings"),
]

for _method, _display, _notes in _CONDITIONS:
    register_method(
        MethodSpec(
            name=_method.name,
            status=STATUS_IMPLEMENTED,
            divergence="controlled time-representation condition, not a paper reproduction",
            display_name=_display,
            reference="EventFM controlled temporal ablation",
            family="time-ablation",
            factory=_method,
            notes=_notes,
        )
    )

