"""ETAM-2 model configuration and loss weights.

Every Stage 2 mechanism is a switch here, so each ablation changes exactly one
field of the reference configuration (``Etam2Config()`` defaults = the main
proposal: private event/time pathways, lag-aware bridges after layers 2 and 4,
hybrid time encoder with an exponential temporal memory, dual decoder).
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List

MIXERS = ("shared", "stage1", "private", "private_shared", "s2p2", "itpp", "flextpp", "kairos")
DECODERS = ("independent", "time_first", "event_first", "dual")
TIME_DISTS = ("lognormal", "spline", "hazard")
CALENDARS = ("none", "fixed", "learned")
RELATIVES = ("stage1", "bins", "bins_residual")
MEMORIES = ("none", "exp", "exp_learned", "count", "ssm", "ode")
COUNT_DISTS = ("none", "stage1", "poisson", "negbin")
INPUTS = ("both", "event_only", "time_only")
BRIDGE_BIASES = ("none", "lag", "lag_mark")

#: Memory timescales in seconds: 1 min, 10 min, 1 h, 6 h, 1 d, 7 d, 30 d, 180 d.
DEFAULT_TIMESCALES = [60.0, 600.0, 3600.0, 21600.0, 86400.0, 604800.0, 2592000.0, 15552000.0]


@dataclass
class Etam2Config:
    num_event_types: int = 53
    field_sizes: List[int] = field(default_factory=list)
    #: Index of the amount field among the store's feature fields (-1: none).
    amount_field: int = -1
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_multiplier: float = 8.0 / 3.0
    dropout: float = 0.1
    max_positions: int = 4096

    # -- middle architecture ------------------------------------------------
    mixer: str = "private"
    #: 1-indexed layers after which an event<->time bridge runs (private mixer).
    bridge_layers: List[int] = field(default_factory=lambda: [2, 4])
    bridge_bias: str = "lag_mark"
    bridge_gate: bool = True
    num_lag_basis: int = 16
    #: Stage-1 mixer options (interleaved two-token layout).
    placement: str = "deep"
    rope: str = "index"
    inputs: str = "both"

    # -- time tokenizer -----------------------------------------------------
    calendar: str = "learned"
    relative: str = "bins_residual"
    gap_bin_edges: List[float] = field(default_factory=list)
    age_bin_edges: List[float] = field(default_factory=list)
    memory: str = "exp"
    memory_timescales: List[float] = field(default_factory=lambda: list(DEFAULT_TIMESCALES))
    memory_dim: int = 32
    ssm_modes: int = 16
    ssm_channels: int = 8
    ode_dim: int = 64
    ode_steps: int = 2
    #: Event content feeds the memory (False: decayed event counts only).
    memory_event_input: bool = True
    time_vq: int = 0

    # -- event tokenizer ----------------------------------------------------
    attr: str = "mlp"
    amount_scalar: bool = True
    amount_bucket_centers: List[float] = field(default_factory=list)

    # -- decoders -----------------------------------------------------------
    decoder: str = "dual"
    time_dist: str = "lognormal"
    num_mixture: int = 8
    spline_bins: int = 32
    hazard_edges: List[float] = field(default_factory=list)
    grid_bins: int = 48

    # -- auxiliary heads ----------------------------------------------------
    unimodal_heads: bool = True
    feature_head: bool = True
    #: Multi-event prediction: heads for events i+2 .. i+L (0 = off).
    mtp_horizon: int = 0
    mtp_target: str = "soft"
    mtp_time: bool = False
    count_dist: str = "negbin"
    count_horizons_days: List[float] = field(default_factory=lambda: [1.0, 7.0, 30.0])
    repr_dim: int = 0
    repr_block: int = 4
    #: Which representations the RePR objective trains (private ones need a private mixer).
    repr_modalities: List[str] = field(default_factory=lambda: ["joint", "event", "time"])

    # -- baselines ----------------------------------------------------------
    s2p2_states: int = 64
    s2p2_layers: int = 4
    itpp_dim: int = 32

    def __post_init__(self) -> None:
        checks = (
            ("mixer", MIXERS),
            ("decoder", DECODERS),
            ("time_dist", TIME_DISTS),
            ("calendar", CALENDARS),
            ("relative", RELATIVES),
            ("memory", MEMORIES),
            ("count_dist", COUNT_DISTS),
            ("inputs", INPUTS),
            ("bridge_bias", BRIDGE_BIASES),
        )
        for name, allowed in checks:
            if getattr(self, name) not in allowed:
                raise ValueError("{} must be one of {}, got {!r}".format(name, allowed, getattr(self, name)))
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mtp_target not in ("hard", "soft"):
            raise ValueError("mtp_target must be hard or soft")
        bad = [layer for layer in self.bridge_layers if not 1 <= layer <= self.num_layers]
        if bad:
            raise ValueError("bridge_layers {} outside 1..{}".format(bad, self.num_layers))

    @property
    def private(self) -> bool:
        return self.mixer in ("private", "private_shared")

    @property
    def intermediate_size(self) -> int:
        return int(math.ceil(self.hidden_size * self.ffn_multiplier / 64.0) * 64)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class Etam2Weights:
    #: -log p_joint(y, k | H) at every event slot (the inference distribution).
    joint: float = 1.0
    #: 1/2 [-log p_A - log p_B] (each branch a proper model on its own).
    dual: float = 1.0
    #: JS(p_A || p_B) on the discretised joint.
    cons: float = 0.1
    #: Cheap marginal mark head of a time-first decoder (CE).
    aux_mark: float = 1.0
    #: Attributes of the next event given its mark and time (generation).
    feature: float = 0.5
    #: Next event measured from "now" at the query slot.
    query: float = 1.0
    #: Elapsed-silence residual likelihood from the last event's distribution.
    survival: float = 0.5
    #: Right-censoring at the observation cutoff, inside the joint/query likelihood (0 = off, 1 = on).
    censor: float = 1.0
    #: Multiscale future event counts at the query slot.
    count: float = 0.5
    #: Event-only / time-only next-event heads on the private states.
    unimodal: float = 0.25
    #: Multi-event mark heads (hard or Soft-MTP) and cumulative-time heads.
    mtp: float = 0.0
    horizon: float = 0.0
    #: Predictive-information representation objective.
    repr: float = 0.0
    #: Vector-quantised time tokens (commitment + codebook).
    vq: float = 1.0
    #: Fine-tuning label.
    cls: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
