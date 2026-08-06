"""Neural temporal point processes (Shchur et al., arXiv:2104.03528).

Two entries from that review are benchmarked:

``ntpp-gru``
    The canonical recipe: embed (mark, inter-arrival time), summarise the
    history with an RNN, and decode the next event with an intensity-free
    mixture density over the inter-arrival time.

``thp``
    The attention variant in the Transformer-Hawkes family: the same event
    embedding and decoder, but the history is mixed with causal self-attention
    over continuous-time position encodings instead of recurrence.

Both use a log-normal mixture for *when*, which the review recommends over
hand-designed intensities because it gives a closed-form likelihood and a
closed-form mean prediction.
"""

from eventfm.methods.base import STATUS_APPROXIMATION, MethodSpec, register_method
from eventfm.methods.flat_method import FlatMethod
from eventfm.models.configuration_flat import FlatEventConfig


class NeuralTppGru(FlatMethod):
    name = "ntpp-gru"
    backbone = "gru"
    pooling = "last"


class TransformerHawkes(FlatMethod):
    name = "thp"
    backbone = "bert-causal"
    pooling = "last"

    def configure(self, config: FlatEventConfig) -> FlatEventConfig:
        # Transformer-Hawkes leans on continuous-time encodings rather than
        # calendar periodicity, so the calendar branch is switched off.
        config.use_calendar_features = False
        config.time_encoding_frequencies = 32
        return config


class NeuralHawkesProcess(FlatMethod):
    name = "nhp"
    backbone = "nhp"
    pooling = "last"
    time_loss = "exponential_intensity"


class SelfAttentiveHawkesProcess(FlatMethod):
    name = "sahp"
    backbone = "bert-causal"
    pooling = "last"
    time_loss = "exponential_intensity"


class AttentiveNeuralHawkesProcess(FlatMethod):
    name = "attnhp"
    backbone = "attnhp"
    pooling = "last"
    time_loss = "exponential_intensity"

    def configure(self, config: FlatEventConfig) -> FlatEventConfig:
        config.time_encoding_frequencies = 32
        config.use_calendar_features = False
        return config


class IntensityFreeTpp(FlatMethod):
    name = "iftpp"
    backbone = "gru"
    pooling = "last"
    time_loss = "lognormal_mixture"


class Cotic(FlatMethod):
    name = "cotic"
    backbone = "cotic"
    pooling = "last"
    time_loss = "lognormal_mixture"


register_method(
    MethodSpec(
        name="ntpp-gru",
        status=STATUS_APPROXIMATION,
        divergence=(
            "GRU history with a log-normal-mixture gap head; not the continuous-time "
            "LSTM/intensity formulation of NHP"
        ),
        display_name="Neural TPP (GRU)",
        reference="Shchur et al., Neural temporal point processes: a review (2021)",
        family="neural-tpp",
        factory=NeuralTppGru,
        notes="RNN history encoder with a log-normal mixture inter-arrival decoder.",
    )
)

register_method(
    MethodSpec(
        name="nhp",
        status=STATUS_APPROXIMATION,
        divergence=(
            "continuous-time decayed GRU state with a piecewise exponential intensity; it does "
            "not reproduce the original continuous-time LSTM cell parameter-for-parameter"
        ),
        display_name="Neural Hawkes Process",
        reference="Mei & Eisner, The Neural Hawkes Process (2017)",
        family="neural-tpp",
        factory=NeuralHawkesProcess,
        notes="Learned continuous-time recurrent decay and a proper intensity likelihood.",
    )
)

register_method(
    MethodSpec(
        name="sahp",
        status=STATUS_APPROXIMATION,
        divergence=(
            "self-attentive continuous-time encoder with a piecewise exponential decoder under "
            "the shared EventFM event embedding"
        ),
        display_name="Self-Attentive Hawkes Process",
        reference="Zhang et al., Self-Attentive Hawkes Process (2020)",
        family="neural-tpp",
        factory=SelfAttentiveHawkesProcess,
        notes="Causal self-attention and an intensity likelihood.",
    )
)

register_method(
    MethodSpec(
        name="attnhp",
        status=STATUS_APPROXIMATION,
        divergence=(
            "attention history encoder and intensity likelihood implemented under the shared "
            "contract rather than the EasyTPP reference code"
        ),
        display_name="Attentive Neural Hawkes Process",
        reference="Yang et al., Attentive Neural Hawkes Process (2022)",
        family="neural-tpp",
        factory=AttentiveNeuralHawkesProcess,
        notes="Causal attention over continuous time with an intensity decoder.",
    )
)

register_method(
    MethodSpec(
        name="iftpp",
        status=STATUS_APPROXIMATION,
        divergence=(
            "direct log-normal-mixture conditional interval density using the shared GRU encoder"
        ),
        display_name="Intensity-Free TPP",
        reference="Shchur et al., Intensity-Free Learning of TPPs (2020)",
        family="neural-tpp",
        factory=IntensityFreeTpp,
        notes="Direct conditional gap distribution without numerical intensity integration.",
    )
)

register_method(
    MethodSpec(
        name="cotic",
        status=STATUS_APPROXIMATION,
        divergence=(
            "dilated continuous-time convolution with shared next-event heads, not the complete "
            "reference COTIC training recipe"
        ),
        display_name="COTIC",
        reference="Karpachev et al., COTIC (2023)",
        family="continuous-convolution",
        factory=Cotic,
        notes="Dilated history convolution modulated by irregular gaps.",
    )
)

register_method(
    MethodSpec(
        name="thp",
        status=STATUS_APPROXIMATION,
        divergence=(
            "causal attention over Fourier time features with a shared intensity-free decoder; "
            "not a faithful THP/SAHP intensity implementation"
        ),
        display_name="Transformer Hawkes",
        reference="Shchur et al. (2021), attention-based intensity family",
        family="neural-tpp",
        factory=TransformerHawkes,
        notes="Causal self-attention over continuous-time encodings.",
    )
)
