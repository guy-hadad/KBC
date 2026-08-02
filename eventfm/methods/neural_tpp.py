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
