"""Mambular (Thielmann et al., arXiv:2408.06291).

Mamba replaces attention with a selective state-space scan: linear rather than
quadratic in sequence length, recurrent in spirit but hardware-parallel in
practice. The project note flags this as the scalability lever for long client
histories, so the only change relative to the other flat models is the mixer.

The ``transformers`` Mamba implementation is used, which falls back to a pure
PyTorch scan when the fused CUDA kernels are absent — the same code therefore
runs in CPU smoke tests and in GPU jobs.
"""

from eventfm.methods.base import STATUS_APPROXIMATION, MethodSpec, register_method
from eventfm.methods.flat_method import FlatMethod
from eventfm.models.configuration_flat import FlatEventConfig


class Mambular(FlatMethod):
    name = "mambular"
    backbone = "mamba"
    pooling = "last"

    def configure(self, config: FlatEventConfig) -> FlatEventConfig:
        # Mamba blocks already carry positional information through the scan,
        # and the reference implementation expects an even hidden size.
        config.hidden_size = int(config.hidden_size)
        return config


register_method(
    MethodSpec(
        name="mambular",
        status=STATUS_APPROXIMATION,
        divergence=(
            "selective state-space history mixer under the shared EventFM event contract rather "
            "than the library's tabular pipeline"
        ),
        display_name="Mambular (Mamba SSM)",
        reference="Thielmann et al., Mambular: a sequential model for tabular deep learning (2024)",
        family="state-space",
        factory=Mambular,
        notes="Selective state-space mixer with linear cost in history length.",
    )
)
