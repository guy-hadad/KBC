"""Every architecture family named in the KBC/TPP project note, side by side.

The note surveys nine modelling directions. Each one is available here under a
short registry key, and all of them are evaluated on the same splits, with the
same metrics, for both downstream tasks:

===================  ==========================================================
key                  paper / system
===================  ==========================================================
count-logistic       count-feature logistic regression (non-neural control)
markov               first-order marked Markov chain (non-neural control)
ntpp-gru             Shchur et al., *Neural temporal point processes: a review*
thp                  Transformer Hawkes-style attention intensity model
pragma               Ostroukhov et al., *PRAGMA: Revolut foundation model*
tabbert              Padhi et al., *Tabular transformers* (BERT variant)
tabgpt               Padhi et al., *Tabular transformers* (GPT variant)
coles                Babaev et al., *CoLES: contrastive learning for event seqs*
mambular             Thielmann et al., *Mambular* (Mamba state-space tabular)
nvidia-tfm           NVIDIA Transaction Foundation Model blueprint
tpp-llm              Liu & Quan, *TPP-LLM*
language-tpp         Kong et al., *Byte-token enhanced LMs for TPP analysis*
mm-tpp               Li et al., *Long-range modeling of multimodal event seqs*
===================  ==========================================================
"""

from eventfm.methods import (  # noqa: F401  (import for registration side effects)
    classic,
    coles,
    llm_tpp,
    mambular,
    neural_tpp,
    nvidia_tfm,
    pragma,
    tabformer,
)
from eventfm.methods.base import (
    METHOD_REGISTRY,
    MethodContext,
    MethodSpec,
    build_method,
    method_names,
)

__all__ = [
    "METHOD_REGISTRY",
    "MethodContext",
    "MethodSpec",
    "build_method",
    "method_names",
]
