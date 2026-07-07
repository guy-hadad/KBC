"""Model configs and PRAGMA-style model classes."""

from eventfm.models.configuration_pragma import PragmaConfig
from eventfm.models.modeling_pragma import (
    PragmaForMaskedEventModeling,
    PragmaForNextEventPrediction,
    PragmaForSequenceClassification,
)

__all__ = [
    "PragmaConfig",
    "PragmaForMaskedEventModeling",
    "PragmaForNextEventPrediction",
    "PragmaForSequenceClassification",
]
