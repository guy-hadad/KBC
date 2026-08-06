"""Configuration shared by every non-hierarchical backbone in the zoo."""

from typing import Optional

from transformers import PretrainedConfig


class FlatEventConfig(PretrainedConfig):
    """One config for all flat event encoders.

    `backbone` selects the sequence mixer, which is the only thing that differs
    between the neural-TPP, TabFormer, CoLES, Mambular and NVIDIA-blueprint
    entries: they all embed an event the same way and only disagree on how the
    history is mixed and what objective sits on top.
    """

    model_type = "flat_event"

    def __init__(
        self,
        vocab_size: int = 512,
        num_event_types: int = 16,
        num_input_event_types: Optional[int] = None,
        num_feature_fields: int = 1,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        backbone: str = "gru",
        max_position_embeddings: int = 512,
        calendar_feature_size: int = 6,
        time_encoding_frequencies: int = 16,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        num_time_mixture_components: int = 8,
        time_loss: str = "lognormal_mixture",
        tpp_loss_weight: float = 1.0,
        pad_token_id: int = 0,
        use_calendar_features: bool = True,
        use_time_features: bool = True,
        time_feature_mode: str = "fourier",
        type_conditioned_features: bool = False,
        num_time_buckets: int = 128,
        use_position_embeddings: bool = True,
        adapter_size: int = 0,
        pretraining_objectives=None,
        pretraining_loss_weights=None,
        pooling: str = "last",
        **kwargs,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, num_labels=num_labels, **kwargs)
        self.vocab_size = int(vocab_size)
        self.num_event_types = int(num_event_types)
        self.num_input_event_types = int(num_input_event_types or num_event_types)
        self.num_feature_fields = int(num_feature_fields)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.dropout = float(dropout)
        self.layer_norm_eps = float(layer_norm_eps)
        self.initializer_range = float(initializer_range)
        self.backbone = str(backbone)
        self.max_position_embeddings = int(max_position_embeddings)
        self.calendar_feature_size = int(calendar_feature_size)
        self.time_encoding_frequencies = int(time_encoding_frequencies)
        self.classifier_dropout = float(classifier_dropout)
        self.num_time_mixture_components = int(num_time_mixture_components)
        self.time_loss = str(time_loss)
        self.tpp_loss_weight = float(tpp_loss_weight)
        self.use_calendar_features = bool(use_calendar_features)
        self.use_time_features = bool(use_time_features)
        self.time_feature_mode = str(time_feature_mode)
        self.type_conditioned_features = bool(type_conditioned_features)
        self.num_time_buckets = int(num_time_buckets)
        self.use_position_embeddings = bool(use_position_embeddings)
        self.adapter_size = int(adapter_size)
        self.pretraining_objectives = list(pretraining_objectives or [])
        self.pretraining_loss_weights = dict(pretraining_loss_weights or {})
        self.pooling = str(pooling)
