"""Hugging Face configuration for PRAGMA-style event models."""

from transformers import PretrainedConfig


class PragmaConfig(PretrainedConfig):
    model_type = "pragma"

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        num_attention_heads: int = 4,
        profile_num_hidden_layers: int = 1,
        event_num_hidden_layers: int = 2,
        history_num_hidden_layers: int = 2,
        max_features_per_event: int = 8,
        max_profile_features: int = 32,
        feature_position_vocab_size: int = 32,
        calendar_feature_size: int = 6,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        unk_token_id: int = 2,
        usr_token_id: int = 3,
        evt_token_id: int = 4,
        num_event_types: int = 50,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        label_smoothing: float = 0.0,
        tpp_loss_weight: float = 1.0,
        time_encoding_frequencies: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.profile_num_hidden_layers = profile_num_hidden_layers
        self.event_num_hidden_layers = event_num_hidden_layers
        self.history_num_hidden_layers = history_num_hidden_layers
        self.max_features_per_event = max_features_per_event
        self.max_profile_features = max_profile_features
        self.feature_position_vocab_size = feature_position_vocab_size
        self.calendar_feature_size = calendar_feature_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id
        self.usr_token_id = usr_token_id
        self.evt_token_id = evt_token_id
        self.num_event_types = num_event_types
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.label_smoothing = label_smoothing
        self.tpp_loss_weight = tpp_loss_weight
        self.time_encoding_frequencies = time_encoding_frequencies
