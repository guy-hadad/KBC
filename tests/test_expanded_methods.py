import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="expanded model tests require torch",
)

import torch  # noqa: E402

from eventfm.data.pretraining import EventPretrainingCollator  # noqa: E402
from eventfm.data.schema import Event, EventSequence  # noqa: E402
from eventfm.data.tokenizer import EventTokenizer, EventVocabulary  # noqa: E402
from eventfm.models.configuration_flat import FlatEventConfig  # noqa: E402
from eventfm.models.modeling_flat import FlatEventBackbone, FlatForNextEventPrediction  # noqa: E402
from eventfm.models.modeling_pretraining import FlatForEventPretraining  # noqa: E402


def _sequences():
    return [
        EventSequence(
            user_id="u{}".format(row),
            events=[
                Event(
                    event_type=("purchase", "transfer", "withdrawal")[index % 3],
                    timestamp=1_700_000_000.0 + index * (900 + row * 60),
                    features={"amount": "b_{}".format(index % 4)},
                )
                for index in range(7)
            ],
            label=row % 2,
        )
        for row in range(3)
    ]


def _tokenizer(sequences):
    event_types = ["purchase", "transfer", "withdrawal"]
    vocab = EventVocabulary.from_sequences(sequences, event_types=event_types)
    return EventTokenizer(
        vocab=vocab,
        max_features_per_event=2,
        event_type_to_index={name: index for index, name in enumerate(event_types)},
    )


@pytest.mark.parametrize("backbone", ["nhp", "attnhp", "cotic"])
def test_new_temporal_backbones_preserve_the_flat_contract(backbone):
    config = FlatEventConfig(
        vocab_size=32,
        num_event_types=3,
        num_feature_fields=1,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        backbone=backbone,
    )
    model = FlatEventBackbone(config)
    batch = {
        "event_type_ids": torch.tensor([[0, 1, 2, 0], [1, 2, 0, 0]]),
        "feature_value_ids": torch.ones(2, 4, 1, dtype=torch.long),
        "delta_log": torch.rand(2, 4),
        "time_since_start": torch.rand(2, 4),
        "calendar_features": torch.rand(2, 4, 6),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }

    output = model(**batch)

    assert output.hidden_states.shape == (2, 4, 16)
    assert output.pooled.shape == (2, 16)
    assert torch.isfinite(output.hidden_states).all()


def test_exponential_intensity_head_produces_a_finite_likelihood():
    config = FlatEventConfig(
        vocab_size=32,
        num_event_types=3,
        num_feature_fields=1,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        backbone="nhp",
        time_loss="exponential_intensity",
    )
    model = FlatForNextEventPrediction(config)
    output = model(
        event_type_ids=torch.tensor([[0, 1, 2, 0]]),
        feature_value_ids=torch.ones(1, 4, 1, dtype=torch.long),
        delta_log=torch.rand(1, 4),
        time_since_start=torch.rand(1, 4),
        calendar_features=torch.rand(1, 4, 6),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        next_event_type_labels=torch.tensor([1]),
        next_delta_log=torch.tensor([8.0]),
    )

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.time_nll).all()


def test_flat_backbone_exposes_a_trainable_bottleneck_adapter():
    config = FlatEventConfig(
        vocab_size=32,
        num_event_types=3,
        num_feature_fields=1,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        backbone="gru",
        adapter_size=4,
    )

    model = FlatEventBackbone(config)

    assert model.adapter is not None
    assert sum(parameter.numel() for parameter in model.adapter.parameters()) > 0


def test_cross_schema_model_separates_input_and_target_mark_spaces():
    config = FlatEventConfig(
        vocab_size=32,
        num_event_types=3,
        num_input_event_types=7,
        num_feature_fields=1,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        backbone="gru",
        pretraining_objectives=("next",),
    )

    task_model = FlatForNextEventPrediction(config)
    pretrain_model = FlatForEventPretraining(config)

    assert task_model.backbone.embeddings.mark_embeddings.num_embeddings == 7
    assert task_model.type_head.out_features == 3
    assert pretrain_model.next_mark_decoder.out_features == 7


def test_composable_pretraining_objectives_have_a_finite_loss():
    sequences = _sequences()
    tokenizer = _tokenizer(sequences)
    collator = EventPretrainingCollator(
        tokenizer=tokenizer,
        task="pretrain",
        max_events=8,
        feature_fields=["amount"],
        objectives=("masked", "next", "next-feature", "marked-tte", "motor"),
        num_event_types=3,
    )
    batch = collator(sequences)
    config = FlatEventConfig(
        vocab_size=len(tokenizer.vocab),
        num_event_types=3,
        num_feature_fields=1,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        backbone="bert-causal",
        pretraining_objectives=("masked", "next", "next-feature", "marked-tte", "motor"),
    )
    model = FlatForEventPretraining(config)

    output = model(**batch)

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
