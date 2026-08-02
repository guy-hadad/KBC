"""Tests for the two batch contracts the method zoo is built on."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="collator tests require torch",
)

from eventfm.data.schema import Event, EventSequence  # noqa: E402
from eventfm.data.tokenizer import EventTokenizer, EventVocabulary  # noqa: E402

EVENT_TYPES = ["purchase", "transfer", "withdrawal"]


def _sequence(user_id: str = "u1", num_events: int = 5, label: int = 1) -> EventSequence:
    return EventSequence(
        user_id=user_id,
        events=[
            Event(
                event_type=EVENT_TYPES[index % len(EVENT_TYPES)],
                # One hour apart, so log1p(delta) is a known constant.
                timestamp=1_640_995_200.0 + index * 3600.0,
                features={"amount": "b_{}".format(index % 4)},
            )
            for index in range(num_events)
        ],
        label=label,
    )


def _tokenizer(sequences) -> EventTokenizer:
    vocab = EventVocabulary.from_sequences(sequences, event_types=EVENT_TYPES)
    return EventTokenizer(
        vocab=vocab,
        max_features_per_event=2,
        event_type_to_index={name: index for index, name in enumerate(EVENT_TYPES)},
    )


def test_flat_collator_shapes_and_padding():
    from eventfm.data.flat import FlatEventCollator

    sequences = [_sequence("u1", 5), _sequence("u2", 3)]
    collator = FlatEventCollator(
        tokenizer=_tokenizer(sequences),
        task="classification",
        max_events=8,
        feature_fields=["amount"],
    )

    batch = collator(sequences)

    assert batch["event_type_ids"].shape == (2, 5)
    assert batch["feature_value_ids"].shape == (2, 5, 1)
    assert batch["calendar_features"].shape == (2, 5, 6)
    # The shorter history is right-padded and masked out.
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0]
    assert batch["labels"].tolist() == [1, 1]


def test_flat_collator_encodes_inter_arrival_times():
    from eventfm.data.flat import FlatEventCollator

    sequences = [_sequence("u1", 4)]
    collator = FlatEventCollator(
        tokenizer=_tokenizer(sequences),
        task="classification",
        max_events=8,
        feature_fields=["amount"],
    )

    batch = collator(sequences)

    # First event has no predecessor; the rest are one hour apart.
    assert batch["delta_log"][0, 0].item() == pytest.approx(0.0)
    assert batch["delta_log"][0, 1].item() == pytest.approx(np.log1p(3600.0), abs=1e-4)
    assert batch["time_since_start"][0, 3].item() == pytest.approx(np.log1p(3 * 3600.0), abs=1e-4)


def test_flat_collator_maps_marks_through_the_pinned_index():
    from eventfm.data.flat import FlatEventCollator

    sequences = [_sequence("u1", 3)]
    collator = FlatEventCollator(
        tokenizer=_tokenizer(sequences),
        task="classification",
        max_events=8,
        feature_fields=["amount"],
    )

    batch = collator(sequences)

    assert batch["event_type_ids"][0].tolist() == [0, 1, 2]


def test_flat_collator_rejects_plain_sequences_for_tpp():
    from eventfm.data.flat import FlatEventCollator

    sequences = [_sequence("u1", 5)]
    collator = FlatEventCollator(
        tokenizer=_tokenizer(sequences), task="tpp", max_events=8, feature_fields=["amount"]
    )

    with pytest.raises(ValueError, match="NextEventExample"):
        collator(sequences)


def test_byte_time_tokens_round_trip_a_float():
    import struct

    from eventfm.data.text import float_to_byte_tokens

    tokens = float_to_byte_tokens(9.5)

    assert len(tokens) == 4
    recovered = struct.unpack(
        ">f", bytes(int(token[len("<byte_") : -1]) for token in tokens)
    )[0]
    assert recovered == pytest.approx(9.5)


def test_temporal_compression_merges_similar_intervals():
    from eventfm.data.text import compress_by_temporal_similarity

    # Three near-identical gaps, then a clearly different one.
    runs = compress_by_temporal_similarity([1.0, 1.05, 1.02, 8.0, 8.01], threshold=0.25)

    assert runs == [3, 2]
    assert sum(runs) == 5


def test_serialiser_styles_differ_in_how_time_is_carried():
    from eventfm.data.text import EventTextSerialiser

    sequence = _sequence("u1", 4)

    type_text = EventTextSerialiser("type_text", feature_fields=["amount"])(sequence)
    byte_time = EventTextSerialiser("byte_time", feature_fields=["amount"])(sequence)

    # TPP-LLM keeps time out of the prompt; Language-TPP puts it in as bytes.
    assert "<byte_" not in type_text.text
    assert "<byte_" in byte_time.text
    assert "purchase" in type_text.text
    assert type_text.num_events_kept == 4


def test_mm_tpp_compression_shortens_the_prompt():
    from eventfm.data.text import EventTextSerialiser

    # Perfectly regular hourly events are exactly the case compression targets.
    sequence = _sequence("u1", 12)
    plain = EventTextSerialiser("byte_time", feature_fields=["amount"])(sequence)
    compressed = EventTextSerialiser("byte_time_compressed", feature_fields=["amount"])(sequence)

    assert compressed.num_events_kept < plain.num_events_kept
    assert compressed.num_events_compressed > 0
    assert len(compressed.text) < len(plain.text)


def test_unknown_serialiser_style_is_rejected():
    from eventfm.data.text import EventTextSerialiser

    with pytest.raises(ValueError, match="Unknown serialisation style"):
        EventTextSerialiser("no_such_style")
