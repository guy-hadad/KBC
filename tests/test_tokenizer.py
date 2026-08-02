from eventfm.data.synthetic import generate_synthetic_sequences
from eventfm.data.tokenizer import EventTokenizer, EventVocabulary


def test_synthetic_tokenizer_encodes_event_type_and_time_features():
    sequence = generate_synthetic_sequences(
        num_users=1, num_event_types=50, min_events=8, max_events=8
    )[0]
    vocab = EventVocabulary.synthetic(num_event_types=50)
    tokenizer = EventTokenizer(vocab=vocab, max_features_per_event=8)

    encoded = tokenizer.encode_sequence(sequence, max_events=8)

    assert encoded.key_ids.shape == (8, 8)
    assert encoded.value_ids.shape == (8, 8)
    assert encoded.calendar_features.shape == (8, 6)
    assert encoded.feature_attention_mask[:, 0].sum() == 8
    assert vocab.event_type_id("type_0") != vocab.unk_token_id


def test_event_type_index_is_stable_for_synthetic_names():
    vocab = EventVocabulary.synthetic(num_event_types=50)
    tokenizer = EventTokenizer(vocab=vocab)

    assert tokenizer.event_type_index("type_17") == 17
