import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="collator smoke test requires torch",
)


def test_pretrain_collator_masks_some_values():
    from eventfm.data.collator import PragmaDataCollator
    from eventfm.data.synthetic import generate_synthetic_sequences
    from eventfm.data.tokenizer import EventTokenizer, EventVocabulary

    vocab = EventVocabulary.synthetic(num_event_types=50)
    tokenizer = EventTokenizer(vocab=vocab, max_features_per_event=8)
    collator = PragmaDataCollator(
        tokenizer=tokenizer,
        task="pretrain",
        max_events=16,
        mlm_probability=1.0,
        event_mask_probability=0.0,
        key_mask_probability=0.0,
        unk_probability=0.0,
    )
    batch = collator(generate_synthetic_sequences(num_users=2, min_events=8, max_events=8))

    assert batch["key_ids"].shape[:2] == (2, 8)
    assert (batch["labels"] != -100).sum().item() > 0
