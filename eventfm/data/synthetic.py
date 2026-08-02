"""Synthetic event-stream generator used for smoke tests and examples."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from eventfm.data.io import write_jsonl
from eventfm.data.schema import Event, EventSequence
from eventfm.data.tokenizer import EventVocabulary


def _sample_next_type(
    rng: np.random.Generator,
    previous_type: int,
    latent_segment: int,
    risky_user: bool,
    num_event_types: int,
) -> int:
    if risky_user and rng.random() < 0.25:
        return int(rng.integers(0, min(8, num_event_types)))

    segment_size = max(1, num_event_types // 5)
    segment_start = latent_segment * segment_size
    segment_end = min(num_event_types, segment_start + segment_size)
    if rng.random() < 0.68:
        local = (previous_type - segment_start + 1 + int(rng.integers(0, 3))) % max(
            1, segment_end - segment_start
        )
        return int(segment_start + local)
    if rng.random() < 0.82:
        return int(rng.integers(segment_start, segment_end))
    return int(rng.integers(0, num_event_types))


def generate_synthetic_sequences(
    num_users: int,
    num_event_types: int = 50,
    min_events: int = 16,
    max_events: int = 64,
    seed: int = 13,
    start_timestamp: float = 1704067200.0,
    label_noise_std: float = 0.0,
) -> List[EventSequence]:
    """Generate sequences with learnable labels and next-event dynamics."""

    rng = np.random.default_rng(seed)
    sequences: List[EventSequence] = []
    segment_count = 5
    segment_size = max(1, num_event_types // segment_count)

    for user_idx in range(num_users):
        risky_user = bool(rng.random() < 0.35)
        latent_segment = int(rng.integers(0, segment_count))
        length = int(rng.integers(min_events, max_events + 1))
        timestamp = float(start_timestamp + rng.integers(0, 14 * 24 * 3600))
        current_type = int(
            latent_segment * segment_size + rng.integers(0, min(segment_size, num_event_types))
        )

        events: List[Event] = []
        short_gap_count = 0
        risky_count = 0
        for event_idx in range(length):
            if event_idx > 0:
                current_type = _sample_next_type(
                    rng=rng,
                    previous_type=current_type,
                    latent_segment=latent_segment,
                    risky_user=risky_user,
                    num_event_types=num_event_types,
                )

            event_scale = 0.65 if current_type < min(8, num_event_types) else 1.0
            user_scale = 0.55 if risky_user else 1.25
            seasonal_scale = 0.8 + 0.4 * np.sin(2.0 * np.pi * (event_idx % 24) / 24.0)
            gap = float(rng.exponential(6.0 * 3600.0 * user_scale * event_scale * seasonal_scale))
            if event_idx > 0:
                timestamp += max(60.0, gap)
            if gap < 2.0 * 3600.0:
                short_gap_count += 1
            if current_type < min(8, num_event_types):
                risky_count += 1

            events.append(Event(event_type="type_{}".format(current_type), timestamp=timestamp))

        risk_ratio = risky_count / float(length)
        burst_ratio = short_gap_count / float(max(1, length - 1))
        label_logit = 4.5 * risk_ratio + 2.0 * burst_ratio + (0.8 if risky_user else -0.4)
        if label_noise_std > 0.0:
            label_logit += float(rng.normal(0.0, label_noise_std))
        label = int(label_logit > 1.2)

        sequences.append(
            EventSequence(
                user_id="user_{:06d}".format(user_idx),
                events=events,
                label=label,
                profile={"latent_segment": latent_segment},
                metadata={"risky_user": risky_user},
            )
        )

    return sequences


def split_sequences(
    sequences: List[EventSequence],
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> Dict[str, List[EventSequence]]:
    train_end = int(len(sequences) * train_fraction)
    validation_end = train_end + int(len(sequences) * validation_fraction)
    return {
        "train": sequences[:train_end],
        "validation": sequences[train_end:validation_end],
        "test": sequences[validation_end:],
    }


def write_synthetic_dataset(
    output_dir: str,
    num_users: int = 1000,
    num_event_types: int = 50,
    min_events: int = 16,
    max_events: int = 64,
    seed: int = 13,
) -> Tuple[Dict[str, str], str]:
    sequences = generate_synthetic_sequences(
        num_users=num_users,
        num_event_types=num_event_types,
        min_events=min_events,
        max_events=max_events,
        seed=seed,
    )
    splits = split_sequences(sequences)
    output = Path(output_dir)
    paths: Dict[str, str] = {}
    for split, split_sequences_ in splits.items():
        path = output / "{}.jsonl".format(split)
        write_jsonl(str(path), split_sequences_)
        paths[split] = str(path)

    vocab = EventVocabulary.synthetic(num_event_types=num_event_types)
    vocab_path = output / "vocab.json"
    vocab.save(str(vocab_path))
    return paths, str(vocab_path)
