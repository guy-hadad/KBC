"""Serializable event-sequence schema.

The first milestone uses only event type and timestamp, but the schema already
has `features` and `profile` dictionaries so bank-specific fields can be added
without changing the model API.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Event:
    """One timestamped event in a user history."""

    event_type: str
    timestamp: float
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSequence:
    """A sequence of events belonging to one entity or user."""

    user_id: str
    events: List[Event]
    label: Optional[Any] = None
    profile: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _event_from_dict(record: Dict[str, Any]) -> Event:
    event_type = record.get("event_type", record.get("type"))
    if event_type is None:
        raise ValueError("Event record must contain `event_type` or `type`.")
    if "timestamp" not in record:
        raise ValueError("Event record must contain `timestamp`.")

    features = dict(record.get("features", {}))
    for key, value in record.items():
        if key not in {"event_type", "type", "timestamp", "features"}:
            features[key] = value

    return Event(
        event_type=str(event_type),
        timestamp=float(record["timestamp"]),
        features=features,
    )


def _event_to_dict(event: Event) -> Dict[str, Any]:
    return {
        "event_type": event.event_type,
        "timestamp": float(event.timestamp),
        "features": dict(event.features),
    }


def sequence_from_json(record: Dict[str, Any]) -> EventSequence:
    """Convert a JSON-compatible dictionary to an `EventSequence`."""

    if "user_id" not in record:
        raise ValueError("Sequence record must contain `user_id`.")
    events = [_event_from_dict(event) for event in record.get("events", [])]
    events.sort(key=lambda event: event.timestamp)
    return EventSequence(
        user_id=str(record["user_id"]),
        events=events,
        label=record.get("label"),
        profile=dict(record.get("profile", {})),
        metadata=dict(record.get("metadata", {})),
    )


def sequence_to_json(sequence: EventSequence) -> Dict[str, Any]:
    """Convert an `EventSequence` to a JSON-compatible dictionary."""

    output = {
        "user_id": sequence.user_id,
        "events": [_event_to_dict(event) for event in sequence.events],
        "profile": dict(sequence.profile),
        "metadata": dict(sequence.metadata),
    }
    if sequence.label is not None:
        output["label"] = sequence.label
    return output


def validate_non_empty_sequences(sequences: Iterable[EventSequence]) -> None:
    for sequence in sequences:
        if not sequence.events:
            raise ValueError("Empty event sequences are not supported by the current collators.")
