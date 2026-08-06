"""Collators for controlled event-sequence pretraining objectives."""

from typing import Any, Dict, Sequence, Set

import torch

from eventfm.data.flat import FlatEventCollator


class EventPretrainingCollator(FlatEventCollator):
    """Attach reconstruction, next-event, marked-TTE, and MOTOR targets."""

    def __init__(
        self,
        *args,
        objectives: Sequence[str],
        num_event_types: int,
        mask_probability: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.objectives: Set[str] = set(objectives)
        self.num_event_types = int(num_event_types)
        self.mask_probability = float(mask_probability)

    def __call__(self, examples: Sequence[Any]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        marks = batch["event_type_ids"]
        features = batch["feature_value_ids"]
        valid = batch["attention_mask"].bool()

        if {"autoencode", "masked"} & self.objectives:
            selected = valid.clone()
            if "masked" in self.objectives or "autoencode" in self.objectives:
                selected = valid & (torch.rand(marks.shape) < self.mask_probability)
                if not bool(selected.any()) and bool(valid.any()):
                    first = valid.nonzero(as_tuple=False)[0]
                    selected[first[0], first[1]] = True
            mark_labels = torch.full_like(marks, -100)
            mark_labels[selected] = marks[selected]
            feature_labels = torch.full_like(features, -100)
            feature_selected = selected.unsqueeze(-1).expand_as(features)
            feature_labels[feature_selected] = features[feature_selected]
            corrupted_marks = marks.clone()
            corrupted_features = features.clone()
            corrupted_marks[selected] = 0
            corrupted_features[feature_selected] = self.tokenizer.vocab.mask_token_id
            batch["event_type_ids"] = corrupted_marks
            batch["feature_value_ids"] = corrupted_features
            batch["mark_labels"] = mark_labels
            batch["feature_labels"] = feature_labels

        if {"next", "next-feature", "next-time", "marked-tte"} & self.objectives:
            next_valid = valid[:, :-1] & valid[:, 1:]
            next_mark_labels = torch.full_like(marks, -100)
            next_mark_labels[:, :-1] = torch.where(
                next_valid, marks[:, 1:], torch.full_like(marks[:, 1:], -100)
            )
            next_feature_labels = torch.full_like(features, -100)
            next_feature_labels[:, :-1, :] = torch.where(
                next_valid.unsqueeze(-1),
                features[:, 1:, :],
                torch.full_like(features[:, 1:, :], -100),
            )
            next_delta_labels = torch.zeros_like(batch["delta_log"])
            next_delta_labels[:, :-1] = batch["delta_log"][:, 1:]
            next_delta_mask = torch.zeros_like(valid)
            next_delta_mask[:, :-1] = next_valid
            batch["next_mark_labels"] = next_mark_labels
            batch["next_feature_labels"] = next_feature_labels
            batch["next_delta_labels"] = next_delta_labels
            batch["next_delta_mask"] = next_delta_mask

        if "motor" in self.objectives:
            batch.update(self._motor_targets(marks, batch["delta_log"], valid))
        return batch

    def _motor_targets(
        self, marks: torch.Tensor, delta_log: torch.Tensor, valid: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size, length = marks.shape
        values = torch.zeros(
            batch_size, length, self.num_event_types, dtype=delta_log.dtype
        )
        observed = torch.zeros(
            batch_size, length, self.num_event_types, dtype=torch.bool
        )
        mask = torch.zeros(batch_size, length, self.num_event_types, dtype=torch.bool)
        # Convert per-event gaps to cumulative physical seconds. The targets are
        # log1p seconds, but censoring is represented for every code.
        raw_gaps = torch.expm1(delta_log).clamp(min=0.0)
        for row in range(batch_size):
            size = int(valid[row].sum().item())
            if size < 2:
                continue
            timestamps = torch.cumsum(raw_gaps[row, :size], dim=0)
            for index in range(size - 1):
                horizon = timestamps[-1] - timestamps[index]
                values[row, index, :] = torch.log1p(horizon)
                mask[row, index, :] = True
                seen = set()
                for future in range(index + 1, size):
                    mark = int(marks[row, future].item())
                    if mark in seen or not 0 <= mark < self.num_event_types:
                        continue
                    seen.add(mark)
                    values[row, index, mark] = torch.log1p(
                        timestamps[future] - timestamps[index]
                    )
                    observed[row, index, mark] = True
        return {
            "code_time_labels": values,
            "code_time_observed": observed,
            "code_time_mask": mask,
        }

