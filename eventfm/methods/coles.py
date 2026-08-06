"""CoLES (Babaev et al., SIGMOD 2022) via the pytorch-lifestream recipe.

CoLES avoids both autoregressive generation and bidirectional masked modelling.
It splits one client's history into several sub-sequences, treats sub-sequences
of the same client as positives and sub-sequences of other clients as
negatives, and trains an RNN encoder with a contrastive loss. The result is a
user-level embedding obtained at a fraction of the compute of token prediction,
which is exactly the trade-off the project note highlights.

Here the contrastive stage is run first, then the frozen-then-finetuned encoder
is handed to the same task heads as every other method, so the numbers stay
comparable.
"""

import random
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from eventfm.data.dataset import JsonlEventDataset
from eventfm.data.flat import FlatEventCollator
from eventfm.data.schema import EventSequence
from eventfm.methods.base import STATUS_APPROXIMATION, MethodSpec, register_method, resolve_device
from eventfm.methods.flat_method import FlatMethod
from eventfm.models.configuration_flat import FlatEventConfig
from eventfm.models.modeling_flat import FlatEventBackbone


class _SubSequenceView(Dataset):
    """Yield ``num_views`` random contiguous slices per client."""

    def __init__(
        self,
        sequences: Sequence[EventSequence],
        num_views: int = 2,
        min_length: int = 8,
        max_length: int = 32,
        seed: int = 13,
    ) -> None:
        self.sequences = list(sequences)
        self.num_views = int(num_views)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> List[EventSequence]:
        sequence = self.sequences[index]
        views: List[EventSequence] = []
        for _ in range(self.num_views):
            length = min(len(sequence.events), self.rng.randint(self.min_length, self.max_length))
            length = max(2, length)
            start = self.rng.randint(0, max(0, len(sequence.events) - length))
            views.append(
                EventSequence(
                    user_id=sequence.user_id,
                    events=sequence.events[start : start + length],
                    label=sequence.label,
                    profile=dict(sequence.profile),
                )
            )
        return views


class _ContrastiveCollator:
    """Flatten the per-client views into one batch plus client ids."""

    def __init__(self, inner: FlatEventCollator) -> None:
        self.inner = inner

    def __call__(self, examples: Sequence[List[EventSequence]]) -> Dict[str, torch.Tensor]:
        flat: List[EventSequence] = []
        group_ids: List[int] = []
        for group_index, views in enumerate(examples):
            for view in views:
                flat.append(view)
                group_ids.append(group_index)
        batch = self.inner(flat)
        batch.pop("labels", None)
        batch["group_ids"] = torch.as_tensor(group_ids, dtype=torch.long)
        return batch


def contrastive_loss(embeddings: torch.Tensor, group_ids: torch.Tensor, temperature: float = 0.1):
    """InfoNCE over sub-sequence views, positives = same client."""

    embeddings = F.normalize(embeddings, dim=-1)
    similarity = embeddings @ embeddings.t() / temperature
    diagonal = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    similarity = similarity.masked_fill(diagonal, torch.finfo(similarity.dtype).min)

    positives = group_ids.unsqueeze(0).eq(group_ids.unsqueeze(1)) & ~diagonal
    log_prob = similarity - torch.logsumexp(similarity, dim=-1, keepdim=True)
    positive_counts = positives.sum(dim=-1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return similarity.sum() * 0.0
    mean_log_prob = (log_prob * positives).sum(dim=-1)[valid] / positive_counts[valid]
    return -mean_log_prob.mean()


class Coles(FlatMethod):
    name = "coles"
    backbone = "gru"
    pooling = "mean"
    contrastive_epochs = 3
    num_views = 4
    temperature = 0.1

    def configure(self, config: FlatEventConfig) -> FlatEventConfig:
        config.pooling = "mean"
        return config

    def pretrain(self, model: torch.nn.Module) -> None:
        context = self.context
        if self.contrastive_epochs <= 0:
            return

        device = resolve_device()
        config = model.backbone.config if hasattr(model, "backbone") else self.build_config()
        encoder = FlatEventBackbone(config).to(device)
        # Hybrid objectives continue from the encoder learned by the preceding
        # stage instead of silently replacing it with a fresh random encoder.
        if hasattr(model, "backbone"):
            encoder.load_state_dict(model.backbone.state_dict())
        projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        ).to(device)

        sequences = JsonlEventDataset(
            str(context.extra.get("pretrain_path", context.train_path))
        ).sequences
        dataset = _SubSequenceView(
            sequences,
            num_views=self.num_views,
            max_length=min(32, context.training.max_events),
            seed=context.training.seed,
        )
        collator = _ContrastiveCollator(
            FlatEventCollator(
                tokenizer=context.tokenizer,
                task="pretrain",
                max_events=context.training.max_events,
                feature_fields=context.model_feature_fields,
            )
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(4, context.training.batch_size // self.num_views),
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )
        parameters = list(encoder.parameters()) + list(projection.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=context.training.learning_rate)

        encoder.train()
        for _ in range(int(self.contrastive_epochs)):
            for batch in loader:
                group_ids = batch.pop("group_ids").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                pooled = encoder(**batch).pooled
                loss = contrastive_loss(projection(pooled), group_ids, self.temperature)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

        model.backbone.load_state_dict({
            key: value.cpu() for key, value in encoder.state_dict().items()
        })


register_method(
    MethodSpec(
        name="coles",
        status=STATUS_APPROXIMATION,
        divergence=(
            "same-entity subsequence contrastive pretraining; augmentations and loss are not "
            "audited against the reference pytorch-lifestream recipe"
        ),
        display_name="CoLES",
        reference="Babaev et al., CoLES: contrastive learning for event sequences (2022)",
        family="contrastive",
        factory=Coles,
        notes="Sub-sequence contrastive pretraining (pytorch-lifestream recipe), then a task head.",
    )
)
