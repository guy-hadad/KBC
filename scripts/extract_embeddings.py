"""Extract user embeddings from a trained checkpoint."""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from eventfm.data.collator import PragmaDataCollator
from eventfm.data.dataset import JsonlEventDataset
from eventfm.data.tokenizer import EventTokenizer, EventVocabulary
from eventfm.models import PragmaForMaskedEventModeling


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = to_absolute_path(cfg.extract.checkpoint_path)
    vocab_path = Path(checkpoint) / "vocab.json"
    if not vocab_path.exists():
        vocab_path = Path(to_absolute_path(cfg.data.vocab_path))
    vocab = EventVocabulary.load(str(vocab_path))
    tokenizer = EventTokenizer(vocab=vocab, max_features_per_event=int(cfg.data.max_features_per_event))
    split_path = to_absolute_path(getattr(cfg.data, "{}_path".format(cfg.extract.split)))
    dataset = JsonlEventDataset(split_path)
    collator = PragmaDataCollator(
        tokenizer=tokenizer,
        task="classification",
        max_events=int(cfg.data.max_events_per_sequence),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.per_device_eval_batch_size),
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PragmaForMaskedEventModeling.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    embeddings = []
    user_ids = [sequence.user_id for sequence in dataset.sequences]
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in {"labels", "next_event_type_labels", "next_delta_log"}
            }
            output = model.backbone(**batch)
            embeddings.append(output.user_embedding.cpu().numpy())

    array = np.concatenate(embeddings, axis=0)
    output_path = Path(to_absolute_path(cfg.extract.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    with (output_path.with_suffix(".json")).open("w", encoding="utf-8") as handle:
        json.dump({"user_ids": user_ids, "embedding_path": str(output_path)}, handle, indent=2)
    print(json.dumps({"output_path": str(output_path), "shape": list(array.shape)}, indent=2))


if __name__ == "__main__":
    main()
