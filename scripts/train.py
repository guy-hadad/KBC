"""Hydra entrypoint for pretraining and downstream tasks."""

import inspect
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from eventfm.data.collator import PragmaDataCollator
from eventfm.data.dataset import JsonlEventDataset, NextEventJsonlDataset
from eventfm.data.synthetic import write_synthetic_dataset
from eventfm.data.tokenizer import EventTokenizer, EventVocabulary
from eventfm.models import (
    PragmaConfig,
    PragmaForMaskedEventModeling,
    PragmaForNextEventPrediction,
    PragmaForSequenceClassification,
)
from eventfm.training.metrics import classification_metrics, tpp_metrics
from eventfm.training.utils import prepare_wandb, seed_everything


def _ensure_synthetic_data(cfg: DictConfig) -> None:
    expected = [
        to_absolute_path(cfg.data.train_path),
        to_absolute_path(cfg.data.validation_path),
        to_absolute_path(cfg.data.test_path),
        to_absolute_path(cfg.data.vocab_path),
    ]
    if not bool(cfg.data.auto_generate):
        return
    if all(Path(path).exists() for path in expected):
        return
    write_synthetic_dataset(
        output_dir=to_absolute_path(cfg.data.data_dir),
        num_users=int(cfg.data.num_users),
        num_event_types=int(cfg.data.num_event_types),
        min_events=int(cfg.data.min_events),
        max_events=int(cfg.data.max_events),
        seed=int(cfg.data.seed),
    )


def _load_vocab(cfg: DictConfig) -> EventVocabulary:
    vocab_path = to_absolute_path(cfg.data.vocab_path)
    if Path(vocab_path).exists():
        return EventVocabulary.load(vocab_path)
    return EventVocabulary.synthetic(num_event_types=int(cfg.data.num_event_types))


def _model_config(cfg: DictConfig, vocab: EventVocabulary) -> PragmaConfig:
    return PragmaConfig(
        vocab_size=len(vocab),
        hidden_size=int(cfg.model.hidden_size),
        intermediate_size=int(cfg.model.intermediate_size),
        num_attention_heads=int(cfg.model.num_attention_heads),
        profile_num_hidden_layers=int(cfg.model.profile_num_hidden_layers),
        event_num_hidden_layers=int(cfg.model.event_num_hidden_layers),
        history_num_hidden_layers=int(cfg.model.history_num_hidden_layers),
        max_features_per_event=int(cfg.data.max_features_per_event),
        feature_position_vocab_size=int(cfg.model.feature_position_vocab_size),
        dropout=float(cfg.model.dropout),
        classifier_dropout=float(cfg.model.classifier_dropout),
        pad_token_id=vocab.pad_token_id,
        mask_token_id=vocab.mask_token_id,
        unk_token_id=vocab.unk_token_id,
        usr_token_id=vocab.usr_token_id,
        evt_token_id=vocab.evt_token_id,
        num_event_types=int(cfg.data.num_event_types),
        num_labels=int(cfg.task.get("num_labels", 2)),
        label_smoothing=float(cfg.task.get("label_smoothing", 0.0)),
        tpp_loss_weight=float(cfg.task.get("tpp_loss_weight", 1.0)),
        time_encoding_frequencies=int(cfg.model.time_encoding_frequencies),
    )


def _maybe_from_checkpoint(model_cls, checkpoint_path: Optional[str], config: PragmaConfig):
    if checkpoint_path:
        return model_cls.from_pretrained(
            to_absolute_path(checkpoint_path),
            config=config,
            ignore_mismatched_sizes=True,
        )
    return model_cls(config)


def _datasets_and_model(
    cfg: DictConfig,
    model_config: PragmaConfig,
) -> Tuple[object, object, object, Optional[Callable]]:
    task_name = str(cfg.task.name)
    train_path = to_absolute_path(cfg.data.train_path)
    validation_path = to_absolute_path(cfg.data.validation_path)

    if task_name == "pretrain":
        train_dataset = JsonlEventDataset(train_path)
        eval_dataset = JsonlEventDataset(validation_path)
        model = PragmaForMaskedEventModeling(model_config)
        metrics = None
    elif task_name == "classification":
        train_dataset = JsonlEventDataset(train_path)
        eval_dataset = JsonlEventDataset(validation_path)
        model = _maybe_from_checkpoint(
            PragmaForSequenceClassification,
            cfg.task.get("checkpoint_path"),
            model_config,
        )
        if bool(cfg.task.get("freeze_backbone", False)):
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
        metrics = classification_metrics
    elif task_name == "tpp":
        train_dataset = NextEventJsonlDataset(
            train_path,
            min_prefix_events=int(cfg.task.min_prefix_events),
            max_prefixes_per_user=cfg.task.get("max_prefixes_per_user"),
            seed=int(cfg.seed),
        )
        eval_dataset = NextEventJsonlDataset(
            validation_path,
            min_prefix_events=int(cfg.task.min_prefix_events),
            max_prefixes_per_user=cfg.task.get("max_prefixes_per_user"),
            seed=int(cfg.seed) + 1,
        )
        model = _maybe_from_checkpoint(
            PragmaForNextEventPrediction,
            cfg.task.get("checkpoint_path"),
            model_config,
        )
        metrics = tpp_metrics
    else:
        raise ValueError("Unknown task: {}".format(task_name))

    return train_dataset, eval_dataset, model, metrics


def _training_arguments(cfg: DictConfig) -> TrainingArguments:
    label_names = None
    if str(cfg.task.name) == "tpp":
        label_names = ["next_event_type_labels", "next_delta_log"]
    kwargs = {
        "output_dir": to_absolute_path(cfg.training.output_dir),
        "overwrite_output_dir": False,
        "num_train_epochs": float(cfg.training.num_train_epochs),
        "max_steps": int(cfg.training.max_steps),
        "learning_rate": float(cfg.training.learning_rate),
        "weight_decay": float(cfg.training.weight_decay),
        "warmup_ratio": float(cfg.training.warmup_ratio),
        "optim": str(cfg.training.optim),
        "per_device_train_batch_size": int(cfg.training.per_device_train_batch_size),
        "per_device_eval_batch_size": int(cfg.training.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
        "logging_steps": int(cfg.training.logging_steps),
        "eval_steps": int(cfg.training.eval_steps),
        "save_steps": int(cfg.training.save_steps),
        "save_total_limit": int(cfg.training.save_total_limit),
        "save_strategy": "steps",
        "report_to": ["wandb"] if bool(cfg.logging.wandb.enabled) else [],
        "run_name": "{}_{}_{}".format(cfg.project_name, cfg.task.name, cfg.model.name),
        "remove_unused_columns": False,
        "fp16": bool(cfg.training.fp16),
        "bf16": bool(cfg.training.bf16),
        "use_cpu": bool(cfg.training.use_cpu),
        "disable_tqdm": bool(cfg.training.disable_tqdm),
        "dataloader_num_workers": int(cfg.training.dataloader_num_workers),
        "gradient_checkpointing": bool(cfg.training.gradient_checkpointing),
        "label_names": label_names,
        "do_train": True,
        "do_eval": True,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    if "label_names" not in signature.parameters:
        kwargs.pop("label_names")
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return TrainingArguments(**kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    prepare_wandb(cfg)
    _ensure_synthetic_data(cfg)

    vocab = _load_vocab(cfg)
    tokenizer = EventTokenizer(
        vocab=vocab,
        max_features_per_event=int(cfg.data.max_features_per_event),
    )
    model_config = _model_config(cfg, vocab)
    train_dataset, eval_dataset, model, metrics = _datasets_and_model(cfg, model_config)
    collator = PragmaDataCollator(
        tokenizer=tokenizer,
        task=str(cfg.task.name),
        max_events=int(cfg.data.max_events_per_sequence),
        mlm_probability=float(cfg.task.get("mlm_probability", 0.0)),
        event_mask_probability=float(cfg.task.get("event_mask_probability", 0.0)),
        key_mask_probability=float(cfg.task.get("key_mask_probability", 0.0)),
        unk_probability=float(cfg.task.get("unk_probability", 0.0)),
    )
    args = _training_arguments(cfg)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=metrics,
    )
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    vocab.save(str(Path(args.output_dir) / "vocab.json"))
    print(json.dumps({"output_dir": args.output_dir, "task": str(cfg.task.name)}, indent=2))


if __name__ == "__main__":
    main()
