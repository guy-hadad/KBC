"""Pure causal-LM temporal-tokenization methods from Liu et al. (GEM 2026)."""

import os
from typing import Dict, List

import numpy as np
import torch

from eventfm.data.dataset import JsonlEventDataset, NextEventJsonlDataset
from eventfm.data.gem_text import (
    STRUCTURAL_TOKENS,
    GemCausalCollator,
    GemTemporalSerialiser,
    fit_temporal_tokenizer,
    parse_generated_event,
)
from eventfm.data.temporal_tokenizers import (
    TEMPORAL_TOKENIZER_SPECS,
    build_temporal_tokenizer,
)
from eventfm.methods.base import (
    STATUS_APPROXIMATION,
    Method,
    MethodSpec,
    _training_arguments,
    register_method,
)
from eventfm.models.modeling_llm_tpp import _lora_targets
from eventfm.training.metrics import macro_f1

DEFAULT_BASE_MODEL = os.environ.get("KBC_LLM_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M")


def method_key_for_tokenizer(tokenizer_key: str) -> str:
    return "gem-{}".format(tokenizer_key)


class GemTemporalTokenMethod(Method):
    """Fine-tune a causal LM to generate the next event type and time tokens."""

    tokenizer_key = "time-numeric-p6"
    max_text_length = 768
    max_events_in_prompt = 32

    def __init__(self, context) -> None:
        super().__init__(context)
        from transformers import AutoTokenizer

        self.base_model_name = str(context.extra.get("llm_base_model", DEFAULT_BASE_MODEL))
        self.max_text_length = int(
            context.extra.get("gem_max_text_length", self.max_text_length)
        )
        self.max_events_in_prompt = int(
            context.extra.get("gem_max_events", self.max_events_in_prompt)
        )
        self.temporal_tokenizer = build_temporal_tokenizer(self.tokenizer_key)
        train_sequences = JsonlEventDataset(context.train_path).sequences
        fit_temporal_tokenizer(self.temporal_tokenizer, train_sequences)

        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.hf_tokenizer.truncation_side = "left"
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        temporal_tokens = self.temporal_tokenizer.vocabulary()
        self.num_added_tokens = self.hf_tokenizer.add_special_tokens(
            {"additional_special_tokens": list(dict.fromkeys(STRUCTURAL_TOKENS + temporal_tokens))}
        )
        aliases = {}
        if bool(context.extra.get("anonymize_event_types", False)):
            aliases = {
                event_type: "event_type_{}".format(index)
                for index, event_type in enumerate(context.dataset.event_types)
            }
        self.generated_type_to_index = {
            aliases.get(event_type, event_type): index
            for index, event_type in enumerate(context.dataset.event_types)
        }
        self.serialiser = GemTemporalSerialiser(
            temporal_tokenizer=self.temporal_tokenizer,
            feature_fields=context.dataset.feature_fields,
            max_events=min(self.max_events_in_prompt, context.training.max_events),
            template_order=str(context.extra.get("gem_template_order", "type-time")),
            event_type_aliases=aliases,
        )

    def _build_model(self):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(self.base_model_name, dtype=torch.float32)
        model.resize_token_embeddings(len(self.hf_tokenizer))
        model.config.pad_token_id = self.hf_tokenizer.pad_token_id

        if bool(self.context.extra.get("freeze_base", True)):
            for parameter in model.parameters():
                parameter.requires_grad = False

        if bool(self.context.extra.get("use_lora", True)):
            try:
                from peft import LoraConfig, TaskType, get_peft_model

                targets = _lora_targets(model)
                if targets:
                    model = get_peft_model(
                        model,
                        LoraConfig(
                            r=int(self.context.extra.get("lora_r", 16)),
                            lora_alpha=int(self.context.extra.get("lora_alpha", 32)),
                            lora_dropout=float(self.context.extra.get("lora_dropout", 0.05)),
                            bias="none",
                            target_modules=targets,
                            task_type=TaskType.CAUSAL_LM,
                        ),
                    )
            except ImportError:
                pass

        # GEM explicitly trains newly introduced temporal embeddings. PEFT APIs
        # differ across versions, so unfreezing the tied embedding table is the
        # portable implementation; trainable-parameter counts expose its cost.
        for parameter in model.get_input_embeddings().parameters():
            parameter.requires_grad = True
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            for parameter in output_embeddings.parameters():
                parameter.requires_grad = True
        return model

    def _dataset(self, path: str, split: str):
        return NextEventJsonlDataset(
            path,
            min_prefix_events=int(self.context.extra.get("tpp_min_prefix_events", 4)),
            max_prefixes_per_user=int(self.context.extra.get("tpp_max_prefixes_per_user", 8)),
            seed=self.context.training.seed + (0 if split == "train" else 1),
        )

    def _evaluate_generation(self, model, dataset) -> Dict[str, float]:
        max_examples = self.context.extra.get("gem_max_eval_examples")
        examples = dataset.examples[: int(max_examples)] if max_examples else dataset.examples
        batch_size = int(self.context.extra.get("gem_generation_batch_size", 8))
        max_new_tokens = int(self.context.extra.get("gem_max_new_tokens", 48))
        event_type_to_index = self.generated_type_to_index

        predicted_types: List[int] = []
        true_types: List[int] = []
        predicted_delta_logs: List[float] = []
        true_delta_logs: List[float] = []
        invalid = 0
        time_token_counts: List[int] = []
        truncated = 0

        original_padding_side = self.hf_tokenizer.padding_side
        self.hf_tokenizer.padding_side = "left"
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                chunk = examples[start : start + batch_size]
                prompts = [self.serialiser.prompt(example) for example in chunk]
                raw_lengths = [
                    len(self.hf_tokenizer(prompt, add_special_tokens=False)["input_ids"])
                    for prompt in prompts
                ]
                truncated += sum(length > self.max_text_length for length in raw_lengths)
                encoded = self.hf_tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.hf_tokenizer.pad_token_id,
                    eos_token_id=self.hf_tokenizer.eos_token_id,
                )
                continuation = generated[:, encoded["input_ids"].shape[1] :].detach().cpu()
                for example, token_ids in zip(chunk, continuation):
                    parsed = parse_generated_event(
                        token_ids.tolist(),
                        self.hf_tokenizer,
                        self.temporal_tokenizer,
                        self.serialiser.template_order,
                        event_type_to_index,
                        previous_timestamp=example.sequence.events[-1].timestamp,
                    )
                    invalid += int(parsed["invalid"])
                    predicted_types.append(int(parsed["type_index"]))
                    true_type_text = self.serialiser.event_type_aliases.get(
                        example.next_event_type, example.next_event_type
                    )
                    true_types.append(int(event_type_to_index.get(true_type_text, 0)))
                    predicted_delta_logs.append(float(parsed["delta_log"]))
                    true_delta_logs.append(float(np.log1p(example.next_delta_seconds)))
                    time_token_counts.append(
                        len(self.temporal_tokenizer.encode(self.serialiser.true_time_value(example)))
                    )
        self.hf_tokenizer.padding_side = original_padding_side

        predicted_types_array = np.asarray(predicted_types, dtype=np.int64)
        true_types_array = np.asarray(true_types, dtype=np.int64)
        residual = np.asarray(predicted_delta_logs) - np.asarray(true_delta_logs)
        count = max(1, len(examples))
        return {
            "next_type_accuracy": float((predicted_types_array == true_types_array).mean()),
            "next_type_macro_f1": macro_f1(
                true_types_array, predicted_types_array, self.context.num_event_types
            ),
            "delta_log_rmse": float(np.sqrt(np.mean(residual**2))),
            "delta_log_mae": float(np.mean(np.abs(residual))),
            "invalid_decode_rate": float(invalid / count),
            "time_tokens_per_event": float(np.mean(time_token_counts)),
            "prompt_truncation_rate": float(truncated / count),
        }

    def run(self) -> Dict[str, float]:
        from transformers import Trainer

        model = self._build_model()
        self._model = model
        train_dataset = self._dataset(self.context.train_path, "train")
        test_dataset = self._dataset(self.context.test_path, "test")
        trainer = Trainer(
            model=model,
            args=_training_arguments(self.context, label_names=None),
            train_dataset=train_dataset,
            data_collator=GemCausalCollator(
                self.hf_tokenizer, self.serialiser, max_length=self.max_text_length
            ),
        )
        trainer.train()
        metrics = self._evaluate_generation(model, test_dataset)
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        metrics["num_parameters"] = float(trainable_parameters)
        metrics["num_trainable_parameters"] = float(trainable_parameters)
        metrics["num_total_parameters"] = float(
            sum(parameter.numel() for parameter in model.parameters())
        )
        metrics["temporal_vocabulary_size"] = float(
            len(self.temporal_tokenizer.vocabulary())
        )
        return metrics


def _factory(tokenizer_key: str):
    class _ConfiguredGemMethod(GemTemporalTokenMethod):
        pass

    _ConfiguredGemMethod.tokenizer_key = tokenizer_key
    _ConfiguredGemMethod.__name__ = "Gem{}".format(
        "".join(part.title() for part in tokenizer_key.split("-"))
    )
    return _ConfiguredGemMethod


for _tokenizer_key, _tokenizer_spec in sorted(TEMPORAL_TOKENIZER_SPECS.items()):
    _method_key = method_key_for_tokenizer(_tokenizer_key)
    register_method(
        MethodSpec(
            name=_method_key,
            status=STATUS_APPROXIMATION,
            divergence=(
                "pure causal generation and the paper's tokenizer logic, but the default EventFM "
                "backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden"
            ),
            display_name="GEM {}".format(_tokenizer_key.replace("time-", "")),
            reference="Liu et al., Temporal Tokenization Strategies (GEM 2026)",
            family="gem-{}".format(_tokenizer_spec.family),
            factory=_factory(_tokenizer_key),
            supports=("tpp",),
            requires_gpu=True,
            notes="Pure next-token generation with `{}`.".format(_tokenizer_key),
        )
    )
