"""The three language-model entries from the project note.

``tpp-llm``       Liu & Quan, *TPP-LLM* (arXiv:2410.02062)
``language-tpp``  Kong et al., *Byte-token enhanced LMs for TPP analysis* (WWW 2026)
``mm-tpp``        Li et al., *Long-range modeling of multimodal event sequences*

All three share a pretrained causal LM adapted with LoRA. They differ only in
how the history is presented to it, which is precisely the tokenisation
question the note poses:

* TPP-LLM writes the event types as text and injects time as a continuous
  temporal embedding;
* Language-TPP serialises each inter-arrival interval into byte tokens placed
  in the text stream;
* MM-TPP keeps the byte-token times and adds temporal-similarity compression so
  a longer history fits in the same context window.
"""

import os
from typing import Any, Callable, Dict, Sequence

import torch
from torch.utils.data import Dataset

from eventfm.data.dataset import JsonlEventDataset, NextEventJsonlDataset
from eventfm.data.text import EventTextSerialiser, TextEventCollator, extend_tokenizer
from eventfm.methods.base import STATUS_APPROXIMATION, MethodSpec, TorchMethod, register_method
from eventfm.models.modeling_llm_tpp import (
    LlmTppConfig,
    LlmTppForNextEventPrediction,
    LlmTppForSequenceClassification,
)

DEFAULT_BASE_MODEL = os.environ.get("KBC_LLM_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M")


class _TextLlmMethod(TorchMethod):
    style = "type_text"
    use_temporal_embedding = True
    max_text_length = 640
    max_events_in_prompt = 32
    compression_threshold = 0.25

    def __init__(self, context) -> None:
        super().__init__(context)
        from transformers import AutoTokenizer

        self.base_model_name = str(context.extra.get("llm_base_model", DEFAULT_BASE_MODEL))
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.num_added_tokens = extend_tokenizer(self.hf_tokenizer, self.style)

    def build_config(self) -> LlmTppConfig:
        context = self.context
        return LlmTppConfig(
            base_model_name=self.base_model_name,
            num_event_types=max(2, context.num_event_types),
            num_labels=context.num_labels,
            num_added_tokens=self.num_added_tokens,
            use_temporal_embedding=self.use_temporal_embedding,
            use_lora=bool(context.extra.get("use_lora", True)),
            freeze_base=bool(context.extra.get("freeze_base", True)),
        )

    def build_model(self) -> torch.nn.Module:
        config = self.build_config()
        if self.context.task == "classification":
            return LlmTppForSequenceClassification(config)
        return LlmTppForNextEventPrediction(config)

    def build_collator(self) -> Callable[[Sequence[Any]], Dict[str, torch.Tensor]]:
        context = self.context
        serialiser = EventTextSerialiser(
            style=self.style,
            feature_fields=context.dataset.feature_fields,
            max_events=min(self.max_events_in_prompt, context.training.max_events),
            compression_threshold=self.compression_threshold,
        )
        return TextEventCollator(
            hf_tokenizer=self.hf_tokenizer,
            serialiser=serialiser,
            task=context.task,
            event_type_to_index=context.dataset.event_type_to_index(),
            max_length=self.max_text_length,
        )

    def build_dataset(self, path: str, split: str) -> Dataset:
        if self.context.task == "classification":
            return JsonlEventDataset(path)
        return NextEventJsonlDataset(
            path,
            min_prefix_events=int(self.context.extra.get("tpp_min_prefix_events", 4)),
            max_prefixes_per_user=int(self.context.extra.get("tpp_max_prefixes_per_user", 8)),
            seed=self.context.training.seed + (0 if split == "train" else 1),
        )


class TppLlm(_TextLlmMethod):
    name = "tpp-llm"
    style = "type_text"
    use_temporal_embedding = True


class LanguageTpp(_TextLlmMethod):
    name = "language-tpp"
    style = "byte_time"
    # Timing already lives in the token stream, so no continuous side channel.
    use_temporal_embedding = False
    max_text_length = 768


class MmTpp(_TextLlmMethod):
    name = "mm-tpp"
    style = "byte_time_compressed"
    use_temporal_embedding = False
    max_text_length = 768
    # Compression buys context, so a longer raw history is fed in.
    max_events_in_prompt = 64


register_method(
    MethodSpec(
        name="tpp-llm",
        status=STATUS_APPROXIMATION,
        divergence=(
            "text history, continuous Fourier side channel and LoRA on a 135M base model; far "
            "below the paper's model scale"
        ),
        display_name="TPP-LLM",
        reference="Liu & Quan, TPP-LLM (2024)",
        family="llm",
        factory=TppLlm,
        requires_gpu=True,
        notes="Textual event types + continuous temporal embedding, LoRA-adapted LM.",
    )
)

register_method(
    MethodSpec(
        name="language-tpp",
        status=STATUS_APPROXIMATION,
        divergence=(
            "byte-tokenised log1p gaps as input, but time is decoded by the shared probabilistic "
            "head rather than autoregressive byte generation"
        ),
        display_name="Language-TPP",
        reference="Kong et al., Byte-token enhanced language models for TPP analysis (2026)",
        family="llm",
        factory=LanguageTpp,
        requires_gpu=True,
        notes="Inter-arrival times serialised as byte tokens inside the prompt.",
    )
)

register_method(
    MethodSpec(
        name="mm-tpp",
        status=STATUS_APPROXIMATION,
        divergence=(
            "byte-time input plus repository-specific temporal-similarity run compression; no "
            "visual modality"
        ),
        display_name="MM-TPP",
        reference="Li et al., Long-range modeling of multimodal event sequences (2026)",
        family="llm",
        factory=MmTpp,
        requires_gpu=True,
        notes="Language-TPP plus temporal-similarity compression for longer histories.",
    )
)
