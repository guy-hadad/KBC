# Runnable research manifest

This file is generated from the method, dataset, tokenizer, and experiment
registries. Regenerate it with `python scripts/export_research_manifest.py`.
It inventories runnable code, not completed results or paper-faithful
reproductions.

## Coverage summary

- 62 registered methods: 18 implemented controls and 44 disclosed approximations;
- 17 prepared-dataset definitions;
- 20 GEM temporal tokenizer configurations;
- 24 named experiment suites.

`Implemented` means an in-repository control with a runnable contract and
smoke coverage. Paper-named systems are deliberately marked `Approximation`
when data, scale, likelihood, architecture, or training recipe differs.

## Methods

| Key | Display name | Family | Tasks | Fidelity | GPU | Reference | Disclosed divergence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `attnhp` | Attentive Neural Hawkes Process | `neural-tpp` | classification, tpp | Approximation | no | Yang et al., Attentive Neural Hawkes Process (2022) | attention history encoder and intensity likelihood implemented under the shared contract rather than the EasyTPP reference code |
| `autoencoder` | Transaction autoencoder | `reconstruction` | classification, tpp | Implemented | no | standard denoising sequence autoencoder baseline | controlled denoising autoencoder rather than a named paper reproduction |
| `autoregressive-transformer` | Autoregressive transaction Transformer | `generative` | classification, tpp | Implemented | no | generic causal next-event modeling baseline | generic controlled next-event baseline |
| `cmlm-coles` | CMLM + CoLES | `hybrid` | classification, tpp | Approximation | no | Babaev et al., Uniting contrastive and generative learning (2024) | sequential masked-event and CoLES stages under a shared benchmark encoder |
| `coles` | CoLES | `contrastive` | classification, tpp | Approximation | no | Babaev et al., CoLES: contrastive learning for event sequences (2022) | same-entity subsequence contrastive pretraining; augmentations and loss are not audited against the reference pytorch-lifestream recipe |
| `cotic` | COTIC | `continuous-convolution` | classification, tpp | Approximation | no | Karpachev et al., COTIC (2023) | dilated continuous-time convolution with shared next-event heads, not the complete reference COTIC training recipe |
| `count-logistic` | Count + logistic regression | `classic` | classification | Implemented | no | non-neural control | no paper to be faithful to; this is an engineered control |
| `engineered-gbdt` | Engineered features + GBDT | `classic` | classification | Implemented | no | non-neural production control | no single paper recipe; this is the registered production-style aggregate control |
| `eventfm-joint` | EventFM joint objective | `candidate` | classification, tpp | Approximation | no | candidate method specified in docs/research/method_catalog.md | candidate EventFM contribution; not a published baseline |
| `gem-time-bin-linear-k256` | GEM bin-linear-k256 | `gem-scale-bin` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-bin-log-k256` | GEM bin-log-k256 | `gem-scale-bin` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-byte-f32` | GEM byte-f32 | `gem-byte` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-calendar-abs-day` | GEM calendar-abs-day | `gem-calendar` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-calendar-abs-hour` | GEM calendar-abs-hour | `gem-calendar` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-calendar-abs-minute` | GEM calendar-abs-minute | `gem-calendar` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-calendar-abs-second` | GEM calendar-abs-second | `gem-calendar` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-calendar-rel-day` | GEM calendar-rel-day | `gem-calendar` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-calendar-rel-hour` | GEM calendar-rel-hour | `gem-calendar` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-calendar-rel-minute` | GEM calendar-rel-minute | `gem-calendar` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-calendar-rel-second` | GEM calendar-rel-second | `gem-calendar` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-numeric-p6` | GEM numeric-p6 | `gem-numeric` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-rsq-linear-128-128` | GEM rsq-linear-128-128 | `gem-rsq` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-rsq-linear-256` | GEM rsq-linear-256 | `gem-rsq` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-rsq-linear-64-64-64-64` | GEM rsq-linear-64-64-64-64 | `gem-rsq` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-rsq-linear-85-85-86` | GEM rsq-linear-85-85-86 | `gem-rsq` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-rsq-log-128-128` | GEM rsq-log-128-128 | `gem-rsq` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-rsq-log-256` | GEM rsq-log-256 | `gem-rsq` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-rsq-log-64-64-64-64` | GEM rsq-log-64-64-64-64 | `gem-rsq` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `gem-time-rsq-log-85-85-86` | GEM rsq-log-85-85-86 | `gem-rsq` | tpp | Approximation | yes | Liu et al., Temporal Tokenization Strategies (GEM 2026) | pure causal generation and the paper's tokenizer logic, but the default EventFM backbone/compute budget differs from Llama-3.2-1B QLoRA unless overridden |
| `iftpp` | Intensity-Free TPP | `neural-tpp` | classification, tpp | Approximation | no | Shchur et al., Intensity-Free Learning of TPPs (2020) | direct log-normal-mixture conditional interval density using the shared GRU encoder |
| `language-tpp` | Language-TPP | `llm` | classification, tpp | Approximation | yes | Kong et al., Byte-token enhanced language models for TPP analysis (2026) | byte-tokenised log1p gaps as input, but time is decoded by the shared probabilistic head rather than autoregressive byte generation |
| `mambular` | Mambular (Mamba SSM) | `state-space` | classification, tpp | Approximation | no | Thielmann et al., Mambular: a sequential model for tabular deep learning (2024) | selective state-space history mixer under the shared EventFM event contract rather than the library's tabular pipeline |
| `markov` | Marked Markov chain | `classic` | tpp | Implemented | no | non-neural control | no paper to be faithful to; this is a marked temporal control |
| `mlem` | MLEM | `hybrid` | classification, tpp | Approximation | no | Rubachev et al., MLEM (2024) | sequential local generative and global contrastive stages under one encoder rather than the reference dual-modality architecture |
| `mm-tpp` | MM-TPP | `llm` | classification, tpp | Approximation | yes | Li et al., Long-range modeling of multimodal event sequences (2026) | byte-time input plus repository-specific temporal-similarity run compression; no visual modality |
| `motor` | MOTOR-style TTE | `time-to-event` | classification, tpp | Approximation | no | Steinberg et al., MOTOR (2023) | code-specific exponential time-to-event pretraining on banking events; the original medical ontology, censoring horizon, and scale are unavailable |
| `nhp` | Neural Hawkes Process | `neural-tpp` | classification, tpp | Approximation | no | Mei & Eisner, The Neural Hawkes Process (2017) | continuous-time decayed GRU state with a piecewise exponential intensity; it does not reproduce the original continuous-time LSTM cell parameter-for-parameter |
| `nppr` | NPPR | `generative` | classification, tpp | Approximation | no | Zuo et al., Towards a Foundation Purchasing Model (2024) | combines past masked reconstruction and future event prediction under the shared EventFM decoder, without the original purchasing corpus or scale |
| `ntpp-gru` | Neural TPP (GRU) | `neural-tpp` | classification, tpp | Approximation | no | Shchur et al., Neural temporal point processes: a review (2021) | GRU history with a log-normal-mixture gap head; not the continuous-time LSTM/intensity formulation of NHP |
| `nvidia-tfm` | NVIDIA TFM blueprint | `tabular-transformer` | classification, tpp | Approximation | no | NVIDIA Transaction Foundation Model blueprint (decoder-only, causal LM) | controlled implementation of the public blueprint, not the industrial pretrained checkpoint |
| `ora` | ORA-style marked TTE | `time-to-event` | classification, tpp | Approximation | no | One Loss to Rule Them All: Marked Time-to-Event (2026) | joint next-time and bucketed feature-value factorization under the EventFM schema, rather than the reference EHR value distributions |
| `pragma` | PRAGMA | `hierarchical` | classification, tpp | Approximation | no | Ostroukhov et al., PRAGMA: Revolut foundation model (2026) | supervised PRAGMA-style hierarchy without foundation-scale pretraining |
| `pragma-mlm` | PRAGMA + masked pretraining | `hierarchical` | classification, tpp | Approximation | no | Ostroukhov et al. (2026), full recipe | masked-value pretraining on the downstream training split, not a separate large unlabelled corpus |
| `sahp` | Self-Attentive Hawkes Process | `neural-tpp` | classification, tpp | Approximation | no | Zhang et al., Self-Attentive Hawkes Process (2020) | self-attentive continuous-time encoder with a piecewise exponential decoder under the shared EventFM event embedding |
| `sohet` | SOHET | `heterogeneous-event` | classification, tpp | Approximation | no | SOHET: Sequence of Heterogeneous Events Transformer (2026) | type-conditioned feature gating approximates event-specific FT-Transformer encoders within the shared parameter budget |
| `supervised-gru` | Supervised GRU | `supervised` | classification, tpp | Implemented | no | standard supervised recurrent baseline | controlled supervised baseline with no paper-specific pretraining |
| `tabbert` | TabFormer (TabBERT) | `tabular-transformer` | classification, tpp | Approximation | no | Padhi et al., Tabular transformers for modeling multivariate time series (2021) | field-pooled events with masked-field pretraining at benchmark scale |
| `tabgpt` | TabFormer (TabGPT) | `tabular-transformer` | classification, tpp | Approximation | no | Padhi et al. (2021), generative variant | field-pooled events with a causal decoder at benchmark scale |
| `thp` | Transformer Hawkes | `neural-tpp` | classification, tpp | Approximation | no | Shchur et al. (2021), attention-based intensity family | causal attention over Fourier time features with a shared intensity-free decoder; not a faithful THP/SAHP intensity implementation |
| `time-calendar-fourier` | Calendar Fourier | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-continuous-rope` | Continuous-time RoPE | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-discrete-buckets` | Discrete time buckets | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-functional` | Functional time | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-gap-age` | Gap + cumulative time | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-gap-age-calendar` | Gap + age + calendar | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-log-gap` | Log inter-event gap | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-no-time` | No time | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-ordinal` | Ordinal position | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-raw-gap` | Raw inter-event gap | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `time-time2vec` | Time2Vec | `time-ablation` | classification, tpp | Implemented | no | EventFM controlled temporal ablation | controlled time-representation condition, not a paper reproduction |
| `tpp-llm` | TPP-LLM | `llm` | classification, tpp | Approximation | yes | Liu & Quan, TPP-LLM (2024) | text history, continuous Fourier side channel and LoRA on a 135M base model; far below the paper's model scale |
| `transaction-mlm` | Transaction MLM | `reconstruction` | classification, tpp | Implemented | no | generic masked event/field modeling baseline | generic controlled masked-transaction baseline |

## Datasets

| Key | Dataset | Supported tasks | Split | Source | Entity | Mark | Target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `banksim` | BankSim | classification, tpp | `hash` | github:atavci/fraud-detection-on-banksim-data | customer | merchant category (15 types) | fraud occurs in the held-out tail of the history |
| `paysim` | PaySim | classification, tpp | `hash` | hf:theman10/paysim | destination account | transfer type (5 types) | fraud occurs in the held-out tail of the history |
| `ibm_aml` | IBM AML (HI-Small) | classification, tpp | `hash` | hf:OsamaMIT/IBM-AML-HI-Small | receiving account | payment format (7 types) | laundering occurs in the held-out tail of the history |
| `mbd_mini` | MBD-mini | classification, tpp | `hash` | hf:ai-lab/MBD-mini | bank client | transaction event type | product-propensity target shipped with the benchmark |
| `mbd` | MBD (full) | classification, tpp | `hash` | hf:ai-lab/MBD | bank client | transaction event type | product-propensity target shipped with the benchmark |
| `synthea` | Synthea EHR | classification, tpp | `hash` | hf:richardyoung/synthea-575k-patients | patient | SNOMED condition description | major adverse cardiovascular or renal event in the held-out tail |
| `amazon_beauty` | Amazon Beauty 2014 | classification, tpp | `hash` | hf:milistu/Amazon_Beauty_2014 | reviewer | item level-2 product category | negative review (rating <= 2) in the held-out tail |
| `synthetic` | Synthetic | classification, tpp | `hash` | eventfm.data.synthetic | synthetic user | type_<idx> (50 types) | generator-assigned binary label |
| `stackoverflow` | Stack Overflow | tpp | `hash` | hf:tppllm/stack-overflow-description | user sequence | badge text (25 types) | next event type and time only |
| `chicago_crime` | Chicago Crime | tpp | `hash` | hf:tppllm/chicago-crime-description | spatial sequence | crime event text (20 types) | next event type and time only |
| `nyc_taxi` | NYC Taxi | tpp | `hash` | hf:tppllm/nyc-taxi-description | taxi trajectory sequence | taxi event text (8 types) | next event type and time only |
| `us_earthquake` | US Earthquake | tpp | `hash` | hf:tppllm/us-earthquake-description | earthquake sequence | earthquake event text (3 types) | next event type and time only |
| `amazon_review` | Amazon Review | tpp | `hash` | hf:tppllm/amazon-review-description | reviewer sequence | review category text (18 types) | next event type and time only |
| `banksim_chrono` | BankSim (chronological) | classification, tpp | `chronological` | github:atavci/fraud-detection-on-banksim-data | customer | merchant category (15 types) | fraud occurs in the held-out tail of the history |
| `paysim_chrono` | PaySim (chronological) | classification, tpp | `chronological` | hf:theman10/paysim | destination account | transfer type (5 types) | fraud occurs in the held-out tail of the history |
| `ibm_aml_chrono` | IBM AML (HI-Small) (chronological) | classification, tpp | `chronological` | hf:OsamaMIT/IBM-AML-HI-Small | receiving account | payment format (7 types) | laundering occurs in the held-out tail of the history |
| `mbd_mini_chrono` | MBD-mini (chronological) | classification, tpp | `chronological` | hf:ai-lab/MBD-mini | bank client | transaction event type | product-propensity target shipped with the benchmark |

## Experiment suites

Cell counts include task-support filtering and all seeds, regimes, and variants.
They describe scheduled runs; they do not imply that the cells have completed.

| Suite | Cells | Datasets | Tasks | Methods | Sizes | Seeds | Regimes | Variants | Purpose |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |
| `adaptation-regimes` | 1080 | 4 | classification, tpp | 9 | 2048 | 13, 29, 43, 71, 101 | frozen, peft, full | default | Frozen probes, bottleneck-adapter PEFT, and full tuning after pretraining. |
| `chronological-robustness` | 300 | 4 | classification, tpp | 8 | 2048 | 13, 29, 43, 71, 101 | full | default | Headline methods on whole-entity splits ordered by final observed timestamp. |
| `context-length-ablation` | 480 | 4 | classification, tpp | 4 | 2048 | 13, 29, 43, 71, 101 | full | events-32, events-64, events-128 | Short, medium, and long event contexts with truncation and efficiency metrics. |
| `core-reproduction` | 1872 | 4 | classification, tpp | 14 | 64, 128, 256, 512, 1024, 2048 | 13, 29, 43 | full | default | Original 624-cell architecture-screening grid, repeated with three seeds. |
| `cross-schema-transfer` | 2880 | 4 | classification, tpp | 4 | 64, 256, 2048 | 13, 29, 43, 71, 101 | frozen, peft, full | in-domain, leave-one-out | In-domain versus leave-one-banking-dataset-out pretraining at fixed target labels. |
| `gem-context-budget` | 480 | 4 | tpp | 12 | 2048 | 13, 29, 43, 71, 101 | full | equal-events, equal-tokens | Equal-event versus equal-token prompt budgets for the main GEM tokenizers. |
| `gem-event-text` | 480 | 4 | tpp | 12 | 2048 | 13, 29, 43, 71, 101 | full | descriptive, anonymous | Descriptive event names versus anonymized type IDs for language-prior analysis. |
| `gem-full-banking` | 420 | 4 | tpp | 21 | 2048 | 13, 29, 43, 71, 101 | full | default | All calendar resolutions and RSQ level allocations on banking datasets. |
| `gem-main-banking` | 260 | 4 | tpp | 13 | 2048 | 13, 29, 43, 71, 101 | full | default | The GEM 2026 main tokenizer table on the four banking datasets. |
| `gem-model-scale` | 480 | 4 | tpp | 12 | 2048 | 13, 29, 43, 71, 101 | full | smollm-135m, smollm-360m | Tokenizer ranking on controlled small and larger causal-LM backbones. |
| `gem-reference-datasets` | 325 | 5 | tpp | 13 | 2048 | 13, 29, 43, 71, 101 | full | default | GEM tokenizer reproduction on Stack Overflow, crime, taxi, earthquake, and reviews. |
| `gem-template-time-type` | 120 | 2 | tpp | 12 | 2048 | 13, 29, 43, 71, 101 | full | default | Prompt-order ablation with time tokens before event type tokens. |
| `labeled-adaptation-scaling` | 576 | 4 | classification, tpp | 4 | 64, 128, 256, 512, 1024, 2048 | 13, 29, 43 | full | default | Nested target-label sizes after a fixed-size in-domain self-supervised stage. |
| `model-scale-ablation` | 360 | 4 | classification, tpp | 3 | 2048 | 13, 29, 43, 71, 101 | full | small, base, large | Small, base, and larger shared backbones under fixed data and tasks. |
| `objective-ablation` | 1584 | 4 | classification, tpp | 11 | 64, 128, 256, 512, 1024, 2048 | 13, 29, 43 | full | default | Single and hybrid representation-learning objectives under shared budgets. |
| `objective-factorial` | 360 | 4 | classification, tpp | 1 | 2048 | 13, 29, 43, 71, 101 | full | masked, next, marked-tte, contrastive, masked-next, next-tte, masked-contrastive, next-contrastive, full | Single, pairwise, and full EventFM objectives under the same encoder and budget. |
| `paper-headline` | 580 | 4 | classification, tpp | 15 | 2048 | 13, 29, 43, 71, 101 | full | default | Compute-manageable five-seed suite spanning every central method family. |
| `pooling-ablation` | 180 | 4 | classification | 3 | 2048 | 13, 29, 43, 71, 101 | full | last, mean, max | Last-event, mean, and max customer readouts under otherwise fixed objectives. |
| `scale-datasets` | 2496 | 3 | classification, tpp | 14 | mbd: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 30665; synthea: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 93305; amazon_beauty: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 34023 | 13, 29, 43 | full | default | Sample-scaling curves from 64 sequences to the whole training pool on full MBD, Synthea EHR and Amazon Beauty 2014. |
| `scale-datasets-baselines` | 864 | 3 | classification, tpp | 5 | mbd: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 30665; synthea: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 93305; amazon_beauty: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 34023 | 13, 29, 43 | full | default | The strong supervised baselines that the core zoo does not already cover, over the same large-dataset scaling grid. |
| `strong-baselines` | 936 | 4 | classification, tpp | 8 | 64, 128, 256, 512, 1024, 2048 | 13, 29, 43 | full | default | Non-neural, supervised, reconstructive, autoregressive, and contrastive controls. |
| `temporal-models` | 720 | 4 | tpp | 10 | 64, 128, 256, 512, 1024, 2048 | 13, 29, 43 | full | default | Recurrent, attention, intensity-free, convolutional, and LLM temporal models. |
| `time-representation-ablation` | 1584 | 4 | classification, tpp | 11 | 64, 128, 256, 512, 1024, 2048 | 13, 29, 43 | full | default | All continuous, calendar, positional, bucket, and rotary time conditions. |
| `unlabeled-pretraining-scaling` | 384 | 4 | classification, tpp | 4 | 256 | 13, 29, 43 | full | pretrain-64, pretrain-256, pretrain-1024, pretrain-2048 | Source-pretraining scale at a fixed target adaptation size. |

## Interpretation boundary

The repository now covers every central comparison in the research plan
with a runnable control or a disclosed approximation, including all GEM
tokenizers. It does not make every external system a faithful reproduction.
Faithful status requires reference-recipe parity, reference-data parity where
available, validation against published numbers, and completed multi-seed runs.
Related-work-only industrial systems remain outside the runnable manifest.
