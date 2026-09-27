# PRAGMA: the paper vs. this implementation

This folder lists every difference found between

* **the paper**: Ostroukhov et al., *PRAGMA: Revolut Foundation Model*,
  [arXiv:2604.08649v1](https://arxiv.org/html/2604.08649v1) (9 Apr 2026), and
* **this repo**: the `pragma` and `pragma-mlm` registry entries at commit
  `5b86137`, i.e. [`eventfm/methods/pragma.py`](../../../eventfm/methods/pragma.py),
  [`eventfm/models/modeling_pragma.py`](../../../eventfm/models/modeling_pragma.py),
  [`eventfm/data/tokenizer.py`](../../../eventfm/data/tokenizer.py),
  [`eventfm/data/collator.py`](../../../eventfm/data/collator.py) and the data
  conversion in [`eventfm/datasets/`](../../../eventfm/datasets/).

Two code paths reach the PRAGMA model:

| Path | Entry point | Used for |
| --- | --- | --- |
| **Benchmark** | `scripts/run_experiments.py` → `Pragma` / `PragmaMlm` | every reported result |
| **Legacy Hydra** | `scripts/train.py` + `configs/` | synthetic data only |

Everything below describes the benchmark path unless it says otherwise.

## How to read the entries

Each difference has an ID (used for cross-references), a **kind** and an
**impact**.

| Kind | Meaning |
| --- | --- |
| Scale | Follows from using public data and academic compute. Cannot be closed inside this repo. |
| Substitution | A paper component is replaced by a different mechanism. |
| Missing | A paper component is absent. |
| Bug | The code does not do what the repo says it does. Collected in [06](06_implementation_issues.md). |

| Impact | Meaning |
| --- | --- |
| High | Changes what the model is, or stops a paper conclusion from carrying over. |
| Medium | Likely to move the numbers. |
| Low | A detail, or an efficiency-only difference. |

Where the paper does not specify something, the entry says **unspecified**
rather than counting it as a difference.

## The seven differences that matter most

1. **The profile-state branch never runs** ([A3](03_architecture.md#a3-profile-state-encoder),
   [B1](06_implementation_issues.md#b1-profile-data-never-reaches-the-model)). No
   collator builds profile tensors, so the `[USR]` input is a constant learned
   vector and the profile encoder gets no gradient. Every PRAGMA result here
   therefore corresponds to the paper's **event-only ablation** (paper Table 6),
   not to the full model.
2. **No RoPE** ([A5](03_architecture.md#a5-time-encoding-in-the-history-encoder)).
   Time enters as an additive Fourier-feature MLP over `log1p(seconds/3600)`,
   not as rotary positions over `8·ln(1+t/8)`.
3. **Post-norm layers** ([A1](03_architecture.md#a1-layer-normalisation-placement)).
   The encoders are Hugging Face `BertEncoder` stacks, which are post-norm. The
   paper uses pre-norm.
4. **No foundation-scale pretraining** ([D1](01_data.md#d1-pre-training-corpus),
   [P1](04_pretraining_and_infrastructure.md#p1-pre-training-budget)).
   `pragma-mlm` runs 3 epochs of MLM on the same labelled training subset it
   then fine-tunes on. The paper pretrains on 207 B tokens from 26 M users.
5. **No LoRA and no linear probe** ([E1](05_adaptation_and_evaluation.md#e1-adaptation-mode),
   [E2](05_adaptation_and_evaluation.md#e2-embedding-probe)). All PRAGMA
   numbers come from full fine-tuning. The paper's headline results use LoRA
   on PRAGMA-L.
6. **About 1 M parameters instead of 10 M to 1 B** ([A7](03_architecture.md#a7-model-size-and-shape)),
   with histories capped at 64 events instead of 6 500
   ([P10](04_pretraining_and_infrastructure.md#p10-truncation)).
7. **No text values, so there is only one token per field** ([T3](02_tokenisation.md#t3-categorical-vs-textual-values),
   [A2](03_architecture.md#a2-within-field-positional-embedding)). Within-field
   positions, key replication and subword values never occur. The position
   embedding that does exist indexes *fields*, which the paper explicitly says
   it does not do.

## Full index

| ID | Area | Paper | This repo | Kind | Impact |
| --- | --- | --- | --- | --- | --- |
| [D1](01_data.md#d1-pre-training-corpus) | Data | 26 M users, 24 B events, 207 B tokens; separate pretraining corpus | Pretrains on the downstream training subset (64 to 93 305 sequences) | Scale | High |
| [D2](01_data.md#d2-event-sources-and-schemas) | Data | Multi-source events (transactions, app, trading, comms), ~60 keys | One source per dataset, 2 to 6 keys | Scale | High |
| [D3](01_data.md#d3-profile-state) | Data | Profile key–value pairs at the evaluation point | Only BankSim and Synthea have profiles, taken from the first event row, and they are never used | Missing | High |
| [D4](01_data.md#d4-life-long-events) | Data | Timestamped first-occurrence milestones | None | Missing | Medium |
| [D5](01_data.md#d5-record-and-evaluation-point) | Data | Record = history up to an evaluation point | Record = history minus a held-out 25 % tail, or up to a fixed cutoff | Substitution | Low |
| [D6](01_data.md#d6-time-range-and-timestamp-quality) | Data | 25 months (2023–2025) of real timestamps | Per-dataset spans; intra-day time synthetic on 4 of 7 datasets | Scale | Medium |
| [D7](01_data.md#d7-filtering-and-pre-processing) | Data | No filtering or vocabulary pruning | Min-length filter, top-64 category pruning, negative downsampling, entity caps | Substitution | Medium |
| [D8](01_data.md#d8-splits) | Data | Pre-defined task folds | Hash or chronological entity splits and nested subsamples | Scale | Low |
| [T1](02_tokenisation.md#t1-keys) | Tokens | ~60 key tokens shared by events and profile | 2 to 6 key tokens per dataset | Scale | Low |
| [T2](02_tokenisation.md#t2-numerical-values) | Tokens | Percentile buckets plus a dedicated zero bucket | 16 quantile buckets, no zero bucket; the Hydra path does not bucket at all | Substitution | Low |
| [T3](02_tokenisation.md#t3-categorical-vs-textual-values) | Tokens | Cardinality threshold; BPE subwords for text fields | No text path; top-64 levels plus `other` | Missing | Medium |
| [T4](02_tokenisation.md#t4-value-vocabulary) | Tokens | ~28 k value tokens, one global vocabulary | Hundreds of tokens, rebuilt for every cell from the training subset | Scale | Low |
| [T5](02_tokenisation.md#t5-elapsed-time-transform) | Tokens | `8·ln(1+t/8)` of seconds to the last event | `log1p(t/3600)` of the same quantity | Substitution | Medium |
| [T6](02_tokenisation.md#t6-calendar-features) | Tokens | Hour, weekday, day of month; sin/cos; 2-layer MLP | Same, in UTC | Match (minor) | Low |
| [A1](03_architecture.md#a1-layer-normalisation-placement) | Model | Pre-norm | Post-norm (`BertEncoder`), plus an embedding LayerNorm | Substitution | Medium |
| [A2](03_architecture.md#a2-within-field-positional-embedding) | Model | Static sinusoidal, indexes values *within* a field | Learned, indexes fields *within* an event | Substitution | Low |
| [A3](03_architecture.md#a3-profile-state-encoder) | Model | Profile encoder with RoPE over life-long-event time | Built but never executed; no time input | Missing / Bug | High |
| [A4](03_architecture.md#a4-event-encoder) | Model | Per-event encoder, `[EVT]` + calendar MLP | Same structure; padded rather than packed | Match | Low |
| [A5](03_architecture.md#a5-time-encoding-in-the-history-encoder) | Model | RoPE over log-seconds | Additive Fourier MLP; no RoPE anywhere | Substitution | High |
| [A6](03_architecture.md#a6-history-encoder-inputs-and-outputs) | Model | `[z_a : z_e]` → `z_h` | Same | Match | – |
| [A7](03_architecture.md#a7-model-size-and-shape) | Model | S / M / L = 10 M / 100 M / 1 B | ~0.7 to 1.0 M; a PRAGMA-S-shaped config exists but only on the Hydra path | Scale | High |
| [P1](04_pretraining_and_infrastructure.md#p1-pre-training-budget) | Pretrain | Days to weeks on 16 to 32 H100s | 3 epochs on the downstream subset, on CPU | Scale | High |
| [P2](04_pretraining_and_infrastructure.md#p2-mlm-head) | Pretrain | Concatenate three vectors (3d) → project to d → tied logits | Same, plus GELU, LayerNorm and an output bias | Match (minor) | Low |
| [P3](04_pretraining_and_infrastructure.md#p3-loss) | Pretrain | Cross-entropy with label smoothing | No label smoothing on the benchmark path | Missing | Low |
| [P4](04_pretraining_and_infrastructure.md#p4-masking) | Pretrain | Token 15 %, event 10 %, key 10 %, some `[UNK]` | Same rates; key masking applies per *user*; `[UNK]` = 10 % | Match (ambiguous) | Low |
| [P5](04_pretraining_and_infrastructure.md#p5-optimiser-and-precision) | Pretrain | Muon + AdamW, bf16 | AdamW only; fp32 on CPU | Substitution | Medium |
| [P6](04_pretraining_and_infrastructure.md#p6-hardware) | Pretrain | 16 to 32 H100s | 8 CPU cores per cell (GPU lane not used for PRAGMA) | Scale | – |
| [P7](04_pretraining_and_infrastructure.md#p7-data-storage) | Infra | LMDB user index + Parquet shards by event count | JSONL fully loaded into memory | Scale | Low |
| [P8](04_pretraining_and_infrastructure.md#p8-batching) | Infra | Token-budget dynamic batching within length shards | Fixed batch of 32, padded | Substitution | Low |
| [P9](04_pretraining_and_infrastructure.md#p9-sequence-packing) | Infra | Varlen FlashAttention over packed events | Padded dense attention | Substitution | Low |
| [P10](04_pretraining_and_infrastructure.md#p10-truncation) | Infra | ≤24 tokens/event, ≤200 profile tokens, ≤6 500 events | ≤6 tokens/event, no profile, ≤64 events | Scale | High |
| [E1](05_adaptation_and_evaluation.md#e1-adaptation-mode) | Eval | LoRA r=8, α=8 on QKV and MLP; Adam | Full fine-tuning; the `peft` regime crashes for PRAGMA | Missing | High |
| [E2](05_adaptation_and_evaluation.md#e2-embedding-probe) | Eval | Standard-scaled `[USR]` / last `[EVT]` / both → L-BFGS logistic regression | No probe; the extraction script exports `[USR]` only | Missing | Medium |
| [E3](05_adaptation_and_evaluation.md#e3-task-head) | Eval | Classification head (form unspecified) | Dropout + linear on `[USR]`; no multilabel | Match (minor) | Low |
| [E4](05_adaptation_and_evaluation.md#e4-scratch-vs-pretrained-comparison) | Eval | LoRA vs full training from scratch | Full fine-tuning after in-domain MLM vs from scratch | Substitution | Medium |
| [E5](05_adaptation_and_evaluation.md#e5-downstream-tasks) | Eval | Six banking tasks, plus AML and uplift | One binary label per dataset, plus a TPP task | Scale | High |
| [E6](05_adaptation_and_evaluation.md#e6-next-event-tpp-task-not-in-the-paper) | Eval | – | Next-event TPP head (MSE on log Δt) | Extra | – |
| [E7](05_adaptation_and_evaluation.md#e7-metrics-and-reporting) | Eval | Relative change vs internal baselines | Absolute metrics and scaling-law fits | Scale | Low |
| [E8](05_adaptation_and_evaluation.md#e8-paper-experiments-not-reproduced) | Eval | Scale, LoRA, profile, uplift, Nemotron and AML studies | Mostly not run for PRAGMA | Missing | Medium |
| [B1–B7](06_implementation_issues.md) | Bugs | – | Repo-internal defects found during the comparison | Bug | varies |

## What already matches the paper

These are listed so the gaps above are not mistaken for a wholesale mismatch:

* The key–value–time decomposition, with keys and values in **one shared
  embedding table** summed as `E(k) + E(v)`
  ([`modeling_pragma.py:191-195`](../../../eventfm/models/modeling_pragma.py#L191-L195)).
* The event type is modelled as an ordinary key–value pair (`key:event_type`).
* The three-block hierarchy: a per-event encoder with a prepended `[EVT]`, a
  profile encoder with a prepended `[USR]`, and a history encoder over
  `[z_a : z_e]` ([`modeling_pragma.py:197-247`](../../../eventfm/models/modeling_pragma.py#L197-L247)).
* Calendar features: hour, day of week and day of month as sin/cos, passed
  through a two-layer MLP and **added to the `[EVT]` output**, on events only
  ([`modeling_pragma.py:218`](../../../eventfm/models/modeling_pragma.py#L218)).
* The MLM head input is the concatenation of the event-encoder token output,
  the history output at that event's `[EVT]`, and the history output at
  `[USR]`, projected from 3d to d and scored against the tied embedding table
  ([`modeling_pragma.py:299-310`](../../../eventfm/models/modeling_pragma.py#L299-L310)).
* Masking rates of 15 % (token), 10 % (event) and 10 % (key), with `[UNK]`
  corruption excluded from the loss
  ([`collator.py:127-166`](../../../eventfm/data/collator.py#L127-L166)).
* GELU activations, dropout 0.1, bidirectional attention everywhere, and the
  most recent events kept when truncating.

## Files

| File | Covers | Paper sections |
| --- | --- | --- |
| [01_data.md](01_data.md) | Corpus, sources, profile state, life-long events, time range, filtering, splits | §2.1, §3.1.3 |
| [02_tokenisation.md](02_tokenisation.md) | Keys, values, vocabulary, temporal features | §2.2 |
| [03_architecture.md](03_architecture.md) | Embeddings, the three encoders, normalisation, time encoding, sizes | §2.3–2.3.4, Table 1 |
| [04_pretraining_and_infrastructure.md](04_pretraining_and_infrastructure.md) | Objective, MLM head, masking, optimiser, storage, batching, packing, truncation | §2.3.5, §2.4 |
| [05_adaptation_and_evaluation.md](05_adaptation_and_evaluation.md) | Probe, LoRA, heads, tasks, metrics, ablations | §3 |
| [06_implementation_issues.md](06_implementation_issues.md) | Repo defects found along the way, each with a proposed fix | – |
