# 04 — Pre-training and training infrastructure (paper §2.3.5, §2.4)

[← index](README.md)

## Pre-training

### P1. Pre-training budget

**Kind:** Scale · **Impact:** High

| | Paper | This repo (`pragma-mlm`) |
| --- | --- | --- |
| Data | 207 B tokens, 24 B events | The downstream training subset: 64 to 93 305 sequences × ≤64 events × ≤6 tokens |
| Duration | PRAGMA-S ≈2 days on 16 H100s; M and L ≈2 weeks on 16 and 32 H100s | **3 epochs**, fixed as a class attribute ([`pragma.py:111`](../../../eventfm/methods/pragma.py#L111)) |
| Stopping and checkpoint choice | Run to convergence; probes are used to pick checkpoints (§3.1.1) | No evaluation set, no checkpoint selection; the last step is used |
| Fine-tuning vs pretraining compute | LoRA fine-tuning takes about 1/8 of pretraining wall-clock time | **Fine-tuning runs 8 epochs against 3 for pretraining**, so the ratio is inverted |

`context.extra["pretrain_epochs"]` and `extra["pretrain_max_steps"]` are
ignored for PRAGMA. See
[B4](06_implementation_issues.md#b4-pragmapretrain-ignores-the-pretraining-pool-and-epoch-overrides).

### P2. MLM head

**Kind:** Match (minor additions) · **Impact:** Low

**Paper:** For each masked token, the head takes
`[ẑ_e(token) ; z_h,i ([EVT] of its event) ; z_h,0 ([USR])]` (3d), "projected back
to d dimensions and matched against the embedding table".

**Repo** ([`modeling_pragma.py:267-310`](../../../eventfm/models/modeling_pragma.py#L267-L310)):
the same three vectors in the same order. The projection is
`Linear(3d→d) → GELU → LayerNorm`, the BERT prediction-head transform, followed
by a decoder tied to the token embeddings and a separate output bias. The paper
does not mention the GELU, the LayerNorm or the bias. Other points:

* Logits cover the **whole vocabulary**: keys, special tokens, and the values
  of every other field. The paper does not say whether it restricts candidates
  to the masked key's value set.
* Because the profile branch never runs ([A3](03_architecture.md#a3-profile-state-encoder)),
  `z_h,0` is a summary of the events only. It contains no profile context.

### P3. Loss

**Kind:** Missing (benchmark path) · **Impact:** Low

**Paper:** Cross-entropy with label smoothing. The smoothing value is
unspecified.

**Repo:** `F.cross_entropy(..., label_smoothing=config.label_smoothing)`
([`modeling_pragma.py:313-318`](../../../eventfm/models/modeling_pragma.py#L313-L318)).

* The **benchmark path never sets the value**, so it stays at the default 0.0
  ([`configuration_pragma.py:33`](../../../eventfm/models/configuration_pragma.py#L33);
  `Pragma.build_config` in [`pragma.py:40-63`](../../../eventfm/methods/pragma.py#L40-L63)).
* The Hydra path uses 0.05
  ([`configs/task/pretrain.yaml`](../../../configs/task/pretrain.yaml)).

See [B5](06_implementation_issues.md#b5-label-smoothing-is-not-wired-into-the-benchmark-path).

### P4. Masking

**Kind:** Match, with one ambiguity · **Impact:** Low

Implementation: [`collator.py:127-166`](../../../eventfm/data/collator.py#L127-L166).

| Component | Paper | Repo |
| --- | --- | --- |
| Token-level | 15 % of event tokens | 15 % of event value tokens ✓ |
| Event-level | 10 % of events, the whole event is reconstructed | 10 % of events; all value tokens masked. Keys, `[EVT]`, time and calendar stay visible ✓ |
| Key-level | 10 %; "all values of the selected keys are masked" | For **each user**, each distinct key is chosen with p = 0.10, and **every occurrence of that key in the whole history** (up to 64 events) is masked |
| How sources combine | unspecified | Union. The expected selection rate is ≈ 1 − 0.85·0.9·0.9 ≈ **31 %** of value tokens |
| `[UNK]` replacement | "a small fraction" of selected positions; excluded from the loss | 10 % of selected positions (`unk_probability = 0.10`); excluded from the loss ✓ |
| Random-token replacement (BERT 80/10/10) | not mentioned | not used ✓ |
| What gets masked | Event tokens only | Event values only; keys and profile are never masked ✓ |
| Empty-selection fallback | – | If nothing in the batch is selected, one random token is forced |

**The ambiguity.** The paper does not say whether key masking is applied per
event or per record. In this repo every field has exactly one token
([T3](02_tokenisation.md#t3-categorical-vs-textual-values)), so a per-event
key mask would be the same thing as token masking. The per-record reading is
therefore the only one that adds anything new. It is also a strong
corruption: with 2 to 6 keys per dataset, about 10 % of users lose, for
example, **all** their event types or **all** their amounts at once.

### P5. Optimiser and precision

**Kind:** Substitution · **Impact:** Medium

| | Paper | Repo |
| --- | --- | --- |
| Optimiser | **Muon combined with AdamW** | AdamW only (Hugging Face `Trainer` default) |
| Learning rate, schedule, batch | unspecified | lr 3e-4, weight decay 0.01, 6 % warmup then linear decay, batch 32 ([`base.py:27-49`](../../../eventfm/methods/base.py#L27-L49)) |
| Precision | bf16 mixed precision | **fp32**: PRAGMA cells run only on the CPU array (`--cpu`, [`run_experiment_cpu_array.sbatch:52-66`](../../../slurm/run_experiment_cpu_array.sbatch#L52-L66)) |

Pretraining and fine-tuning use the same optimiser settings.

### P6. Hardware

**Kind:** Scale

The paper uses 16 H100s (S, M) or 32 H100s (L). In the repo, PRAGMA is not
marked `requires_gpu`, so each cell runs on 8 CPU cores with 48 GB of RAM. The
GPU array runs only `--only-gpu-only` methods (the LLM entries).

## Training infrastructure

These differences affect throughput, not what the model computes. The one
exception is [P10](#p10-truncation).

### P7. Data storage

**Kind:** Scale · **Impact:** Low

**Paper:** An LMDB user index (tokenised profile state plus per-user token
statistics), and Parquet event shards **partitioned by event count**, so that
each file holds users with the same number of events.

**Repo:** Each split is a JSONL file, loaded entirely into memory as Python
`EventSequence` objects (`JsonlEventDataset`,
[`dataset.py:12-24`](../../../eventfm/data/dataset.py#L12-L24)). Tokenisation
happens inside the collator on every step.

### P8. Batching

**Kind:** Substitution · **Impact:** Low

**Paper:** Records from one length shard are packed greedily up to a **fixed
token budget**, so the history axis is never padded and the number of records
per batch changes with history length.

**Repo:** A fixed 32 records per batch, padded on the right to the longest
history in the batch
([`collator.py:57-83`](../../../eventfm/data/collator.py#L57-L83)). Besides the
wasted compute, the batch composition differs: the paper puts many short users
or a few long users in a batch, while the repo always uses 32 records.

### P9. Sequence packing

**Kind:** Substitution · **Impact:** Low (numerically equivalent)

**Paper:** All event tokens go into one flat buffer, processed with
variable-length FlashAttention, so there is no padding on either the event or
the token axis. This gives a 2 to 5× throughput gain.

**Repo:** A dense `(B·L, W+1)` tensor with additive attention masks, run
through a standalone `BertEncoder`. Padding events are encoded too
([A4](03_architecture.md#a4-event-encoder)).

### P10. Truncation

**Kind:** Scale · **Impact:** High (for long-range signal)

| Limit | Paper | Repo |
| --- | --- | --- |
| Tokens per event | ≤24 (affects 0.01 % of events) | `n_fields + 1`, i.e. 2 to 6 ([`runner.py:385`](../../../eventfm/benchmark/runner.py#L385)). No truncation in practice. Fields beyond the limit would be dropped in alphabetical order ([`tokenizer.py:256-257`](../../../eventfm/data/tokenizer.py#L256-L257)). The Hydra path uses 8 |
| Profile tokens | ≤200 | Profile not used |
| Events per user | ≤6 500, most recent kept | **≤64** at training time (`--max-events 64`), most recent kept ✓. Conversion already caps at 64 to 128 |
| Minimum history | ≥1 event | ≥8 events, and ≥2 observed after the label hold-out |

A context of 64 events is about 100× shorter than the paper's. That is exactly
the situation the life-long events ([D4](01_data.md#d4-life-long-events)) were
designed to compensate for, and they are missing here.
