# PRAGMA: paper vs. repo (short version)

Paper: [arXiv:2604.08649v1](https://arxiv.org/html/2604.08649v1). Repo: `pragma` / `pragma-mlm` at `5b86137`.
Full details per ID: [README](README.md).

**Bottom line:** every PRAGMA result here comes from a ~1 M-param, post-norm,
no-RoPE, **event-only** model, fully fine-tuned after at most 3 epochs of
in-domain MLM.

## Data ([01](01_data.md))

| ID | Paper | Repo |
| --- | --- | --- |
| D1 | Separate 207 B-token corpus, 26 M users | MLM on the downstream training subset |
| D2 | Multi-source events, ~60 keys | One source, 2–6 keys per dataset |
| D3 | Profile at the evaluation point | BankSim/Synthea only, taken from the first row, never used |
| D4 | Life-long milestone events | None |
| D5 | History up to an evaluation point | History minus a 25 % tail, or up to a fixed cutoff |
| D6 | 25 months of real timestamps | Native spans; hour is synthetic on 4/7 datasets |
| D7 | No filtering or pruning | ≥8 events, top-64 categories, negative downsampling, entity caps |
| D8 | Fixed task folds | Hash/chrono entity splits, nested subsamples |

## Tokenisation ([02](02_tokenisation.md))

| ID | Paper | Repo |
| --- | --- | --- |
| T1 | ~60 key tokens | 2–6 key tokens |
| T2 | Percentile buckets + zero bucket | 16 quantile buckets, no zero bucket (Hydra path: raw strings) |
| T3 | BPE subwords for text fields | No text path; top-64 + `other`; always 1 token per field |
| T4 | ~28 k global vocab | Hundreds of tokens, rebuilt per cell |
| T5 | `8·ln(1+t/8)` | `log1p(t/3600)` |
| T6 | Calendar sin/cos + MLP | Same (UTC) ✓ |

## Architecture ([03](03_architecture.md))

| ID | Paper | Repo |
| --- | --- | --- |
| A1 | Pre-norm | Post-norm `BertEncoder` + embedding LayerNorm |
| A2 | Sinusoidal position *within* a field | Learned position over field order |
| A3 | Profile encoder with RoPE | Built but never runs (constant `[USR]`); no time input |
| A4 | Event encoder | Same ✓ (padded, not packed) |
| A5 | RoPE over log-seconds | Additive Fourier MLP; no RoPE anywhere |
| A6 | History over `[z_a : z_e]` | Same ✓ |
| A7 | 10 M / 100 M / 1 B | ~0.7–1.0 M; event depth = history depth; FFN 2× instead of 4× |

## Pretraining and infrastructure ([04](04_pretraining_and_infrastructure.md))

| ID | Paper | Repo |
| --- | --- | --- |
| P1 | Days to weeks on 16–32 H100s | 3 epochs on CPU; no checkpoint selection |
| P2 | 3d → d MLM head, tied logits | Same ✓ (+ GELU, LayerNorm, bias) |
| P3 | Label smoothing | 0.0 on benchmark (0.05 on Hydra) |
| P4 | Masking 15 / 10 / 10 % | Same ✓; key masking per user (≈31 % masked overall) |
| P5 | Muon + AdamW, bf16 | AdamW, fp32 |
| P6 | H100s | 8 CPU cores |
| P7 | LMDB + Parquet shards | JSONL in memory |
| P8 | Token-budget batching | Fixed batch of 32, padded |
| P9 | Varlen FlashAttention | Padded dense attention |
| P10 | ≤6 500 events, ≤24 tok/event, ≤200 profile tok | ≤64 events, ≤6 tok/event, no profile |

## Adaptation and evaluation ([05](05_adaptation_and_evaluation.md))

| ID | Paper | Repo |
| --- | --- | --- |
| E1 | LoRA r=8, α=8 | Full fine-tuning; `peft` regime crashes |
| E2 | Scaled L-BFGS probe on `[USR]` / `[EVT]` / both | None |
| E3 | Classification head | Same ✓; no multilabel |
| E4 | LoRA vs scratch | Full FT after in-domain MLM vs scratch |
| E5 | 6 banking tasks + AML + uplift | One binary label per dataset + TPP |
| E6 | – | Extra TPP head (MSE on log Δt) |
| E7 | Relative to internal baselines | Absolute metrics + scaling fits |
| E8 | Scale, LoRA, profile, uplift, Nemotron, AML studies | Mostly not run; profile ablation impossible |

## Repo bugs ([06](06_implementation_issues.md))

| ID | Issue |
| --- | --- |
| B1 | Profile tensors are never built → event-only model; 132 k dead params counted |
| B2 | Fourier frequencies span [1, 8760] but input spans [0, 10.5] → 12/16 wasted |
| B3 | `peft` regime raises for PRAGMA |
| B4 | `Pragma.pretrain` ignores pretrain pool and epoch overrides → fake "transfer" |
| B5 | Label smoothing not wired into benchmark path |
| B6 | TPP head ≠ shared log-normal mixture; `time_nll` not comparable |
| B7 | Registry `divergence` strings understate the gap |

## Already matches

Key–value–time split with a shared embedding table; `event_type` as a
key–value pair; three-block hierarchy; calendar MLP added to `[EVT]`; MLM head
inputs; masking rates; GELU, dropout 0.1, bidirectional attention, most recent
events kept.

## Fix order

B1 → A1 (pre-norm) → A5/B2 (RoPE) → B3/E1 (LoRA) + E2 (probe) → B4 → A2/T2/B5.
Scale gaps (A7, P1, D1, D2, E5) stay open.
