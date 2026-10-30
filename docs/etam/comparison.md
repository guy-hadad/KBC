# ETAM vs strong baselines on full MBD

Generated 2026-10-30 08:24 by `make_comparison.py` from the campaign run directories (`/groups/bshapira_group/guyhada/campaigns/etam_mbd_phaseB_20261029`) and the benchmark's `../benchmark_scale/results.csv`. Do not edit by hand; rerun the script. Commentary is in [results.md](results.md); the protocol is in [methodology.md](methodology.md).

**How to read the tables.**

- Every test number uses the benchmark's own examples: 40,000 test histories at natural prevalence for classification, and the benchmark's 320,000 TPP prefixes (seed + 1 replay) for TPP. All models are trained on the same 30,665 labeled clients.
- Cells are mean ± sample sd over seeds 13, 29 and 43; `n` counts finished seeds.
- ETAM selects its checkpoint on validation (early stopping, patience 3); the baselines train a fixed 8 epochs with no selection. The *at final step* column is where ETAM stopped, which is by construction up to 3 non-improving epochs past its best: a pessimistic bound, **not** a like-for-like number. Like-for-like would need the baselines rerun with the same validation selection.
- **better** / worse means the gap exceeds the larger of the two arms' seed sd (the pre-registered rule); otherwise *tie*. With n = 3 this is a screening rule, not a test.
- Every pretrained arm starts from **one** pretraining run (seed 13), so its sd covers fine-tuning noise only. The frozen probe's AUC is identical across seeds (one checkpoint, deterministic logistic regression); its seeds vary only the TPP prefixes.
- Time RMSE and MAE are on log1p seconds: lower is better.
- Pretraining-stage numbers are on 10,000 **validation** users with the pretrained heads and no fine-tuning. Use them to compare pretraining runs with each other, not with test numbers.

## Best benchmark baseline per metric (MBD, n = 30,665)

| Task | Metric | Best benchmark method | Mean ± sd (n = 3) |
| --- | --- | --- | ---: |
| classification | AUC | PRAGMA + masked pretraining | 0.7309 ± 0.0017 |
| tpp | next-type acc. | PRAGMA + masked pretraining | 0.5860 ± 0.0013 |
| tpp | macro-F1 | Autoregressive transaction Transformer | 0.2067 ± 0.0045 |
| tpp | time RMSE | CoLES | 1.526 ± 0.003 |
| tpp | time MAE | PRAGMA | 1.149 ± 0.003 |

## Headline: one model, best on both tasks?

Pre-registered pass rule: an arm must beat the best baseline on AUC **and** on next-type accuracy by more than the seed spread.

| ETAM arm (one model for both tasks) | AUC | Δ AUC | next-type acc. | Δ acc. | n | both better? |
| --- | ---: | --- | ---: | --- | --- | --- |
| ETAM-M multi (M1, headline) | 0.7406 ± 0.0033 | +0.0097 **better** | 0.5976 ± 0.0015 | +0.0116 **better** | 3 | **yes** |
| ETAM-M frozen probe (M3) | 0.7432 ± 0.0000 | +0.0122 **better** | 0.5994 ± 0.0009 | +0.0134 **better** | 3 | **yes** |
| ETAM-M multi, HPO fine-tune (H2) | – | – | – | – | 0/3 (queued) | – |
| ETAM-M multi, HPO both stages (H1+H2) | – | – | – | – | 0/3 (queued) | – |
| ETAM-M multi (M4) | – | – | – | – | 0/3 (queued) | – |
| ETAM-M multi, scratch | 0.7304 ± 0.0044 | -0.0005 tie | 0.5814 ± 0.0048 | -0.0047 tie | 3 | no |
| ETAM Qwen3-0.6B + LoRA multi (L1) | – | – | – | – | 0/3 (queued) | – |

## Classification (test AUC)

ETAM arms, then the four baselines pretrained on the same unlabeled pool (B1), then the 8 best benchmark methods (the full table is in [`../benchmark_scale/results.md`](../benchmark_scale/results.md)). Arms with no classification output (`tpp` fine-tunes) are omitted.

| Model | Pretraining | AUC | AP | AUC at final step | n | Δ AUC vs PRAGMA + masked pretraining |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ETAM-M multi (M1, headline) | pool, 20 ep | 0.7406 ± 0.0033 | 0.0328 ± 0.0005 | 0.7400 ± 0.0033 | 3 | +0.0097 **better** |
| ETAM-M cls (M2) | pool, 20 ep | 0.7449 ± 0.0027 | 0.0339 ± 0.0006 | 0.7105 ± 0.0012 | 3 | +0.0139 **better** |
| ETAM-M frozen probe (M3) | pool, 20 ep | 0.7432 ± 0.0000 | 0.0324 ± 0.0000 | – | 3 | +0.0122 **better** |
| ETAM-M multi, HPO fine-tune (H2) | pool, 20 ep | – | – | – | 0/3 (queued) | – |
| ETAM-M multi, HPO both stages (H1+H2) | pool, 20 ep, HPO | – | – | – | 0/3 (queued) | – |
| ETAM-M multi (M4) | 30,665 labeled only | – | – | – | 0/3 (queued) | – |
| ETAM-M cls (M4) | 30,665 labeled only | – | – | – | 0/3 (queued) | – |
| ETAM-M multi, scratch | none | 0.7304 ± 0.0044 | 0.0304 ± 0.0010 | 0.7275 ± 0.0025 | 3 | -0.0005 tie |
| ETAM-M cls, scratch | none | 0.7318 ± 0.0033 | 0.0309 ± 0.0018 | 0.6993 ± 0.0129 | 3 | +0.0009 tie |
| ETAM Qwen3-0.6B + LoRA multi (L1) | pool, 2 ep | – | – | – | 0/3 (queued) | – |
| PRAGMA + masked pretraining (B1) | pool, 1 ep | 0.7295 ± 0.0048 | 0.0322 ± 0.0018 | 0.7295 ± 0.0048 | 3 | -0.0015 tie |
| Transaction MLM (B1) | pool, 1 ep | 0.7218 ± 0.0033 | 0.0292 ± 0.0008 | 0.7218 ± 0.0033 | 3 | -0.0091 worse |
| Autoregressive transaction Transformer (B1) | pool, 1 ep | 0.7149 ± 0.0046 | 0.0276 ± 0.0011 | 0.7149 ± 0.0046 | 3 | -0.0161 worse |
| CoLES (B1) | pool, 1 ep | 0.7086 ± 0.0023 | 0.0281 ± 0.0003 | 0.7086 ± 0.0023 | 3 | -0.0223 worse |
| PRAGMA + masked pretraining | per method | 0.7309 ± 0.0017 | 0.0305 ± 0.0004 | 0.7309 ± 0.0017 | 3 | reference |
| PRAGMA | per method | 0.7296 ± 0.0024 | 0.0302 ± 0.0008 | 0.7296 ± 0.0024 | 3 | -0.0013 tie |
| Supervised GRU | per method | 0.7253 ± 0.0014 | 0.0282 ± 0.0005 | 0.7253 ± 0.0014 | 3 | -0.0057 worse |
| Neural TPP (GRU) | per method | 0.7253 ± 0.0014 | 0.0282 ± 0.0005 | 0.7253 ± 0.0014 | 3 | -0.0057 worse |
| Transaction MLM | per method | 0.7198 ± 0.0010 | 0.0311 ± 0.0007 | 0.7198 ± 0.0010 | 3 | -0.0111 worse |
| Transformer Hawkes | per method | 0.7195 ± 0.0036 | 0.0267 ± 0.0005 | 0.7195 ± 0.0036 | 3 | -0.0114 worse |
| Engineered features + GBDT | per method | 0.7182 ± 0.0017 | 0.0292 ± 0.0006 | 0.7182 ± 0.0017 | 3 | -0.0128 worse |
| NVIDIA TFM blueprint | per method | 0.7170 ± 0.0081 | 0.0281 ± 0.0007 | 0.7170 ± 0.0081 | 3 | -0.0139 worse |

## TPP (test next event)

| Model | Pretraining | next-type acc. | macro-F1 | time RMSE | time MAE | next-type acc. at final step | n | Δ next-type acc. vs PRAGMA + masked pretraining |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETAM-M multi (M1, headline) | pool, 20 ep | 0.5976 ± 0.0015 | 0.2533 ± 0.0092 | 1.491 ± 0.002 | 1.122 ± 0.002 | 0.5984 ± 0.0007 | 3 | +0.0116 **better** |
| ETAM-M tpp (M2) | pool, 20 ep | 0.5981 ± 0.0002 | 0.2552 ± 0.0047 | 1.491 ± 0.003 | 1.122 ± 0.002 | 0.5979 ± 0.0003 | 3 | +0.0121 **better** |
| ETAM-M frozen probe (M3) | pool, 20 ep | 0.5994 ± 0.0009 | 0.2618 ± 0.0019 | 1.489 ± 0.002 | 1.116 ± 0.002 | – | 3 | +0.0134 **better** |
| ETAM-M multi, HPO fine-tune (H2) | pool, 20 ep | – | – | – | – | – | 0/3 (queued) | – |
| ETAM-M multi, HPO both stages (H1+H2) | pool, 20 ep, HPO | – | – | – | – | – | 0/3 (queued) | – |
| ETAM-M multi (M4) | 30,665 labeled only | – | – | – | – | – | 0/3 (queued) | – |
| ETAM-M tpp (M4) | 30,665 labeled only | – | – | – | – | – | 0/3 (queued) | – |
| ETAM-M multi, scratch | none | 0.5814 ± 0.0048 | 0.1845 ± 0.0030 | 1.528 ± 0.010 | 1.162 ± 0.021 | 0.5855 ± 0.0038 | 3 | -0.0047 tie |
| ETAM-M tpp, scratch | none | 0.5892 ± 0.0006 | 0.2055 ± 0.0040 | 1.511 ± 0.003 | 1.141 ± 0.002 | 0.5892 ± 0.0006 | 3 | +0.0032 **better** |
| ETAM Qwen3-0.6B + LoRA multi (L1) | pool, 2 ep | – | – | – | – | – | 0/3 (queued) | – |
| PRAGMA + masked pretraining (B1) | pool, 1 ep | 0.5864 ± 0.0007 | 0.2043 ± 0.0039 | 1.527 ± 0.002 | 1.151 ± 0.005 | 0.5864 ± 0.0007 | 3 | +0.0003 tie |
| Autoregressive transaction Transformer (B1) | pool, 1 ep | 0.5859 ± 0.0010 | 0.2172 ± 0.0021 | 1.526 ± 0.002 | 1.154 ± 0.003 | 0.5859 ± 0.0010 | 3 | -0.0002 tie |
| CoLES (B1) | pool, 1 ep | 0.5854 ± 0.0010 | 0.1948 ± 0.0019 | 1.526 ± 0.003 | 1.156 ± 0.002 | 0.5854 ± 0.0010 | 3 | -0.0007 tie |
| Transaction MLM (B1) | pool, 1 ep | 0.5852 ± 0.0005 | 0.2067 ± 0.0035 | 1.526 ± 0.002 | 1.151 ± 0.002 | 0.5852 ± 0.0005 | 3 | -0.0008 tie |
| PRAGMA + masked pretraining | per method | 0.5860 ± 0.0013 | 0.2008 ± 0.0024 | 1.528 ± 0.002 | 1.150 ± 0.002 | 0.5860 ± 0.0013 | 3 | reference |
| Transaction autoencoder | per method | 0.5857 ± 0.0012 | 0.1934 ± 0.0028 | 1.526 ± 0.003 | 1.156 ± 0.003 | 0.5857 ± 0.0012 | 3 | -0.0004 tie |
| CoLES | per method | 0.5856 ± 0.0012 | 0.1958 ± 0.0011 | 1.526 ± 0.003 | 1.156 ± 0.002 | 0.5856 ± 0.0012 | 3 | -0.0004 tie |
| Neural TPP (GRU) | per method | 0.5855 ± 0.0013 | 0.1951 ± 0.0030 | 1.526 ± 0.003 | 1.157 ± 0.002 | 0.5855 ± 0.0013 | 3 | -0.0005 tie |
| Supervised GRU | per method | 0.5855 ± 0.0013 | 0.1951 ± 0.0030 | 1.526 ± 0.003 | 1.157 ± 0.002 | 0.5855 ± 0.0013 | 3 | -0.0005 tie |
| PRAGMA | per method | 0.5854 ± 0.0005 | 0.2023 ± 0.0041 | 1.530 ± 0.003 | 1.149 ± 0.003 | 0.5854 ± 0.0005 | 3 | -0.0006 tie |
| Transaction MLM | per method | 0.5846 ± 0.0007 | 0.2014 ± 0.0026 | 1.527 ± 0.002 | 1.152 ± 0.002 | 0.5846 ± 0.0007 | 3 | -0.0014 worse |
| NVIDIA TFM blueprint | per method | 0.5842 ± 0.0005 | 0.2042 ± 0.0032 | 1.527 ± 0.002 | 1.154 ± 0.003 | 0.5842 ± 0.0005 | 3 | -0.0018 worse |

## Pretraining ablations (A1–A5): 10 pool epochs each

Pretraining columns are validation, zero-shot. The fine-tune columns are test, `multi` task, 3 seeds.

| Variant | Change | next-type acc. | macro-F1 | time RMSE | time MAE | joint NLL | GPU min | state | test AUC (multi) | test next-type acc. (multi) | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| M10 (reference) | deep experts, token layout, abs+rel time, 128 events | 0.5994 | 0.2589 | 1.495 | 1.123 | 2.542 | 58 | done | – | – | 0/3 (queued) |
| shared | no experts (A1) | 0.5984 | 0.2614 | 1.497 | 1.124 | 2.551 | 52 | done | – | – | 0/3 (queued) |
| ffn | FFN experts only (A1) | 0.5979 | 0.2479 | 1.496 | 1.124 | 2.549 | 55 | done | – | – | 0/3 (queued) |
| attn_ffn | attention + FFN experts (A1) | 0.5988 | 0.2584 | 1.495 | 1.124 | 2.546 | 57 | done | – | – | 0/3 (queued) |
| additive | time added to the event token (A2) | 0.5974 | 0.2504 | 1.499 | 1.128 | 2.561 | 24 | done | – | – | 0/3 (queued) |
| tokshared_nocond | token layout, shared weights, no cond loss (A2 control) | 0.5977 | 0.2521 | 1.498 | 1.126 | 2.562 | 51 | done | – | – | 0/3 (queued) |
| relonly | relative time only (A3) | 0.5975 | 0.2524 | 1.517 | 1.141 | 2.591 | 61 | done | – | – | 0/3 (queued) |
| absonly | absolute time only (A3) | 0.5983 | 0.2600 | 1.522 | 1.182 | 2.565 | 61 | done | – | – | 0/3 (queued) |
| ropetime | index + continuous-time RoPE (A4) | 0.5996 | 0.2592 | 1.495 | 1.124 | 2.544 | 62 | done | – | – | 0/3 (queued) |
| ev64 | 64 events, window-relative time (A5) | 0.5981 | 0.2572 | 1.506 | 1.133 | 2.566 | 33 | done | – | – | 0/3 (queued) |

## Model scale (S1): 10 pool epochs each

| Size | Hidden / layers | next-type acc. | macro-F1 | time RMSE | time MAE | joint NLL | GPU min | state | test AUC (multi) | test next-type acc. (multi) | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| XS | 128 / 2 | 0.5930 | 0.2528 | 1.504 | 1.131 | 2.597 | 13 | done | – | – | 0/3 (queued) |
| S | 192 / 4 | 0.5977 | 0.2582 | 1.498 | 1.123 | 2.558 | 32 | done | – | – | 0/3 (queued) |
| M | 256 / 6 | 0.5994 | 0.2589 | 1.495 | 1.123 | 2.542 | 58 | done | – | – | 0/3 (queued) |
| L | 512 / 8 | 0.5986 | 0.2598 | 1.495 | 1.120 | 2.544 | 110 | done | – | – | 0/3 (queued) |
| XL | 768 / 12 | – | – | – | – | – | – | failed | – | – | 0/3 (failed) |

## Unlabeled data (S2): fixed 20,670 steps

| Unlabeled pretraining data | next-type acc. | macro-F1 | time RMSE | time MAE | joint NLL | GPU min | state | test AUC (multi) | test next-type acc. (multi) | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 30,665 labeled | 0.5942 | 0.2259 | 1.498 | 1.128 | 2.622 | 59 | done | – | – | 0/3 (queued) |
| 10 % pool (52,918) | 0.5972 | 0.2516 | 1.496 | 1.124 | 2.516 | 62 | done | – | – | 0/3 (queued) |
| 30 % pool (158,754) | 0.5996 | 0.2627 | 1.495 | 1.125 | 2.511 | 62 | done | – | – | 0/3 (queued) |
| 100 % pool (529,180) | 0.5994 | 0.2589 | 1.495 | 1.123 | 2.542 | 58 | done | – | – | 0/3 (queued) |

## Other pretraining runs

| Run | next-type acc. | macro-F1 | time RMSE | time MAE | joint NLL | GPU min | state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETAM-M, 20 pool epochs (`pre_main`) | 0.6010 | 0.2696 | 1.492 | 1.118 | 2.523 | 59 | done |
| ETAM-M, 20 epochs, HPO recipe (`pre_final`) | – | – | – | – | – | – | queued |
| Qwen3-0.6B-Base + LoRA r = 64, 2 pool epochs (`pre_llm`) | 0.5931 | 0.2358 | 1.504 | 1.130 | 2.559 | 292 | done |

## Pretraining HPO (H1): 10 pool epochs, frozen probe on validation

| Trial | valid probe AUC | valid zero-shot next-type acc. | score (sum) | hyper-parameters |
| --- | ---: | ---: | ---: | --- |
| t6 | 0.7335 | 0.6015 | 1.3351 | `learning_rate=0.000331 batch_size=256 weight_decay=0.05 model.dropout=0.0 p_full=0.5 weights.future=2.0 weights.cond_feature=1.0 weights.user_next=0.5` |
| t3 | 0.7282 | 0.6027 | 1.3309 | `learning_rate=0.00176 batch_size=256 weight_decay=0.01 model.dropout=0.0 p_full=0.3 weights.future=2.0 weights.cond_feature=1.0 weights.user_next=0.5` |
| t4 | 0.7302 | 0.5970 | 1.3273 | `learning_rate=0.00174 batch_size=256 weight_decay=0.1 model.dropout=0.0 p_full=0.5 weights.future=2.0 weights.cond_feature=0.25 weights.user_next=2.0` |
| t2 | 0.7308 | 0.5954 | 1.3261 | `learning_rate=0.00116 batch_size=128 weight_decay=0.1 model.dropout=0.1 p_full=0.5 weights.future=2.0 weights.cond_feature=0.25 weights.user_next=2.0` |
| t1 | 0.7273 | 0.5978 | 1.3251 | `learning_rate=0.000479 batch_size=128 weight_decay=0.01 model.dropout=0.1 p_full=0.3 weights.future=1.0 weights.cond_feature=0.5 weights.user_next=2.0` |
| t0 | 0.7255 | 0.5965 | 1.3220 | `learning_rate=0.000273 batch_size=128 weight_decay=0.05 model.dropout=0.0 p_full=0.5 weights.future=1.0 weights.cond_feature=0.5 weights.user_next=2.0` |
| t5 | 0.7248 | 0.5965 | 1.3213 | `learning_rate=0.000333 batch_size=256 weight_decay=0.05 model.dropout=0.2 p_full=0.3 weights.future=1.0 weights.cond_feature=0.25 weights.user_next=2.0` |
| t7 | 0.7224 | 0.5964 | 1.3188 | `learning_rate=0.000307 batch_size=512 weight_decay=0.05 model.dropout=0.0 p_full=0.7 weights.future=1.0 weights.cond_feature=1.0 weights.user_next=2.0` |
