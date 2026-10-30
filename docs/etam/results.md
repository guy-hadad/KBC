# ETAM results on full MBD

Design and protocol: [methodology.md](methodology.md). The benchmark reference is
[`../benchmark_scale/results.md`](../benchmark_scale/results.md) (n = 30,665, 3-seed means):
classification AUC 0.731 (`pragma-mlm`); TPP next-type accuracy 0.586, macro-F1 0.207, time
RMSE 1.526, time MAE 1.149 (log1p seconds).

**All tables comparing ETAM with the strong baselines are in [comparison.md](comparison.md).**
They are regenerated from the run directories by
`conda run -n KBC python docs/etam/make_comparison.py`; rerun it as cells finish.

## Findings so far (2026-10-30)

Test set, 3 fine-tuning seeds, mean ± sd. The reference is the best benchmark method per
metric at n = 30,665. Every pretrained arm comes from **one** pretraining run (seed 13), so
the sd covers fine-tuning noise only.

1. **The headline passes its pre-registered rule.** Pool-pretrained ETAM-M (20 epochs),
   fine-tuned as one `multi` model (M1), beats the best baseline on both tasks by more than
   the seed spread.

   | metric | ETAM-M multi (M1) | best baseline | Δ |
   |---|---:|---:|---:|
   | AUC | 0.7406 ± 0.0033 | 0.7309 ± 0.0017 (PRAGMA + MLM) | +0.0097 |
   | next-type acc. | 0.5976 ± 0.0015 | 0.5860 ± 0.0013 (PRAGMA + MLM) | +0.0116 |
   | next-type macro-F1 | 0.2533 ± 0.0092 | 0.2067 ± 0.0045 (AR Transformer) | +0.0467 |
   | time RMSE (lower is better) | 1.491 ± 0.002 | 1.526 ± 0.003 (CoLES) | −0.035 |
   | time MAE (lower is better) | 1.122 ± 0.002 | 1.149 ± 0.003 (PRAGMA) | −0.027 |

   This does not depend on checkpoint selection. At its final step M1 scores AUC
   0.7400 ± 0.0033 and next-type accuracy 0.5984 ± 0.0007.
2. **Pretraining is what does it.** The same model trained from scratch (`multi`) only ties
   the baselines: AUC 0.7304 ± 0.0044, next-type accuracy 0.5814 ± 0.0048. Pretraining adds
   +0.0102 AUC and +0.0163 next-type accuracy.
3. **Sharing one model costs little.** The specialist fine-tunes (M2) reach AUC
   0.7449 ± 0.0027 (`cls`, +0.0043 over M1) and next-type accuracy 0.5981 ± 0.0002 (`tpp`,
   +0.0005, a tie).
   - The `cls` specialist's AUC needs validation early stopping. It peaks after 2 epochs, and
     its final-step AUC is 0.7105 ± 0.0012.
4. **The frozen embedding is already as good as fine-tuning.** The frozen probe (M3) gives
   test AUC 0.7432 from a logistic regression on `u`, plus zero-shot next-type accuracy
   0.5994 ± 0.0009 from the pretrained heads. Both are at or above M1 (+0.0026 AUC, +0.0018
   accuracy).
   - The probe's AUC is the same for every seed: one checkpoint and a deterministic probe, so
     effectively n = 1.
   - On this evidence, multitask fine-tuning does not improve on the pretrained `u`.
5. **Giving the baselines the pool does not raise the bar (B1).** PRAGMA-MLM, Transaction
   MLM, CoLES and the AR Transformer were each pretrained one epoch on the same 529,180-client
   pool. Every one stays within its seed spread of its benchmark self on AUC (PRAGMA + MLM:
   0.7295 vs 0.7309).
   - The exceptions are on TPP. The AR Transformer improves next-type accuracy, 0.5859 vs
     0.5841 (+0.0017), which only brings it level with the other strong baselines.
   - Its macro-F1, 0.2172 vs 0.2067, becomes the best of any baseline, but it is still well
     below ETAM's 0.2533.
   - Transaction MLM's macro-F1 also rises, 0.2067 vs 0.2014.
6. **Caveats that bear on the headline.**
   - *Pool budget.* B1 gave the baselines one pool epoch of their own objective. ETAM-M got 20
     pool epochs of an objective that includes TPP itself. A B1 arm with a matched epoch
     budget has not been run.
   - *Selection.* ETAM selects on validation; the baselines run a fixed 8 epochs. M1's
     final-step numbers show this does not matter for the headline, but it does for the `cls`
     specialist.
   - *Final step.* With patience 3, "final step" is up to 3 non-improving epochs past the
     best. It is a pessimistic bound, not the baselines' protocol (see
     [methodology §6](methodology.md)).
   - *Open data caveats.* The label-window overlap and the absolute-time features
     (methodology §6) are still open. The `relonly` fine-tune, which bounds them, is queued.

### Pretraining-stage signals (validation, zero-shot, one seed each)

These numbers rank pretraining runs against each other; they are not test results. Full
tables are in [comparison.md](comparison.md).

- **Data.** At a fixed 20,670 steps, pool data beats the 30,665 labeled clients: joint NLL
  2.542 vs 2.622, macro-F1 0.2589 vs 0.2259. Next-type accuracy saturates by 30 % of the
  pool (0.5996 vs 0.5994).
  - The 10 % and 30 % subsets have a *lower* joint NLL than the full pool (2.516 and 2.511 vs
    2.542). The gap is driven by time NLL (1.397 and 1.395 vs 1.427), and it is unexplained.
- **Scale.** Joint NLL goes 2.597 (XS) → 2.558 (S) → 2.542 (M) → 2.544 (L): flat beyond M at
  10 epochs. Training M for 20 epochs instead (`pre_main`, 2.523) gives the best full-pool
  run, and the best run on every metric except joint NLL, where the 10 % and 30 % subsets
  are lower.
- **Ablations.** Most variants land within 0.02 NLL of the M10 reference (2.542).
  - Time content matters most. Relative-only time gives 2.591, its time NLL 1.473 vs 1.427.
    Absolute-only time gives 2.565, its time MAE 1.182 vs 1.123.
  - Experts: deep 2.542, attn_ffn 2.546, ffn 2.549, shared 2.551.
  - Layout: the additive layout (2.561) equals its token-layout control `tokshared_nocond`
    (2.562).
  - The A1/A2 pass criteria are judged on the fine-tuned test results, which are queued.
- **LLM backbone (L1).** Qwen3-0.6B-Base with LoRA experts reaches 2.559 after 292 GPU-min.
  ETAM-M10 reaches 2.542 in 58 GPU-min.
- **Pretraining HPO (H1).** The best trial is t6: LR 3.3e-4, batch 256, dropout 0, weights
  future 2 / cond-feature 1 / user-next 0.5. It scores validation probe AUC 0.7335 and
  zero-shot next-type accuracy 0.6015. Its 20-epoch retrain (`pre_final`) is queued.

## Status

| campaign | contents | state |
|---|---|---|
| `etam_mbd_phaseA_20261029` | pool build; GPU smoke | done |
| `etam_mbd_phaseB_20261029` | main, ablations, scale ladders, two-level HPO, pool baselines | 165 tasks: pretraining, scratch, pool baselines, M1–M3 done; other fine-tunes queued (AUTO block below) |

**Path bug, fixed 2026-10-30.**
- *Cause.* The campaign driver names each run dir `<group>__<id>` (e.g.
  `runs/pre_main__pre_main`), but `campaigns/make_phaseB.py` pointed every downstream stage
  at `runs/<id>`.
- *Effect.* 102 downstream tasks failed within seconds: every fine-tune, probe, HPO select
  and confirm task, plus the HPO retrain `pre_final`. They failed with `FileNotFoundError`,
  or because the select step they depend on had failed. No finished pretraining was lost.
- *Fix.* The generator now uses the full names. The re-plan changed only those paths; the
  frozen code snapshot is identical to the tree.
- *Resubmission.* Jobs 23692671–23692679, plus 23692691 for `pre_final`, which the feeder
  submitted once its select step had finished. Feeder 23692680 submits the remaining
  dependent stages and re-harvests the AUTO block.

**XL pretraining: out of GPU memory.** `pre_scale__XL` (768/12, batch 256) exhausted the
48 GB RTX 6000 after 45 s. It was not resubmitted, because fitting it needs a recipe change:
a smaller batch, or gradient accumulation, which `eventfm/etam/train.py` does not have. Its
3 fine-tunes stay failed.

**Smoke (plumbing only, not a result).** 400 pretraining steps plus 300 multitask steps on
the 30,665 labeled clients, 3 GPU-minutes in total, gave test AUC 0.719, next-type accuracy
0.581, RMSE 1.539 and MAE 1.160. It shows that the pipeline and the benchmark-parity
evaluation run end to end.

## Campaign tables

Generated by the campaign harvester. Rows are seed means ± std with `n`. The text outside
the AUTO block is hand-written.

<!-- AUTO:etam_mbd_phaseB_20261029:START -->
### etam_mbd_phaseB_20261029

harvested 2026-10-30T08:23:01 · root `/groups/bshapira_group/guyhada/campaigns/etam_mbd_phaseB_20261029` · git `0712fc0` (dirty tree)

**72/165 done** · running 7 · pending 75 · failed 11

_Incomplete: tables use finished tasks only; `n` shows finished/planned where they differ._

#### pre_main

Fixed: `size=M`

| task | valid.next_type_accuracy | valid.next_type_macro_f1 | valid.delta_log_rmse | valid.delta_log_mae | valid.joint_nll | train_minutes |
|---|---|---|---|---|---|---|
| pre_main__pre_main | 0.6010 | 0.2696 | 1.4919 | 1.1180 | 2.5233 | 58.50 |

#### hpo_pre

| trial | hp | valid.score | valid.probe_auc | valid.tpp_next_type_accuracy |
|---|---|---|---|---|
| 0 | learning_rate=0.000273 batch_size=128 weight_decay=0.05 model.dropout=0.0 p_full=0.5 weights.future=1.0 weights.cond_feature=0.5 weights.user_next=2.0 | 1.3220 | 0.7255 | 0.5965 |
| 1 | learning_rate=0.000479 batch_size=128 weight_decay=0.01 model.dropout=0.1 p_full=0.3 weights.future=1.0 weights.cond_feature=0.5 weights.user_next=2.0 | 1.3251 | 0.7273 | 0.5978 |
| 2 | learning_rate=0.00116 batch_size=128 weight_decay=0.1 model.dropout=0.1 p_full=0.5 weights.future=2.0 weights.cond_feature=0.25 weights.user_next=2.0 | 1.3261 | 0.7308 | 0.5954 |
| 3 | learning_rate=0.00176 batch_size=256 weight_decay=0.01 model.dropout=0.0 p_full=0.3 weights.future=2.0 weights.cond_feature=1.0 weights.user_next=0.5 | 1.3309 | 0.7282 | 0.6027 |
| 4 | learning_rate=0.00174 batch_size=256 weight_decay=0.1 model.dropout=0.0 p_full=0.5 weights.future=2.0 weights.cond_feature=0.25 weights.user_next=2.0 | 1.3273 | 0.7302 | 0.5970 |
| 5 | learning_rate=0.000333 batch_size=256 weight_decay=0.05 model.dropout=0.2 p_full=0.3 weights.future=1.0 weights.cond_feature=0.25 weights.user_next=2.0 | 1.3213 | 0.7248 | 0.5965 |
| 6 | learning_rate=0.000331 batch_size=256 weight_decay=0.05 model.dropout=0.0 p_full=0.5 weights.future=2.0 weights.cond_feature=1.0 weights.user_next=0.5 | 1.3351 | 0.7335 | 0.6015 |
| 7 | learning_rate=0.000307 batch_size=512 weight_decay=0.05 model.dropout=0.0 p_full=0.7 weights.future=1.0 weights.cond_feature=1.0 weights.user_next=2.0 | 1.3188 | 0.7224 | 0.5964 |

#### pre_scale_large

| size | arch | valid.next_type_accuracy | valid.next_type_macro_f1 | valid.delta_log_rmse | valid.delta_log_mae | valid.joint_nll | train_minutes |
|---|---|---|---|---|---|---|---|
| L | model.hidden_size=512 model.num_layers=8 model.num_heads=8 | 0.5986 | 0.2598 | 1.4952 | 1.1205 | 2.5441 | 109.6 |
| XL | model.hidden_size=768 model.num_layers=12 model.num_heads=12 | – | – | – | – | – | – |

#### pre_ablate

| variant | arch | valid.next_type_accuracy | valid.next_type_macro_f1 | valid.delta_log_rmse | valid.delta_log_mae | valid.joint_nll | train_minutes |
|---|---|---|---|---|---|---|---|
| M10 |  | 0.5994 | 0.2589 | 1.4953 | 1.1227 | 2.5421 | 57.80 |
| shared | model.placement=shared | 0.5984 | 0.2614 | 1.4967 | 1.1239 | 2.5510 | 52.17 |
| ffn | model.placement=ffn | 0.5979 | 0.2479 | 1.4958 | 1.1244 | 2.5488 | 54.56 |
| attn_ffn | model.placement=attn_ffn | 0.5988 | 0.2584 | 1.4952 | 1.1236 | 2.5462 | 57.38 |
| additive | model.layout=additive | 0.5974 | 0.2504 | 1.4992 | 1.1284 | 2.5608 | 23.74 |
| relonly | model.time_absolute=false | 0.5975 | 0.2524 | 1.5165 | 1.1410 | 2.5908 | 61.05 |
| absonly | model.time_relative=false | 0.5983 | 0.2600 | 1.5220 | 1.1816 | 2.5652 | 61.09 |
| ropetime | model.rope=index+time | 0.5996 | 0.2592 | 1.4947 | 1.1238 | 2.5438 | 62.44 |
| ev64 | max_events=64 window_time=true | 0.5981 | 0.2572 | 1.5058 | 1.1331 | 2.5656 | 32.93 |
| tokshared_nocond | model.placement=shared 'objectives=["tpp","user","future"]' | 0.5977 | 0.2521 | 1.4976 | 1.1263 | 2.5621 | 50.82 |

#### pre_scale_small

| size | arch | valid.next_type_accuracy | valid.next_type_macro_f1 | valid.delta_log_rmse | valid.delta_log_mae | valid.joint_nll | train_minutes |
|---|---|---|---|---|---|---|---|
| XS | model.hidden_size=128 model.num_layers=2 model.num_heads=4 | 0.5930 | 0.2528 | 1.5036 | 1.1311 | 2.5970 | 12.74 |
| S | model.hidden_size=192 model.num_layers=4 model.num_heads=6 | 0.5977 | 0.2582 | 1.4983 | 1.1234 | 2.5579 | 32.36 |

#### pre_data

| data | arch | valid.next_type_accuracy | valid.next_type_macro_f1 | valid.delta_log_rmse | valid.delta_log_mae | valid.joint_nll | train_minutes |
|---|---|---|---|---|---|---|---|
| matched | pretrain_store=train | 0.5942 | 0.2259 | 1.4983 | 1.1280 | 2.6221 | 58.52 |
| pool10pct | pretrain_fraction=0.1 | 0.5972 | 0.2516 | 1.4962 | 1.1238 | 2.5161 | 61.97 |
| pool30pct | pretrain_fraction=0.3 | 0.5996 | 0.2627 | 1.4949 | 1.1247 | 2.5108 | 62.12 |

#### ft_main

| task | classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|---|
| multi | 0.7406 ± 0.0033 | 0.5976 ± 0.0015 | 0.2533 ± 0.0092 | 1.4909 ± 0.0021 | 1.1224 ± 0.0015 | 1.3298 ± 0.0027 | 3 |
| cls | 0.7449 ± 0.0027 | – | – | – | – | 0.7350 ± 0.0018 | 3 |
| tpp | – | 0.5981 ± 0.0002 | 0.2552 ± 0.0047 | 1.4907 ± 0.0031 | 1.1222 ± 0.0018 | -2.5354 ± 0.0090 | 3 |

_mean ± sample std over `seed`._

#### probe_main

| classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|
| 0.7432 ± 0.0000 | 0.5994 ± 0.0009 | 0.2618 ± 0.0019 | 1.4886 ± 0.0020 | 1.1164 ± 0.0017 | 1.3291 ± 0.0030 | 3 |

_mean ± sample std over `seed`._

#### hpo_ft

Fixed: `task=multi`, `seed=13`

_No metrics yet (0/24 done)._

#### hpo_ft_select

Fixed: `stage=finetune`

_No metrics yet (0/1 done)._

#### hpo_pre_select

Fixed: `stage=pretrain`

_No metrics yet (1/1 done)._

#### pre_final

Fixed: `size=M`

_No metrics yet (0/1 done)._

#### confirm

_No metrics yet (0/6 done)._

#### ft_ablate

Fixed: `task=multi`

_No metrics yet (0/30 done)._

#### ft_scale_small

Fixed: `task=multi`

_No metrics yet (0/6 done)._

#### ft_scale_large

Fixed: `task=multi`

_No metrics yet (0/6 done)._

#### ft_data

_No metrics yet (0/15 done)._

#### ft_scratch

| task | classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|---|
| multi | 0.7304 ± 0.0044 | 0.5814 ± 0.0048 | 0.1845 ± 0.0030 | 1.5279 ± 0.0103 | 1.1623 ± 0.0212 | 1.3108 ± 0.0061 | 3 |
| cls | 0.7318 ± 0.0033 | – | – | – | – | 0.7309 ± 0.0033 | 3 |
| tpp | – | 0.5892 ± 0.0006 | 0.2055 ± 0.0040 | 1.5110 ± 0.0035 | 1.1406 ± 0.0020 | -2.6493 ± 0.0312 | 3 |

_mean ± sample std over `seed`._

#### pre_llm

Fixed: `size=qwen0.6b`

| task | valid.next_type_accuracy | valid.next_type_macro_f1 | valid.delta_log_rmse | valid.delta_log_mae | valid.joint_nll | train_minutes |
|---|---|---|---|---|---|---|
| pre_llm__pre_llm | 0.5931 | 0.2358 | 1.5042 | 1.1298 | 2.5591 | 291.9 |

#### ft_llm

Fixed: `task=multi`

_No metrics yet (0/3 done)._

#### baselines_pool

| cell | auc | next_type_accuracy | next_type_macro_f1 | delta_log_rmse | delta_log_mae |
|---|---|---|---|---|---|
| 0 | 0.7300 | – | – | – | – |
| 1 | 0.7340 | – | – | – | – |
| 2 | 0.7244 | – | – | – | – |
| 3 | 0.7212 | – | – | – | – |
| 4 | 0.7253 | – | – | – | – |
| 5 | 0.7189 | – | – | – | – |
| 6 | 0.7106 | – | – | – | – |
| 7 | 0.7061 | – | – | – | – |
| 8 | 0.7091 | – | – | – | – |
| 9 | 0.7198 | – | – | – | – |
| 10 | 0.7140 | – | – | – | – |
| 11 | 0.7107 | – | – | – | – |
| 12 | – | 0.5872 | 0.2008 | 1.5281 | 1.1518 |
| 13 | – | 0.5859 | 0.2085 | 1.5249 | 1.1458 |
| 14 | – | 0.5860 | 0.2037 | 1.5289 | 1.1566 |
| 15 | – | 0.5857 | 0.2056 | 1.5265 | 1.1498 |
| 16 | – | 0.5847 | 0.2107 | 1.5237 | 1.1506 |
| 17 | – | 0.5854 | 0.2039 | 1.5281 | 1.1535 |
| 18 | – | 0.5863 | 0.1933 | 1.5271 | 1.1554 |
| 19 | – | 0.5855 | 0.1970 | 1.5229 | 1.1536 |
| 20 | – | 0.5843 | 0.1941 | 1.5282 | 1.1585 |
| 21 | – | 0.5868 | 0.2181 | 1.5270 | 1.1526 |
| 22 | – | 0.5859 | 0.2187 | 1.5238 | 1.1514 |
| 23 | – | 0.5848 | 0.2148 | 1.5281 | 1.1571 |

#### Not done (93)

| task | state | exit | node | elapsed | log |
|---|---|---|---|---|---|
| pre_scale_large__pre_scale__XL | failed | 1 | ise-6000-04 | 00:00:45 | `runs/pre_scale_large__pre_scale__XL/task.log` |
| hpo_ft__hpo_ft__t0 | running |  | ise-4090-20 |  |  |
| hpo_ft__hpo_ft__t1 | running |  | ise-4090-19 |  |  |
| hpo_ft__hpo_ft__t2 | running |  | cs-4090-07 |  |  |
| hpo_ft__hpo_ft__t3 | running |  | ise-4090-16 |  |  |
| hpo_ft__hpo_ft__t4 | running |  | ise-4090-18 |  |  |
| hpo_ft__hpo_ft__t5 | running |  | ise-4090-19 |  |  |
| hpo_ft__hpo_ft__t6 | running |  | ise-4090-19 |  |  |
| hpo_ft__hpo_ft__t7 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t8 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t9 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t10 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t11 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t12 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t13 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t14 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t15 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t16 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t17 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t18 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t19 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t20 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t21 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t22 | pending |  |  |  |  |
| hpo_ft__hpo_ft__t23 | pending |  |  |  |  |

_… 68 more in `failed.txt` / `results.csv`._

Per-task data (every metric, every state): `results.csv`.
<!-- AUTO:etam_mbd_phaseB_20261029:END -->
