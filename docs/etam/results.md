# ETAM results on full MBD

Design and protocol: [methodology.md](methodology.md). The benchmark reference is
[`../benchmark_scale/results.md`](../benchmark_scale/results.md) (n = 30,665, 3-seed means):
classification AUC 0.731 (`pragma-mlm`); TPP next-type accuracy 0.586, macro-F1 0.207, time
RMSE 1.526, time MAE 1.149 (log1p seconds).

**All tables comparing ETAM with the strong baselines are in [comparison.md](comparison.md).**
They are regenerated from the run directories by
`conda run -n KBC python docs/etam/make_comparison.py`; rerun it as cells finish.

## Findings (2026-10-31, phase B complete)

Test set, 3 fine-tuning seeds, mean ± sd. The reference is the best benchmark method per
metric at n = 30,665. Every pretrained arm comes from **one** pretraining run (seed 13), so
the sd covers fine-tuning noise only. "Beyond the spread" means the gap exceeds the larger of
the two arms' sd, the pre-registered rule; with n = 3 it is a screening rule, not a test.

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
   0.7400 ± 0.0033 and next-type accuracy 0.5984 ± 0.0007. M1 also beats the
   pool-pretrained baselines (B1, the rule's second condition): +0.0111 AUC and +0.0113
   next-type accuracy over PRAGMA + MLM (B1).
2. **Pretraining is what does it.** The same model trained from scratch (`multi`) only ties
   the baselines: AUC 0.7304 ± 0.0044, next-type accuracy 0.5814 ± 0.0048. Pretraining adds
   +0.0102 AUC and +0.0163 next-type accuracy. Most of that gain survives pretraining on the
   30,665 labeled clients alone (item 9).
3. **Sharing one model costs little.** The specialist fine-tunes (M2) reach AUC
   0.7449 ± 0.0027 (`cls`, +0.0043 over M1) and next-type accuracy 0.5981 ± 0.0002 (`tpp`,
   +0.0005, a tie).
   - The `cls` specialist's AUC needs validation early stopping. It peaks after 2 epochs, and
     its final-step AUC is 0.7105 ± 0.0012.
   - With the tuned pretraining recipe (item 6) the AUC gap closes: the multitask model
     reaches 0.7456 ± 0.0021, a tie with the `cls` specialist.
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
6. **Tuning the pretraining recipe raises AUC; tuning the fine-tune recipe does not (H1,
   H2).**
   - *H2 alone.* The 24 fine-tune trials score within a narrow validation band: best 1.3305,
     median 1.3260, against 1.3298 for M1's default recipe. The winner, t13 (LR 1.45e-4,
     batch 256, dropout 0.2, up to 20 epochs, `p_full` 0.7), was confirmed on M1's
     checkpoint. It ties M1: AUC 0.7392 ± 0.0038, next-type accuracy 0.5977 ± 0.0013.
   - *H1 + H2.* The same fine-tune recipe on `pre_final` (H1 winner t6, retrained for 20
     epochs) gives AUC **0.7456 ± 0.0021**, +0.0050 over M1 and beyond the spread. Next-type
     accuracy ties, 0.5977 ± 0.0002.
     - Against the best baseline that is +0.0147 AUC and +0.0117 accuracy. It is the highest
       test AUC of any arm in the campaign.
     - At its final step it scores 0.7451 ± 0.0012 and 0.5974 ± 0.0006.
     - Time errors trade off, each just beyond the spread: MAE improves (1.113 vs 1.122)
       and RMSE worsens (1.499 vs 1.491). Macro-F1 ties (0.246 ± 0.014 vs 0.253 ± 0.009).
   - H2 alone changes nothing, so the AUC gain is attributed to the H1 pretraining recipe.
     `pre_final` with the default fine-tune recipe was not run, so this is an inference.
   - Every selection used validation only (`test_eval=false` for trials).
7. **Ablations (A1–A5).** Fine-tuned `multi`, against the 10-epoch reference M10 (AUC
   0.7409 ± 0.0016, next-type accuracy 0.5952 ± 0.0007):
   - *A1, experts: kept, for classification.* `deep` beats `shared` by +0.0042 AUC (+0.0034
     at final step), beyond the spread, which meets the pre-registered rule.
     - `ffn` alone gets the same gain (+0.0040), so it sits in the FFN experts.
     - Next-type accuracy does not move (+0.0003).
   - *A2, time as its own token: passes the rule, but only with what the layout enables.*
     M10 beats `additive` on both tasks, +0.0039 AUC and +0.0019 accuracy. The controls
     show where the gain comes from:
     - With shared weights and no time-conditioned loss (`tokshared_nocond`), the token
       layout is *worse* than additive on AUC (−0.0059) and ties on accuracy.
     - Adding the conditioned loss (`shared`) ties additive on AUC and beats it on accuracy
       (+0.0016). Adding experts (M10) then lifts AUC.
     - So the gain does not come from the layout by itself. It comes from the
       time-conditioned objective and the per-modality experts that the layout makes
       possible.
   - *A3, time content.* Absolute time carries classification. Relative time carries the
     prediction of when the next event happens.
     - `relonly`: AUC −0.0074, accuracy −0.0015, RMSE +0.021, MAE +0.020.
     - `absonly`: AUC +0.0037 (0.7446 ± 0.0008, higher than with both), accuracy ties, RMSE
       +0.032, MAE +0.065.
     - Item 11 covers what this implies for the label-window caveat.
   - *A4, time RoPE: helps classification.* `ropetime` gives AUC 0.7449 ± 0.0019, +0.0040
     (+0.0025 at final step), with accuracy and time errors tied. It is a candidate for the
     default recipe.
   - *A5, context length is not the source of the gain.* At 64 events (`ev64`,
     window-relative time as in the baselines' collator), AUC drops by 0.0019 and macro-F1
     by 0.014. ETAM still beats the best baseline by +0.0081 AUC and +0.0087 accuracy.
8. **Model scale saturates at M (S1).** Fine-tuned AUC goes 0.7296 (XS) → 0.7360 (S) →
   0.7409 (M) → 0.7406 (L). Next-type accuracy goes 0.5903 → 0.5944 → 0.5952 → 0.5952. L
   ties M on both. XL did not run (see Status).
   - XS has the baselines' width and depth (128/2). It ties the best baseline on AUC
     (−0.0013) and beats it on next-type accuracy (+0.0043).
9. **The pool adds little beyond pretraining itself (S2, M4).** At a fixed 20,670
   pretraining steps:
   - 10 %, 30 % and 100 % of the pool tie on both tasks after fine-tuning (AUC
     0.7408 / 0.7399 / 0.7409).
   - Pretraining on only the 30,665 labeled clients (M4) costs 0.0017 AUC and 0.0025
     next-type accuracy relative to the full pool, and macro-F1 falls from 0.245 to 0.216.
   - M4 still beats the best baseline on both tasks, +0.0083 AUC and +0.0067 accuracy, and
     beats scratch by +0.0088 AUC and +0.0113 accuracy.
   - So most of ETAM's edge is available at the baselines' *data* budget. Its pretraining
     *compute* is not matched to theirs (item 11).
   - The matched `cls` specialist overfits: its final-step AUC is 0.6776 ± 0.0033, against
     0.7396 ± 0.0076 at the selected checkpoint.
10. **The LLM backbone adds nothing (L1).** Qwen3-0.6B-Base with LoRA experts, fine-tuned
    `multi`, ties M10 on AUC (0.7388 ± 0.0033, −0.0021) and next-type accuracy (0.5948,
    −0.0004).
    - Its time errors are worse (RMSE +0.006, MAE +0.009), and it costs 292 pretraining
      GPU-minutes against M10's 58.
    - Its average precision, 0.0363 ± 0.0008, is the highest of any arm (M1: 0.0328).
11. **Caveats that bear on the headline.**
    - *Pretraining budget.* B1 gave the baselines one pool epoch of their own objective.
      ETAM-M got 20 pool epochs of an objective that includes TPP itself.
      - Item 9 shows that the pool *data* is not what matters, but pretraining compute is
        still unmatched. The matched arm runs 20,670 steps on the 30,665 clients.
      - A B1 arm with a matched step budget has not been run.
    - *Selection.* ETAM selects on validation; the baselines run a fixed 8 epochs. No
      multitask arm moves by more than 0.0043 AUC at its final step (the largest is M4,
      0.7392 → 0.7349). The `cls` specialists depend on selection (items 3 and 9).
    - *Final step.* With patience 3, "final step" is up to 3 non-improving epochs past the
      best. It is a pessimistic bound, not the baselines' protocol (see
      [methodology §6](methodology.md)).
    - *Absolute time and the label window* (methodology §6). The `relonly` result bounds this
      only loosely.
      - Without absolute time, ETAM's AUC edge over the best baseline shrinks from +0.0100
        (M10) to +0.0026 (0.7335 ± 0.0021), at the edge of the spread. It ties PRAGMA + MLM
        (B1): +0.0040, with sd 0.0048.
      - Its next-type accuracy edge mostly survives: +0.0077 of +0.0091.
      - Dropping relative time instead (`absonly`) raises AUC above the reference, to
        0.7446. Label-window overlap would produce this pattern, but so would genuine
        seasonality, and the ablations cannot separate the two.
      - `relonly` also removes the calendar features (up to day-of-month) that the
        baselines receive, so it understates what a baseline-matched calendar would score.
        An arm with exactly the baselines' calendar features would separate the
        explanations; it has not been run.
      - **Net:** the TPP gain does not depend on absolute time. Most of the classification
        gain does, until the label definition is checked.

### Pretraining-stage signals (validation, zero-shot, one seed each)

These numbers rank pretraining runs against each other; they are not test results. Full
tables are in [comparison.md](comparison.md).

- **Data.** At a fixed 20,670 steps, pool data beats the 30,665 labeled clients: joint NLL
  2.542 vs 2.622, macro-F1 0.2589 vs 0.2259. Next-type accuracy saturates by 30 % of the
  pool (0.5996 vs 0.5994).
  - The 10 % and 30 % subsets have a *lower* joint NLL than the full pool (2.516 and 2.511 vs
    2.542). The gap is driven by time NLL (1.397 and 1.395 vs 1.427), and it is unexplained.
    After fine-tuning, the three tie (item 9).
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
  - Pretraining NLL does not predict the fine-tuned ranking. `absonly` (2.565) and
    `ropetime` (2.544) are no better than M10 on NLL, yet they are the two best ablations on
    test AUC (item 7).
- **LLM backbone (L1).** Qwen3-0.6B-Base with LoRA experts reaches 2.559 after 292 GPU-min.
  ETAM-M10 reaches 2.542 in 58 GPU-min.
- **Pretraining HPO (H1).** The best trial is t6: LR 3.3e-4, batch 256, dropout 0, weights
  future 2 / cond-feature 1 / user-next 0.5. It scores validation probe AUC 0.7335 and
  zero-shot next-type accuracy 0.6015.
  - Its 20-epoch retrain, `pre_final` (118 GPU-min), has the lowest validation joint NLL of
    any run: 2.476, against 2.523 for `pre_main`. Its next-type accuracy is 0.6032.
  - Its fine-tuned results are in item 6.

## Status

| campaign | contents | state |
|---|---|---|
| `etam_mbd_phaseA_20261029` | pool build; GPU smoke | done |
| `etam_mbd_phaseB_20261029` | main, ablations, scale ladders, two-level HPO, pool baselines | done: 161/165 tasks; XL pretraining and its 3 fine-tunes failed (AUTO block below) |

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
  submitted once its select step had finished. Feeder 23692680 submitted the remaining
  dependent stages. Every resubmitted task finished.

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

harvested 2026-10-31T20:15:58 · root `/groups/bshapira_group/guyhada/campaigns/etam_mbd_phaseB_20261029` · git `0712fc0` (dirty tree)

**161/165 done** · failed 4

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

| trial | hp | valid.score | valid.cls_auc | valid.tpp_next_type_accuracy |
|---|---|---|---|---|
| 0 | learning_rate=0.000857 batch_size=64 weight_decay=0.0 model.dropout=0.0 epochs=12 p_full=0.3 weights.cls=2.0 head_learning_rate=0.001 warmup_steps=100 | 1.3155 | 0.7265 | 0.5890 |
| 1 | learning_rate=0.000252 batch_size=256 weight_decay=0.01 model.dropout=0.3 epochs=20 p_full=0.7 weights.cls=2.0 head_learning_rate=0.001 warmup_steps=300 | 1.3277 | 0.7303 | 0.5974 |
| 2 | learning_rate=0.000707 batch_size=64 weight_decay=0.05 model.dropout=0.3 epochs=12 p_full=0.5 weights.cls=4.0 head_learning_rate=0.0 warmup_steps=100 | 1.3213 | 0.7338 | 0.5875 |
| 3 | learning_rate=6.87e-05 batch_size=64 weight_decay=0.01 model.dropout=0.2 epochs=8 p_full=0.3 weights.cls=2.0 head_learning_rate=0.0 warmup_steps=300 | 1.3295 | 0.7285 | 0.6010 |
| 4 | learning_rate=0.00049 batch_size=256 weight_decay=0.05 model.dropout=0.2 epochs=12 p_full=0.5 weights.cls=1.0 head_learning_rate=0.001 warmup_steps=300 | 1.3283 | 0.7302 | 0.5981 |
| 5 | learning_rate=0.000298 batch_size=64 weight_decay=0.1 model.dropout=0.2 epochs=20 p_full=0.7 weights.cls=2.0 head_learning_rate=0.001 warmup_steps=300 | 1.3231 | 0.7309 | 0.5921 |
| 6 | learning_rate=0.000103 batch_size=256 weight_decay=0.1 model.dropout=0.3 epochs=8 p_full=0.5 weights.cls=1.0 head_learning_rate=0.001 warmup_steps=300 | 1.3295 | 0.7293 | 0.6002 |
| 7 | learning_rate=8.88e-05 batch_size=256 weight_decay=0.1 model.dropout=0.2 epochs=8 p_full=0.5 weights.cls=2.0 head_learning_rate=0.0 warmup_steps=300 | 1.3284 | 0.7281 | 0.6003 |
| 8 | learning_rate=0.000383 batch_size=64 weight_decay=0.0 model.dropout=0.0 epochs=4 p_full=0.5 weights.cls=1.0 head_learning_rate=0.0 warmup_steps=100 | 1.3260 | 0.7292 | 0.5968 |
| 9 | learning_rate=0.000599 batch_size=64 weight_decay=0.01 model.dropout=0.0 epochs=20 p_full=0.7 weights.cls=0.5 head_learning_rate=0.0 warmup_steps=300 | 1.3131 | 0.7301 | 0.5831 |
| 10 | learning_rate=0.000106 batch_size=64 weight_decay=0.0 model.dropout=0.0 epochs=4 p_full=0.3 weights.cls=0.5 head_learning_rate=0.0 warmup_steps=100 | 1.3221 | 0.7225 | 0.5997 |
| 11 | learning_rate=0.000111 batch_size=64 weight_decay=0.01 model.dropout=0.1 epochs=4 p_full=0.5 weights.cls=0.5 head_learning_rate=0.0 warmup_steps=100 | 1.3250 | 0.7254 | 0.5995 |
| 12 | learning_rate=0.000903 batch_size=64 weight_decay=0.05 model.dropout=0.0 epochs=12 p_full=0.5 weights.cls=4.0 head_learning_rate=0.0 warmup_steps=300 | 1.3144 | 0.7263 | 0.5881 |
| 13 | learning_rate=0.000145 batch_size=256 weight_decay=0.0 model.dropout=0.2 epochs=20 p_full=0.7 weights.cls=1.0 head_learning_rate=0.001 warmup_steps=100 | 1.3305 | 0.7308 | 0.5997 |
| 14 | learning_rate=4.16e-05 batch_size=256 weight_decay=0.05 model.dropout=0.0 epochs=4 p_full=0.5 weights.cls=1.0 head_learning_rate=0.001 warmup_steps=300 | 1.3260 | 0.7252 | 0.6008 |
| 15 | learning_rate=0.000182 batch_size=64 weight_decay=0.05 model.dropout=0.2 epochs=12 p_full=0.7 weights.cls=4.0 head_learning_rate=0.0 warmup_steps=100 | 1.3274 | 0.7305 | 0.5969 |
| 16 | learning_rate=0.000315 batch_size=128 weight_decay=0.0 model.dropout=0.1 epochs=8 p_full=0.3 weights.cls=0.5 head_learning_rate=0.001 warmup_steps=100 | 1.3238 | 0.7253 | 0.5985 |
| 17 | learning_rate=0.000178 batch_size=256 weight_decay=0.0 model.dropout=0.1 epochs=8 p_full=0.7 weights.cls=4.0 head_learning_rate=0.0 warmup_steps=300 | 1.3288 | 0.7295 | 0.5993 |
| 18 | learning_rate=3.98e-05 batch_size=64 weight_decay=0.05 model.dropout=0.2 epochs=20 p_full=0.5 weights.cls=0.5 head_learning_rate=0.0 warmup_steps=100 | 1.3299 | 0.7293 | 0.6006 |
| 19 | learning_rate=0.000116 batch_size=64 weight_decay=0.0 model.dropout=0.0 epochs=8 p_full=0.3 weights.cls=0.5 head_learning_rate=0.0 warmup_steps=100 | 1.3273 | 0.7270 | 0.6003 |
| 20 | learning_rate=0.000417 batch_size=64 weight_decay=0.01 model.dropout=0.0 epochs=20 p_full=0.5 weights.cls=2.0 head_learning_rate=0.001 warmup_steps=100 | 1.3226 | 0.7310 | 0.5916 |
| 21 | learning_rate=0.000331 batch_size=64 weight_decay=0.1 model.dropout=0.3 epochs=4 p_full=0.7 weights.cls=0.5 head_learning_rate=0.001 warmup_steps=100 | 1.3273 | 0.7293 | 0.5979 |
| 22 | learning_rate=0.000757 batch_size=256 weight_decay=0.1 model.dropout=0.2 epochs=4 p_full=0.7 weights.cls=0.5 head_learning_rate=0.001 warmup_steps=300 | 1.3266 | 0.7287 | 0.5978 |
| 23 | learning_rate=0.000338 batch_size=128 weight_decay=0.05 model.dropout=0.0 epochs=20 p_full=0.3 weights.cls=0.5 head_learning_rate=0.001 warmup_steps=100 | 1.3208 | 0.7242 | 0.5966 |

#### hpo_ft_select

Fixed: `stage=finetune`

_No metrics yet (1/1 done)._

#### hpo_pre_select

Fixed: `stage=pretrain`

_No metrics yet (1/1 done)._

#### pre_final

Fixed: `size=M`

| task | valid.next_type_accuracy | valid.next_type_macro_f1 | valid.delta_log_rmse | valid.delta_log_mae | valid.joint_nll | train_minutes |
|---|---|---|---|---|---|---|
| pre_final__pre_final | 0.6032 | 0.2709 | 1.4892 | 1.1222 | 2.4756 | 117.7 |

#### confirm

| init | classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|---|
| pre_main | 0.7392 ± 0.0038 | 0.5977 ± 0.0013 | 0.2527 ± 0.0060 | 1.4943 ± 0.0041 | 1.1146 ± 0.0063 | 1.3306 ± 0.0028 | 3 |
| pre_final | 0.7456 ± 0.0021 | 0.5977 ± 0.0002 | 0.2459 ± 0.0142 | 1.4986 ± 0.0073 | 1.1132 ± 0.0073 | 1.3304 ± 0.0033 | 3 |

_mean ± sample std over `seed`._

#### ft_ablate

Fixed: `task=multi`

| variant | ftx | classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|---|---|
| M10 |  | 0.7409 ± 0.0016 | 0.5952 ± 0.0007 | 0.2451 ± 0.0095 | 1.4948 ± 0.0020 | 1.1279 ± 0.0041 | 1.3244 ± 0.0034 | 3 |
| shared |  | 0.7367 ± 0.0015 | 0.5949 ± 0.0007 | 0.2471 ± 0.0027 | 1.4957 ± 0.0031 | 1.1246 ± 0.0008 | 1.3260 ± 0.0031 | 3 |
| ffn |  | 0.7407 ± 0.0015 | 0.5950 ± 0.0011 | 0.2450 ± 0.0078 | 1.4957 ± 0.0013 | 1.1295 ± 0.0070 | 1.3255 ± 0.0014 | 3 |
| attn_ffn |  | 0.7407 ± 0.0012 | 0.5958 ± 0.0014 | 0.2463 ± 0.0051 | 1.4934 ± 0.0026 | 1.1246 ± 0.0009 | 1.3278 ± 0.0029 | 3 |
| additive |  | 0.7370 ± 0.0002 | 0.5933 ± 0.0009 | 0.2406 ± 0.0025 | 1.4969 ± 0.0010 | 1.1318 ± 0.0046 | 1.3198 ± 0.0007 | 3 |
| relonly |  | 0.7335 ± 0.0021 | 0.5937 ± 0.0001 | 0.2434 ± 0.0021 | 1.5157 ± 0.0027 | 1.1475 ± 0.0032 | 1.3287 ± 0.0010 | 3 |
| absonly |  | 0.7446 ± 0.0008 | 0.5954 ± 0.0012 | 0.2401 ± 0.0147 | 1.5264 ± 0.0104 | 1.1923 ± 0.0151 | 1.3254 ± 0.0022 | 3 |
| ropetime |  | 0.7449 ± 0.0019 | 0.5954 ± 0.0006 | 0.2441 ± 0.0035 | 1.4936 ± 0.0024 | 1.1254 ± 0.0032 | 1.3247 ± 0.0020 | 3 |
| ev64 | max_events=64 window_time=true | 0.7390 ± 0.0004 | 0.5947 ± 0.0016 | 0.2315 ± 0.0042 | 1.5040 ± 0.0011 | 1.1368 ± 0.0045 | 1.3210 ± 0.0025 | 3 |
| tokshared_nocond | 'objectives=["cls","tpp","user"]' | 0.7312 ± 0.0011 | 0.5938 ± 0.0017 | 0.2489 ± 0.0042 | 1.4957 ± 0.0012 | 1.1288 ± 0.0033 | 1.3241 ± 0.0010 | 3 |

_mean ± sample std over `seed`._

#### ft_scale_small

Fixed: `task=multi`

| size | classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|---|
| XS | 0.7296 ± 0.0004 | 0.5903 ± 0.0012 | 0.2366 ± 0.0060 | 1.5023 ± 0.0019 | 1.1345 ± 0.0057 | 1.3203 ± 0.0033 | 3 |
| S | 0.7360 ± 0.0024 | 0.5944 ± 0.0016 | 0.2503 ± 0.0074 | 1.4962 ± 0.0017 | 1.1274 ± 0.0050 | 1.3278 ± 0.0016 | 3 |

_mean ± sample std over `seed`._

#### ft_scale_large

Fixed: `task=multi`

| size | classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|---|
| L | 0.7406 ± 0.0016 | 0.5952 ± 0.0012 | 0.2425 ± 0.0023 | 1.4953 ± 0.0016 | 1.1314 ± 0.0049 | 1.3290 ± 0.0019 | 3 |
| XL | – | – | – | – | – | – | 0/3 |

_mean ± sample std over `seed`._

#### ft_data

| data | task | classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|---|---|
| matched | multi | 0.7392 ± 0.0006 | 0.5927 ± 0.0013 | 0.2155 ± 0.0105 | 1.4964 ± 0.0015 | 1.1282 ± 0.0013 | 1.3215 ± 0.0041 | 3 |
| matched | cls | 0.7396 ± 0.0076 | – | – | – | – | 0.7303 ± 0.0033 | 3 |
| matched | tpp | – | 0.5939 ± 0.0010 | 0.2207 ± 0.0122 | 1.4943 ± 0.0019 | 1.1264 ± 0.0015 | -2.6262 ± 0.0060 | 3 |
| pool10pct | multi | 0.7408 ± 0.0018 | 0.5949 ± 0.0017 | 0.2456 ± 0.0063 | 1.4949 ± 0.0018 | 1.1268 ± 0.0050 | 1.3238 ± 0.0028 | 3 |
| pool30pct | multi | 0.7399 ± 0.0011 | 0.5956 ± 0.0014 | 0.2450 ± 0.0090 | 1.4939 ± 0.0017 | 1.1258 ± 0.0046 | 1.3246 ± 0.0016 | 3 |

_mean ± sample std over `seed`._

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

| classification.auc | tpp.next_type_accuracy | tpp.next_type_macro_f1 | tpp.delta_log_rmse | tpp.delta_log_mae | valid.score | n |
|---|---|---|---|---|---|---|
| 0.7388 ± 0.0033 | 0.5948 ± 0.0010 | 0.2426 ± 0.0064 | 1.5009 ± 0.0026 | 1.1372 ± 0.0040 | 1.3287 ± 0.0011 | 3 |

_mean ± sample std over `seed`._

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

#### Not done (4)

| task | state | exit | node | elapsed | log |
|---|---|---|---|---|---|
| pre_scale_large__pre_scale__XL | failed | 1 | ise-6000-04 | 00:00:45 | `runs/pre_scale_large__pre_scale__XL/task.log` |
| ft_scale_large__ft_scale__XL_s13 | failed | 1 | ise-cpu256-28 | 00:00:18 | `runs/ft_scale_large__ft_scale__XL_s13/task.log` |
| ft_scale_large__ft_scale__XL_s29 | failed | 1 | ise-cpu256-28 | 00:00:17 | `runs/ft_scale_large__ft_scale__XL_s29/task.log` |
| ft_scale_large__ft_scale__XL_s43 | failed | 1 | ise-6000-08 | 00:00:31 | `runs/ft_scale_large__ft_scale__XL_s43/task.log` |

Per-task data (every metric, every state): `results.csv`.
<!-- AUTO:etam_mbd_phaseB_20261029:END -->
