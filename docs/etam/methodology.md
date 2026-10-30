# ETAM: Events and Time As Modalities

Status **2026-10-30**: implemented, unit-tested (15 tests; the full suite passes),
independently reviewed, and parity-checked against the benchmark on real MBD data.
Phase A is done. In phase B every pretraining run except XL has finished, and the
fine-tunes are running after a path fix. First results are in [results.md](results.md) and
[comparison.md](comparison.md). The code is in `eventfm/etam/` and runs through
`scripts/etam.py`.

## 1. Question

Can **one** event-sequence model, whose user embedding serves both tasks, be at least as
strong as the best specialist on each task of full MBD? The two tasks are classification
(the product-propensity label) and the marked TPP task (next event type and log-gap).

The approach transfers the modality-expert recipe from vision–language models to event
streams: Mono-InternVL, EVEv2, NEO, Mixture-of-Transformers, and this group's RAM
(`~/ST/docs_RAM/methodology.md`). The *two modalities are time and event*. Each event becomes
two soft tokens. Each token type has its own parameters in every layer. The two token types
meet only in one shared attention softmax.

## 2. The model

```
slot i:   τ_i = TimeTok(abs(t_i), rel(t_i − t_{i−1}, t_i − t_first))   ε_i = EventTok(k_i, f_i)
query:    τ_q = TimeTok(abs(now), rel(now − t_n, now − t_first)) + q   USR (learned)
sequence: τ_1 ε_1 τ_2 ε_2 … τ_n ε_n τ_q USR          (left-padded by slot)
layer:    a = RMSNorm^m(x); q,k,v = QKNorm^m(W^m_qkv a); RoPE; o = W^m_o · Attn(one causal softmax over all tokens)
          x = x + o;  x = x + SwiGLU^m(RMSNorm^m(x))      m ∈ {time, event} by token type
```

- **Time token**, built from absolute and relative time:
  - *Absolute*: hour, weekday, day-of-month and month embeddings; their cyclic coordinates;
    Fourier features of the absolute day count (1 d / 7 d / 30.4 d / 91.3 d / 365.25 d periods);
    and a linear trend.
  - *Relative*: log gap to the previous event, as Fourier features plus a learned log-scale
    bucket embedding (the GEM-2026 finding that log bins suit heavy-tailed gaps); log age
    since the first observed event; and a has-previous flag.
  - In the query token, "gap" means the time elapsed since the last event.
- **Event token**: the mark embedding plus one embedding per feature value (amount bucket,
  currency, source and destination type, subtype), mixed by a residual MLP.
- **Experts** (`placement`): `deep` routes norms, Q/K/V/O and the FFN (EVEv2/NEO default);
  `attn_ffn`, `ffn` (Mono-InternVL V1) and `shared` (no experts) are ablations. Routing is
  gather/scatter, so each token pays for one expert.
- **Positions** (`rope`): `index` uses event-index RoPE, and both tokens of a slot share an
  index. `index+time` additionally rotates half of each head by continuous time (periods 1 h
  to 2 y), so attention scores depend on time differences.
- **Heads.** The interleaving yields the marked-TPP likelihood's own factorisation:
  `p(t_{i+1}, k_{i+1} | H_i) = p(t_{i+1} | H_i) · p(k_{i+1} | t_{i+1}, H_i)`.
  - `ε_i` predicts *when* the next event happens (log-normal mixture on `log1p Δt`, the head
    the baselines share) and the marginal next mark.
  - `τ_{i+1}`, which has seen that time but not that event, predicts its mark and features.
  - `USR` carries the user embedding `u`. It predicts the next event measured from "now",
    the multi-hot of marks and the log-count in the next 30 days, and the label (`cls_head`).

The single-token control `layout=additive` adds the time embedding onto the event embedding.
That is how every baseline in this repo encodes time, at equal width and depth.

Default: hidden 256, 6 layers, 8 heads, SwiGLU 704, dropout 0.1, 10.05 M parameters.

## 3. Objectives

| stage | data | losses |
|---|---|---|
| pretrain | unlabeled store: the MBD train pool, ~529k clients, or the 30,665 labeled train clients for the matched arm | dense time NLL + marginal mark CE at every `ε_i`; time-conditioned mark/feature CE at every `τ_i`; next event at `USR`; 30-day future window at `USR` |
| finetune `cls` | the 30,665 labeled train clients | label CE on `u` |
| finetune `tpp` | same clients, random cuts | pretraining TPP terms |
| finetune `multi` | same clients; half full histories (label), half random cuts | label CE + TPP terms: **one model, one `u`, both tasks** |
| probe | frozen `u` | logistic regression (C chosen on validation) for the label; pretrained heads unchanged (zero-shot) for TPP |

Pretraining and TPP views cut each history at a random point. With probability 0.5, "now"
falls uniformly inside the gap before the next event, so `USR` learns time-to-next under
elapsed silence.

The label only attaches to full-history views. The future-window target is used only when
`now + 30 d` is at or before the observation cutoff (2022-09-30 for MBD), so it is never
censored.

## 4. Protocol (parity with the benchmark)

- **Test examples are the benchmark's own.**
  - Classification uses all 40,000 test histories, at natural prevalence (1.14 %).
  - TPP replays `NextEventJsonlDataset`: prefixes of at least 4 events, at most 8 per user,
    RNG seed = seed + 1. Checked on the real MBD test split: 320,000 views, **0 mismatches**
    in user, prefix, next mark or next gap.
  - Both are scored with `eventfm.training.metrics`. Results are written as benchmark
    `RunResult` JSON files to `KBC_OUTPUT_ROOT/paper_runs/etam/results/`.
- **Seeds** 13, 29 and 43, as in `scale-datasets`. The TPP test prefixes therefore match the
  baselines' seed by seed.
- **Selection is on validation only** (40,000 validation histories; never test).
  - `cls`: validation AUC.
  - `tpp`: validation joint NLL at `USR`.
  - `multi`: validation AUC + validation next-type accuracy.
  - The baseline grid trains for a fixed number of epochs with no selection. That is a
    difference in the baselines' disfavour, and it is reported.
- **Known input differences**, reported alongside every result:
  - ETAM reads up to 128 events; the baselines read 64. A 64-event ETAM arm controls this.
  - ETAM's absolute time includes month and day count; the baselines' calendar features stop
    at day-of-month.
  - ETAM's first window event takes its gap from its true predecessor.
  - All three are information already present in the timestamps every method receives.
- **Pool fairness.** The pool is every MBD train-split client (hash split, so no validation
  or test client), with labels withheld.
  - The 30,665 benchmark clients must come out event-for-event identical to `train.jsonl`
    (checked by `scripts/build_mbd_pool.py`).
  - The strongest self-supervised baselines are rerun with the same pool (`mbd-pool-baselines`
    suite: PRAGMA-MLM, Transaction MLM, CoLES and the autoregressive Transformer; one
    pretraining epoch, then the benchmark recipe).

## 5. Arms

Reference to beat (benchmark at n = 30,665, 3-seed means): classification AUC **0.731**
(`pragma-mlm`); TPP next-type accuracy **0.586** (5-way tie), macro-F1 **0.207**, time
RMSE **1.526**, time MAE **1.149** (log1p seconds).

| id | arm | question |
|---|---|---|
| M1 | pool pretrain → `multi` | Headline: one model and one embedding, best on both tasks? |
| M2 | pool pretrain → `cls`, → `tpp` | What does sharing cost each task, compared with specialist fine-tunes? |
| M3 | pool pretrain → probe | Is the frozen `u` already a universal embedding? |
| M4 | matched pretrain (30,665) → `multi`/`cls`/`tpp`; scratch `multi` | Architecture gain at the baselines' data budget |
| A1 | `placement` ∈ {shared, ffn, attn_ffn, deep} | Do modality experts help, and which parts? |
| A2 | `layout=additive` | Time as its own token vs time added to the event (the central claim) |
| A3 | time content: relative only / absolute only / both | The user's absolute + relative design |
| A4 | `rope=index+time` | Relative-time attention |
| A5 | `max_events=64` | Context-length control vs the baselines |
| B1 | `mbd-pool-baselines` | Baselines given the same unlabeled pool |
| S1 | model scale XS (128/2) · S (192/4) · M (256/6) · L (512/8) · XL (768/12), 10 pool epochs each | Does quality grow with parameters? (XS has the baselines' width and depth) |
| S2 | pretraining data: 30,665 · 10 % · 30 % · 100 % of the pool, at a fixed 20,670 steps | Does it grow with unique unlabeled data at equal compute? |
| H1 | pretraining HPO: 8 random configurations (LR, batch, dropout, WD, cut probability, loss weights), each scored by a frozen probe on validation; the winner is retrained for 20 epochs | Pretraining recipe |
| H2 | fine-tuning HPO: 24 random multitask configurations from the M20 checkpoint, validation only (`test_eval=false`); the winner is confirmed with 3 seeds on both the M20 and H1 checkpoints | Fine-tuning recipe, with no test peeking |
| L1 | frozen Qwen3-0.6B-Base with time/event LoRA experts r = 64 (RAM-faithful); 2 pool epochs | Does a pretrained LLM backbone add anything here? |

**Pass criteria, pre-registered:**
- The headline holds only if M1 beats the best baseline on classification AUC *and* on
  next-type accuracy, each by more than the seed spread (3 seeds), and also beats B1 on both.
- A2 supports "time as a modality" only if the token layout beats the additive layout on both
  tasks at equal width and depth.
- A1 keeps experts only if `deep` or `ffn` beats `shared` by more than the seed spread.

## 6. Review findings and open caveats (independent review, 2026-10-29)

The review found no leakage into any input. Its TPP replay check on real test users also
found 0 mismatches.

Fixed before phase B:
- **Short-context control.** With `max_events=64`, age and the first gap now come from
  inside the window (`window_time=true`), as in the baselines' collator. Before this fix
  they leaked the span of the full 128-event history.
- **Layout ablation confound.** The additive layout has one expert and no time-conditioned
  objective, so it is now also compared with `tokshared_nocond`: token layout, shared
  weights, no `cond` loss.
- **Result files.** Each run writes under its own variant, regime follows
  `freeze_backbone`, and a cell file written by another run raises instead of being
  overwritten.
- **Resume.** Resuming from a `last.pt` with a different run config raises.
- **Pool builder.** Nulls are factorised explicitly; this did not affect MBD.
- **Future-window target.** It is disabled whenever the dataset has no known observation
  cutoff.
- **Final-step scores.** Every fine-tune also reports test scores at its final step
  (`final_step`). *Correction (2026-10-30):* with early stopping (patience 3) the final step
  is up to 3 non-improving epochs past the best. It is therefore a pessimistic bound, not the
  baselines' fixed-schedule protocol. Like-for-like needs the baselines rerun with the same
  validation selection.

Open, to read alongside the results:
- **Label window may overlap the history.** `MBD_LABEL_MONTHS` includes `2022-09-30`, and
  history runs to that date. If the September target covers September activity, the label
  overlaps the last month of history for *every* method. ETAM's month and day-count
  features isolate September events more easily than the baselines' calendar can. The
  `relonly` ablation (no absolute time) bounds how much of any classification gain this
  could explain.
- **Cutoff bound on the TPP target.** Every TPP target satisfies next ≤ cutoff, and only
  ETAM sees absolute day counts. The reviewer measured the effect on top of the prefix's
  mean gap: RMSE 1.572 → 1.568 for a simple predictor, which is negligible.
  `relonly` covers this too.
- **Unequal selection.** ETAM selects checkpoints and HPO configurations on validation;
  the baselines do not. Both the selected and the final-step numbers are reported.

## 7. Code map

| file | role |
|---|---|
| `eventfm/etam/store.py` | columnar store from JSONL; vocabulary from `meta.json` |
| `eventfm/etam/batching.py` | views, collator (inputs and targets), benchmark-replay view generators |
| `eventfm/etam/model.py` | tokenizers, routed MoT layers, rotary, heads, losses |
| `eventfm/etam/train.py` | pretrain / finetune / probe, evaluation, result files |
| `scripts/etam.py` | `build-store`, `run --config … key=value` |
| `scripts/build_mbd_pool.py` | raw MBD → pool JSONL + store, with the parity check |
| `eventfm/etam/llm_backbone.py` | L1: frozen LM mixer with token-conditional LoRA experts; checkpoints keep only trainable weights |
| `scripts/etam_hpo.py` | HPO search spaces, trial sampling, validation-only selection |
| `scripts/cell_result.py` | benchmark cell JSON → campaign `result.json` (fails on a failed cell) |
| `tests/test_etam.py` | calendar, store, benchmark replay, causality/leakage, routing, end-to-end (incl. an HPO trial) |
| `docs/etam/campaigns/*.yaml` | cluster campaigns (slurm-submit driver) |
