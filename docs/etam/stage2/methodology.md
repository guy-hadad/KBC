# ETAM Stage 2 methodology

Snapshot: **2026-11-10 (Asia/Jerusalem)**. ETAM-2 is implemented in
`eventfm/etam/stage2/` and runs through `scripts/etam2.py`. The campaign is incomplete;
recorded task states and every available numeric result are in [results.md](results.md).
Read [positives.md](positives.md), [negatives.md](negatives.md) and
[analysis.md](analysis.md) for the supported findings and their limits.
Stage 1's design is in [../methodology.md](../methodology.md), and its original results
are in [../results.md](../results.md).

The sections below preserve the implemented protocol and its pre-registered decisions.
A scheduled arm is not evidence that its run completed. Equal steps and sequence exposure
are the intended controls; FLOPs, parameter counts and elapsed training costs differ.

## II.1 Questions

Stage 1 showed that one pretrained ETAM-M embedding beats the strongest baselines on both
MBD tasks. It left four open issues:
- the classification gain rides on absolute time, while the label window (Sep 2022 – Jan
  2023) overlaps the last month of history;
- pretraining compute and checkpoint selection were not matched to the baselines;
- every pretrained arm had a single pretraining seed;
- the data-scaling curve was non-monotonic.

Stage 2 asks five questions:
1. **P0.** Do the Stage 1 gains survive a fair comparison, with strictly prospective
   histories, matched compute, matched selection and several pretraining seeds?
2. **P1.** Do a predictive temporal encoding (A) and a joint event–time decoder (D) improve
   the time representation and the conditional likelihood?
3. **P2.** Does selective, lag-aware interaction between private event and time pathways
   (C) beat simpler fusion at reported parameters and compute?
4. **P3.** Do multi-event prediction (E), a predictive-information objective (F) and the new
   pretraining terms (G) improve transfer, long-horizon prediction and scaling?
5. **H.** Is the pretrained representation label-efficient under parameter-efficient
   adaptation?

## II.2 Model

```
history (t_j, k_j, a_j), j <= n; query "now" >= t_n
z^T_i = RMSNorm(W_c c(t_i) + W_r r(gap_i, age_i) + W_m m(t_i^-))             time token   (A)
z^E_i = RMSNorm(E_k + sum_f E_f(a_f) + MLP_attr([a_f]_f, amount residual))     event token  (B)
h^E = F_E(z^E), h^T = F_T(z^T): two 6-layer causal transformers (private pathways)
bridges after layers 2 and 4 (both directions):
   h^E_i += g_i * W_o sum_{j<=i} softmax(<q_i,k_j>/sqrt(d) + b(t_i - t_j, k_i)) v_j       (C)
h^J = RMSNorm(g * h^E + (1-g) * h^T);  u = W_u [u_E; u_T; u_J] at the query slot
decoders on h^J: time-first f_A(y|H) pi_A(k|y,H) and event-first pi_B(k|H) f_B(y|k,H);
inference p = 1/2 p_A + 1/2 p_B                                                          (D)
```

Here y = log1p(gap in seconds), as in Stage 1 and the baselines.

- **Calendar `c`.**
  - Hour, weekday, day-of-month and month embeddings, plus their cyclic coordinates.
  - Fourier features of the absolute day count, with periods initialised at 1 h, 1 d, 7 d,
    30.44 d, 91.31 d and 365.25 d, plus a linear trend.
  - `calendar=learned` lets each period move by at most ±25 %
    (P = P₀ · exp(0.25 tanh θ)), so learned periodicity cannot drift to an arbitrary
    timestamp scale.
- **Relative `r`** (`bins_residual`).
  - 64 adaptive quantile bins of y_gap, with edges taken from the pretraining store.
  - Plus a residual term f_θ(y − μ_b), with μ_b the bin centre. Log age gets the same
    treatment.
  - `bins` drops the residual. `stage1` is the Stage 1 encoder (Fourier features plus linear
    log buckets).
- **Temporal memory `m`** (`exp`).
  - m_r(t_i⁻) = Σ_{j<i} exp(−(t_i − t_j)/τ_r) · V_r z^E_j, for τ ∈ {1 min, 10 min, 1 h, 6 h,
    1 d, 7 d, 30 d, 180 d}.
  - Each state is divided by its decayed event count; log1p of that count is also an input.
  - It is computed exactly and in parallel (a B × R × n × n decay tensor). The memory at slot
    i only uses events j < i.
  - Alternatives:
    - `exp_learned`: learned timescales.
    - `count`: decayed counts only, an event-agnostic Hawkes-like state.
    - `ssm`: a learned complex-diagonal continuous-time state-space model with 16
      oscillating modes and 8 channels.
    - `ode`: an ODE-RNN, with GRU jumps at events and an RK4 flow between them on the clock
      log1p(Δ / 60 s).
- **Event token.**
  - Mark and field embeddings, plus an attribute MLP over small per-field embeddings.
  - The amount is also used as a continuous feature: signed log1p(amount) minus the mean of
    its amount bucket (a quantile-plus-residual encoding).
  - The raw amounts were recovered from the MBD parquet files and match every store event
    for event (0 mismatches across 62 M events).
- **Bridges.**
  - Queries come from one pathway, keys and values from the other, with causal attention
    (j ≤ i).
  - Per-head lag bias b = Σ_r w_{h,r}(k_i) φ_r(log1p lag): 16 RBF bases, weights per query
    mark, zero-initialised.
  - A vector gate σ(W_g [h^E; h^T; φ(gap_i)]).
- **Middle-architecture controls** (`mixer`):
  - `shared`: one transformer over z^E + z^T.
  - `stage1`: Stage 1's interleaved two-token layout with deep experts.
  - `private` with no bridges: two pathways and gated fusion.
  - `private_shared`: adds a shared stream that reads both pathways at every layer.
  - Bridges after every layer.
- **Decoders.**
  - Four kinds: `independent` = f(y|H) π(k|H), `time_first`, `event_first` and `dual`.
    - π_A(k|y,H) = softmax(W GELU(W_h h + W_y φ(y))).
    - f_B(y|k,H) is an MLP on h + E_k.
  - Time families: `lognormal` (an 8-component Gaussian mixture on y, the Stage 1 and
    baseline head), `spline` (a conditional monotone rational-quadratic spline flow with 32
    bins) and `hazard` (a piecewise-constant hazard on adaptive y bins, with exact
    cumulative hazard, survival and sampling).
  - Marginals:
    - p_A(k|H) = ∫ π_A(k|y,H) f_A(y|H) dy is computed from exact bin masses: CDF differences
      on a 48-bin y grid, with the tails folded into the end bins and π_A evaluated at bin
      centres.
    - p_B(y|H) = Σ_k π_B(k|H) f_B(y|k,H) is computed exactly.
  - Units: every density is over y. A time NLL converts to seconds as NLL_s = NLL_y + E[y].
- **Auxiliary heads.**
  - Next-event attributes p(a | k, y, H), needed for generation.
  - Event-only and time-only heads on the private states before the first bridge.
  - Multi-event heads (E), count heads (G), the user head (H) and the representation
    projections (F).

## II.3 Objectives

L = L_joint + λ_d L_dual + λ_c L_cons + L_aux + L_feature + L_query + L_survival + L_censor + L_count
+ λ_u L_unimodal + λ_m L_MTP + λ_h L_horizon + λ_r L_repr (+ L_vq; + L_cls when fine-tuning).

Reference weights: joint 1, dual 1, cons 0.1, feature 0.5, query 1, survival 0.5, censor 1
(an on/off switch), count 0.5 and unimodal 0.25. MTP, horizon and repr are 0 in the reference; the P3 arms switch
them on.

| term | definition |
|---|---|
| L_joint | −log p_joint(y_{i+1}, k_{i+1} \| H_i) at every event slot, under the inference distribution (½p_A + ½p_B for `dual`) |
| L_dual | ½[−log p_A − log p_B], which keeps each branch a proper model on its own |
| L_cons | JS(p_A ‖ p_B) on the discretised joint (48 y bins × K marks), at 256 random positions per batch |
| L_query | joint NLL of the next event measured from "now", at the query slot |
| L_survival (G 8.2) | when "now" falls inside the gap: −log[p(s + v, k) / S(s)] from the last event's distribution, with the y-unit Jacobians (the elapsed-silence likelihood) |
| L_censor (G 8.3) | histories that reach the cutoff: −log S(T_cut − t_n \| H_n) at the last event and −log S(T_cut − now) at the query slot, T_cut = 2022-09-30 (the MBD observation end). Each censored interval enters the same per-interval mean as the observed gaps (joint, dual and query terms), so the likelihood is proper |
| L_count (G 8.4) | negative-binomial (or Poisson) log-likelihood of per-mark and total counts in (now, now + h], h ∈ {1, 7, 30} d, used only when now + h ≤ T_cut |
| L_MTP (E 6.1) | heads for events i+2 … i+L that share the mark projection. *hard*: CE on k_{i+ℓ}. *Soft-MTP* (Teoh 2026, arXiv:2610.38331): KL(sg(q) ‖ p_ℓ), with q = π_B at the aligned later prefix i+ℓ−1. See note 1 |
| L_horizon (E 6.2) | −log q_ℓ(log1p(t_{i+ℓ} − t_i) \| H_i), one head shared across horizons with a horizon embedding |
| L_repr (F) | RePR, bootstrapped form (Baade et al. 2026, arXiv:2610.20533). See note 2 |
| L_unimodal | CE and NLL of the event-only and time-only heads |
| L_vq | VQ-VAE codebook and commitment terms (`time_vq` arm only) |

1. **Soft-MTP.** Head and trunk receive only the KL gradient. It is implemented as an explicit
   KL, which differs from the paper's routing trick only in the gradient that reaches the
   shared projection.
2. **RePR.** The loss is Huber(⟨R(H), R(y_{≤k}) − R(y_{<k})⟩ − sg(PMI_k)), with
   PMI_k = log p(y_k | y_{<k}, H) − log p(y_k | y_{<k}).
   - The model itself is the generator. The first term comes from the main forward pass; the
     second from a forward pass over the last 4 events alone (history dropout).
   - There are joint, event-only and time-only versions, on u_J, u_E and u_T.

**Views and curriculum.**
- Each sequence gets one view per draw (`Views2`):
  - the full history, with p = 0.5;
  - otherwise, with p = 0.5, a *calendar-time query*: "now" is uniform in calendar time
    between the 2nd event and the observation cutoff, and the history is every event up to
    it;
  - else an *event cut*: the history ends at a uniformly chosen event, and "now" is that
    event.
- A gap then contains "now" with probability proportional to its length, as for a forecast
  requested at a random time. So p(gap | elapsed s) = p(gap)/S(s) holds exactly, and the
  survival-conditioned term is a proper likelihood. Queries after the last event are
  right-censored.
- Count targets are built only for calendar-time queries, whose "now" never depends on the
  future.
- Stage 1's views put "now" at a uniform *fraction* of the next gap, which gives
  p(gap | s) ∝ p(gap)/gap instead (§II.9).
- **Staged curriculum** (G 8.1): c0 = joint plus the data terms → c1 adds dual → c2 adds
  cons → c3 adds Soft-MTP → c4 adds horizon → c5 adds repr → c6 adds unimodal (the full
  objective). The reference model is c2 plus unimodal.

## II.4 Protocol

Shared with Stage 1 ([protocol §I.4](../methodology.md)):
- the benchmark's own test examples (40,000 histories; 320,000 TPP prefixes replayed with
  seed + 1);
- selection on validation only;
- pretraining on the 529,180-client pool;
- seeds 13, 29 and 43.

What changes:
- **Budget.** Every pretraining arm, ETAM-2 and Stage 1 alike, gets Stage 1's M10 budget.
  - 20,670 steps at an effective batch of 256. ETAM-2 runs micro-batches of 128 with 2-step
    gradient accumulation.
  - Learning rate 6e-4 with a cosine schedule and 2,000 warm-up steps.
  - Validation joint NLL at the query slot, every 4,134 steps, on 10,000 validation users.
  - So each Stage 2 arm sits in the same table as Stage 1's ablations.
- **Pretraining seeds (P0).** 3-seed arms are pretrained three times, so their sd covers
  pretraining noise, not just fine-tuning noise.
- **Strictly prospective classification (P0).** The label window starts in September 2022,
  so histories are cut before 2022-09-01.
  - Rebuilt from raw MBD: each history is the last 128 events before the cut. 63 % of users
    regain older events that the stored 128-event window had dropped.
  - Same users and labels. Users with fewer than 2 events before the cut are dropped: 40
    train, 24 validation and 14 test users. **33 of the 40 dropped train users are
    positives.**
  - TPP evaluation stays on the standard stores.
- **Matched selection (P0).** The `final8` fine-tunes train a fixed 8 epochs and report the
  last step, the baselines' protocol.
- **Matched context.** `ev64` uses 64 events, with window-relative time.
- **Baselines.**
  1. Strict labels: PRAGMA-MLM, Transaction MLM, CoLES and the AR Transformer, each with the
     benchmark recipe and pool-pretrained for 1 epoch, 3 seeds (`mbd-strict-baselines`).
  2. Matched pool compute: PRAGMA-MLM and the AR Transformer pretrained for 10 pool epochs
     (ETAM-M10's sequence exposure), both tasks, 3 seeds (`mbd-pool-baselines-matched`).
  3. S2P2, ITPP, FlexTPP and Kairos, reimplemented inside this framework with the same data,
     views and evaluation (divergences in §II.8).
- **Transfer.**
  - Datasets:
    - Synthea: synthetic EHR, 93,305 training patients, 65 marks, a prospective adverse-event
      label at 6.3 %.
    - Stack Overflow: 2,668 users, 25 badge marks, TPP only. Both use the official splits.
  - ETAM-2 and Stage 1 are each pretrained from scratch on the training split, with 3 seeds:
    7,290 steps for Synthea, 912 for Stack Overflow.
- **Compute.** The GPU quota (5 at a time) was shared with other campaigns.
  - ETAM-2 runs are therefore bundled three per 48 GB GPU (`scripts/etam2_bundle.py`). A run
    that runs out of memory is retried alone.
  - On these CPUs (AMD EPYC 7763, AVX2 only), ETAM-M trains at about a minute per step, so CPU
    jobs only score checkpoints, fit probes and run the small JSONL baselines.

## II.5 Evaluation battery

Every pretraining arm is scored in the same job, on validation (10,000 users) and on the full
test set. Scores are read at the query slot, with "now" at the prefix's last event, on the
benchmark prefixes.

- **Benchmark metrics.** Next-type accuracy, macro-F1, and time RMSE / MAE (the mean of the
  marginal time distribution, in log1p seconds).
- **Likelihoods.** Joint NLL −log p(y,k|H), time NLL −log p(y|H) and type NLL −log p(k|H),
  where p(k|H) is marginalised over time (by quadrature for time-first decoders).
- **Calibration.**
  - PIT: KS statistic and the probabilistic calibration error (PCE, Bosser & Ben Taieb
    2023, 50 levels).
  - Coverage of the central 50 / 80 / 90 % intervals.
  - Survival calibration: ECE and Brier score of P(gap > h), h = 1 h, 1 d, 7 d.
  - Top-1 ECE of the mark distribution.
- **Event–time dependency.**
  - E[PMI] = E[log p(k|y,H) − log p(k|H)], overall and for the 10 most frequent marks. By
    Bayes, this equals E[log p(y|k,H) − log p(y|H)].
  - Oracle-time accuracy, argmax p(k | y_true, H). It uses the true next time, so it is a
    diagnostic, not a forecast.
  - Accuracy of the cheap direct branch, π_B (or π_aux for time-first).
- **Unimodal versus joint.** Type NLL and time NLL of the event-only and time-only heads,
  compared with the joint decoder at the same (last-event) slot.
- **Multi-event prediction.** Accuracy on k_{n+ℓ}, ℓ = 2..4, and the NLL of the cumulative
  time.
- **Strata** (the data-scaling question). Joint, time and type NLL and accuracy by user
  activity tercile, calendar month of the target, and target-mark frequency tercile.
- **Elapsed silence.**
  - One seeded calendar-time query per user, strictly inside an observed gap. It compares
    the query slot's direct forecast with the survival-conditioned forecast from the last
    event.
  - Both are scored in residual-time units.
- **Counts.**
  - At a fixed cut, T_cut − 30 d, the NB or Poisson count heads predict counts for 1 / 7 /
    30 d.
  - Scored by CRPS, log score, 80 % coverage and the Brier score of P(N = 0), against a
    Poisson at the history rate.
- **Frozen probes.**
  - Logistic regression (C chosen on validation) on u and u_J, for the label, on standard
    and strict histories.
  - Reported on test: AUC, AP, P@K and R@K (K = 100 … 2,000), and prior-corrected ECE and
    Brier.
- **Efficiency.** Parameters, forward GFLOPs per sequence (`torch.utils.flop_counter`), and
  latency, timed for all checkpoints in one job on one RTX 4090.

**Downstream evaluation.**
- **Fine-tunes:**
  - `multi`, `cls` and `tpp` fine-tunes, standard and strict;
  - `final8`;
  - multitask with separate learning rates (heads 5×, encoders 0.3×) and gradient-conflict
    logging (cosine between the classification and TPP gradients on shared parameters,
    every 50 steps);
  - TPP adaptation (heads only, or LoRA rank 8; frozen = the zero-shot scores).
- **Label efficiency (strict labels):**
  - LoRA and full fine-tuning at 1 / 5 / 25 % of labels;
  - linear and MLP probes at 1 / 5 / 10 / 25 / 100 % (stratified subsamples, 3 subsample
    seeds).
- **Frozen cross-task transfer:** probes for the three other MBD product targets (credit,
  salary project, acquiring).
- **Unimodal concatenation:** separately trained event-only and time-only models,
  concatenated, against u_J of one joint model.
- **Long-horizon simulation:**
  - 4,000 test users × 16 rollouts over 30 days from T_cut − 30 d.
  - Scored on count CRPS and coverage, the JS divergence of the mark mix, and the
    W1 / KS distance of the log gaps.
  - Compared against the history-rate Poisson.

## II.6 Arms

Each arm changes one field of the reference configuration (`Etam2Config()` defaults). Arms
marked ×3 have 3 pretraining seeds; all others are single-seed screens.

| phase | arm | change against the reference | question |
|---|---|---|---|
| P0 | `ref` ×3 | – | ETAM-2 reference |
| P0 | `s1` M10 ×3 | Stage 1 code: phase B seed 13, plus new seeds 29 and 43 | Stage 1 under pretraining-seed noise |
| P0 | `ev64` | 64 events, window time | context control |
| A | `t_stage1` ×3, `t_nomem` ×3 | Stage 1 time encoder, no memory / no memory | is the new temporal encoding better? |
| A | `t_bins`, `t_mem_learned`, `t_mem_count`, `t_mem_ssm`, `t_mem_ode`, `t_cal_fixed`, `t_relonly`, `t_vq` | bins without residual / learned τ / event-agnostic memory / continuous-time SSM / neural ODE / fixed periods / no absolute time / VQ-256 time tokens | which temporal component matters |
| D | `d_indep` ×3, `d_time_first` ×3, `d_event_first` ×3 | single decoders | does the dual decoder help? |
| D | `d_nocons`, `d_cons1`, `d_spline`, `d_hazard` | JS weight 0 / 1; spline flow; survival decoder | consistency strength; time likelihood family |
| G | `g_nosurv`, `g_nocensor`, `g_poisson`, `g_count_stage1` | no elapsed-silence term / no censoring / Poisson counts / Stage 1 30-day target | the pretraining-strategy terms |
| C | `a_shared` ×3, `a_stage1` ×3, `a_private` ×3, `a_private_shared` ×3, `a_bridges_all` ×3 | the middle architectures of §II.2 | is selective interaction needed? |
| C | `a_shared_wide`, `a_nobias`, `a_lagonly`, `a_nogate` | parameter-matched shared model (hidden 368) / no lag bias / lag bias without mark / no gate | bridge mechanisms |
| C | `b_s2p2` ×3, `b_itpp` ×3, `b_flextpp` ×3, `b_kairos` ×3 | external baselines (§II.8) | the challenges of the proposal |
| E | `m_hard4` ×3, `m_soft4` ×3, `m_soft2`, `m_soft8`, `m_soft4_time` | hard MTP or Soft-MTP with L = 2 / 4 / 8; plus time heads | multi-event prediction |
| F | `r_all` ×3, `r_joint`, `r_time` | RePR on u_J, u_E, u_T / only u_J / only u_T (a predictive-information-trained temporal encoder) | predictive representation |
| F | `u_event_only`, `u_time_only` | inputs restricted to marks and attributes / to timestamps | does a joint u beat event ⊕ time? |
| G 8.1 | `c0` … `c5`, `c6` ×3 | cumulative curriculum | attribution of each objective |
| G 8.5 | `scaleA_{matched,pool10,pool30}` ×3, `scaleB_*` ×3 | 30,665 / 52,918 / 158,754 users; A: fixed 20,670 steps; B: 10 epochs per user | data scaling (100 % = `ref`) |
| G 8.5 | Stage 1 `pool10`, `pool30` ×3 | Stage 1 code at 10 % and 30 % of the pool | is Stage 1's non-monotonic result seed noise? |

## II.7 Pre-registered decision rules

- **Credit.** A mechanism is credited only when its arm differs from the reference in that
  mechanism alone and the difference exceeds the larger of the two arms' 3-seed sd.
  Single-seed arms are screens and are labelled as such.
- **C.** The bridged design is retained only if it beats every simpler middle architecture
  (shared, Stage 1 MoT, private with gating, private with a shared expert) and bridges at
  every layer on validation joint NLL, without losing on frozen-probe AUC. Parameters,
  FLOPs and latency are reported next to it.
- **D.** The dual decoder is retained only if it beats independent heads on test joint NLL
  without losing next-type accuracy. A time family is retained only if it improves joint
  NLL and PIT calibration over the log-normal mixture.
- **E/F.** An objective is retained only if it improves its own target (multi-event
  accuracy; probe AUC) without hurting next-event joint NLL beyond the spread.
- **P0.** The Stage 1 classification claim survives only if an ETAM model beats the best
  strict-label baseline on strict test AUC by more than the spread.

## II.8 Baseline reimplementations: divergences from the papers

All four reuse this framework's data, views, evaluation and (for S2P2 and ITPP) decoders.
Parameters at hidden 256: S2P2 1.42 M, ITPP 1.42 M, FlexTPP 5.17 M, Kairos 4.98 M; the ETAM-2
reference has 12.6 M.

- **S2P2** (Chang et al., NeurIPS 2025).
  - The latent-linear-Hawkes stack is kept: 4 layers, 64 complex states, input-dependent
    rates, backward zero-order hold.
  - Divergences:
    - The impulse is Linear(z^E + z^T), not a mark embedding.
    - It has no intensity head or Monte-Carlo compensator; the ETAM-2 decoder sits on its
      states.
    - Δ is measured in hours. Re Λ is initialised to −1/τ over 10 min – 180 d, in the style
      of S4D-Lin, not HiPPO-LegS.
    - It uses post-norm GELU, not the official pre-norm GLU.
- **ITPP** (Zhou et al., AAAI 2026).
  - Kept: one latent per event type, with a GRU-ODE drift and inverted attention over the
    type tokens.
  - Divergences:
    - The drift is integrated with 2 Euler steps on the clock log1p(Δ / 60 s), not an
      adaptive solver.
    - The update-gate bias starts at +2 for stability.
    - The jump input is a projection of z^E (the original takes no input).
    - It has no intensity head or compensator; its type tokens are pooled into h for the
      ETAM-2 decoder.
- **FlexTPP** (Draxler et al., NeurIPS 2025).
  - Kept: interleaved scalar (time, type) tokens with a spline time head.
  - Divergences:
    - No attributes.
    - No beginning-of-sequence token, so τ₁ is never predicted.
    - GELU feed-forward layers.
    - This repo's 32-bin spline on y.
    - Sized 256 / 6 / 8.
- **Kairos** (Gupta et al., ACML 2025).
  - Kept: one shared vocabulary of mark tokens and 256 equal-width time-bin tokens.
  - Divergences:
    - Bins are floored, with an overflow bin, and τ_max is the 99.5 % gap quantile.
    - The softmax is restricted to the kind of token being predicted.
    - The density is uniform in y within each bin (the first bin spans y ≈ 0 – 9.2).
    - Sized 256 / 6 / 8, against the paper's 16-dimensional, 2-layer model.
- **FlexTPP and Kairos.**
  - Both have no "now" token, so their query term is off.
  - Their "direct" elapsed-silence forecast is just the last event's distribution.

## II.9 Independent review before any GPU run (2026-11-08)

An independent correctness review of the Stage 2 code ran before any pretraining job had
started. Its main findings were these, and all are fixed in the code that every campaign
job runs (the frozen snapshots were refreshed before the first job started):

1. **"Now" was drawn as a uniform fraction of the next gap (Stage 1's sampler).**
   - It made the elapsed-silence term improper: on exponential gaps it recovers 2λ.
   - It taught the query slot residuals that are too short for calendar-time queries: on
     the train store the median residual was 14 h under the old scheme, against 40 h at a
     uniform calendar time.
   - It favoured the direct forecast in the elapsed-silence evaluation by construction.
   - Fixed by the calendar-time queries of §II.3.
2. **The `u_event_only` and `u_time_only` arms leaked the excluded modality.** The leak ran
   through the bridges' lag bias and gate (time) and through the bias marks (marks).
   - Fixed. Tests now check that stretching timestamps leaves an event-only model unchanged,
     and that relabelling marks leaves a time-only model unchanged.
3. **Count targets were zero-inflated by full views**, whose "now" is the last event.
   - Fixed: counts are built only at calendar-time queries.
4. **The simulation fed the model inputs it never saw in training.** Real amounts were
   dropped, and the attribute head was untrained at the query slot.
   - Fixed: real amounts are carried through the rollout, generated events get their
     bucket's mean amount, and the attribute loss also covers the query slot.
5. **Stage 1 checkpoints got a different adaptation protocol.** LoRA wrapped their heads,
   and the classification head stayed frozen at random initialisation.
   - Fixed: LoRA is applied only to the mixer, and the heads train, for both stages.
6. **Censored intervals were weighted unlike observed ones.**
   - Fixed: they now enter the same per-interval likelihood (§II.3).
7. **Minor items:**
   - `select=final` reported the best step's validation metrics;
   - the latency job could crash and timed every checkpoint at 128 events;
   - Synthea's 65 mark probabilities were stored in fp16;
   - Stage 1 seeds 29 and 43 were scored on seed 13's test prefixes;
   - Stage 1 re-scoring used 5,000 validation users instead of 10,000.

   All are fixed.

The review also checked these and found them correct:
- causality of every mixer, memory and baseline (including the KV cache);
- target alignment (MTP, counts, censoring, RePR, the Soft-MTP teacher);
- that every density and joint is proper, including the elapsed-silence Jacobians;
- the loss formulas, the evaluation semantics, gradient accumulation, resume and the bundle
  runner.

## II.10 Code map (Stage 2)

| file | role |
|---|---|
| `eventfm/etam/stage2/config.py` | `Etam2Config` (every mechanism is a switch) and `Etam2Weights` |
| `eventfm/etam/stage2/tokenizers.py` | calendar, relative bins with residual, temporal memories (exp / learned / count / SSM / ODE), VQ, event tokenizer |
| `eventfm/etam/stage2/backbone.py` | shared, Stage 1 MoT, private pathways, bridges (lag bias, gates), fusion |
| `eventfm/etam/stage2/distributions.py` | log-normal mixture, spline flow, piecewise hazard, discrete bins; independent / time-first / event-first / mixture joints; JS |
| `eventfm/etam/stage2/heads.py` | decoders, feature, multi-event, count, user and RePR heads |
| `eventfm/etam/stage2/losses.py` | every objective of §II.3 |
| `eventfm/etam/stage2/batching.py` | Stage 2 targets: multi-step, censoring, elapsed silence, counts, amounts, RePR sub-batch |
| `eventfm/etam/stage2/evaluate.py` | the §II.5 battery and the Stage 1 adapter |
| `eventfm/etam/stage2/simulate.py` | long-horizon rollouts and sequence statistics |
| `eventfm/etam/stage2/train.py` | pretrain / finetune (full, LoRA, heads; label fractions; LR groups; gradient conflict) / evaluate / simulate |
| `eventfm/etam/stage2/baselines.py` | S2P2, ITPP, FlexTPP, Kairos |
| `eventfm/etam/stage2/metrics.py` | calibration, ranking, count and distribution metrics |
| `scripts/etam2.py`, `etam2_bundle.py`, `etam2_probe.py`, `etam2_latency.py` | CLI, GPU bundling, frozen probes, efficiency |
| `scripts/build_mbd_stage2_data.py` | raw amounts aligned to the stores; strict stores and JSONL; all four MBD targets |
| `scripts/etam2_public_stores.py` | Synthea and Stack Overflow stores |
| `docs/etam/campaigns/make_stage2.py` | campaign generator (`etam2_mbd_20261108.yaml`) |
| `tests/test_etam2*.py` | proper densities, coherent joints, causality of every mixer and memory, collator targets, baselines, metrics |
