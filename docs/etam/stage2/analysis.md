# ETAM Stage 2 analysis

Snapshot: **2026-11-10 (Asia/Jerusalem)**.
[Results and complete metrics](results.md) · [Methodology](methodology.md) ·
[Positives](positives.md) · [Negatives](negatives.md)

The clearest supported improvement is the dual event–time decoder within the
Stage 2 framework. The completed results do not establish ETAM-2 as a better
replacement for Stage 1 across prediction, calibration and representation
quality. The proposed sparse-bridge architecture also fails its full retention
rule. Those conclusions can coexist because the decoder ablation and the
comparison between stages answer different questions.

## What the completed evidence decides

| question | evidence | decision at this snapshot |
|---|---|---|
| P0: strict classification | reference AUC 0.7204 ± 0.0059; best strict baseline 0.7151 ± 0.0002 | reference gap is within spread; other arms are promising, and matched-selection ETAM runs are absent |
| D: dual versus independent decoder | joint NLL improves by 0.1512; accuracy by 0.0018 | passes the planned decoder rule |
| C: selective bridges versus every alternative | private/shared stream ties; all-layer bridges have better validation NLL | sparse-bridge retention rule fails |
| A: new time representation and memory | Stage 1-time and no-memory controls tie reference NLL | no demonstrated gain from the proposed package |
| E: Soft-MTP | next-event NLL worsens by 0.0097; probe AUC ties | does not pass the objective rule |
| F, G scaling, H adaptation | required arms have no results | unresolved |
| G: count forecasting | head CRPS beats history Poisson; nominal 80% intervals over-cover | useful predictor; mechanism attribution and calibration remain open |
| transfer and efficiency | only Stage 1 Synthea completed; common-device latency absent | no Stage 2 transfer or controlled speed claim |

“Beyond the spread” compares the absolute difference with the larger sample
sd. It is a pre-registered screening rule, not a hypothesis-test result.

## Better joint likelihood can accompany worse marginals

For the same examples,

`joint NLL = time NLL + type NLL − mean PMI`.

For the reference the recorded means are approximately
**1.4874 + 1.1239 − 0.1019 = 2.5094**. Stage 1 M10 has independent heads,
so mean PMI is effectively zero, and **1.4274 + 1.1166 = 2.5440**.
The Stage 2 dependency contribution outweighs its worse time and type
marginals. Its oracle-time mark accuracy, **0.6432**, is much higher than
its marginal forecast accuracy, **0.5962**, which is consistent with learned
event–time dependence. The oracle number cannot be used as forecast accuracy.

This identity explains how the reported likelihoods fit together. It does
not explain the large time-mean error: reference RMSE is **2.357**, versus
Stage 1 M10's **1.492**. The recorded values should be retained while the
decoder's marginal means, tails, numerical calculations and calibration
are investigated. The current evidence does not identify a cause.

## Separate the useful interaction result from the bridge-placement claim

The reference beats shared, Stage 1-layout and unbridged-private controls on
validation NLL beyond their spreads. That supports cross-pathway interaction
relative to those particular controls. It does not establish the proposed
bridge placement as optimal: all-layer bridges score **2.5031 ± 0.0047** on
validation, versus **2.5127 ± 0.0072**, and the private/shared-stream model
ties at **2.5114 ± 0.0041**.

The parameter-matched shared control and the bias/gate ablations are needed
before crediting lag-aware sparse bridges. The existing shared model uses
less than half the reference parameter count. The `a_stage1` layout control
uses Stage 2's other components and training terms; it is not the original
Stage 1 checkpoint (`phaseB_M10`).

## Strict classification narrows the claim

The strongest completed strict baseline is pool PRAGMA-MLM, AUC **0.7151**.
Reference strict frozen-probe AUC is **0.7204**, but its seed spread is
**0.0059**. Hard-MTP (**0.7259 ± 0.0039**) and the Stage 1-time control
(**0.7254 ± 0.0015**) exceed the baseline spread screen. Neither establishes
its own mechanism as better than the reference beyond the reference spread.

These are frozen probes; the baseline scores come from its supervised
benchmark recipe. The planned ETAM strict fine-tunes and final-eight-epoch
controls will answer the intended adaptation and selection questions more
directly. An arm chosen after looking at this test set should be confirmed
on a fresh evaluation before becoming the new headline.

Strict inputs move every classification history before September 2022 and
restore older events from raw data. The resulting change in AUC combines
calendar cutoff, recency, context and a small cohort change. It cannot by
itself separate seasonality from label-window overlap. Stage 1 M10's strict
probe AUC (**0.7252**, one pretraining seed) remains a relevant comparator.

## Count prediction is useful without settling the pretraining hypotheses

Reference 30-day head CRPS is **2.3568**, versus **5.0740** for a
history-rate Poisson. Its 80% coverage is **0.9234**, so better probabilistic
scores do not imply well-calibrated intervals. The Poisson-head, no-survival,
no-censoring and Stage 1-count-target ablations are needed to attribute the
gain. No completed simulation measures whether it survives autoregressive
rollouts over 30 days.

Likewise, a direct elapsed-silence joint NLL of **2.3081**, versus **2.4413**
after survival conditioning, compares two fitted forecasts. It is not an
ablation of the survival loss. The no-survival arm is still unavailable.

## What remains needed to settle the open questions

1. Complete Stage 1 pretraining seeds and the strict/final-eight-epoch
   adaptations, plus the missing matched-exposure PRAGMA baseline cells.
2. Check the time-marginal and point-error discrepancy, preserving the
   current recorded metrics and separately reporting any corrected scoring.
3. Complete parameter-matched shared, bridge-bias and gate controls before
   retaining a particular interaction mechanism.
4. Complete RePR, curriculum, data scaling, external baselines and the
   label-efficiency studies before claiming improved foundation representations.
5. Resolve the Stage 1 attribute-free Stack Overflow failure and obtain
   ETAM-2 transfer, simulation and common-device latency results.

These are outstanding experiments and checks, not work performed for this
documentation snapshot. Recorded task states, complete metrics and failed
versus missing outcomes are available in [results.md](results.md).
