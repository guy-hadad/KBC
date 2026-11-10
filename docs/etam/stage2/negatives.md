# ETAM Stage 2: negative findings and limitations

Snapshot: **2026-11-10 (Asia/Jerusalem)**.
[Results and complete metrics](results.md) · [Methodology](methodology.md) ·
[Positives](positives.md) · [Analysis](analysis.md)

## Stage 2 does not yet improve the full Stage 1 metric set

The reference improves test joint NLL over the single Stage 1 M10 checkpoint
re-evaluated with the same battery, but its time marginal and point predictions
are worse:

| test metric | Stage 2 reference, 3 pretraining seeds | Stage 1 M10, seed 13 |
|---|---:|---:|
| joint NLL | 2.5094 ± 0.0045 | 2.5440 |
| time NLL | 1.4874 ± 0.0073 | 1.4274 |
| next-type accuracy | 0.5962 ± 0.0009 | 0.5979 |
| time RMSE | 2.357 ± 0.029 | 1.492 |
| time MAE | 1.541 ± 0.008 | 1.121 |
| PIT KS | 0.0448 ± 0.0034 | 0.0055 |
| standard frozen-probe AUC | 0.7342 ± 0.0008 | 0.7392 |
| strict frozen-probe AUC | 0.7204 ± 0.0059 | 0.7252 |

Lower is better for the NLLs, errors and PIT KS. Time errors use log1p seconds.
The missing Stage 1 pretraining seeds prevent a completed three-seed stage
comparison. The discrepancy between joint likelihood and time point error
needs investigation before attributing it to a particular component.

The Stage 1 HPO checkpoint has joint NLL **2.4787**, accuracy **0.6022**, and
RMSE **1.487**. It is a stronger observed comparator, but has a different
recipe, a larger training/tuning budget, and one pretraining seed.

## The reference does not pass the strict-classification spread rule

Strict frozen-probe AUC is **0.7204 ± 0.0059**, compared with
**0.7151 ± 0.0002** for pool PRAGMA-MLM. The **0.0053** gap is smaller than
the reference sd. Standard probe AUC also falls from **0.7342** to **0.7204**
when histories become strict. This sensitivity is not evidence by itself of
label leakage: the cutoff changes history length, recency and the cohort.

Some other Stage 2 arms exceed the strict baseline's spread, but they do not
establish a reference-model win or a confirmed best recipe. ETAM strict
fine-tunes and the fixed-eight-epoch selection controls have no results yet.

## Sparse bridges fail the complete architecture-retention rule

The reference's validation joint NLL (**2.5127 ± 0.0072**) ties the private
model with a shared stream (**2.5114 ± 0.0041**) and is worse than bridges at
every layer (**2.5031 ± 0.0047**) beyond the larger sd. All-layer bridges
also improve standard probe AUC, **0.7377 ± 0.0009** versus
**0.7342 ± 0.0008**. The proposed two-bridge design therefore does not meet
the rule requiring it to beat every alternative.

The parameter-matched shared model, bridge-bias ablations and gate ablation
have no completed outputs. The shared trunk has **5.77 M** parameters versus
the reference's **12.64 M**. Equal steps do not make these comparisons equal
in parameter count, training FLOPs or latency.

## No demonstrated benefit from the new temporal encoding

Replacing the new time representation and memory with Stage 1's encoding
gives test joint NLL **2.5039 ± 0.0062**, compared with reference
**2.5094 ± 0.0045**; the difference is within the larger sd.
Removing memory gives **2.5139 ± 0.0065**, also within the spread.
The strict probe of the Stage 1-time control is higher (**0.7254 ± 0.0015**),
but its gap over the reference is within the reference sd.

The `t_stage1` arm changes calendar, relative encoding and memory together.
Its result cannot assign credit to one of those mechanisms. The individual
memory-family, calendar, residual-bin and VQ screens are unfinished.

## Soft-MTP does not meet the objective-retention rule

Soft four-event prediction raises next-event joint NLL to
**2.5191 ± 0.0012**, a worsening of **0.0097** beyond the reference spread,
and its standard probe AUC (**0.7343 ± 0.0006**) ties the reference.
Its multi-event accuracies are **0.5891**, **0.5844**, **0.5812** at horizons
2, 3, 4, versus hard-MTP's **0.5900**, **0.5851**, **0.5818**.
The reference has no MTP heads, so those missing reference scores cannot be
used as zero-valued comparisons.

RePR, combined curriculum, cumulative-time prediction and the other MTP
horizons have no completed results. Their effectiveness is unresolved.

## Count intervals and silence forecasts still need controls

The reference's 30-day count interval covers **0.9234 ± 0.0011** of outcomes
at nominal 80%, despite its better CRPS than history Poisson. Count accuracy
and calibration should be assessed separately.

The direct elapsed-silence forecast has joint NLL **2.3081 ± 0.0176** versus
**2.4413 ± 0.0151** for survival conditioning. This observed difference does
not isolate the survival loss: the no-survival and no-censoring controls
have no results. The direct query head and the last-event distribution
are different fitted predictors.

## The campaign is incomplete, with recorded failures and terminations

The latest saved MBD task statuses show **51/131 done**, **14 terminated**,
and **66 without a status file**. No live scheduler state is inferred.
The 51 completed tasks include 13 GPU bundles containing 39 model runs.

- All 24 strict-baseline cells finished. Only the six AR Transformer cells
  finished for the 10-pool-epoch baselines; PRAGMA-MLM cells were terminated.
- Stage 1 MBD seeds 29/43, data scaling, external baselines, ETAM fine-tunes,
  label-efficiency and cross-target probes, simulations, and common-device
  latency have no completed results.
- Transfer has three completed Stage 1 Synthea seeds, two terminated ETAM-2
  bundle tasks, and three failed Stage 1 Stack Overflow seeds. Each Stack
  Overflow log ends with `RuntimeError: stack expects a non-empty TensorList`
  in Stage 1's conditional-feature loss on a dataset without attributes.
- Data preparation finished. Neither CPU timing job produced a result:
  the earlier harvest records one OOM and one cancellation; saved task
  statuses are `failed` and `terminated`.
- The two pilot tasks have no recorded result or status.

Missing outputs are documented as pending or unavailable, never as model
failures on predictive metrics. The complete state counts are regenerated
in [results.md](results.md).

## Statistical and protocol limits

Three seeds support the pre-registered spread screen, not a significance
test. Many comparisons use the same test set. The eight archived Stage 1
checkpoints each represent one pretraining seed. Baseline standard deviations
and ETAM pretraining-seed standard deviations cover different sources of noise.

The strict cohort excludes 40 training users (33 positive), 24 validation
users (3 positive), and 14 test users (none positive). Strict train prevalence
is **19.92%**, versus **1.18%** on strict test; probability calibration must
account for that training prior. Strict histories make classification inputs
prospective; they do not establish a separately time-cut pretraining corpus.

The 10-epoch baselines match intended sequence exposure, not training FLOPs.
The external named models are framework reimplementations with documented
differences from their papers, and have no completed scores here. Per-run
timings may use different devices and sharing conditions; the common RTX 4090
comparison is unfinished.
