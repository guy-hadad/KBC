# ETAM Stage 2: positive findings

Snapshot: **2026-11-10 (Asia/Jerusalem)**. Numbers are test means ± sample sd
across pretraining seeds 13, 29 and 43, unless stated otherwise.
[Results and complete metrics](results.md) · [Methodology](methodology.md) ·
[Negatives](negatives.md) · [Analysis](analysis.md)

## Dual decoding passes the planned decoder rule

The reference's dual decoder has joint NLL **2.5094 ± 0.0045**, compared with
**2.6606 ± 0.0056** for independent event and time heads, an improvement of
**0.1512**. Next-type accuracy also improves: **0.5962 ± 0.0009** versus
**0.5944 ± 0.0009**, a gain of **0.0018**. Both exceed the larger arm sd.
This meets the pre-registered D rule for these completed arms.

Dual decoding also has lower joint NLL than either conditional decoder alone:
time-first **2.5861 ± 0.0079**, event-first **2.5723 ± 0.0180**. These comparisons
support the complete dual-decoder recipe; they do not separately identify the
effect of branch averaging, the extra branch loss or consistency regularization.

The dependency diagnostic is consistent with this result: reference mean PMI
is **0.1019 ± 0.0025** nats, and oracle-time mark accuracy is
**0.6432 ± 0.0018**, above marginal accuracy **0.5962 ± 0.0009**.
Oracle-time accuracy uses the true future gap and is a diagnostic.

## Interaction helps against several simpler architectures

Validation joint NLL is **2.5127 ± 0.0072** for the reference, versus
**2.5355 ± 0.0123** for a shared trunk, **2.5313 ± 0.0125** for the Stage 1
layout inside the Stage 2 framework, and **2.5365 ± 0.0104** for private
pathways without bridges. Each gap exceeds the larger sd. The same ordering
holds on test joint NLL.

Strict frozen-probe AUC is **0.7204 ± 0.0059**, above the shared trunk's
**0.7079 ± 0.0038** and the Stage 1 layout control's **0.7101 ± 0.0040**
beyond their spreads. The reference does not beat every architecture; the
full C retention rule is not met, as explained in [negatives.md](negatives.md).

## Learned count heads beat the history-rate Poisson

Reference count CRPS, lower is better:

| horizon | reference head | history-rate Poisson |
|---|---:|---:|
| 1 day | 0.2968 ± 0.0001 | 0.3300 |
| 7 days | 0.9030 ± 0.0012 | 1.3425 |
| 30 days | 2.3568 ± 0.0628 | 5.0740 |

The Poisson comparator is identical across these seeds. The improvement is
useful predictive evidence, but does not isolate the negative-binomial family
or any particular training term; those ablations have no completed results.
The head's nominal 80% intervals over-cover at 30 days (0.9234 ± 0.0011).

## There is a promising strict-classification arm

The best completed strict baseline is pool-pretrained PRAGMA-MLM, AUC
**0.7151 ± 0.0002**. The hard four-event prediction arm's frozen probe gives
**0.7259 ± 0.0039**, a gap of **0.0108**, exceeding both spreads.
The Stage 1-time-encoder control also reaches **0.7254 ± 0.0015**.

These are useful candidates from the completed arms. The hard-MTP gain over
the reference itself is only **0.0055**, within the reference sd of **0.0059**,
so the result does not establish that hard MTP caused better classification.
The reference does not pass the strict-baseline spread rule, and choosing an
arm after inspecting many test results needs a fresh confirmation.

## Data preparation completed with explicit integrity checks

Raw amounts align with **65,797,418 stored event entries** across pool, train,
validation and test, with zero missing, mismatched or length-mismatched users.
These are entries across stores, not a count of unique raw transactions.

Strict histories were rebuilt before **2022-09-01** from raw MBD records,
recovering older events for 19,030 training users. The builder reports zero
consistency violations, zero differing lines in a 1,000-line JSONL contract
check, and successful round trips for 1,000 sequences per split.
The strict cohort and its dropped positives are reported in
[methodology.md](methodology.md) and the complete result appendix.

## The Stage 1 Synthea reference is available

Three independent Stage 1 pretraining seeds completed on Synthea: test joint
NLL **2.3935 ± 0.0703**, next-type accuracy **0.4361 ± 0.0043**, and frozen
probe AUC **0.8580 ± 0.0023**. This establishes a transfer comparator;
ETAM-2 has no completed transfer result yet.
