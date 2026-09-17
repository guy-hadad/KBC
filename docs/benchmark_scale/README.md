# Large-dataset scaling results (interim)

Generated from the `scale-datasets` and `scale-datasets-baselines` suites on the
three large benchmarks: full MBD, Synthea EHR, and Amazon Beauty 2014. Kept
separate from [`../benchmark/`](../benchmark/), which holds the original
624-cell four-dataset screen, so neither overwrites the other.

**Interim: 3 013 of 3 360 cells, zero failures** (regenerated 2026-09-17).
Complete: all 864 `scale-datasets-baselines` cells. Outstanding: 152 CPU cells
of `scale-datasets`, still running, and 195 GPU cells whose lane was cancelled
(see [`../research/gpu_resume.md`](../research/gpu_resume.md)). Treat every number here as provisional
until the campaign closes — see
[`../research/campaign_status.md`](../research/campaign_status.md).

| File | What it holds |
| --- | --- |
| [`results.md`](results.md) | All methods per dataset and task, at the largest sample size. |
| [`full_results.md`](full_results.md) | **Everything**: every method x sample size x metric, per dataset and task, plus all metrics at the full pool. Regenerate with `python docs/benchmark_scale/make_full_results.py`. |
| [`results.csv`](results.csv) | Every cell: metric x method x dataset x sample size x seed. |
| [`scaling_exponents.md`](scaling_exponents.md) | Fitted `E(N) = a · N^-b` per method and dataset. |
| `figures/by_method/`, `figures/by_dataset/`, `figures/scaling_laws/` | 496 SVGs, light and dark. |

## What the curves say so far

Scores are the mean over three seeds at each dataset's largest sample point
(MBD 30 665, Synthea 93 305, Amazon Beauty 34 023).

| Dataset | Task | Best method | Score | Non-neural control | Gain |
| --- | --- | --- | ---: | ---: | ---: |
| MBD (full) | classification (AUC) | `pragma-mlm` | 0.731 | 0.696 | +0.035 |
| Synthea EHR | classification (AUC) | `supervised-gru` | 0.882 | 0.833 | +0.049 |
| Amazon Beauty | classification (AUC) | `autoregressive-transformer` ¹ | 0.708 | 0.589 | **+0.119** |
| MBD (full) | TPP (next-type acc.) | `pragma-mlm` | 0.586 | 0.489 | +0.097 |
| Synthea EHR | TPP (next-type acc.) | `autoencoder` | 0.439 | 0.319 | **+0.120** |
| Amazon Beauty | TPP (next-type acc.) | `transaction-mlm` | 0.492 | 0.463 | +0.028 |

¹ A statistical tie: `thp`, `pragma-mlm` and `autoencoder` are within 0.0004 AUC. The earlier `thp` lead was a one-seed mean. The control has two of three seeds at this point.

Four things stand out.

1. **Every fit passes the R² gate, on every dataset, for every method except the
   Markov control.** The original screen could not say that: PaySim failed the
   gate for nearly all of them. Ten to twelve points spanning 64 to 93 305
   sequences is what bought the difference, and it is the main methodological
   gain from these datasets.
2. **The Markov control is flat at this scale too** — `b` of 0.004, 0.001, 0.001,
   all three failing the gate. It does not convert data into accuracy at all,
   which is what a first-order transition table should do. Every neural method
   has an order of magnitude more slope. This reproduces the legacy finding on
   three new datasets and one new domain.
3. **Classification still scales about twice as fast as the TPP task**
   (`b ≈ 0.05-0.13` against `0.04-0.05`). More labelled entities helps a
   sequence-level label more than it helps next-mark prediction, where the mark
   distribution binds.
4. **Rank by slope and rank by score still disagree.** `autoencoder` has the
   steepest classification exponent (0.126) without topping any cell, while
   `supervised-gru` and `ntpp-gru` win Synthea outright on a middling slope
   (0.094). Reading only the largest-`n` table would miss that.

## Caveats that do not go away with scale

* Synthea is **synthetic**. Same fidelity class as BankSim and PaySim; no result
  here transfers to real clinical data.
* Amazon Beauty histories average 6.0 observed events, so its curves measure
  label scale, not long-range temporal structure.
* The LLM entries are still a frozen 135M base with LoRA adapters, far below the
  scale their papers used. Their exponents are informative; their absolute
  scores are a statement about budget.
* Paper-named methods remain **approximations** — see
  [`../research/method_catalog.md`](../research/method_catalog.md).
