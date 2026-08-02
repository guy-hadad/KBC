# Scaling Law Sweep (superseded — synthetic data only)

> **Superseded by [`docs/benchmark/`](../benchmark/).** This page is the earlier
> sweep: one model (`pragma_tiny`) plus two controls, on the in-repo *synthetic*
> generator, produced by `scripts/run_scaling_laws.py`. The current benchmark
> covers every architecture in the project note across the four open banking
> datasets, on real data. Kept for reference only; the numbers below are not
> comparable to the current report.

CPU-friendly sample-scaling sweep on the synthetic downstream tasks.

The x-axis in the plots is the number of labeled training users. The default run is CPU-friendly and designed to show directional sample-scaling behavior; increase epochs, sample sizes, and model scale for research-grade curves.

## Classification

| samples | method | accuracy |
| --- | --- | --- |
| 32 | count-logistic | 0.8984 |
| 32 | pragma-tiny | 0.5352 |
| 64 | count-logistic | 0.8984 |
| 64 | pragma-tiny | 0.7109 |
| 256 | count-logistic | 0.9062 |
| 256 | pragma-tiny | 0.9336 |

## Temporal Point Process

| samples | method | next_type_accuracy | delta_log_rmse |
| --- | --- | --- | --- |
| 32 | markov | 0.2087 | 1.4148 |
| 32 | pragma-tiny | 0.0195 | 1.5613 |
| 64 | markov | 0.2160 | 1.3970 |
| 64 | pragma-tiny | 0.0488 | 1.5594 |
| 256 | markov | 0.2184 | 1.3795 |
| 256 | pragma-tiny | 0.1094 | 1.4136 |

## Plots

- [classification_accuracy.svg](classification_accuracy.svg)
- [tpp_next_type_accuracy.svg](tpp_next_type_accuracy.svg)
- [tpp_delta_log_rmse.svg](tpp_delta_log_rmse.svg)
