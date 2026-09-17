# Full results: large-dataset scaling campaign

Every method, sample size, metric and scaling fit on full MBD, Synthea EHR and Amazon Beauty 2014. Generated 2026-09-17 from [`results.csv`](results.csv) (3013 cells) by [`make_full_results.py`](make_full_results.py).

**Interim.** The campaign is not closed. Cells still missing show as `—`. See [`../research/campaign_status.md`](../research/campaign_status.md) for what remains.

How to read the tables:

* The scaling matrices give the **mean over seeds** at each training-set size. A number in parentheses is the seed count when fewer than 3 seeds finished.
* Rows are ranked by the metric at the dataset's full training pool (the last column).
* The full-pool table gives **mean ± sample standard deviation** over seeds for every metric.
* Paper-named methods are **approximations**. Read the `Fidelity` column and [`../research/method_catalog.md`](../research/method_catalog.md) before citing a number.
* The per-seed values behind every cell are in [`results.csv`](results.csv).

## Coverage

| Dataset | Task | Methods | Sample sizes | Cells |
| --- | --- | ---: | --- | ---: |
| MBD (full) | classification | 18 | 10 (64–30665) | 530 |
| MBD (full) | tpp | 17 | 10 (64–30665) | 491 |
| Synthea EHR | classification | 18 | 12 (64–93305) | 620 |
| Synthea EHR | tpp | 17 | 12 (64–93305) | 573 |
| Amazon Beauty 2014 | classification | 18 | 10 (64–34023) | 417 |
| Amazon Beauty 2014 | tpp | 17 | 10 (64–34023) | 382 |

Contents: [MBD (full)](#mbd-full) · [Synthea EHR](#synthea-ehr) · [Amazon Beauty 2014](#amazon-beauty-2014) · [Scaling analysis](#scaling-analysis)

## Scaling analysis

Each curve's error is fitted to `E(N) = a · N^-b`, where `N` is the number of training sequences, using the seed means at each size. **`b` is the slope on log-log axes: how fast the error falls as data is added.** Larger `b` means the method is still converting data into accuracy. `b` near zero means it has flattened. Read `b` as a local slope over the measured range, not an asymptotic claim.

`*` marks a fit with `R² < 0.70` or fewer than four points, which should not be read as a scaling rate. `Mean b (reliable)` averages only the unflagged fits, with their count in parentheses, and rows are sorted by it. The primary-metric exponents match [`scaling_exponents.md`](scaling_exponents.md). Per-dataset fit details, including `a`, `R²` and the spread of per-seed slopes, are under each dataset below.

### Classification — `b` fitted on 1 − ROC-AUC

| Method | Family | MBD (full) | Synthea EHR | Amazon Beauty 2014 | Mean b (all) | Mean b (reliable) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Transaction autoencoder | reconstruction | 0.069 | 0.197 | 0.112 | 0.126 | **0.126** (3) |
| Transformer Hawkes | neural-tpp | 0.062 | 0.188 | 0.067 | 0.106 | **0.106** (3) |
| NVIDIA TFM blueprint | tabular-transformer | 0.070 | 0.151 | 0.081 | 0.101 | **0.101** (3) |
| Autoregressive transaction Transformer | generative | 0.068 | 0.148 | 0.082 | 0.099 | **0.099** (3) |
| TabFormer (TabGPT) | tabular-transformer | 0.076 | 0.137 | 0.073 | 0.095 | **0.095** (3) |
| Neural TPP (GRU) | neural-tpp | 0.051 | 0.144 | 0.087 | 0.094 | **0.094** (3) |
| Supervised GRU | supervised | 0.051 | 0.144 | 0.086 | 0.094 | **0.094** (3) |
| CoLES | contrastive | 0.042 | 0.138 | 0.095 | 0.092 | **0.092** (3) |
| MM-TPP | llm | 0.101 | 0.131 | 0.024 | 0.085 | **0.085** (3) |
| Language-TPP | llm | 0.088 | 0.137 | 0.025 | 0.083 | **0.083** (3) |
| Mambular (Mamba SSM) | state-space | 0.050 | 0.129 | 0.047 | 0.075 | **0.075** (3) |
| TPP-LLM | llm | 0.060 | 0.131 | 0.031 | 0.074 | **0.074** (3) |
| PRAGMA + masked pretraining | hierarchical | 0.066 | 0.105 | 0.048 | 0.073 | **0.073** (3) |
| Engineered features + GBDT | classic | 0.070 | 0.090 | 0.059 | 0.073 | **0.073** (3) |
| PRAGMA | hierarchical | 0.065 | 0.096 | 0.043 | 0.068 | **0.068** (3) |
| Transaction MLM | reconstruction | 0.037 | 0.091 | 0.046 | 0.058 | **0.058** (3) |
| Count + logistic regression | classic | 0.064 | 0.084 | 0.014 | 0.054 | **0.054** (3) |
| TabFormer (TabBERT) | tabular-transformer | 0.051 | 0.057 | 0.047 | 0.052 | **0.052** (3) |

### Classification — `b` fitted on 1 − avg. precision

| Method | Family | MBD (full) | Synthea EHR | Amazon Beauty 2014 | Mean b (all) | Mean b (reliable) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Transaction autoencoder | reconstruction | 0.003 | 0.043 | 0.067 | 0.037 | **0.037** (3) |
| Supervised GRU | supervised | 0.002 | 0.046 | 0.055 | 0.034 | **0.034** (3) |
| Neural TPP (GRU) | neural-tpp | 0.002 | 0.046 | 0.055 | 0.034 | **0.034** (3) |
| Autoregressive transaction Transformer | generative | 0.002 | 0.041 | 0.057 | 0.033 | **0.033** (3) |
| NVIDIA TFM blueprint | tabular-transformer | 0.002 | 0.039 | 0.055 | 0.032 | **0.032** (3) |
| CoLES | contrastive | 0.002 | 0.036 | 0.059 | 0.032 | **0.032** (3) |
| Transformer Hawkes | neural-tpp | 0.002 | 0.039 | 0.049 | 0.030 | **0.030** (3) |
| PRAGMA + masked pretraining | hierarchical | 0.003 | 0.037 | 0.041 | 0.027 | **0.027** (3) |
| TabFormer (TabGPT) | tabular-transformer | 0.002 | 0.031 | 0.047 | 0.027 | **0.027** (3) |
| Engineered features + GBDT | classic | 0.002 | 0.027 | 0.045 | 0.025 | **0.025** (3) |
| Transaction MLM | reconstruction | 0.002 | 0.035 | 0.035 | 0.024 | **0.024** (3) |
| PRAGMA | hierarchical | 0.003 | 0.027 | 0.039 | 0.023 | **0.023** (3) |
| Mambular (Mamba SSM) | state-space | 0.001 | 0.031 | 0.035 | 0.022 | **0.022** (3) |
| TabFormer (TabBERT) | tabular-transformer | 0.002 | 0.025 | 0.036 | 0.021 | **0.021** (3) |
| TPP-LLM | llm | 0.002 | 0.025 | 0.020 | 0.016 | **0.016** (3) |
| Language-TPP | llm | 0.002 | 0.022 | 0.016 | 0.013 | **0.013** (3) |
| MM-TPP | llm | 0.002 | 0.020 | 0.016 | 0.013 | **0.013** (3) |
| Count + logistic regression | classic | 0.002 | 0.018 | 0.014 | 0.011 | **0.011** (3) |

### Temporal point process (TPP) — `b` fitted on 1 − next-type acc.

| Method | Family | MBD (full) | Synthea EHR | Amazon Beauty 2014 | Mean b (all) | Mean b (reliable) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CoLES | contrastive | 0.052 | 0.042 | 0.067 | 0.053 | **0.053** (3) |
| Supervised GRU | supervised | 0.056 | 0.042 | 0.051 | 0.050 | **0.050** (3) |
| Neural TPP (GRU) | neural-tpp | 0.056 | 0.042 | 0.049 | 0.049 | **0.049** (3) |
| TPP-LLM | llm | 0.052 | 0.037 | 0.051 | 0.047 | **0.047** (3) |
| Autoregressive transaction Transformer | generative | 0.037 | 0.037 | 0.063 | 0.046 | **0.046** (3) |
| Transaction autoencoder | reconstruction | 0.054 | 0.042 | 0.042 | 0.046 | **0.046** (3) |
| NVIDIA TFM blueprint | tabular-transformer | 0.038 | 0.038 | 0.059 | 0.045 | **0.045** (3) |
| Transaction MLM | reconstruction | 0.037 | 0.038 | 0.060 | 0.045 | **0.045** (3) |
| Transformer Hawkes | neural-tpp | 0.036 | 0.037 | 0.055 | 0.043 | **0.043** (3) |
| PRAGMA | hierarchical | 0.043 | 0.035 | 0.050 | 0.043 | **0.043** (3) |
| MM-TPP | llm | 0.040 | 0.014 | 0.071 | 0.042 | **0.042** (3) |
| TabFormer (TabBERT) | tabular-transformer | 0.028 | 0.032 | 0.065 | 0.042 | **0.042** (3) |
| PRAGMA + masked pretraining | hierarchical | 0.043 | 0.038 | 0.043 | 0.041 | **0.041** (3) |
| Mambular (Mamba SSM) | state-space | 0.027 | 0.040 | 0.053 | 0.040 | **0.040** (3) |
| TabFormer (TabGPT) | tabular-transformer | 0.030 | 0.031 | 0.058 | 0.040 | **0.040** (3) |
| Language-TPP | llm | 0.035 | 0.018 | 0.058 | 0.037 | **0.037** (3) |
| Marked Markov chain | classic | 0.004* | 0.001* | 0.001* | 0.002 | — (0) |

### Temporal point process (TPP) — `b` fitted on Time RMSE (log1p s)

| Method | Family | MBD (full) | Synthea EHR | Amazon Beauty 2014 | Mean b (all) | Mean b (reliable) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Neural TPP (GRU) | neural-tpp | 0.040* | 0.069 | 0.024 | 0.044 | **0.047** (2) |
| CoLES | contrastive | 0.038* | 0.070 | 0.023 | 0.043 | **0.046** (2) |
| Transaction autoencoder | reconstruction | 0.039* | 0.069 | 0.023 | 0.044 | **0.046** (2) |
| Supervised GRU | supervised | 0.040* | 0.069 | 0.023 | 0.044 | **0.046** (2) |
| TPP-LLM | llm | 0.038 | 0.055 | 0.017 | 0.037 | **0.037** (3) |
| Autoregressive transaction Transformer | generative | 0.024 | 0.054 | 0.022 | 0.033 | **0.033** (3) |
| NVIDIA TFM blueprint | tabular-transformer | 0.024 | 0.052 | 0.021 | 0.033 | **0.033** (3) |
| Transaction MLM | reconstruction | 0.022 | 0.052 | 0.022 | 0.032 | **0.032** (3) |
| Mambular (Mamba SSM) | state-space | 0.018 | 0.051 | 0.020 | 0.030 | **0.030** (3) |
| Transformer Hawkes | neural-tpp | 0.018 | 0.050 | 0.022 | 0.030 | **0.030** (3) |
| PRAGMA | hierarchical | 0.035 | 0.037 | 0.016 | 0.029 | **0.029** (3) |
| PRAGMA + masked pretraining | hierarchical | 0.034* | 0.038 | 0.015 | 0.029 | **0.027** (2) |
| TabFormer (TabGPT) | tabular-transformer | 0.022 | 0.044 | 0.009 | 0.025 | **0.025** (3) |
| TabFormer (TabBERT) | tabular-transformer | 0.022 | 0.042 | 0.009 | 0.024 | **0.024** (3) |
| MM-TPP | llm | 0.016 | 0.026 | 0.011 | 0.018 | **0.018** (3) |
| Language-TPP | llm | 0.016 | 0.026 | 0.007 | 0.016 | **0.016** (3) |
| Marked Markov chain | classic | 0.003* | 0.001* | 0.001* | 0.002 | — (0) |

## MBD (full)

### Classification

#### All metrics at the full pool (n = 30665)

| Method | Fidelity | Seeds | ROC-AUC | Avg. precision | Accuracy | Macro F1 | Wall s (median) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | Approximation | 3 | 0.731 ± 0.002 | 0.031 ± 0.000 | 0.963 ± 0.002 | 0.523 ± 0.002 | 4,565 |
| PRAGMA | Approximation | 3 | 0.730 ± 0.002 | 0.030 ± 0.001 | 0.968 ± 0.004 | 0.520 ± 0.001 | 5,741 |
| Supervised GRU | Implemented | 3 | 0.725 ± 0.001 | 0.028 ± 0.000 | 0.963 ± 0.002 | 0.520 ± 0.002 | 1,070 |
| Neural TPP (GRU) | Approximation | 3 | 0.725 ± 0.001 | 0.028 ± 0.000 | 0.963 ± 0.002 | 0.520 ± 0.002 | 1,284 |
| Transaction MLM | Implemented | 3 | 0.720 ± 0.001 | 0.031 ± 0.001 | 0.961 ± 0.003 | 0.521 ± 0.004 | 1,216 |
| Transformer Hawkes | Approximation | 3 | 0.720 ± 0.004 | 0.027 ± 0.000 | 0.969 ± 0.002 | 0.515 ± 0.003 | 1,431 |
| Engineered features + GBDT | Implemented | 3 | 0.718 ± 0.002 | 0.029 ± 0.001 | 0.978 ± 0.002 | 0.514 ± 0.002 | 173 |
| NVIDIA TFM blueprint | Approximation | 3 | 0.717 ± 0.008 | 0.028 ± 0.001 | 0.962 ± 0.004 | 0.519 ± 0.003 | 2,129 |
| Autoregressive transaction Transformer | Implemented | 3 | 0.717 ± 0.004 | 0.029 ± 0.001 | 0.957 ± 0.004 | 0.519 ± 0.002 | 2,287 |
| Transaction autoencoder | Implemented | 3 | 0.716 ± 0.004 | 0.028 ± 0.001 | 0.969 ± 0.001 | 0.517 ± 0.001 | 1,484 |
| TabFormer (TabBERT) | Approximation | 3 | 0.715 ± 0.003 | 0.028 ± 0.001 | 0.969 ± 0.000 | 0.520 ± 0.003 | 1,842 |
| TPP-LLM | Approximation | 3 | 0.710 ± 0.002 | 0.027 ± 0.001 | 0.982 ± 0.002 | 0.509 ± 0.006 | 3,348 |
| Language-TPP | Approximation | 3 | 0.709 ± 0.001 | 0.025 ± 0.001 | 0.979 ± 0.001 | 0.509 ± 0.003 | 3,984 |
| CoLES | Approximation | 3 | 0.708 ± 0.005 | 0.028 ± 0.001 | 0.965 ± 0.001 | 0.520 ± 0.002 | 2,418 |
| TabFormer (TabGPT) | Approximation | 3 | 0.707 ± 0.000 | 0.027 ± 0.000 | 0.969 ± 0.002 | 0.523 ± 0.003 | 1,963 |
| Mambular (Mamba SSM) | Approximation | 3 | 0.704 ± 0.008 | 0.025 ± 0.002 | 0.941 ± 0.003 | 0.511 ± 0.001 | 17,582 |
| Count + logistic regression | Implemented | 3 | 0.696 ± 0.000 | 0.027 ± 0.000 | 0.981 ± 0.000 | 0.512 ± 0.000 | 62 |
| MM-TPP | Approximation | 3 | 0.692 ± 0.001 | 0.023 ± 0.001 | 0.985 ± 0.002 | 0.499 ± 0.004 | 3,809 |

#### Scaling fits

`E(N) = a · N^-b` on seed means, as in the summary above. The per-seed column fits each seed's own curve separately (four or more points) and gives the spread of `b` across seeds. `*` flags an unreliable fit.

| Method | b (1 − ROC-AUC) | a | R² | per-seed b (mean ± sd, n) | b (1 − avg. precision) | a | R² | per-seed b (mean ± sd, n) | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | 0.066 | 0.515 | 0.95 | 0.066 ± 0.004 (3) | 0.003 | 0.998 | 0.95 | 0.003 ± 0.000 (3) | 10 |
| PRAGMA | 0.065 | 0.510 | 0.96 | 0.065 ± 0.004 (3) | 0.003 | 0.997 | 0.95 | 0.003 ± 0.000 (3) | 10 |
| Supervised GRU | 0.051 | 0.448 | 0.81 | 0.051 ± 0.007 (3) | 0.002 | 0.988 | 0.92 | 0.002 ± 0.000 (3) | 10 |
| Neural TPP (GRU) | 0.051 | 0.448 | 0.81 | 0.051 ± 0.007 (3) | 0.002 | 0.988 | 0.92 | 0.002 ± 0.000 (3) | 10 |
| Transaction MLM | 0.037 | 0.404 | 0.87 | 0.037 ± 0.008 (3) | 0.002 | 0.988 | 0.98 | 0.002 ± 0.001 (3) | 10 |
| Transformer Hawkes | 0.062 | 0.507 | 0.90 | 0.062 ± 0.012 (3) | 0.002 | 0.990 | 0.91 | 0.002 ± 0.000 (3) | 10 |
| Engineered features + GBDT | 0.070 | 0.576 | 0.99 | 0.070 ± 0.008 (3) | 0.002 | 0.995 | 0.97 | 0.002 ± 0.001 (3) | 10 |
| NVIDIA TFM blueprint | 0.070 | 0.545 | 0.81 | 0.070 ± 0.009 (3) | 0.002 | 0.991 | 0.91 | 0.002 ± 0.000 (3) | 10 |
| Autoregressive transaction Transformer | 0.068 | 0.536 | 0.88 | 0.068 ± 0.002 (3) | 0.002 | 0.992 | 0.97 | 0.002 ± 0.000 (3) | 10 |
| Transaction autoencoder | 0.069 | 0.542 | 0.80 | 0.069 ± 0.004 (3) | 0.003 | 0.997 | 0.90 | 0.003 ± 0.000 (3) | 10 |
| TabFormer (TabBERT) | 0.051 | 0.465 | 0.90 | 0.051 ± 0.013 (3) | 0.002 | 0.993 | 0.99 | 0.002 ± 0.000 (3) | 10 |
| TPP-LLM | 0.060 | 0.529 | 0.93 | 0.060 ± 0.003 (3) | 0.002 | 0.995 | 0.91 | 0.002 ± 0.000 (3) | 10 |
| Language-TPP | 0.088 | 0.684 | 0.87 | 0.096 ± 0.039 (3) | 0.002 | 0.997 | 0.92 | 0.002 ± 0.001 (3) | 10 |
| CoLES | 0.042 | 0.427 | 0.73 | 0.042 ± 0.005 (3) | 0.002 | 0.989 | 0.91 | 0.002 ± 0.000 (3) | 10 |
| TabFormer (TabGPT) | 0.076 | 0.597 | 0.91 | 0.076 ± 0.004 (3) | 0.002 | 0.995 | 0.97 | 0.002 ± 0.000 (3) | 10 |
| Mambular (Mamba SSM) | 0.050 | 0.484 | 0.82 | 0.049 ± 0.005 (3) | 0.001 | 0.987 | 0.79 | 0.001 ± 0.000 (3) | 10 |
| Count + logistic regression | 0.064 | 0.563 | 0.94 | 0.064 ± 0.017 (3) | 0.002 | 0.993 | 0.97 | 0.002 ± 0.001 (3) | 10 |
| MM-TPP | 0.101 | 0.863 | 0.90 | 0.105 ± 0.008 (3) | 0.002 | 1.000 | 0.89 | 0.002 ± 0.000 (3) | 10 |

#### ROC-AUC vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 30 665 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | hierarchical | 0.614 | 0.603 | 0.651 | 0.655 | 0.681 | 0.695 | 0.714 | 0.721 | 0.724 | 0.731 |
| PRAGMA | hierarchical | 0.608 | 0.610 | 0.653 | 0.661 | 0.676 | 0.697 | 0.714 | 0.720 | 0.723 | 0.730 |
| Supervised GRU | supervised | 0.593 | 0.651 | 0.685 | 0.691 | 0.695 | 0.699 | 0.710 | 0.715 | 0.720 | 0.725 |
| Neural TPP (GRU) | neural-tpp | 0.593 | 0.651 | 0.685 | 0.691 | 0.695 | 0.699 | 0.710 | 0.715 | 0.720 | 0.725 |
| Transaction MLM | reconstruction | 0.648 | 0.645 | 0.677 | 0.686 | 0.700 | 0.708 | 0.707 | 0.709 | 0.711 | 0.720 |
| Transformer Hawkes | neural-tpp | 0.579 | 0.614 | 0.650 | 0.674 | 0.682 | 0.695 | 0.710 | 0.711 | 0.713 | 0.720 |
| Engineered features + GBDT | classic | 0.577 | 0.580 | 0.618 | 0.622 | 0.648 | 0.662 | 0.679 | 0.693 | 0.714 | 0.718 |
| NVIDIA TFM blueprint | tabular-transformer | 0.536 | 0.595 | 0.654 | 0.678 | 0.687 | 0.693 | 0.705 | 0.707 | 0.712 | 0.717 |
| Autoregressive transaction Transformer | generative | 0.561 | 0.598 | 0.643 | 0.665 | 0.682 | 0.698 | 0.710 | 0.703 | 0.713 | 0.717 |
| Transaction autoencoder | reconstruction | 0.550 | 0.579 | 0.653 | 0.683 | 0.688 | 0.690 | 0.703 | 0.708 | 0.712 | 0.716 |
| TabFormer (TabBERT) | tabular-transformer | 0.605 | 0.625 | 0.656 | 0.675 | 0.689 | 0.697 | 0.700 | 0.707 | 0.711 | 0.715 |
| TPP-LLM | llm | 0.603 | 0.609 | 0.615 | 0.616 | 0.629 (2) | 0.674 (2) | 0.690 | 0.700 | 0.706 | 0.710 |
| Language-TPP | llm | 0.566 (1) | 0.517 | 0.531 | 0.593 | 0.653 | 0.678 (1) | 0.686 | 0.695 | 0.703 | 0.709 |
| CoLES | contrastive | 0.597 | 0.652 | 0.677 | 0.690 | 0.690 | 0.698 | 0.707 | 0.707 | 0.705 | 0.708 |
| TabFormer (TabGPT) | tabular-transformer | 0.527 | 0.580 | 0.611 | 0.646 | 0.666 | 0.684 | 0.693 | 0.702 | 0.707 | 0.707 |
| Mambular (Mamba SSM) | state-space | 0.565 | 0.616 | 0.664 | 0.659 | 0.663 | 0.667 | 0.682 | 0.688 | 0.693 | 0.704 |
| Count + logistic regression | classic | 0.569 | 0.581 | 0.595 | 0.614 | 0.654 | 0.668 | 0.684 | 0.692 | 0.693 | 0.696 |
| MM-TPP | llm | 0.492 (2) | 0.485 (2) | 0.485 | 0.488 | 0.524 | 0.588 | 0.648 (1) | 0.674 | 0.684 | 0.692 |

#### Avg. precision vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 30 665 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | hierarchical | 0.016 | 0.015 | 0.018 | 0.018 | 0.021 | 0.024 | 0.028 | 0.030 | 0.030 | 0.031 |
| PRAGMA | hierarchical | 0.016 | 0.016 | 0.019 | 0.018 | 0.019 | 0.024 | 0.027 | 0.028 | 0.030 | 0.030 |
| Supervised GRU | supervised | 0.017 | 0.020 | 0.022 | 0.023 | 0.023 | 0.024 | 0.025 | 0.026 | 0.028 | 0.028 |
| Neural TPP (GRU) | neural-tpp | 0.017 | 0.020 | 0.022 | 0.023 | 0.023 | 0.024 | 0.025 | 0.026 | 0.028 | 0.028 |
| Transaction MLM | reconstruction | 0.020 | 0.020 | 0.022 | 0.023 | 0.024 | 0.026 | 0.026 | 0.027 | 0.029 | 0.031 |
| Transformer Hawkes | neural-tpp | 0.016 | 0.018 | 0.021 | 0.022 | 0.023 | 0.024 | 0.026 | 0.028 | 0.026 | 0.027 |
| Engineered features + GBDT | classic | 0.015 | 0.016 | 0.019 | 0.019 | 0.021 | 0.022 | 0.024 | 0.025 | 0.029 | 0.029 |
| NVIDIA TFM blueprint | tabular-transformer | 0.015 | 0.017 | 0.021 | 0.023 | 0.023 | 0.024 | 0.027 | 0.026 | 0.027 | 0.028 |
| Autoregressive transaction Transformer | generative | 0.015 | 0.018 | 0.020 | 0.022 | 0.023 | 0.025 | 0.025 | 0.026 | 0.029 | 0.029 |
| Transaction autoencoder | reconstruction | 0.013 | 0.013 | 0.019 | 0.022 | 0.023 | 0.023 | 0.025 | 0.028 | 0.027 | 0.028 |
| TabFormer (TabBERT) | tabular-transformer | 0.016 | 0.017 | 0.018 | 0.020 | 0.022 | 0.022 | 0.025 | 0.026 | 0.028 | 0.028 |
| TPP-LLM | llm | 0.016 | 0.016 | 0.017 | 0.017 | 0.018 (2) | 0.023 (2) | 0.025 | 0.027 | 0.027 | 0.027 |
| Language-TPP | llm | 0.014 (1) | 0.013 | 0.013 | 0.015 | 0.019 | 0.021 (1) | 0.022 | 0.023 | 0.024 | 0.025 |
| CoLES | contrastive | 0.016 | 0.019 | 0.022 | 0.023 | 0.023 | 0.024 | 0.025 | 0.026 | 0.027 | 0.028 |
| TabFormer (TabGPT) | tabular-transformer | 0.014 | 0.015 | 0.018 | 0.019 | 0.020 | 0.023 | 0.024 | 0.026 | 0.028 | 0.027 |
| Mambular (Mamba SSM) | state-space | 0.016 | 0.018 | 0.022 | 0.021 | 0.022 | 0.021 | 0.022 | 0.023 | 0.023 | 0.025 |
| Count + logistic regression | classic | 0.016 | 0.017 | 0.018 | 0.019 | 0.021 | 0.023 | 0.024 | 0.026 | 0.026 | 0.027 |
| MM-TPP | llm | 0.012 (2) | 0.012 (2) | 0.011 | 0.012 | 0.013 | 0.015 | 0.019 (1) | 0.021 | 0.022 | 0.023 |

#### Accuracy vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 30 665 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | hierarchical | 0.988 | 0.988 | 0.978 | 0.987 | 0.984 | 0.964 | 0.969 | 0.967 | 0.960 | 0.963 |
| PRAGMA | hierarchical | 0.988 | 0.988 | 0.948 | 0.988 | 0.987 | 0.983 | 0.974 | 0.972 | 0.967 | 0.968 |
| Supervised GRU | supervised | 0.988 | 0.988 | 0.988 | 0.980 | 0.970 | 0.960 | 0.961 | 0.965 | 0.965 | 0.963 |
| Neural TPP (GRU) | neural-tpp | 0.988 | 0.988 | 0.988 | 0.980 | 0.970 | 0.960 | 0.961 | 0.965 | 0.965 | 0.963 |
| Transaction MLM | reconstruction | 0.988 | 0.988 | 0.988 | 0.986 | 0.976 | 0.975 | 0.963 | 0.963 | 0.962 | 0.961 |
| Transformer Hawkes | neural-tpp | 0.988 | 0.988 | 0.981 | 0.967 | 0.966 | 0.949 | 0.960 | 0.969 | 0.968 | 0.969 |
| Engineered features + GBDT | classic | 0.881 | 0.879 | 0.896 | 0.925 | 0.943 | 0.940 | 0.950 | 0.959 | 0.981 | 0.978 |
| NVIDIA TFM blueprint | tabular-transformer | 0.988 | 0.988 | 0.988 | 0.971 | 0.959 | 0.939 | 0.942 | 0.953 | 0.961 | 0.962 |
| Autoregressive transaction Transformer | generative | 0.988 | 0.988 | 0.988 | 0.988 | 0.973 | 0.964 | 0.966 | 0.955 | 0.959 | 0.957 |
| Transaction autoencoder | reconstruction | 0.988 | 0.988 | 0.988 | 0.986 | 0.985 | 0.983 | 0.984 | 0.980 | 0.972 | 0.969 |
| TabFormer (TabBERT) | tabular-transformer | 0.988 | 0.988 | 0.985 | 0.984 | 0.985 | 0.964 | 0.962 | 0.967 | 0.971 | 0.969 |
| TPP-LLM | llm | 0.986 | 0.988 | 0.988 | 0.987 | 0.988 (2) | 0.987 (2) | 0.953 | 0.974 | 0.975 | 0.982 |
| Language-TPP | llm | 0.988 (1) | 0.988 | 0.988 | 0.988 | 0.988 | 0.986 (1) | 0.986 | 0.987 | 0.987 | 0.979 |
| CoLES | contrastive | 0.988 | 0.988 | 0.986 | 0.982 | 0.980 | 0.970 | 0.969 | 0.970 | 0.968 | 0.965 |
| TabFormer (TabGPT) | tabular-transformer | 0.988 | 0.988 | 0.984 | 0.959 | 0.964 | 0.931 | 0.952 | 0.956 | 0.967 | 0.969 |
| Mambular (Mamba SSM) | state-space | 0.988 | 0.988 | 0.969 | 0.919 | 0.931 | 0.920 | 0.929 | 0.936 | 0.943 | 0.941 |
| Count + logistic regression | classic | 0.833 | 0.860 | 0.876 | 0.922 | 0.948 | 0.963 | 0.976 | 0.979 | 0.981 | 0.981 |
| MM-TPP | llm | 0.798 (2) | 0.950 (2) | 0.979 | 0.987 | 0.988 | 0.986 | 0.988 (1) | 0.987 | 0.988 | 0.985 |

#### Macro F1 vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 30 665 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | hierarchical | 0.497 | 0.497 | 0.500 | 0.499 | 0.501 | 0.509 | 0.518 | 0.521 | 0.520 | 0.523 |
| PRAGMA | hierarchical | 0.497 | 0.497 | 0.503 | 0.497 | 0.497 | 0.504 | 0.518 | 0.518 | 0.521 | 0.520 |
| Supervised GRU | supervised | 0.497 | 0.497 | 0.497 | 0.504 | 0.508 | 0.513 | 0.514 | 0.516 | 0.520 | 0.520 |
| Neural TPP (GRU) | neural-tpp | 0.497 | 0.497 | 0.497 | 0.504 | 0.508 | 0.513 | 0.514 | 0.516 | 0.520 | 0.520 |
| Transaction MLM | reconstruction | 0.497 | 0.497 | 0.497 | 0.502 | 0.501 | 0.515 | 0.516 | 0.519 | 0.522 | 0.521 |
| Transformer Hawkes | neural-tpp | 0.497 | 0.497 | 0.507 | 0.512 | 0.510 | 0.511 | 0.513 | 0.517 | 0.514 | 0.515 |
| Engineered features + GBDT | classic | 0.483 | 0.484 | 0.493 | 0.503 | 0.509 | 0.508 | 0.512 | 0.514 | 0.516 | 0.514 |
| NVIDIA TFM blueprint | tabular-transformer | 0.497 | 0.497 | 0.497 | 0.511 | 0.509 | 0.506 | 0.512 | 0.514 | 0.515 | 0.519 |
| Autoregressive transaction Transformer | generative | 0.497 | 0.497 | 0.497 | 0.497 | 0.507 | 0.514 | 0.514 | 0.515 | 0.520 | 0.519 |
| Transaction autoencoder | reconstruction | 0.497 | 0.497 | 0.497 | 0.500 | 0.504 | 0.507 | 0.506 | 0.517 | 0.518 | 0.517 |
| TabFormer (TabBERT) | tabular-transformer | 0.497 | 0.497 | 0.503 | 0.502 | 0.499 | 0.506 | 0.515 | 0.514 | 0.518 | 0.520 |
| TPP-LLM | llm | 0.497 | 0.497 | 0.497 | 0.497 | 0.497 (2) | 0.499 (2) | 0.511 | 0.516 | 0.515 | 0.509 |
| Language-TPP | llm | 0.497 (1) | 0.497 | 0.497 | 0.497 | 0.497 | 0.498 (1) | 0.499 | 0.498 | 0.499 | 0.509 |
| CoLES | contrastive | 0.497 | 0.497 | 0.499 | 0.507 | 0.507 | 0.511 | 0.514 | 0.518 | 0.520 | 0.520 |
| TabFormer (TabGPT) | tabular-transformer | 0.497 | 0.497 | 0.499 | 0.505 | 0.509 | 0.504 | 0.512 | 0.513 | 0.521 | 0.523 |
| Mambular (Mamba SSM) | state-space | 0.498 | 0.497 | 0.510 | 0.500 | 0.504 | 0.502 | 0.505 | 0.506 | 0.508 | 0.511 |
| Count + logistic regression | classic | 0.471 | 0.480 | 0.484 | 0.498 | 0.505 | 0.509 | 0.511 | 0.511 | 0.511 | 0.512 |
| MM-TPP | llm | 0.454 (2) | 0.492 (2) | 0.499 | 0.497 | 0.497 | 0.497 | 0.497 (1) | 0.499 | 0.497 | 0.499 |

### Temporal point process (TPP)

#### All metrics at the full pool (n = 30665)

| Method | Fidelity | Seeds | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) | Wall s (median) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | Approximation | 3 | 0.586 ± 0.001 | 0.201 ± 0.002 | 1.528 ± 0.002 | 1.150 ± 0.002 | 47,717 |
| Transaction autoencoder | Implemented | 3 | 0.586 ± 0.001 | 0.193 ± 0.003 | 1.526 ± 0.003 | 1.156 ± 0.003 | 15,135 |
| CoLES | Approximation | 3 | 0.586 ± 0.001 | 0.196 ± 0.001 | 1.526 ± 0.003 | 1.156 ± 0.002 | 16,530 |
| Supervised GRU | Implemented | 3 | 0.586 ± 0.001 | 0.195 ± 0.003 | 1.526 ± 0.003 | 1.157 ± 0.002 | 15,419 |
| Neural TPP (GRU) | Approximation | 3 | 0.586 ± 0.001 | 0.195 ± 0.003 | 1.526 ± 0.003 | 1.157 ± 0.002 | 14,922 |
| PRAGMA | Approximation | 3 | 0.585 ± 0.001 | 0.202 ± 0.004 | 1.530 ± 0.003 | 1.149 ± 0.003 | 44,843 |
| Transaction MLM | Implemented | 3 | 0.585 ± 0.001 | 0.201 ± 0.003 | 1.527 ± 0.002 | 1.152 ± 0.002 | 6,353 |
| NVIDIA TFM blueprint | Approximation | 3 | 0.584 ± 0.000 | 0.204 ± 0.003 | 1.527 ± 0.002 | 1.154 ± 0.003 | 11,549 |
| Autoregressive transaction Transformer | Implemented | 3 | 0.584 ± 0.001 | 0.207 ± 0.004 | 1.527 ± 0.002 | 1.156 ± 0.001 | 8,055 |
| Transformer Hawkes | Approximation | 3 | 0.584 ± 0.001 | 0.203 ± 0.004 | 1.539 ± 0.003 | 1.169 ± 0.002 | 6,673 |
| Mambular (Mamba SSM) | Approximation | 3 | 0.582 ± 0.001 | 0.201 ± 0.002 | 1.527 ± 0.003 | 1.156 ± 0.003 | 186,990 |
| Language-TPP | Approximation | 2 | 0.581 ± 0.001 | 0.196 ± 0.004 | 1.739 ± 0.000 | 1.329 ± 0.001 | 33,062 |
| TPP-LLM | Approximation | 3 | 0.581 ± 0.001 | 0.201 ± 0.004 | 1.565 ± 0.001 | 1.198 ± 0.007 | 26,183 |
| TabFormer (TabGPT) | Approximation | 3 | 0.580 ± 0.000 | 0.198 ± 0.004 | 1.717 ± 0.005 | 1.309 ± 0.003 | 8,517 |
| TabFormer (TabBERT) | Approximation | 3 | 0.580 ± 0.000 | 0.199 ± 0.003 | 1.715 ± 0.004 | 1.309 ± 0.004 | 10,140 |
| MM-TPP | Approximation | 3 | 0.569 ± 0.000 | 0.175 ± 0.003 | 1.749 ± 0.003 | 1.340 ± 0.000 | 31,736 |
| Marked Markov chain | Implemented | 3 | 0.489 ± 0.000 | 0.088 ± 0.000 | 1.857 ± 0.000 | 1.424 ± 0.000 | 95 |

#### Scaling fits

`E(N) = a · N^-b` on seed means, as in the summary above. The per-seed column fits each seed's own curve separately (four or more points) and gives the spread of `b` across seeds. `*` flags an unreliable fit.

| Method | b (1 − next-type acc.) | a | R² | per-seed b (mean ± sd, n) | b (Time RMSE (log1p s)) | a | R² | per-seed b (mean ± sd, n) | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | 0.043 | 0.619 | 0.88 | 0.043 ± 0.005 (3) | 0.034* | 2.088 | 0.69 | 0.034 ± 0.001 (3) | 10 |
| Transaction autoencoder | 0.054 | 0.684 | 0.86 | 0.053 ± 0.001 (3) | 0.039* | 2.167 | 0.67 | 0.039 ± 0.000 (3) | 10 |
| CoLES | 0.052 | 0.673 | 0.86 | 0.052 ± 0.003 (3) | 0.038* | 2.139 | 0.65 | 0.038 ± 0.001 (3) | 10 |
| Supervised GRU | 0.056 | 0.699 | 0.85 | 0.056 ± 0.001 (3) | 0.040* | 2.189 | 0.70 | 0.040 ± 0.001 (3) | 10 |
| Neural TPP (GRU) | 0.056 | 0.699 | 0.85 | 0.056 ± 0.001 (3) | 0.040* | 2.189 | 0.70 | 0.040 ± 0.001 (3) | 10 |
| PRAGMA | 0.043 | 0.617 | 0.88 | 0.042 ± 0.004 (3) | 0.035 | 2.093 | 0.70 | 0.035 ± 0.002 (3) | 10 |
| Transaction MLM | 0.037 | 0.585 | 0.84 | 0.037 ± 0.003 (3) | 0.022 | 1.872 | 0.73 | 0.022 ± 0.001 (3) | 10 |
| NVIDIA TFM blueprint | 0.038 | 0.592 | 0.85 | 0.038 ± 0.003 (3) | 0.024 | 1.906 | 0.73 | 0.024 ± 0.002 (3) | 10 |
| Autoregressive transaction Transformer | 0.037 | 0.590 | 0.86 | 0.037 ± 0.003 (3) | 0.024 | 1.901 | 0.72 | 0.024 ± 0.001 (3) | 10 |
| Transformer Hawkes | 0.036 | 0.580 | 0.82 | 0.036 ± 0.004 (3) | 0.018 | 1.810 | 0.74 | 0.018 ± 0.001 (3) | 10 |
| Mambular (Mamba SSM) | 0.027 | 0.541 | 0.82 | 0.027 ± 0.003 (3) | 0.018 | 1.808 | 0.70 | 0.018 ± 0.001 (3) | 10 |
| Language-TPP | 0.035 | 0.577 | 0.78 | 0.038 ± 0.003 (3) | 0.016 | 2.045 | 0.97 | 0.017 ± 0.001 (3) | 9 |
| TPP-LLM | 0.052 | 0.679 | 0.86 | 0.049 ± 0.004 (3) | 0.038 | 2.247 | 0.91 | 0.038 ± 0.000 (3) | 10 |
| TabFormer (TabGPT) | 0.030 | 0.559 | 0.88 | 0.030 ± 0.005 (3) | 0.022 | 2.134 | 0.95 | 0.022 ± 0.001 (3) | 10 |
| TabFormer (TabBERT) | 0.028 | 0.548 | 0.86 | 0.028 ± 0.004 (3) | 0.022 | 2.128 | 0.95 | 0.022 ± 0.001 (3) | 10 |
| MM-TPP | 0.040 | 0.623 | 0.78 | 0.044 ± 0.004 (3) | 0.016 | 2.042 | 0.98 | 0.016 ± 0.000 (3) | 10 |
| Marked Markov chain | 0.004* | 0.529 | 0.61 | 0.004 ± 0.001 (3) | 0.003* | 1.902 | 0.52 | 0.003 ± 0.002 (3) | 10 |

#### Next-type acc. vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 30 665 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | hierarchical | 0.461 | 0.483 | 0.510 | 0.543 | 0.561 | 0.568 | 0.575 | 0.580 | 0.583 | 0.586 |
| Transaction autoencoder | reconstruction | 0.446 | 0.446 | 0.464 | 0.539 | 0.553 | 0.567 | 0.575 | 0.578 | 0.582 | 0.586 |
| CoLES | contrastive | 0.446 | 0.446 | 0.485 | 0.536 | 0.561 | 0.571 | 0.576 | 0.580 | 0.583 | 0.586 |
| Supervised GRU | supervised | 0.446 | 0.446 | 0.447 | 0.525 | 0.557 | 0.568 | 0.575 | 0.579 | 0.583 | 0.586 |
| Neural TPP (GRU) | neural-tpp | 0.446 | 0.446 | 0.447 | 0.525 | 0.557 | 0.568 | 0.575 | 0.579 | 0.583 | 0.586 |
| PRAGMA | hierarchical | 0.461 | 0.482 | 0.512 | 0.546 | 0.561 | 0.568 | 0.574 | 0.579 | 0.583 | 0.585 |
| Transaction MLM | reconstruction | 0.464 | 0.506 | 0.532 | 0.553 | 0.563 | 0.570 | 0.575 | 0.579 | 0.582 | 0.585 |
| NVIDIA TFM blueprint | tabular-transformer | 0.460 | 0.503 | 0.530 | 0.550 | 0.559 | 0.566 | 0.574 | 0.578 | 0.582 | 0.584 |
| Autoregressive transaction Transformer | generative | 0.461 | 0.505 | 0.530 | 0.549 | 0.557 | 0.566 | 0.574 | 0.577 | 0.581 | 0.584 |
| Transformer Hawkes | neural-tpp | 0.464 | 0.507 | 0.535 | 0.553 | 0.563 | 0.570 | 0.575 | 0.578 | 0.582 | 0.584 |
| Mambular (Mamba SSM) | state-space | 0.488 | 0.525 | 0.547 | 0.557 | 0.564 | 0.566 | 0.570 | 0.575 | 0.579 | 0.582 |
| Language-TPP | llm | 0.462 | 0.507 | 0.540 (1) | 0.551 | 0.565 (2) | 0.568 | 0.573 | — | 0.579 | 0.581 (2) |
| TPP-LLM | llm | 0.446 | 0.436 (1) | 0.478 | 0.532 | 0.552 | 0.566 | 0.573 (2) | 0.576 (2) | 0.580 (2) | 0.581 |
| TabFormer (TabGPT) | tabular-transformer | 0.484 | 0.513 | 0.533 | 0.550 | 0.557 | 0.564 | 0.569 | 0.573 | 0.577 | 0.580 |
| TabFormer (TabBERT) | tabular-transformer | 0.492 | 0.516 | 0.537 | 0.554 | 0.560 | 0.567 | 0.571 | 0.573 | 0.577 | 0.580 |
| MM-TPP | llm | 0.437 | 0.462 | 0.522 (1) | 0.541 (2) | 0.550 (2) | 0.557 | 0.562 | 0.565 (2) | 0.567 (1) | 0.569 |
| Marked Markov chain | classic | 0.472 | 0.481 | 0.486 | 0.487 | 0.487 | 0.488 | 0.489 | 0.489 | 0.489 | 0.489 |

#### Next-type macro F1 vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 30 665 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | hierarchical | 0.019 | 0.037 | 0.054 | 0.071 | 0.094 | 0.109 | 0.130 | 0.164 | 0.190 | 0.201 |
| Transaction autoencoder | reconstruction | 0.012 | 0.012 | 0.018 | 0.050 | 0.062 | 0.083 | 0.109 | 0.132 | 0.167 | 0.193 |
| CoLES | contrastive | 0.012 | 0.012 | 0.027 | 0.047 | 0.074 | 0.101 | 0.123 | 0.149 | 0.180 | 0.196 |
| Supervised GRU | supervised | 0.012 | 0.012 | 0.014 | 0.041 | 0.064 | 0.086 | 0.116 | 0.141 | 0.176 | 0.195 |
| Neural TPP (GRU) | neural-tpp | 0.012 | 0.012 | 0.014 | 0.041 | 0.064 | 0.086 | 0.116 | 0.141 | 0.176 | 0.195 |
| PRAGMA | hierarchical | 0.021 | 0.036 | 0.054 | 0.071 | 0.095 | 0.110 | 0.134 | 0.167 | 0.193 | 0.202 |
| Transaction MLM | reconstruction | 0.024 | 0.044 | 0.066 | 0.079 | 0.105 | 0.118 | 0.140 | 0.168 | 0.190 | 0.201 |
| NVIDIA TFM blueprint | tabular-transformer | 0.019 | 0.041 | 0.061 | 0.077 | 0.102 | 0.115 | 0.138 | 0.167 | 0.194 | 0.204 |
| Autoregressive transaction Transformer | generative | 0.020 | 0.042 | 0.061 | 0.077 | 0.102 | 0.121 | 0.141 | 0.166 | 0.192 | 0.207 |
| Transformer Hawkes | neural-tpp | 0.023 | 0.042 | 0.063 | 0.078 | 0.106 | 0.124 | 0.151 | 0.176 | 0.194 | 0.203 |
| Mambular (Mamba SSM) | state-space | 0.030 | 0.044 | 0.073 | 0.082 | 0.109 | 0.130 | 0.149 | 0.170 | 0.190 | 0.201 |
| Language-TPP | llm | 0.017 | 0.030 | 0.053 (1) | 0.074 | 0.110 (2) | 0.119 | 0.148 | — | 0.188 | 0.196 (2) |
| TPP-LLM | llm | 0.012 | 0.017 (1) | 0.025 | 0.048 | 0.069 | 0.108 | 0.142 (2) | 0.181 (2) | 0.193 (2) | 0.201 |
| TabFormer (TabGPT) | tabular-transformer | 0.032 | 0.046 | 0.073 | 0.085 | 0.107 | 0.123 | 0.151 | 0.172 | 0.189 | 0.198 |
| TabFormer (TabBERT) | tabular-transformer | 0.036 | 0.047 | 0.076 | 0.087 | 0.110 | 0.128 | 0.156 | 0.175 | 0.190 | 0.199 |
| MM-TPP | llm | 0.014 | 0.019 | 0.045 (1) | 0.059 (2) | 0.086 (2) | 0.113 | 0.133 | 0.161 (2) | 0.168 (1) | 0.175 |
| Marked Markov chain | classic | 0.056 | 0.064 | 0.074 | 0.073 | 0.076 | 0.086 | 0.081 | 0.088 | 0.091 | 0.088 |

#### Time RMSE (log1p s) vs. training sequences (lower is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 30 665 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | hierarchical | 1.970 | 1.836 | 1.627 | 1.593 | 1.572 | 1.557 | 1.548 | 1.542 | 1.532 | 1.528 |
| Transaction autoencoder | reconstruction | 1.973 | 1.954 | 1.640 | 1.583 | 1.563 | 1.554 | 1.548 | 1.540 | 1.531 | 1.526 |
| CoLES | contrastive | 1.971 | 1.932 | 1.615 | 1.581 | 1.564 | 1.553 | 1.545 | 1.536 | 1.530 | 1.526 |
| Supervised GRU | supervised | 1.972 | 1.958 | 1.670 | 1.587 | 1.565 | 1.554 | 1.547 | 1.537 | 1.530 | 1.526 |
| Neural TPP (GRU) | neural-tpp | 1.972 | 1.958 | 1.670 | 1.587 | 1.565 | 1.554 | 1.547 | 1.537 | 1.530 | 1.526 |
| PRAGMA | hierarchical | 1.966 | 1.837 | 1.645 | 1.594 | 1.571 | 1.557 | 1.550 | 1.542 | 1.537 | 1.530 |
| Transaction MLM | reconstruction | 1.819 | 1.668 | 1.620 | 1.586 | 1.564 | 1.556 | 1.546 | 1.538 | 1.531 | 1.527 |
| NVIDIA TFM blueprint | tabular-transformer | 1.847 | 1.689 | 1.623 | 1.589 | 1.569 | 1.560 | 1.549 | 1.540 | 1.532 | 1.527 |
| Autoregressive transaction Transformer | generative | 1.847 | 1.682 | 1.623 | 1.586 | 1.571 | 1.558 | 1.548 | 1.539 | 1.532 | 1.527 |
| Transformer Hawkes | neural-tpp | 1.764 | 1.660 | 1.611 | 1.585 | 1.569 | 1.561 | 1.554 | 1.547 | 1.542 | 1.539 |
| Mambular (Mamba SSM) | state-space | 1.777 | 1.628 | 1.603 | 1.576 | 1.570 | 1.560 | 1.549 | 1.539 | 1.531 | 1.527 |
| Language-TPP | llm | 1.933 | 1.890 | 1.853 (1) | 1.841 | 1.821 (2) | 1.799 | 1.778 | — | 1.749 | 1.739 (2) |
| TPP-LLM | llm | 1.975 | 1.892 (1) | 1.847 | 1.717 | 1.672 | 1.632 | 1.605 (2) | 1.589 (2) | 1.576 (2) | 1.565 |
| TabFormer (TabGPT) | tabular-transformer | 1.971 | 1.934 | 1.885 | 1.835 | 1.804 | 1.782 | 1.764 | 1.746 | 1.731 | 1.717 |
| TabFormer (TabBERT) | tabular-transformer | 1.967 | 1.936 | 1.870 | 1.832 | 1.801 | 1.776 | 1.759 | 1.742 | 1.727 | 1.715 |
| MM-TPP | llm | 1.921 | 1.897 | 1.880 (1) | 1.849 (2) | 1.821 (2) | 1.801 | 1.783 | 1.770 (2) | 1.762 (1) | 1.749 |
| Marked Markov chain | classic | 1.905 | 1.872 | 1.864 | 1.861 | 1.859 | 1.858 | 1.857 | 1.857 | 1.857 | 1.857 |

#### Time MAE (log1p s) vs. training sequences (lower is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 30 665 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRAGMA + masked pretraining | hierarchical | 1.510 | 1.419 | 1.250 | 1.199 | 1.195 | 1.178 | 1.170 | 1.162 | 1.154 | 1.150 |
| Transaction autoencoder | reconstruction | 1.511 | 1.501 | 1.259 | 1.208 | 1.199 | 1.186 | 1.184 | 1.176 | 1.164 | 1.156 |
| CoLES | contrastive | 1.510 | 1.485 | 1.246 | 1.210 | 1.199 | 1.185 | 1.180 | 1.171 | 1.163 | 1.156 |
| Supervised GRU | supervised | 1.510 | 1.505 | 1.280 | 1.213 | 1.201 | 1.186 | 1.181 | 1.172 | 1.163 | 1.157 |
| Neural TPP (GRU) | neural-tpp | 1.510 | 1.505 | 1.280 | 1.213 | 1.201 | 1.186 | 1.181 | 1.172 | 1.163 | 1.157 |
| PRAGMA | hierarchical | 1.508 | 1.420 | 1.268 | 1.204 | 1.195 | 1.172 | 1.167 | 1.161 | 1.162 | 1.149 |
| Transaction MLM | reconstruction | 1.388 | 1.275 | 1.245 | 1.203 | 1.189 | 1.178 | 1.172 | 1.164 | 1.157 | 1.152 |
| NVIDIA TFM blueprint | tabular-transformer | 1.408 | 1.290 | 1.248 | 1.206 | 1.195 | 1.179 | 1.172 | 1.167 | 1.159 | 1.154 |
| Autoregressive transaction Transformer | generative | 1.404 | 1.284 | 1.246 | 1.206 | 1.197 | 1.184 | 1.174 | 1.172 | 1.164 | 1.156 |
| Transformer Hawkes | neural-tpp | 1.347 | 1.274 | 1.239 | 1.205 | 1.198 | 1.188 | 1.183 | 1.178 | 1.174 | 1.169 |
| Mambular (Mamba SSM) | state-space | 1.367 | 1.277 | 1.253 | 1.209 | 1.212 | 1.190 | 1.179 | 1.170 | 1.162 | 1.156 |
| Language-TPP | llm | 1.478 | 1.451 | 1.415 (1) | 1.403 | 1.390 (2) | 1.370 | 1.352 | — | 1.337 | 1.329 (2) |
| TPP-LLM | llm | 1.515 | 1.451 (1) | 1.416 | 1.307 | 1.284 | 1.247 | 1.224 (2) | 1.215 (2) | 1.201 (2) | 1.198 |
| TabFormer (TabGPT) | tabular-transformer | 1.511 | 1.489 | 1.459 | 1.410 | 1.384 | 1.361 | 1.347 | 1.329 | 1.319 | 1.309 |
| TabFormer (TabBERT) | tabular-transformer | 1.509 | 1.490 | 1.441 | 1.399 | 1.378 | 1.355 | 1.342 | 1.330 | 1.317 | 1.309 |
| MM-TPP | llm | 1.471 | 1.458 | 1.453 (1) | 1.410 (2) | 1.403 (2) | 1.373 | 1.358 | 1.355 (2) | 1.345 (1) | 1.340 |
| Marked Markov chain | classic | 1.447 | 1.435 | 1.427 | 1.426 | 1.422 | 1.420 | 1.423 | 1.424 | 1.424 | 1.424 |

## Synthea EHR

### Classification

#### All metrics at the full pool (n = 93305)

| Method | Fidelity | Seeds | ROC-AUC | Avg. precision | Accuracy | Macro F1 | Wall s (median) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised GRU | Implemented | 3 | 0.882 ± 0.001 | 0.367 ± 0.007 | 0.904 ± 0.002 | 0.669 ± 0.001 | 5,412 |
| Neural TPP (GRU) | Approximation | 3 | 0.882 ± 0.001 | 0.367 ± 0.007 | 0.904 ± 0.002 | 0.669 ± 0.001 | 5,438 |
| Transaction MLM | Implemented | 3 | 0.881 ± 0.001 | 0.342 ± 0.017 | 0.902 ± 0.003 | 0.666 ± 0.003 | 3,217 |
| Autoregressive transaction Transformer | Implemented | 3 | 0.879 ± 0.000 | 0.336 ± 0.006 | 0.900 ± 0.001 | 0.664 ± 0.000 | 5,540 |
| NVIDIA TFM blueprint | Approximation | 3 | 0.877 ± 0.000 | 0.315 ± 0.011 | 0.898 ± 0.003 | 0.662 ± 0.001 | 5,783 |
| PRAGMA + masked pretraining | Approximation | 3 | 0.876 ± 0.001 | 0.339 ± 0.007 | 0.902 ± 0.003 | 0.659 ± 0.002 | 12,008 |
| CoLES | Approximation | 3 | 0.876 ± 0.001 | 0.300 ± 0.006 | 0.899 ± 0.002 | 0.658 ± 0.001 | 5,449 |
| Transaction autoencoder | Implemented | 3 | 0.876 ± 0.001 | 0.314 ± 0.011 | 0.899 ± 0.001 | 0.661 ± 0.002 | 3,849 |
| Engineered features + GBDT | Implemented | 3 | 0.875 ± 0.000 | 0.300 ± 0.001 | 0.903 ± 0.000 | 0.658 ± 0.000 | 208 |
| Mambular (Mamba SSM) | Approximation | 3 | 0.871 ± 0.003 | 0.295 ± 0.018 | 0.892 ± 0.003 | 0.653 ± 0.002 | 81,690 |
| Transformer Hawkes | Approximation | 3 | 0.870 ± 0.001 | 0.291 ± 0.006 | 0.901 ± 0.002 | 0.658 ± 0.001 | 4,452 |
| PRAGMA | Approximation | 3 | 0.869 ± 0.001 | 0.285 ± 0.008 | 0.897 ± 0.003 | 0.646 ± 0.001 | 9,954 |
| Count + logistic regression | Implemented | 3 | 0.833 ± 0.000 | 0.220 ± 0.000 | 0.910 ± 0.000 | 0.605 ± 0.000 | 115 |
| TPP-LLM | Approximation | 3 | 0.826 ± 0.001 | 0.223 ± 0.002 | 0.908 ± 0.001 | 0.612 ± 0.002 | 10,070 |
| TabFormer (TabBERT) | Approximation | 3 | 0.826 ± 0.001 | 0.275 ± 0.002 | 0.920 ± 0.002 | 0.608 ± 0.001 | 4,843 |
| TabFormer (TabGPT) | Approximation | 3 | 0.823 ± 0.001 | 0.266 ± 0.003 | 0.918 ± 0.002 | 0.607 ± 0.001 | 5,819 |
| Language-TPP | Approximation | 3 | 0.809 ± 0.002 | 0.194 ± 0.010 | 0.906 ± 0.006 | 0.586 ± 0.007 | 11,565 |

#### Scaling fits

`E(N) = a · N^-b` on seed means, as in the summary above. The per-seed column fits each seed's own curve separately (four or more points) and gives the spread of `b` across seeds. `*` flags an unreliable fit.

| Method | b (1 − ROC-AUC) | a | R² | per-seed b (mean ± sd, n) | b (1 − avg. precision) | a | R² | per-seed b (mean ± sd, n) | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised GRU | 0.144 | 0.547 | 0.89 | 0.144 ± 0.012 (3) | 0.046 | 1.117 | 0.97 | 0.046 ± 0.003 (3) | 12 |
| Neural TPP (GRU) | 0.144 | 0.547 | 0.89 | 0.144 ± 0.012 (3) | 0.046 | 1.117 | 0.97 | 0.046 ± 0.003 (3) | 12 |
| Transaction MLM | 0.091 | 0.328 | 0.96 | 0.091 ± 0.001 (3) | 0.035 | 1.014 | 0.93 | 0.035 ± 0.001 (3) | 12 |
| Autoregressive transaction Transformer | 0.148 | 0.591 | 0.92 | 0.148 ± 0.004 (3) | 0.041 | 1.092 | 0.98 | 0.041 ± 0.001 (3) | 12 |
| NVIDIA TFM blueprint | 0.151 | 0.599 | 0.89 | 0.151 ± 0.005 (3) | 0.039 | 1.073 | 0.99 | 0.039 ± 0.001 (3) | 12 |
| PRAGMA + masked pretraining | 0.105 | 0.396 | 0.98 | 0.105 ± 0.006 (3) | 0.037 | 1.047 | 0.93 | 0.037 ± 0.001 (3) | 12 |
| CoLES | 0.138 | 0.519 | 0.80 | 0.137 ± 0.021 (3) | 0.036 | 1.046 | 0.97 | 0.036 ± 0.003 (3) | 12 |
| Transaction autoencoder | 0.197 | 0.975 | 0.80 | 0.197 ± 0.006 (3) | 0.043 | 1.137 | 0.96 | 0.043 ± 0.002 (3) | 12 |
| Engineered features + GBDT | 0.090 | 0.326 | 0.95 | 0.090 ± 0.008 (3) | 0.027 | 0.943 | 1.00 | 0.027 ± 0.001 (3) | 12 |
| Mambular (Mamba SSM) | 0.129 | 0.517 | 0.90 | 0.128 ± 0.006 (3) | 0.031 | 1.024 | 0.98 | 0.031 ± 0.001 (3) | 12 |
| Transformer Hawkes | 0.188 | 0.909 | 0.86 | 0.187 ± 0.008 (3) | 0.039 | 1.095 | 0.96 | 0.039 ± 0.002 (3) | 12 |
| PRAGMA | 0.096 | 0.379 | 0.98 | 0.096 ± 0.005 (3) | 0.027 | 0.983 | 0.97 | 0.027 ± 0.002 (3) | 12 |
| Count + logistic regression | 0.084 | 0.389 | 0.82 | 0.084 ± 0.009 (3) | 0.018 | 0.932 | 0.86 | 0.018 ± 0.001 (3) | 12 |
| TPP-LLM | 0.131 | 0.694 | 0.92 | 0.132 ± 0.009 (3) | 0.025 | 1.033 | 0.98 | 0.026 ± 0.002 (3) | 11 |
| TabFormer (TabBERT) | 0.057 | 0.321 | 0.89 | 0.057 ± 0.005 (3) | 0.025 | 0.981 | 0.98 | 0.025 ± 0.001 (3) | 12 |
| TabFormer (TabGPT) | 0.137 | 0.724 | 0.79 | 0.137 ± 0.009 (3) | 0.031 | 1.060 | 0.96 | 0.031 ± 0.001 (3) | 12 |
| Language-TPP | 0.137 | 0.836 | 0.95 | 0.135 ± 0.003 (3) | 0.022 | 1.030 | 0.96 | 0.021 ± 0.001 (3) | 11 |
| MM-TPP | 0.131 | 0.767 | 0.95 | 0.126 ± 0.004 (3) | 0.020 | 1.013 | 0.95 | 0.020 ± 0.001 (3) | 11 |

#### ROC-AUC vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 | 65 536 | 93 305 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised GRU | supervised | 0.612 | 0.692 | 0.788 | 0.806 | 0.815 | 0.833 | 0.848 | 0.860 | 0.867 | 0.872 | 0.879 | 0.882 |
| Neural TPP (GRU) | neural-tpp | 0.612 | 0.692 | 0.788 | 0.806 | 0.815 | 0.833 | 0.848 | 0.860 | 0.867 | 0.872 | 0.879 | 0.882 |
| Transaction MLM | reconstruction | 0.788 | 0.787 | 0.793 | 0.801 | 0.816 | 0.846 | 0.857 | 0.862 | 0.867 | 0.874 | 0.878 | 0.881 |
| Autoregressive transaction Transformer | generative | 0.611 | 0.669 | 0.769 | 0.791 | 0.808 | 0.820 | 0.844 | 0.855 | 0.862 | 0.870 | 0.876 | 0.879 |
| NVIDIA TFM blueprint | tabular-transformer | 0.600 | 0.657 | 0.766 | 0.797 | 0.809 | 0.832 | 0.846 | 0.860 | 0.865 | 0.869 | 0.875 | 0.877 |
| PRAGMA + masked pretraining | hierarchical | 0.749 | 0.765 | 0.763 | 0.795 | 0.806 | 0.822 | 0.843 | 0.855 | 0.862 | 0.869 | 0.874 | 0.876 |
| CoLES | contrastive | 0.550 | 0.746 | 0.790 | 0.804 | 0.817 | 0.842 | 0.857 | 0.863 | 0.866 | 0.870 | 0.875 | 0.876 |
| Transaction autoencoder | reconstruction | 0.377 | 0.442 | 0.757 | 0.795 | 0.806 | 0.806 | 0.816 | 0.845 | 0.863 | 0.868 | 0.873 | 0.876 |
| Engineered features + GBDT | classic | 0.752 | 0.781 | 0.808 | 0.821 | 0.832 | 0.844 | 0.851 | 0.858 | 0.868 | 0.871 | 0.874 | 0.875 |
| Mambular (Mamba SSM) | state-space | 0.621 | 0.701 | 0.764 | 0.806 | 0.810 | 0.816 | 0.829 | 0.841 | 0.852 | 0.860 | 0.869 | 0.871 |
| Transformer Hawkes | neural-tpp | 0.524 | 0.555 | 0.587 | 0.759 | 0.813 | 0.826 | 0.841 | 0.852 | 0.858 | 0.863 | 0.867 | 0.870 |
| PRAGMA | hierarchical | 0.752 | 0.759 | 0.765 | 0.792 | 0.802 | 0.818 | 0.833 | 0.848 | 0.855 | 0.862 | 0.864 | 0.869 |
| Count + logistic regression | classic | 0.689 | 0.709 | 0.751 | 0.784 | 0.805 | 0.820 | 0.825 | 0.829 | 0.831 | 0.832 | 0.833 | 0.833 |
| TPP-LLM | llm | 0.574 (2) | 0.580 | 0.646 | — | 0.756 | 0.775 (2) | 0.791 | 0.803 | 0.811 (1) | 0.816 (2) | 0.824 (2) | 0.826 |
| TabFormer (TabBERT) | tabular-transformer | 0.722 | 0.751 | 0.763 | 0.792 | 0.796 | 0.802 | 0.807 | 0.812 | 0.817 | 0.820 | 0.823 | 0.826 |
| TabFormer (TabGPT) | tabular-transformer | 0.522 | 0.535 | 0.593 | 0.757 | 0.783 | 0.791 | 0.798 | 0.802 | 0.811 | 0.815 | 0.819 | 0.823 |
| Language-TPP | llm | 0.528 | 0.566 (2) | 0.601 (1) | 0.617 | 0.653 (2) | 0.723 | 0.769 | — | 0.798 | 0.803 (2) | 0.805 | 0.809 |
| MM-TPP | llm | 0.540 (1) | 0.584 (2) | 0.614 (2) | 0.650 | 0.692 | 0.745 (2) | 0.770 (1) | 0.784 | 0.790 (2) | 0.795 | 0.797 | — |

#### Avg. precision vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 | 65 536 | 93 305 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised GRU | supervised | 0.088 | 0.113 | 0.153 | 0.165 | 0.176 | 0.202 | 0.230 | 0.252 | 0.269 | 0.292 | 0.348 | 0.367 |
| Neural TPP (GRU) | neural-tpp | 0.088 | 0.113 | 0.153 | 0.165 | 0.176 | 0.202 | 0.230 | 0.252 | 0.269 | 0.292 | 0.348 | 0.367 |
| Transaction MLM | reconstruction | 0.159 | 0.154 | 0.151 | 0.159 | 0.175 | 0.225 | 0.244 | 0.253 | 0.266 | 0.287 | 0.312 | 0.342 |
| Autoregressive transaction Transformer | generative | 0.093 | 0.108 | 0.144 | 0.154 | 0.170 | 0.181 | 0.219 | 0.243 | 0.260 | 0.277 | 0.320 | 0.336 |
| NVIDIA TFM blueprint | tabular-transformer | 0.089 | 0.107 | 0.147 | 0.158 | 0.175 | 0.202 | 0.228 | 0.250 | 0.262 | 0.277 | 0.312 | 0.315 |
| PRAGMA + masked pretraining | hierarchical | 0.132 | 0.140 | 0.135 | 0.179 | 0.183 | 0.185 | 0.209 | 0.238 | 0.256 | 0.281 | 0.330 | 0.339 |
| CoLES | contrastive | 0.067 | 0.135 | 0.149 | 0.163 | 0.177 | 0.218 | 0.241 | 0.253 | 0.261 | 0.272 | 0.289 | 0.300 |
| Transaction autoencoder | reconstruction | 0.047 | 0.053 | 0.131 | 0.154 | 0.160 | 0.160 | 0.173 | 0.218 | 0.254 | 0.269 | 0.292 | 0.314 |
| Engineered features + GBDT | classic | 0.155 | 0.168 | 0.186 | 0.204 | 0.213 | 0.231 | 0.244 | 0.257 | 0.278 | 0.286 | 0.296 | 0.300 |
| Mambular (Mamba SSM) | state-space | 0.097 | 0.124 | 0.145 | 0.167 | 0.173 | 0.182 | 0.199 | 0.220 | 0.237 | 0.256 | 0.275 | 0.295 |
| Transformer Hawkes | neural-tpp | 0.070 | 0.077 | 0.089 | 0.144 | 0.195 | 0.208 | 0.226 | 0.239 | 0.252 | 0.265 | 0.283 | 0.291 |
| PRAGMA | hierarchical | 0.133 | 0.139 | 0.137 | 0.174 | 0.182 | 0.189 | 0.204 | 0.229 | 0.245 | 0.260 | 0.260 | 0.285 |
| Count + logistic regression | classic | 0.118 | 0.126 | 0.146 | 0.173 | 0.192 | 0.205 | 0.211 | 0.215 | 0.218 | 0.220 | 0.220 | 0.220 |
| TPP-LLM | llm | 0.073 (2) | 0.072 | 0.095 | — | 0.147 | 0.159 (2) | 0.170 | 0.181 | 0.193 (1) | 0.201 (2) | 0.217 (2) | 0.223 |
| TabFormer (TabBERT) | tabular-transformer | 0.125 | 0.133 | 0.139 | 0.171 | 0.178 | 0.187 | 0.200 | 0.209 | 0.228 | 0.241 | 0.266 | 0.275 |
| TabFormer (TabGPT) | tabular-transformer | 0.069 | 0.073 | 0.088 | 0.142 | 0.164 | 0.172 | 0.181 | 0.189 | 0.202 | 0.214 | 0.247 | 0.266 |
| Language-TPP | llm | 0.071 | 0.075 (2) | 0.080 (1) | 0.083 | 0.097 (2) | 0.129 | 0.152 | — | 0.173 | 0.179 (2) | 0.181 | 0.194 |
| MM-TPP | llm | 0.076 (1) | 0.079 (2) | 0.085 (2) | 0.094 | 0.113 | 0.136 (2) | 0.156 (1) | 0.165 | 0.170 (2) | 0.174 | 0.177 | — |

#### Accuracy vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 | 65 536 | 93 305 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised GRU | supervised | 0.936 | 0.936 | 0.936 | 0.894 | 0.885 | 0.893 | 0.896 | 0.887 | 0.891 | 0.890 | 0.901 | 0.904 |
| Neural TPP (GRU) | neural-tpp | 0.936 | 0.936 | 0.936 | 0.894 | 0.885 | 0.893 | 0.896 | 0.887 | 0.891 | 0.890 | 0.901 | 0.904 |
| Transaction MLM | reconstruction | 0.936 | 0.936 | 0.936 | 0.883 | 0.888 | 0.896 | 0.882 | 0.877 | 0.886 | 0.891 | 0.898 | 0.902 |
| Autoregressive transaction Transformer | generative | 0.936 | 0.936 | 0.936 | 0.888 | 0.877 | 0.885 | 0.885 | 0.884 | 0.887 | 0.888 | 0.896 | 0.900 |
| NVIDIA TFM blueprint | tabular-transformer | 0.936 | 0.936 | 0.936 | 0.872 | 0.860 | 0.847 | 0.866 | 0.880 | 0.885 | 0.887 | 0.896 | 0.898 |
| PRAGMA + masked pretraining | hierarchical | 0.936 | 0.936 | 0.936 | 0.898 | 0.892 | 0.905 | 0.889 | 0.878 | 0.889 | 0.890 | 0.901 | 0.902 |
| CoLES | contrastive | 0.936 | 0.936 | 0.934 | 0.873 | 0.903 | 0.894 | 0.887 | 0.882 | 0.890 | 0.892 | 0.898 | 0.899 |
| Transaction autoencoder | reconstruction | 0.936 | 0.936 | 0.936 | 0.912 | 0.925 | 0.936 | 0.926 | 0.885 | 0.887 | 0.888 | 0.896 | 0.899 |
| Engineered features + GBDT | classic | 0.872 | 0.864 | 0.882 | 0.883 | 0.881 | 0.884 | 0.890 | 0.891 | 0.899 | 0.901 | 0.903 | 0.903 |
| Mambular (Mamba SSM) | state-space | 0.936 | 0.936 | 0.930 | 0.822 | 0.809 | 0.831 | 0.869 | 0.868 | 0.872 | 0.877 | 0.887 | 0.892 |
| Transformer Hawkes | neural-tpp | 0.936 | 0.936 | 0.936 | 0.922 | 0.884 | 0.881 | 0.884 | 0.889 | 0.894 | 0.894 | 0.898 | 0.901 |
| PRAGMA | hierarchical | 0.936 | 0.936 | 0.936 | 0.903 | 0.897 | 0.903 | 0.907 | 0.887 | 0.895 | 0.892 | 0.892 | 0.897 |
| Count + logistic regression | classic | 0.841 | 0.816 | 0.831 | 0.861 | 0.876 | 0.891 | 0.897 | 0.903 | 0.906 | 0.909 | 0.909 | 0.910 |
| TPP-LLM | llm | 0.930 (2) | 0.934 | 0.930 | — | 0.915 | 0.874 (2) | 0.887 | 0.890 | 0.912 (1) | 0.898 (2) | 0.904 (2) | 0.908 |
| TabFormer (TabBERT) | tabular-transformer | 0.936 | 0.936 | 0.936 | 0.902 | 0.894 | 0.892 | 0.893 | 0.902 | 0.913 | 0.912 | 0.915 | 0.920 |
| TabFormer (TabGPT) | tabular-transformer | 0.936 | 0.936 | 0.936 | 0.886 | 0.819 | 0.842 | 0.869 | 0.884 | 0.899 | 0.908 | 0.916 | 0.918 |
| Language-TPP | llm | 0.745 | 0.870 (2) | 0.932 (1) | 0.919 | 0.933 (2) | 0.925 | 0.914 | — | 0.903 | 0.894 (2) | 0.901 | 0.906 |
| MM-TPP | llm | 0.608 (1) | 0.891 (2) | 0.882 (2) | 0.915 | 0.930 | 0.915 (2) | 0.911 (1) | 0.894 | 0.895 (2) | 0.895 | 0.894 | — |

#### Macro F1 vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 | 65 536 | 93 305 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised GRU | supervised | 0.483 | 0.483 | 0.483 | 0.556 | 0.575 | 0.594 | 0.616 | 0.635 | 0.648 | 0.654 | 0.664 | 0.669 |
| Neural TPP (GRU) | neural-tpp | 0.483 | 0.483 | 0.483 | 0.556 | 0.575 | 0.594 | 0.616 | 0.635 | 0.648 | 0.654 | 0.664 | 0.669 |
| Transaction MLM | reconstruction | 0.483 | 0.483 | 0.483 | 0.557 | 0.570 | 0.613 | 0.631 | 0.638 | 0.646 | 0.655 | 0.661 | 0.666 |
| Autoregressive transaction Transformer | generative | 0.483 | 0.483 | 0.483 | 0.553 | 0.573 | 0.578 | 0.614 | 0.629 | 0.641 | 0.651 | 0.661 | 0.664 |
| NVIDIA TFM blueprint | tabular-transformer | 0.483 | 0.483 | 0.483 | 0.563 | 0.579 | 0.601 | 0.619 | 0.636 | 0.644 | 0.650 | 0.660 | 0.662 |
| PRAGMA + masked pretraining | hierarchical | 0.483 | 0.483 | 0.483 | 0.573 | 0.585 | 0.563 | 0.601 | 0.627 | 0.636 | 0.646 | 0.657 | 0.659 |
| CoLES | contrastive | 0.483 | 0.483 | 0.494 | 0.557 | 0.559 | 0.608 | 0.629 | 0.637 | 0.642 | 0.650 | 0.657 | 0.658 |
| Transaction autoencoder | reconstruction | 0.483 | 0.483 | 0.483 | 0.524 | 0.507 | 0.483 | 0.518 | 0.609 | 0.637 | 0.647 | 0.656 | 0.661 |
| Engineered features + GBDT | classic | 0.576 | 0.581 | 0.590 | 0.602 | 0.608 | 0.620 | 0.629 | 0.637 | 0.650 | 0.654 | 0.658 | 0.658 |
| Mambular (Mamba SSM) | state-space | 0.483 | 0.483 | 0.494 | 0.573 | 0.577 | 0.587 | 0.604 | 0.615 | 0.629 | 0.639 | 0.649 | 0.653 |
| Transformer Hawkes | neural-tpp | 0.483 | 0.483 | 0.483 | 0.517 | 0.594 | 0.606 | 0.622 | 0.630 | 0.640 | 0.647 | 0.655 | 0.658 |
| PRAGMA | hierarchical | 0.483 | 0.483 | 0.483 | 0.567 | 0.577 | 0.577 | 0.592 | 0.612 | 0.626 | 0.634 | 0.636 | 0.646 |
| Count + logistic regression | classic | 0.546 | 0.548 | 0.563 | 0.584 | 0.596 | 0.603 | 0.604 | 0.606 | 0.604 | 0.606 | 0.605 | 0.605 |
| TPP-LLM | llm | 0.483 (2) | 0.483 | 0.494 | — | 0.542 | 0.577 (2) | 0.558 | 0.581 | 0.581 (1) | 0.599 (2) | 0.611 (2) | 0.612 |
| TabFormer (TabBERT) | tabular-transformer | 0.483 | 0.483 | 0.483 | 0.566 | 0.580 | 0.590 | 0.594 | 0.598 | 0.599 | 0.604 | 0.609 | 0.608 |
| TabFormer (TabGPT) | tabular-transformer | 0.483 | 0.483 | 0.483 | 0.550 | 0.570 | 0.581 | 0.592 | 0.593 | 0.598 | 0.600 | 0.605 | 0.607 |
| Language-TPP | llm | 0.467 | 0.504 (2) | 0.486 (1) | 0.495 | 0.489 (2) | 0.516 | 0.544 | — | 0.570 | 0.584 (2) | 0.580 | 0.586 |
| MM-TPP | llm | 0.445 (1) | 0.498 (2) | 0.509 (2) | 0.502 | 0.498 | 0.529 (2) | 0.548 (1) | 0.573 | 0.576 (2) | 0.578 | 0.581 | — |

### Temporal point process (TPP)

#### All metrics at the full pool (n = 93305)

| Method | Fidelity | Seeds | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) | Wall s (median) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction autoencoder | Implemented | 3 | 0.439 ± 0.000 | 0.189 ± 0.002 | 3.731 ± 0.001 | 2.530 ± 0.008 | 22,486 |
| Neural TPP (GRU) | Approximation | 3 | 0.439 ± 0.000 | 0.186 ± 0.001 | 3.728 ± 0.001 | 2.526 ± 0.005 | 43,844 |
| Supervised GRU | Implemented | 3 | 0.439 ± 0.000 | 0.186 ± 0.001 | 3.728 ± 0.001 | 2.526 ± 0.005 | 41,649 |
| CoLES | Approximation | 3 | 0.438 ± 0.001 | 0.186 ± 0.001 | 3.732 ± 0.005 | 2.540 ± 0.011 | 48,857 |
| PRAGMA + masked pretraining | Approximation | 3 | 0.437 ± 0.001 | 0.185 ± 0.001 | 3.711 ± 0.001 | 2.460 ± 0.005 | 79,023 |
| Transaction MLM | Implemented | 3 | 0.435 ± 0.001 | 0.194 ± 0.001 | 3.741 ± 0.001 | 2.532 ± 0.006 | 18,559 |
| PRAGMA | Approximation | 3 | 0.435 ± 0.001 | 0.185 ± 0.003 | 3.717 ± 0.003 | 2.475 ± 0.002 | 77,142 |
| NVIDIA TFM blueprint | Approximation | 3 | 0.434 ± 0.001 | 0.193 ± 0.001 | 3.751 ± 0.005 | 2.542 ± 0.002 | 33,639 |
| Transformer Hawkes | Approximation | 3 | 0.434 ± 0.001 | 0.192 ± 0.002 | 3.753 ± 0.007 | 2.551 ± 0.010 | 28,764 |
| Autoregressive transaction Transformer | Implemented | 3 | 0.433 ± 0.001 | 0.192 ± 0.002 | 3.750 ± 0.001 | 2.544 ± 0.002 | 32,081 |
| TPP-LLM | Approximation | 3 | 0.424 ± 0.000 | 0.190 ± 0.001 | 3.843 ± 0.004 | 2.672 ± 0.003 | 79,298 |
| TabFormer (TabBERT) | Approximation | 3 | 0.422 ± 0.000 | 0.188 ± 0.002 | 3.836 ± 0.001 | 2.671 ± 0.001 | 30,140 |
| TabFormer (TabGPT) | Approximation | 3 | 0.419 ± 0.000 | 0.184 ± 0.003 | 3.847 ± 0.003 | 2.696 ± 0.004 | 16,399 |
| Language-TPP | Approximation | 3 | 0.352 ± 0.000 | 0.122 ± 0.000 | 4.622 ± 0.004 | 3.590 ± 0.008 | 95,330 |
| Marked Markov chain | Implemented | 3 | 0.319 ± 0.000 | 0.123 ± 0.000 | 4.749 ± 0.000 | 3.875 ± 0.000 | 137 |

#### Scaling fits

`E(N) = a · N^-b` on seed means, as in the summary above. The per-seed column fits each seed's own curve separately (four or more points) and gives the spread of `b` across seeds. `*` flags an unreliable fit.

| Method | b (1 − next-type acc.) | a | R² | per-seed b (mean ± sd, n) | b (Time RMSE (log1p s)) | a | R² | per-seed b (mean ± sd, n) | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction autoencoder | 0.042 | 0.915 | 0.85 | 0.042 ± 0.000 (3) | 0.069 | 7.873 | 0.94 | 0.069 ± 0.002 (3) | 12 |
| Neural TPP (GRU) | 0.042 | 0.911 | 0.89 | 0.042 ± 0.000 (3) | 0.069 | 7.672 | 0.90 | 0.069 ± 0.001 (3) | 12 |
| Supervised GRU | 0.042 | 0.911 | 0.89 | 0.042 ± 0.000 (3) | 0.069 | 7.672 | 0.90 | 0.069 ± 0.001 (3) | 12 |
| CoLES | 0.042 | 0.908 | 0.90 | 0.042 ± 0.000 (3) | 0.070 | 7.750 | 0.89 | 0.069 ± 0.001 (3) | 12 |
| PRAGMA + masked pretraining | 0.038 | 0.870 | 0.97 | 0.038 ± 0.000 (3) | 0.038 | 5.517 | 0.86 | 0.038 ± 0.002 (3) | 12 |
| Transaction MLM | 0.038 | 0.853 | 0.98 | 0.038 ± 0.000 (3) | 0.052 | 6.365 | 0.81 | 0.052 ± 0.001 (3) | 12 |
| PRAGMA | 0.035 | 0.856 | 0.97 | 0.035 ± 0.001 (3) | 0.037 | 5.488 | 0.85 | 0.037 ± 0.002 (3) | 12 |
| NVIDIA TFM blueprint | 0.038 | 0.857 | 0.98 | 0.038 ± 0.000 (3) | 0.052 | 6.413 | 0.81 | 0.052 ± 0.000 (3) | 12 |
| Transformer Hawkes | 0.037 | 0.850 | 0.98 | 0.037 ± 0.001 (3) | 0.050 | 6.265 | 0.78 | 0.050 ± 0.001 (3) | 12 |
| Autoregressive transaction Transformer | 0.037 | 0.857 | 0.98 | 0.037 ± 0.000 (3) | 0.054 | 6.510 | 0.81 | 0.054 ± 0.001 (3) | 12 |
| TPP-LLM | 0.037 | 0.861 | 0.97 | 0.037 ± 0.001 (3) | 0.055 | 6.781 | 0.87 | 0.059 ± 0.008 (3) | 12 |
| TabFormer (TabBERT) | 0.032 | 0.818 | 0.98 | 0.032 ± 0.001 (3) | 0.042 | 5.878 | 0.72 | 0.042 ± 0.001 (3) | 12 |
| TabFormer (TabGPT) | 0.031 | 0.819 | 0.98 | 0.031 ± 0.000 (3) | 0.044 | 6.018 | 0.72 | 0.044 ± 0.001 (3) | 12 |
| Language-TPP | 0.018 | 0.793 | 0.98 | 0.018 ± 0.001 (3) | 0.026 | 6.097 | 0.95 | 0.026 ± 0.001 (3) | 10 |
| Marked Markov chain | 0.001* | 0.688 | 0.57 | 0.001 ± 0.000 (3) | 0.001* | 4.774 | 0.52 | 0.001 ± 0.000 (3) | 12 |
| Mambular (Mamba SSM) | 0.040 | 0.866 | 0.99 | 0.040 ± 0.001 (3) | 0.051 | 6.229 | 0.79 | 0.053 ± 0.004 (3) | 11 |
| MM-TPP | 0.014 | 0.782 | 0.94 | 0.014 ± 0.000 (3) | 0.026 | 6.134 | 0.95 | 0.026 ± 0.002 (3) | 10 |

#### Next-type acc. vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 | 65 536 | 93 305 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction autoencoder | reconstruction | 0.277 | 0.277 | 0.277 | 0.277 | 0.280 | 0.285 | 0.313 | 0.367 | 0.409 | 0.430 | 0.437 | 0.439 |
| Neural TPP (GRU) | neural-tpp | 0.277 | 0.277 | 0.277 | 0.277 | 0.284 | 0.298 | 0.341 | 0.375 | 0.410 | 0.430 | 0.437 | 0.439 |
| Supervised GRU | supervised | 0.277 | 0.277 | 0.277 | 0.277 | 0.284 | 0.298 | 0.341 | 0.375 | 0.410 | 0.430 | 0.437 | 0.439 |
| CoLES | contrastive | 0.277 | 0.277 | 0.277 | 0.277 | 0.280 | 0.304 | 0.348 | 0.376 | 0.406 | 0.427 | 0.435 | 0.438 |
| PRAGMA + masked pretraining | hierarchical | 0.276 | 0.286 | 0.293 | 0.307 | 0.320 | 0.332 | 0.359 | 0.376 | 0.410 | 0.427 | 0.433 | 0.437 |
| Transaction MLM | reconstruction | 0.277 | 0.281 | 0.297 | 0.326 | 0.343 | 0.358 | 0.378 | 0.403 | 0.417 | 0.425 | 0.432 | 0.435 |
| PRAGMA | hierarchical | 0.277 | 0.289 | 0.299 | 0.308 | 0.323 | 0.333 | 0.349 | 0.374 | 0.393 | 0.415 | 0.432 | 0.435 |
| NVIDIA TFM blueprint | tabular-transformer | 0.277 | 0.278 | 0.294 | 0.322 | 0.342 | 0.359 | 0.383 | 0.408 | 0.418 | 0.425 | 0.432 | 0.434 |
| Transformer Hawkes | neural-tpp | 0.277 | 0.279 | 0.300 | 0.328 | 0.346 | 0.359 | 0.380 | 0.401 | 0.417 | 0.425 | 0.431 | 0.434 |
| Autoregressive transaction Transformer | generative | 0.277 | 0.279 | 0.291 | 0.319 | 0.339 | 0.355 | 0.373 | 0.398 | 0.413 | 0.422 | 0.430 | 0.433 |
| TPP-LLM | llm | 0.276 | 0.276 (2) | 0.279 | 0.302 | 0.332 (1) | 0.355 (2) | 0.377 (2) | 0.397 | 0.408 | 0.415 (1) | 0.422 (1) | 0.424 |
| TabFormer (TabBERT) | tabular-transformer | 0.277 | 0.281 | 0.318 | 0.336 | 0.352 | 0.362 | 0.370 | 0.378 | 0.404 | 0.413 | 0.419 | 0.422 |
| TabFormer (TabGPT) | tabular-transformer | 0.277 | 0.278 | 0.312 | 0.336 | 0.350 | 0.361 | 0.369 | 0.380 | 0.399 | 0.411 | 0.416 | 0.419 |
| Language-TPP | llm | 0.271 (2) | 0.274 | 0.277 (2) | — | 0.288 | 0.305 (2) | 0.320 | 0.327 | — | 0.343 (2) | 0.349 (2) | 0.352 |
| Marked Markov chain | classic | 0.312 | 0.314 | 0.318 | 0.318 | 0.319 | 0.319 | 0.319 | 0.319 | 0.319 | 0.319 | 0.319 | 0.319 |
| Mambular (Mamba SSM) | state-space | 0.277 | 0.282 | 0.293 | 0.317 | 0.343 | 0.363 | 0.381 | 0.398 | 0.418 | 0.428 | 0.435 (2) | — |
| MM-TPP | llm | 0.273 | 0.271 (1) | 0.277 (1) | 0.278 | 0.282 (2) | 0.299 | 0.312 (1) | — | 0.324 | 0.329 (2) | 0.332 (2) | — |

#### Next-type macro F1 vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 | 65 536 | 93 305 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction autoencoder | reconstruction | 0.007 | 0.007 | 0.007 | 0.007 | 0.008 | 0.010 | 0.020 | 0.048 | 0.162 | 0.180 | 0.185 | 0.189 |
| Neural TPP (GRU) | neural-tpp | 0.007 | 0.007 | 0.007 | 0.007 | 0.010 | 0.015 | 0.032 | 0.061 | 0.142 | 0.177 | 0.185 | 0.186 |
| Supervised GRU | supervised | 0.007 | 0.007 | 0.007 | 0.007 | 0.010 | 0.015 | 0.032 | 0.061 | 0.141 | 0.177 | 0.185 | 0.186 |
| CoLES | contrastive | 0.007 | 0.007 | 0.007 | 0.007 | 0.008 | 0.016 | 0.035 | 0.068 | 0.129 | 0.176 | 0.183 | 0.186 |
| PRAGMA + masked pretraining | hierarchical | 0.009 | 0.014 | 0.022 | 0.025 | 0.030 | 0.041 | 0.101 | 0.153 | 0.168 | 0.178 | 0.182 | 0.185 |
| Transaction MLM | reconstruction | 0.007 | 0.009 | 0.016 | 0.039 | 0.064 | 0.131 | 0.156 | 0.175 | 0.181 | 0.185 | 0.193 | 0.194 |
| PRAGMA | hierarchical | 0.009 | 0.015 | 0.021 | 0.024 | 0.032 | 0.053 | 0.086 | 0.140 | 0.162 | 0.176 | 0.183 | 0.185 |
| NVIDIA TFM blueprint | tabular-transformer | 0.007 | 0.008 | 0.014 | 0.034 | 0.061 | 0.124 | 0.153 | 0.174 | 0.178 | 0.183 | 0.190 | 0.193 |
| Transformer Hawkes | neural-tpp | 0.007 | 0.008 | 0.018 | 0.041 | 0.070 | 0.135 | 0.155 | 0.171 | 0.179 | 0.185 | 0.190 | 0.192 |
| Autoregressive transaction Transformer | generative | 0.007 | 0.008 | 0.013 | 0.032 | 0.053 | 0.118 | 0.150 | 0.168 | 0.177 | 0.182 | 0.188 | 0.192 |
| TPP-LLM | llm | 0.008 | 0.008 (2) | 0.013 | 0.026 | 0.056 (1) | 0.127 (2) | 0.158 (2) | 0.174 | 0.178 | 0.182 (1) | 0.191 (1) | 0.190 |
| TabFormer (TabBERT) | tabular-transformer | 0.007 | 0.009 | 0.035 | 0.065 | 0.106 | 0.147 | 0.154 | 0.164 | 0.175 | 0.178 | 0.184 | 0.188 |
| TabFormer (TabGPT) | tabular-transformer | 0.007 | 0.007 | 0.031 | 0.063 | 0.095 | 0.139 | 0.152 | 0.161 | 0.170 | 0.174 | 0.182 | 0.184 |
| Language-TPP | llm | 0.008 (2) | 0.008 | 0.007 (2) | — | 0.026 | 0.060 (2) | 0.082 | 0.097 | — | 0.108 (2) | 0.117 (2) | 0.122 |
| Marked Markov chain | classic | 0.120 | 0.114 | 0.116 | 0.118 | 0.122 | 0.122 | 0.122 | 0.122 | 0.122 | 0.123 | 0.123 | 0.123 |
| Mambular (Mamba SSM) | state-space | 0.007 | 0.009 | 0.015 | 0.033 | 0.044 | 0.110 | 0.153 | 0.170 | 0.180 | 0.185 | 0.192 (2) | — |
| MM-TPP | llm | 0.008 | 0.008 (1) | 0.007 (1) | 0.008 | 0.017 (2) | 0.053 | 0.073 (1) | — | 0.099 | 0.104 (2) | 0.110 (2) | — |

#### Time RMSE (log1p s) vs. training sequences (lower is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 | 65 536 | 93 305 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction autoencoder | reconstruction | 5.690 | 5.630 | 5.581 | 5.316 | 4.839 | 4.915 | 4.205 | 3.925 | 3.833 | 3.777 | 3.746 | 3.731 |
| Neural TPP (GRU) | neural-tpp | 5.687 | 5.625 | 5.595 | 5.346 | 4.698 | 4.144 | 3.998 | 3.901 | 3.823 | 3.776 | 3.744 | 3.728 |
| Supervised GRU | supervised | 5.687 | 5.625 | 5.595 | 5.346 | 4.698 | 4.144 | 3.998 | 3.901 | 3.823 | 3.776 | 3.744 | 3.728 |
| CoLES | contrastive | 5.679 | 5.626 | 5.600 | 5.391 | 4.949 | 4.139 | 3.975 | 3.900 | 3.834 | 3.789 | 3.753 | 3.732 |
| PRAGMA + masked pretraining | hierarchical | 5.163 | 4.671 | 4.321 | 4.166 | 4.099 | 4.045 | 3.957 | 3.905 | 3.799 | 3.751 | 3.722 | 3.711 |
| Transaction MLM | reconstruction | 5.645 | 5.350 | 4.787 | 4.236 | 4.099 | 4.026 | 3.953 | 3.892 | 3.836 | 3.792 | 3.757 | 3.741 |
| PRAGMA | hierarchical | 5.164 | 4.659 | 4.300 | 4.172 | 4.096 | 4.050 | 3.975 | 3.913 | 3.823 | 3.768 | 3.729 | 3.717 |
| NVIDIA TFM blueprint | tabular-transformer | 5.643 | 5.456 | 4.766 | 4.249 | 4.123 | 4.039 | 3.963 | 3.894 | 3.839 | 3.799 | 3.764 | 3.751 |
| Transformer Hawkes | neural-tpp | 5.643 | 5.411 | 4.573 | 4.224 | 4.093 | 4.029 | 3.955 | 3.899 | 3.848 | 3.801 | 3.766 | 3.753 |
| Autoregressive transaction Transformer | generative | 5.647 | 5.495 | 4.905 | 4.269 | 4.123 | 4.054 | 3.968 | 3.901 | 3.850 | 3.807 | 3.768 | 3.750 |
| TPP-LLM | llm | 5.740 | 5.322 (2) | 5.233 | 4.781 | 4.309 (1) | 4.112 (2) | 4.011 (2) | 3.957 | 3.914 | 3.882 (1) | 3.848 (1) | 3.843 |
| TabFormer (TabBERT) | tabular-transformer | 5.612 | 5.140 | 4.383 | 4.195 | 4.099 | 4.054 | 4.002 | 3.970 | 3.917 | 3.879 | 3.850 | 3.836 |
| TabFormer (TabGPT) | tabular-transformer | 5.629 | 5.337 | 4.448 | 4.208 | 4.115 | 4.065 | 4.010 | 3.973 | 3.924 | 3.889 | 3.860 | 3.847 |
| Language-TPP | llm | 5.493 (2) | 5.412 | 5.377 (2) | — | 5.141 | 4.908 (2) | 4.802 | 4.743 | — | 4.666 (2) | 4.636 (2) | 4.622 |
| Marked Markov chain | classic | 4.779 | 4.762 | 4.755 | 4.752 | 4.749 | 4.748 | 4.749 | 4.749 | 4.749 | 4.749 | 4.749 | 4.749 |
| Mambular (Mamba SSM) | state-space | 5.609 | 5.165 | 4.481 | 4.199 | 4.072 | 4.011 | 3.945 | 3.885 | 3.833 | 3.788 | 3.750 (2) | — |
| MM-TPP | llm | 5.496 | 5.389 (1) | 5.362 (1) | 5.307 | 5.223 (2) | 4.973 | 4.838 (1) | — | 4.735 | 4.711 (2) | 4.689 (2) | — |

#### Time MAE (log1p s) vs. training sequences (lower is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 | 65 536 | 93 305 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction autoencoder | reconstruction | 5.239 | 4.882 | 4.873 | 4.511 | 3.938 | 4.011 | 3.188 | 2.753 | 2.668 | 2.595 | 2.552 | 2.530 |
| Neural TPP (GRU) | neural-tpp | 5.233 | 4.876 | 4.887 | 4.572 | 3.798 | 3.076 | 2.843 | 2.736 | 2.651 | 2.592 | 2.550 | 2.526 |
| Supervised GRU | supervised | 5.233 | 4.876 | 4.887 | 4.572 | 3.798 | 3.076 | 2.843 | 2.736 | 2.652 | 2.592 | 2.549 | 2.526 |
| CoLES | contrastive | 5.208 | 4.879 | 4.890 | 4.625 | 4.095 | 3.101 | 2.834 | 2.726 | 2.668 | 2.615 | 2.565 | 2.540 |
| PRAGMA + masked pretraining | hierarchical | 4.493 | 3.812 | 3.205 | 3.002 | 2.972 | 2.899 | 2.796 | 2.732 | 2.589 | 2.522 | 2.486 | 2.460 |
| Transaction MLM | reconstruction | 5.129 | 4.514 | 3.862 | 3.129 | 2.982 | 2.878 | 2.783 | 2.714 | 2.655 | 2.592 | 2.558 | 2.532 |
| PRAGMA | hierarchical | 4.485 | 3.778 | 3.194 | 3.008 | 2.966 | 2.910 | 2.822 | 2.740 | 2.622 | 2.554 | 2.504 | 2.475 |
| NVIDIA TFM blueprint | tabular-transformer | 5.112 | 4.656 | 3.759 | 3.088 | 2.949 | 2.857 | 2.783 | 2.709 | 2.651 | 2.605 | 2.561 | 2.542 |
| Transformer Hawkes | neural-tpp | 5.115 | 4.596 | 3.606 | 3.116 | 2.980 | 2.885 | 2.786 | 2.725 | 2.668 | 2.602 | 2.571 | 2.551 |
| Autoregressive transaction Transformer | generative | 5.122 | 4.691 | 3.948 | 3.127 | 2.932 | 2.849 | 2.784 | 2.716 | 2.669 | 2.614 | 2.570 | 2.544 |
| TPP-LLM | llm | 4.904 | 4.359 (2) | 4.404 | 3.762 | 3.262 (1) | 2.989 (2) | 2.871 (2) | 2.802 | 2.770 | 2.721 (1) | 2.688 (1) | 2.672 |
| TabFormer (TabBERT) | tabular-transformer | 5.099 | 4.393 | 3.329 | 3.068 | 2.970 | 2.907 | 2.853 | 2.820 | 2.760 | 2.719 | 2.689 | 2.671 |
| TabFormer (TabGPT) | tabular-transformer | 5.122 | 4.546 | 3.416 | 3.040 | 2.956 | 2.879 | 2.837 | 2.805 | 2.774 | 2.737 | 2.712 | 2.696 |
| Language-TPP | llm | 4.781 (2) | 4.498 | 4.529 (2) | — | 4.277 | 3.970 (2) | 3.812 | 3.749 | — | 3.647 (2) | 3.610 (2) | 3.590 |
| Marked Markov chain | classic | 3.914 | 3.883 | 3.870 | 3.872 | 3.873 | 3.871 | 3.874 | 3.874 | 3.875 | 3.875 | 3.876 | 3.875 |
| Mambular (Mamba SSM) | state-space | 5.060 | 4.323 | 3.505 | 3.057 | 2.933 | 2.833 | 2.758 | 2.702 | 2.637 | 2.592 | 2.549 (2) | — |
| MM-TPP | llm | 4.740 | 4.509 (1) | 4.510 (1) | 4.436 | 4.349 (2) | 4.065 | 3.870 (1) | — | 3.727 | 3.687 (2) | 3.674 (2) | — |

## Amazon Beauty 2014

### Classification

#### All metrics at the full pool (n = 34023)

| Method | Fidelity | Seeds | ROC-AUC | Avg. precision | Accuracy | Macro F1 | Wall s (median) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive transaction Transformer | Implemented | 3 | 0.708 ± 0.000 | 0.484 ± 0.001 | 0.763 ± 0.001 | 0.585 ± 0.009 | 583 |
| Transformer Hawkes | Approximation | 3 | 0.708 ± 0.000 | 0.486 ± 0.000 | 0.764 ± 0.001 | 0.589 ± 0.007 | 688 |
| PRAGMA + masked pretraining | Approximation | 3 | 0.708 ± 0.000 | 0.489 ± 0.001 | 0.766 ± 0.001 | 0.579 ± 0.008 | 1,515 |
| Transaction autoencoder | Implemented | 3 | 0.708 ± 0.001 | 0.485 ± 0.002 | 0.762 ± 0.001 | 0.586 ± 0.004 | 644 |
| PRAGMA | Approximation | 3 | 0.708 ± 0.001 | 0.489 ± 0.002 | 0.765 ± 0.001 | 0.574 ± 0.002 | 2,076 |
| Transaction MLM | Implemented | 3 | 0.707 ± 0.002 | 0.484 ± 0.005 | 0.763 ± 0.000 | 0.590 ± 0.006 | 520 |
| TabFormer (TabGPT) | Approximation | 2 | 0.705 ± 0.002 | 0.475 ± 0.003 | 0.759 ± 0.001 | 0.584 ± 0.008 | 914 |
| Engineered features + GBDT | Implemented | 3 | 0.705 ± 0.000 | 0.479 ± 0.001 | 0.761 ± 0.001 | 0.572 ± 0.002 | 13 |
| Neural TPP (GRU) | Approximation | 2 | 0.705 ± 0.000 | 0.481 ± 0.001 | 0.762 ± 0.002 | 0.590 ± 0.011 | 737 |
| Supervised GRU | Implemented | 3 | 0.705 ± 0.001 | 0.480 ± 0.001 | 0.762 ± 0.001 | 0.589 ± 0.008 | 462 |
| CoLES | Approximation | 2 | 0.704 ± 0.001 | 0.480 ± 0.002 | 0.761 ± 0.001 | 0.591 ± 0.002 | 764 |
| TabFormer (TabBERT) | Approximation | 3 | 0.704 ± 0.000 | 0.474 ± 0.002 | 0.760 ± 0.002 | 0.584 ± 0.004 | 526 |
| Mambular (Mamba SSM) | Approximation | 2 | 0.681 ± 0.001 | 0.445 ± 0.009 | 0.745 ± 0.004 | 0.599 ± 0.004 | 2,406 |
| TPP-LLM | Approximation | 1 | 0.605 | 0.371 | 0.748 | 0.508 | 3,443 |
| Language-TPP | Approximation | 2 | 0.603 ± 0.002 | 0.370 ± 0.001 | 0.748 ± 0.000 | 0.503 ± 0.007 | 3,595 |
| MM-TPP | Approximation | 2 | 0.598 ± 0.001 | 0.366 ± 0.001 | 0.750 ± 0.002 | 0.504 ± 0.010 | 3,606 |
| Count + logistic regression | Implemented | 2 | 0.589 ± 0.000 | 0.356 ± 0.000 | 0.749 ± 0.000 | 0.471 ± 0.000 | 5 |

#### Scaling fits

`E(N) = a · N^-b` on seed means, as in the summary above. The per-seed column fits each seed's own curve separately (four or more points) and gives the spread of `b` across seeds. `*` flags an unreliable fit.

| Method | b (1 − ROC-AUC) | a | R² | per-seed b (mean ± sd, n) | b (1 − avg. precision) | a | R² | per-seed b (mean ± sd, n) | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive transaction Transformer | 0.082 | 0.646 | 0.93 | 0.081 ± 0.002 (3) | 0.057 | 0.909 | 0.97 | 0.056 ± 0.001 (3) | 10 |
| Transformer Hawkes | 0.067 | 0.565 | 0.96 | 0.064 ± 0.009 (3) | 0.049 | 0.844 | 0.98 | 0.048 ± 0.006 (3) | 10 |
| PRAGMA + masked pretraining | 0.048 | 0.457 | 0.78 | 0.043 ± 0.012 (3) | 0.041 | 0.758 | 0.90 | 0.039 ± 0.006 (3) | 9 |
| Transaction autoencoder | 0.112 | 0.832 | 0.86 | 0.112 ± 0.009 (3) | 0.067 | 0.978 | 0.93 | 0.067 ± 0.006 (3) | 10 |
| PRAGMA | 0.043 | 0.441 | 0.86 | 0.040 ± 0.012 (3) | 0.039 | 0.750 | 0.94 | 0.038 ± 0.008 (3) | 9 |
| Transaction MLM | 0.046 | 0.461 | 0.96 | 0.046 ± 0.008 (3) | 0.035 | 0.736 | 0.99 | 0.035 ± 0.006 (3) | 10 |
| TabFormer (TabGPT) | 0.073 | 0.595 | 0.95 | 0.074 ± 0.013 (3) | 0.047 | 0.852 | 0.99 | 0.046 ± 0.004 (3) | 9 |
| Engineered features + GBDT | 0.059 | 0.547 | 0.98 | 0.059 ± 0.006 (3) | 0.045 | 0.836 | 0.98 | 0.045 ± 0.003 (3) | 10 |
| Neural TPP (GRU) | 0.087 | 0.662 | 0.75 | 0.077 ± 0.014 (3) | 0.055 | 0.871 | 0.82 | 0.052 ± 0.003 (3) | 9 |
| Supervised GRU | 0.086 | 0.652 | 0.76 | 0.085 ± 0.005 (3) | 0.055 | 0.870 | 0.82 | 0.055 ± 0.001 (3) | 10 |
| CoLES | 0.095 | 0.713 | 0.80 | 0.107 ± 0.019 (3) | 0.059 | 0.901 | 0.87 | 0.065 ± 0.010 (3) | 10 |
| TabFormer (TabBERT) | 0.047 | 0.468 | 0.86 | 0.048 ± 0.002 (3) | 0.036 | 0.766 | 0.92 | 0.037 ± 0.003 (3) | 10 |
| Mambular (Mamba SSM) | 0.047 | 0.508 | 0.90 | 0.049 ± 0.012 (3) | 0.035 | 0.779 | 0.91 | 0.036 ± 0.006 (3) | 9 |
| TPP-LLM | 0.031 | 0.540 | 0.79 | 0.038 ± 0.030 (3) | 0.020 | 0.774 | 0.89 | 0.026 ± 0.018 (3) | 8 |
| Language-TPP | 0.025 | 0.513 | 0.98 | 0.021 ± 0.011 (3) | 0.016 | 0.747 | 1.00 | 0.015 ± 0.007 (3) | 6 |
| MM-TPP | 0.024 | 0.508 | 0.84 | 0.019 ± 0.009 (3) | 0.016 | 0.740 | 0.87 | 0.012 ± 0.005 (3) | 7 |
| Count + logistic regression | 0.014 | 0.470 | 0.76 | 0.016 ± 0.004 (3) | 0.014 | 0.733 | 0.83 | 0.015 ± 0.005 (3) | 10 |
| NVIDIA TFM blueprint | 0.081 | 0.623 | 0.92 | 0.083 ± 0.019 (3) | 0.055 | 0.885 | 0.96 | 0.056 ± 0.008 (3) | 9 |

#### ROC-AUC vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 34 023 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive transaction Transformer | generative | 0.540 | 0.549 | 0.582 | 0.598 | 0.654 | 0.681 | 0.687 | 0.697 | 0.702 | 0.708 |
| Transformer Hawkes | neural-tpp | 0.558 (1) | 0.590 (2) | 0.600 | 0.644 | 0.659 (2) | 0.666 (2) | 0.683 (1) | 0.697 | 0.702 | 0.708 |
| PRAGMA + masked pretraining | hierarchical | 0.579 (2) | 0.648 (2) | 0.663 (1) | 0.669 | 0.682 | 0.697 | 0.699 (2) | — | 0.706 | 0.708 |
| Transaction autoencoder | reconstruction | 0.443 | 0.448 | 0.542 | 0.629 | 0.666 | 0.679 | 0.690 | 0.699 | 0.706 | 0.708 |
| PRAGMA | hierarchical | 0.605 (2) | — | 0.666 | 0.667 | 0.682 | 0.695 (2) | 0.698 (1) | 0.703 (2) | 0.706 | 0.708 |
| Transaction MLM | reconstruction | 0.613 | 0.629 | 0.641 | 0.653 | 0.676 | 0.682 | 0.692 | 0.698 | 0.703 | 0.707 |
| TabFormer (TabGPT) | tabular-transformer | 0.557 | 0.567 | 0.591 (2) | — | 0.659 | 0.672 | 0.682 | 0.696 (2) | 0.702 (1) | 0.705 (2) |
| Engineered features + GBDT | classic | 0.581 | 0.600 | 0.595 | 0.615 | 0.635 | 0.647 | 0.661 | 0.680 | 0.700 | 0.705 |
| Neural TPP (GRU) | neural-tpp | 0.458 (1) | 0.515 | 0.636 | 0.658 | 0.686 (2) | — | 0.688 | 0.695 | 0.701 | 0.705 (2) |
| Supervised GRU | supervised | 0.470 | 0.515 | 0.636 | 0.658 | 0.681 | 0.684 | 0.688 | 0.695 | 0.701 | 0.705 |
| CoLES | contrastive | 0.473 | 0.474 | 0.602 | 0.650 (2) | 0.669 (1) | 0.687 (2) | 0.691 | 0.698 | 0.699 (2) | 0.704 (2) |
| TabFormer (TabBERT) | tabular-transformer | 0.585 | 0.646 (2) | 0.628 (1) | 0.668 (2) | 0.670 | 0.678 | 0.687 (2) | 0.695 (2) | 0.701 (1) | 0.704 |
| Mambular (Mamba SSM) | state-space | 0.583 (1) | 0.579 | 0.599 | 0.630 | 0.659 (2) | — | 0.659 | 0.674 | 0.673 | 0.681 (2) |
| TPP-LLM | llm | — | 0.510 (2) | 0.572 (2) | 0.545 (2) | 0.570 (1) | — | 0.579 (1) | 0.591 | 0.598 (2) | 0.605 (1) |
| Language-TPP | llm | — | — | 0.557 | 0.557 (2) | 0.566 (2) | — | — | 0.592 (2) | 0.597 (2) | 0.603 (2) |
| MM-TPP | llm | 0.522 (1) | — | 0.566 (1) | 0.562 | 0.576 (2) | 0.585 (1) | — | — | 0.591 | 0.598 (2) |
| Count + logistic regression | classic | 0.565 | 0.547 | 0.563 | 0.570 (2) | 0.580 (1) | 0.588 (2) | 0.588 | 0.589 | 0.589 (2) | 0.589 (2) |
| NVIDIA TFM blueprint | tabular-transformer | 0.549 | 0.561 | 0.591 (2) | 0.633 (2) | 0.679 (1) | 0.673 | 0.683 | 0.696 | 0.702 (2) | — |

#### Avg. precision vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 34 023 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive transaction Transformer | generative | 0.288 | 0.294 | 0.330 | 0.344 | 0.394 | 0.433 | 0.447 | 0.456 | 0.468 | 0.484 |
| Transformer Hawkes | neural-tpp | 0.307 (1) | 0.327 (2) | 0.351 | 0.394 | 0.409 (2) | 0.418 (2) | 0.441 (1) | 0.464 | 0.473 | 0.486 |
| PRAGMA + masked pretraining | hierarchical | 0.325 (2) | 0.391 (2) | 0.409 (1) | 0.406 | 0.430 | 0.466 | 0.468 (2) | — | 0.485 | 0.489 |
| Transaction autoencoder | reconstruction | 0.246 | 0.253 | 0.317 | 0.381 | 0.412 | 0.438 | 0.455 | 0.472 | 0.483 | 0.485 |
| PRAGMA | hierarchical | 0.344 (2) | — | 0.408 | 0.407 | 0.429 | 0.458 (2) | 0.467 (1) | 0.476 (2) | 0.483 | 0.489 |
| Transaction MLM | reconstruction | 0.359 | 0.380 | 0.395 | 0.403 | 0.425 | 0.438 | 0.450 | 0.465 | 0.474 | 0.484 |
| TabFormer (TabGPT) | tabular-transformer | 0.301 | 0.318 | 0.340 (2) | — | 0.392 | 0.399 | 0.416 | 0.442 (2) | 0.461 (1) | 0.475 (2) |
| Engineered features + GBDT | classic | 0.318 | 0.341 | 0.342 | 0.359 | 0.383 | 0.402 | 0.427 | 0.446 | 0.472 | 0.479 |
| Neural TPP (GRU) | neural-tpp | 0.258 (1) | 0.283 | 0.393 | 0.413 | 0.451 (2) | — | 0.463 | 0.469 | 0.477 | 0.481 (2) |
| Supervised GRU | supervised | 0.260 | 0.283 | 0.393 | 0.413 | 0.444 | 0.454 | 0.463 | 0.469 | 0.477 | 0.480 |
| CoLES | contrastive | 0.264 | 0.272 | 0.360 | 0.402 (2) | 0.427 (1) | 0.454 (2) | 0.463 | 0.472 | 0.473 (2) | 0.480 (2) |
| TabFormer (TabBERT) | tabular-transformer | 0.315 | 0.387 (2) | 0.363 (1) | 0.402 (2) | 0.403 | 0.417 | 0.429 (2) | 0.445 (2) | 0.463 (1) | 0.474 |
| Mambular (Mamba SSM) | state-space | 0.320 (1) | 0.321 | 0.352 | 0.383 | 0.417 (2) | — | 0.419 | 0.440 | 0.436 | 0.445 (2) |
| TPP-LLM | llm | — | 0.287 (2) | 0.326 (2) | 0.303 (2) | 0.335 (1) | — | 0.344 (1) | 0.357 | 0.362 (2) | 0.371 (1) |
| Language-TPP | llm | — | — | 0.319 | 0.324 (2) | 0.332 (2) | — | — | 0.356 (2) | 0.362 (2) | 0.370 (2) |
| MM-TPP | llm | 0.292 (1) | — | 0.328 (1) | 0.328 | 0.343 (2) | 0.354 (1) | — | — | 0.357 | 0.366 (2) |
| Count + logistic regression | classic | 0.313 | 0.298 | 0.317 | 0.332 (2) | 0.341 (1) | 0.353 (2) | 0.354 | 0.355 | 0.356 (2) | 0.356 (2) |
| NVIDIA TFM blueprint | tabular-transformer | 0.293 | 0.307 | 0.344 (2) | 0.380 (2) | 0.425 (1) | 0.426 | 0.441 | 0.459 | 0.472 (2) | — |

#### Accuracy vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 34 023 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive transaction Transformer | generative | 0.745 | 0.745 | 0.745 | 0.745 | 0.745 | 0.750 | 0.752 | 0.754 | 0.759 | 0.763 |
| Transformer Hawkes | neural-tpp | 0.745 (1) | 0.745 (2) | 0.745 | 0.748 | 0.745 (2) | 0.744 (2) | 0.751 (1) | 0.754 | 0.759 | 0.764 |
| PRAGMA + masked pretraining | hierarchical | 0.745 (2) | 0.745 (2) | 0.749 (1) | 0.746 | 0.752 | 0.759 | 0.760 (2) | — | 0.764 | 0.766 |
| Transaction autoencoder | reconstruction | 0.745 | 0.745 | 0.745 | 0.745 | 0.746 | 0.752 | 0.755 | 0.759 | 0.763 | 0.762 |
| PRAGMA | hierarchical | 0.745 (2) | — | 0.747 | 0.746 | 0.751 | 0.757 (2) | 0.753 (1) | 0.759 (2) | 0.763 | 0.765 |
| Transaction MLM | reconstruction | 0.745 | 0.745 | 0.745 | 0.748 | 0.749 | 0.752 | 0.753 | 0.754 | 0.762 | 0.763 |
| TabFormer (TabGPT) | tabular-transformer | 0.745 | 0.745 | 0.744 (2) | — | 0.735 | 0.734 | 0.738 | 0.747 (2) | 0.756 (1) | 0.759 (2) |
| Engineered features + GBDT | classic | 0.689 | 0.700 | 0.715 | 0.728 | 0.735 | 0.738 | 0.747 | 0.753 | 0.761 | 0.761 |
| Neural TPP (GRU) | neural-tpp | 0.745 (1) | 0.745 | 0.745 | 0.749 | 0.755 (2) | — | 0.757 | 0.756 | 0.761 | 0.762 (2) |
| Supervised GRU | supervised | 0.745 | 0.745 | 0.745 | 0.749 | 0.755 | 0.754 | 0.757 | 0.756 | 0.761 | 0.762 |
| CoLES | contrastive | 0.745 | 0.745 | 0.745 | 0.751 (2) | 0.754 (1) | 0.756 (2) | 0.758 | 0.759 | 0.760 (2) | 0.761 (2) |
| TabFormer (TabBERT) | tabular-transformer | 0.745 | 0.745 (2) | 0.746 (1) | 0.745 (2) | 0.743 | 0.746 | 0.747 (2) | 0.753 (2) | 0.757 (1) | 0.760 |
| Mambular (Mamba SSM) | state-space | 0.745 (1) | 0.745 | 0.746 | 0.737 | 0.734 (2) | — | 0.735 | 0.740 | 0.741 | 0.745 (2) |
| TPP-LLM | llm | — | 0.726 (2) | 0.739 (2) | 0.737 (2) | 0.746 (1) | — | 0.748 (1) | 0.747 | 0.747 (2) | 0.748 (1) |
| Language-TPP | llm | — | — | 0.745 | 0.746 (2) | 0.747 (2) | — | — | 0.748 (2) | 0.748 (2) | 0.748 (2) |
| MM-TPP | llm | 0.744 (1) | — | 0.746 (1) | 0.742 | 0.747 (2) | 0.749 (1) | — | — | 0.748 | 0.750 (2) |
| Count + logistic regression | classic | 0.708 | 0.734 | 0.739 | 0.746 (2) | 0.748 (1) | 0.748 (2) | 0.748 | 0.749 | 0.748 (2) | 0.749 (2) |
| NVIDIA TFM blueprint | tabular-transformer | 0.745 | 0.745 | 0.745 (2) | 0.747 (2) | 0.747 (1) | 0.748 | 0.751 | 0.753 | 0.760 (2) | — |

#### Macro F1 vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 34 023 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive transaction Transformer | generative | 0.427 | 0.427 | 0.427 | 0.427 | 0.530 | 0.562 | 0.587 | 0.587 | 0.581 | 0.585 |
| Transformer Hawkes | neural-tpp | 0.427 (1) | 0.427 (2) | 0.431 | 0.518 | 0.555 (2) | 0.574 (2) | 0.575 (1) | 0.595 | 0.586 | 0.589 |
| PRAGMA + masked pretraining | hierarchical | 0.427 (2) | 0.427 (2) | 0.543 (1) | 0.525 | 0.548 | 0.562 | 0.588 (2) | — | 0.577 | 0.579 |
| Transaction autoencoder | reconstruction | 0.427 | 0.427 | 0.427 | 0.427 | 0.480 | 0.552 | 0.580 | 0.590 | 0.588 | 0.586 |
| PRAGMA | hierarchical | 0.427 (2) | — | 0.487 | 0.541 | 0.552 | 0.564 (2) | 0.592 (1) | 0.596 (2) | 0.575 | 0.574 |
| Transaction MLM | reconstruction | 0.427 | 0.427 | 0.427 | 0.471 | 0.558 | 0.580 | 0.592 | 0.599 | 0.587 | 0.590 |
| TabFormer (TabGPT) | tabular-transformer | 0.427 | 0.427 | 0.439 (2) | — | 0.555 | 0.566 | 0.580 | 0.595 (2) | 0.581 (1) | 0.584 (2) |
| Engineered features + GBDT | classic | 0.536 | 0.557 | 0.549 | 0.553 | 0.566 | 0.574 | 0.580 | 0.584 | 0.567 | 0.572 |
| Neural TPP (GRU) | neural-tpp | 0.427 (1) | 0.427 | 0.427 | 0.503 | 0.588 (2) | — | 0.598 | 0.598 | 0.592 | 0.590 (2) |
| Supervised GRU | supervised | 0.427 | 0.427 | 0.427 | 0.503 | 0.570 | 0.587 | 0.598 | 0.598 | 0.592 | 0.589 |
| CoLES | contrastive | 0.427 | 0.427 | 0.428 | 0.474 (2) | 0.507 (1) | 0.580 (2) | 0.589 | 0.596 | 0.590 (2) | 0.591 (2) |
| TabFormer (TabBERT) | tabular-transformer | 0.427 | 0.429 (2) | 0.434 (1) | 0.530 (2) | 0.546 | 0.558 | 0.581 (2) | 0.588 (2) | 0.590 (1) | 0.584 |
| Mambular (Mamba SSM) | state-space | 0.427 (1) | 0.430 | 0.442 | 0.538 | 0.596 (2) | — | 0.596 | 0.606 | 0.598 | 0.599 (2) |
| TPP-LLM | llm | — | 0.493 (2) | 0.488 (2) | 0.482 (2) | 0.487 (1) | — | 0.486 (1) | 0.473 | 0.511 (2) | 0.508 (1) |
| Language-TPP | llm | — | — | 0.462 | 0.469 (2) | 0.459 (2) | — | — | 0.481 (2) | 0.495 (2) | 0.503 (2) |
| MM-TPP | llm | 0.448 (1) | — | 0.460 (1) | 0.484 | 0.476 (2) | 0.483 (1) | — | — | 0.497 | 0.504 (2) |
| Count + logistic regression | classic | 0.513 | 0.481 | 0.483 | 0.479 (2) | 0.461 (1) | 0.481 (2) | 0.471 | 0.470 | 0.469 (2) | 0.471 (2) |
| NVIDIA TFM blueprint | tabular-transformer | 0.427 | 0.427 | 0.427 (2) | 0.453 (2) | 0.590 (1) | 0.575 | 0.583 | 0.593 | 0.584 (2) | — |

### Temporal point process (TPP)

#### All metrics at the full pool (n = 34023)

| Method | Fidelity | Seeds | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) | Wall s (median) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction MLM | Implemented | 3 | 0.491 ± 0.001 | 0.326 ± 0.002 | 5.041 ± 0.002 | 4.497 ± 0.006 | 1,360 |
| TPP-LLM | Approximation | 3 | 0.491 ± 0.002 | 0.328 ± 0.002 | 5.175 ± 0.008 | 4.744 ± 0.017 | 6,208 |
| CoLES | Approximation | 3 | 0.490 ± 0.001 | 0.319 ± 0.009 | 5.053 ± 0.007 | 4.530 ± 0.016 | 2,525 |
| Autoregressive transaction Transformer | Implemented | 3 | 0.489 ± 0.001 | 0.323 ± 0.001 | 5.045 ± 0.006 | 4.481 ± 0.006 | 900 |
| Mambular (Mamba SSM) | Approximation | 3 | 0.489 ± 0.001 | 0.324 ± 0.002 | 5.031 ± 0.003 | 4.479 ± 0.008 | 6,250 |
| NVIDIA TFM blueprint | Approximation | 2 | 0.489 ± 0.001 | 0.322 ± 0.000 | 5.051 ± 0.005 | 4.493 ± 0.008 | 1,377 |
| Transformer Hawkes | Approximation | 3 | 0.489 ± 0.002 | 0.323 ± 0.001 | 5.049 ± 0.002 | 4.506 ± 0.004 | 591 |
| TabFormer (TabGPT) | Approximation | 2 | 0.489 ± 0.001 | 0.326 ± 0.001 | 5.435 ± 0.003 | 5.106 ± 0.005 | 767 |
| Language-TPP | Approximation | 2 | 0.489 ± 0.001 | 0.331 ± 0.004 | 5.456 ± 0.005 | 5.158 ± 0.006 | 6,249 |
| PRAGMA + masked pretraining | Approximation | 2 | 0.487 ± 0.000 | 0.323 ± 0.002 | 5.212 ± 0.005 | 4.662 ± 0.004 | 5,190 |
| PRAGMA | Approximation | 3 | 0.483 ± 0.001 | 0.319 ± 0.001 | 5.208 ± 0.002 | 4.671 ± 0.006 | 2,498 |
| Transaction autoencoder | Implemented | 3 | 0.482 ± 0.001 | 0.287 ± 0.001 | 5.063 ± 0.008 | 4.544 ± 0.022 | 1,744 |
| Neural TPP (GRU) | Approximation | 3 | 0.479 ± 0.002 | 0.277 ± 0.004 | 5.061 ± 0.008 | 4.536 ± 0.007 | 1,468 |
| Supervised GRU | Implemented | 3 | 0.479 ± 0.001 | 0.276 ± 0.003 | 5.061 ± 0.008 | 4.537 ± 0.009 | 1,483 |
| Marked Markov chain | Implemented | 3 | 0.463 ± 0.000 | 0.274 ± 0.000 | 5.761 ± 0.000 | 5.559 ± 0.000 | 6 |

#### Scaling fits

`E(N) = a · N^-b` on seed means, as in the summary above. The per-seed column fits each seed's own curve separately (four or more points) and gives the spread of `b` across seeds. `*` flags an unreliable fit.

| Method | b (1 − next-type acc.) | a | R² | per-seed b (mean ± sd, n) | b (Time RMSE (log1p s)) | a | R² | per-seed b (mean ± sd, n) | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction MLM | 0.060 | 0.898 | 0.89 | 0.060 ± 0.004 (3) | 0.022 | 6.214 | 0.91 | 0.022 ± 0.001 (3) | 10 |
| TPP-LLM | 0.051 | 0.820 | 0.85 | 0.045 ± 0.014 (3) | 0.017 | 6.068 | 0.81 | 0.015 ± 0.007 (3) | 7 |
| CoLES | 0.067 | 1.027 | 0.91 | 0.068 ± 0.004 (3) | 0.023 | 6.432 | 0.87 | 0.025 ± 0.002 (3) | 10 |
| Autoregressive transaction Transformer | 0.063 | 0.930 | 0.90 | 0.063 ± 0.003 (3) | 0.022 | 6.249 | 0.91 | 0.022 ± 0.002 (3) | 10 |
| Mambular (Mamba SSM) | 0.053 | 0.845 | 0.87 | 0.052 ± 0.001 (3) | 0.020 | 6.104 | 0.85 | 0.019 ± 0.002 (3) | 9 |
| NVIDIA TFM blueprint | 0.059 | 0.898 | 0.89 | 0.064 ± 0.005 (3) | 0.021 | 6.218 | 0.92 | 0.022 ± 0.001 (3) | 9 |
| Transformer Hawkes | 0.055 | 0.862 | 0.91 | 0.051 ± 0.005 (3) | 0.022 | 6.204 | 0.85 | 0.020 ± 0.001 (3) | 9 |
| TabFormer (TabGPT) | 0.058 | 0.876 | 0.85 | 0.064 ± 0.007 (3) | 0.009 | 5.927 | 0.97 | 0.009 ± 0.000 (3) | 10 |
| Language-TPP | 0.058 | 0.858 | 0.80 | 0.051 ± 0.007 (2) | 0.007 | 5.869 | 0.99 | 0.007 ± 0.002 (2) | 6 |
| PRAGMA + masked pretraining | 0.043 | 0.797 | 0.97 | 0.045 ± 0.004 (3) | 0.015 | 6.083 | 0.95 | 0.015 ± 0.001 (3) | 9 |
| PRAGMA | 0.050 | 0.835 | 0.96 | 0.048 ± 0.002 (3) | 0.016 | 6.104 | 0.95 | 0.015 ± 0.002 (3) | 10 |
| Transaction autoencoder | 0.042 | 0.887 | 0.77 | 0.042 ± 0.006 (3) | 0.023 | 6.402 | 0.93 | 0.023 ± 0.001 (3) | 10 |
| Neural TPP (GRU) | 0.049 | 0.933 | 0.80 | 0.056 ± 0.012 (3) | 0.024 | 6.493 | 0.87 | 0.025 ± 0.002 (3) | 9 |
| Supervised GRU | 0.051 | 0.934 | 0.87 | 0.051 ± 0.010 (3) | 0.023 | 6.432 | 0.89 | 0.023 ± 0.001 (3) | 10 |
| Marked Markov chain | 0.001* | 0.542 | 0.50 | 0.001 ± 0.001 (3) | 0.001* | 5.839 | 0.65 | 0.001 ± 0.000 (3) | 10 |
| MM-TPP | 0.071 | 0.948 | 0.89 | 0.075 ± 0.008 (2) | 0.011 | 5.943 | 0.98 | 0.012 ± 0.004 (2) | 7 |
| TabFormer (TabBERT) | 0.065 | 0.909 | 0.80 | 0.067 ± 0.006 (3) | 0.009 | 5.934 | 0.97 | 0.009 ± 0.000 (3) | 9 |

#### Next-type acc. vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 34 023 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction MLM | reconstruction | 0.294 | 0.305 | 0.317 | 0.401 | 0.444 | 0.463 | 0.475 | 0.485 | 0.488 | 0.491 |
| TPP-LLM | llm | 0.300 (2) | — | — | 0.400 (2) | 0.448 (2) | 0.474 (2) | 0.484 (1) | — | 0.490 (1) | 0.491 |
| CoLES | contrastive | 0.220 (1) | 0.305 (2) | 0.305 | 0.305 | 0.305 (2) | 0.341 (2) | 0.405 (1) | 0.466 | 0.476 | 0.490 |
| Autoregressive transaction Transformer | generative | 0.283 | 0.306 | 0.313 | 0.344 | 0.436 | 0.458 | 0.471 | 0.481 | 0.486 | 0.489 |
| Mambular (Mamba SSM) | state-space | 0.305 (2) | — | 0.331 | 0.409 | 0.446 | 0.466 (2) | 0.473 (1) | 0.482 (2) | 0.488 | 0.489 |
| NVIDIA TFM blueprint | tabular-transformer | 0.307 (1) | 0.305 | 0.308 | 0.378 | 0.449 (2) | — | 0.475 | 0.482 | 0.486 | 0.489 (2) |
| Transformer Hawkes | neural-tpp | 0.305 (2) | 0.313 (2) | 0.357 (1) | 0.382 | 0.443 | 0.460 | 0.472 (2) | — | 0.487 | 0.489 |
| TabFormer (TabGPT) | tabular-transformer | 0.299 | 0.307 | 0.325 | 0.418 (2) | 0.448 (1) | 0.473 (2) | 0.481 | 0.486 | 0.485 (2) | 0.489 (2) |
| Language-TPP | llm | 0.297 (2) | 0.320 (1) | — | — | 0.469 | 0.482 (2) | 0.486 (2) | — | — | 0.489 (2) |
| PRAGMA + masked pretraining | hierarchical | 0.319 | 0.349 | 0.391 (2) | — | 0.406 | 0.436 | 0.444 | 0.461 (2) | 0.471 (1) | 0.487 (2) |
| PRAGMA | hierarchical | 0.310 | 0.343 (2) | 0.348 (1) | 0.399 (2) | 0.409 | 0.444 | 0.461 (2) | 0.472 (2) | 0.482 (1) | 0.483 |
| Transaction autoencoder | reconstruction | 0.276 | 0.305 | 0.305 | 0.305 | 0.306 | 0.315 | 0.371 | 0.371 | 0.395 | 0.482 |
| Neural TPP (GRU) | neural-tpp | 0.288 (2) | — | 0.305 | 0.305 | 0.305 | 0.326 (2) | 0.342 (1) | 0.392 (2) | 0.447 | 0.479 |
| Supervised GRU | supervised | 0.265 | 0.305 | 0.305 | 0.305 | 0.305 | 0.335 | 0.356 | 0.417 | 0.447 | 0.479 |
| Marked Markov chain | classic | 0.461 (1) | 0.458 (2) | 0.462 | 0.463 | 0.462 (2) | 0.462 (2) | 0.463 (1) | 0.463 | 0.463 | 0.463 |
| MM-TPP | llm | 0.296 (2) | 0.316 (2) | 0.330 (1) | — | 0.464 (1) | 0.469 | 0.470 (2) | 0.473 (1) | — | — |
| TabFormer (TabBERT) | tabular-transformer | 0.309 | 0.305 | 0.308 (2) | 0.449 (2) | 0.462 (1) | 0.471 | 0.482 | 0.486 | 0.488 (2) | — |

#### Next-type macro F1 vs. training sequences (higher is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 34 023 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction MLM | reconstruction | 0.083 | 0.067 | 0.085 | 0.182 | 0.224 | 0.240 | 0.274 | 0.310 | 0.321 | 0.326 |
| TPP-LLM | llm | 0.091 (2) | — | — | 0.190 (2) | 0.249 (2) | 0.295 (2) | 0.315 (1) | — | 0.321 (1) | 0.328 |
| CoLES | contrastive | 0.052 (1) | 0.067 (2) | 0.067 | 0.067 | 0.067 (2) | 0.115 (2) | 0.186 (1) | 0.222 | 0.265 | 0.319 |
| Autoregressive transaction Transformer | generative | 0.078 | 0.071 | 0.083 | 0.128 | 0.214 | 0.227 | 0.260 | 0.300 | 0.318 | 0.323 |
| Mambular (Mamba SSM) | state-space | 0.067 (2) | — | 0.102 | 0.193 | 0.229 | 0.231 (2) | 0.254 (1) | 0.299 (2) | 0.319 | 0.324 |
| NVIDIA TFM blueprint | tabular-transformer | 0.096 (1) | 0.067 | 0.072 | 0.163 | 0.230 (2) | — | 0.282 | 0.305 | 0.318 | 0.322 (2) |
| Transformer Hawkes | neural-tpp | 0.067 (2) | 0.087 (2) | 0.149 (1) | 0.166 | 0.223 | 0.237 | 0.277 (2) | — | 0.320 | 0.323 |
| TabFormer (TabGPT) | tabular-transformer | 0.086 | 0.071 | 0.091 | 0.189 (2) | 0.210 (1) | 0.282 (2) | 0.312 | 0.319 | 0.320 (2) | 0.326 (2) |
| Language-TPP | llm | 0.071 (2) | 0.089 (1) | — | — | 0.234 | 0.276 (2) | 0.285 (2) | — | — | 0.331 (2) |
| PRAGMA + masked pretraining | hierarchical | 0.090 | 0.145 | 0.172 (2) | — | 0.196 | 0.192 | 0.211 | 0.248 (2) | 0.275 (1) | 0.323 (2) |
| PRAGMA | hierarchical | 0.078 | 0.144 (2) | 0.135 (1) | 0.188 (2) | 0.199 | 0.219 | 0.255 (2) | 0.265 (2) | 0.320 (1) | 0.319 |
| Transaction autoencoder | reconstruction | 0.062 | 0.067 | 0.067 | 0.067 | 0.063 | 0.100 | 0.148 | 0.138 | 0.158 | 0.287 |
| Neural TPP (GRU) | neural-tpp | 0.064 (2) | — | 0.067 | 0.067 | 0.067 | 0.096 (2) | 0.132 (1) | 0.149 (2) | 0.204 | 0.277 |
| Supervised GRU | supervised | 0.060 | 0.067 | 0.067 | 0.067 | 0.067 | 0.113 | 0.131 | 0.173 | 0.204 | 0.276 |
| Marked Markov chain | classic | 0.238 (1) | 0.237 (2) | 0.250 | 0.262 | 0.256 (2) | 0.256 (2) | 0.274 (1) | 0.274 | 0.274 | 0.274 |
| MM-TPP | llm | 0.081 (2) | 0.093 (2) | 0.095 (1) | — | 0.236 (1) | 0.267 | 0.280 (2) | 0.303 (1) | — | — |
| TabFormer (TabBERT) | tabular-transformer | 0.083 | 0.067 | 0.073 (2) | 0.212 (2) | 0.227 (1) | 0.267 | 0.316 | 0.321 | 0.326 (2) | — |

#### Time RMSE (log1p s) vs. training sequences (lower is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 34 023 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction MLM | reconstruction | 5.725 | 5.655 | 5.587 | 5.365 | 5.228 | 5.176 | 5.128 | 5.104 | 5.074 | 5.041 |
| TPP-LLM | llm | 5.798 (2) | — | — | 5.407 (2) | 5.297 (2) | 5.226 (2) | 5.214 (1) | — | 5.181 (1) | 5.175 |
| CoLES | contrastive | 5.697 (1) | 5.685 (2) | 5.695 | 5.667 | 5.659 (2) | 5.450 (2) | 5.173 (1) | 5.129 | 5.090 | 5.053 |
| Autoregressive transaction Transformer | generative | 5.710 | 5.661 | 5.638 | 5.416 | 5.260 | 5.156 | 5.126 | 5.106 | 5.079 | 5.045 |
| Mambular (Mamba SSM) | state-space | 5.698 (2) | — | 5.579 | 5.305 | 5.177 | 5.137 (2) | 5.108 (1) | 5.085 (2) | 5.061 | 5.031 |
| NVIDIA TFM blueprint | tabular-transformer | 5.703 (1) | 5.662 | 5.615 | 5.383 | 5.236 (2) | — | 5.126 | 5.104 | 5.084 | 5.051 (2) |
| Transformer Hawkes | neural-tpp | 5.702 (2) | 5.607 (2) | 5.690 (1) | 5.352 | 5.212 | 5.152 | 5.123 (2) | — | 5.073 | 5.049 |
| TabFormer (TabGPT) | tabular-transformer | 5.703 | 5.690 | 5.680 | 5.645 (2) | 5.559 (1) | 5.544 (2) | 5.517 | 5.484 | 5.453 (2) | 5.435 (2) |
| Language-TPP | llm | 5.710 (2) | 5.665 (1) | — | — | 5.593 | 5.579 (2) | 5.559 (2) | — | — | 5.456 (2) |
| PRAGMA + masked pretraining | hierarchical | 5.675 | 5.657 | 5.678 (2) | — | 5.420 | 5.364 | 5.362 | 5.290 (2) | 5.238 (1) | 5.212 (2) |
| PRAGMA | hierarchical | 5.675 | 5.685 (2) | 5.657 (1) | 5.552 (2) | 5.411 | 5.366 | 5.310 (2) | 5.280 (2) | 5.242 (1) | 5.208 |
| Transaction autoencoder | reconstruction | 5.720 | 5.682 | 5.702 | 5.668 | 5.524 | 5.381 | 5.187 | 5.146 | 5.111 | 5.063 |
| Neural TPP (GRU) | neural-tpp | 5.720 (2) | — | 5.696 | 5.668 | 5.662 | 5.483 (2) | 5.194 (1) | 5.144 (2) | 5.093 | 5.061 |
| Supervised GRU | supervised | 5.715 | 5.687 | 5.696 | 5.668 | 5.662 | 5.413 | 5.199 | 5.148 | 5.093 | 5.061 |
| Marked Markov chain | classic | 5.831 (1) | 5.794 (2) | 5.786 | 5.769 | 5.764 (2) | 5.765 (2) | 5.762 (1) | 5.762 | 5.762 | 5.761 |
| MM-TPP | llm | 5.693 (2) | 5.596 (2) | 5.586 (1) | — | 5.509 (1) | 5.442 | 5.404 (2) | 5.378 (1) | — | — |
| TabFormer (TabBERT) | tabular-transformer | 5.717 | 5.684 | 5.680 (2) | 5.578 (2) | 5.588 (1) | 5.537 | 5.510 | 5.478 | 5.451 (2) | — |

#### Time MAE (log1p s) vs. training sequences (lower is better)

| Method | Family | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 34 023 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transaction MLM | reconstruction | 5.513 | 5.495 | 5.375 | 5.060 | 4.765 | 4.662 | 4.628 | 4.584 | 4.544 | 4.497 |
| TPP-LLM | llm | 5.532 (2) | — | — | 5.014 (2) | 4.945 (2) | 4.791 (2) | 4.770 (1) | — | 4.763 (1) | 4.744 |
| CoLES | contrastive | 5.524 (1) | 5.522 (2) | 5.522 | 5.519 | 5.516 (2) | 5.169 (2) | 4.741 (1) | 4.667 | 4.591 | 4.530 |
| Autoregressive transaction Transformer | generative | 5.515 | 5.496 | 5.430 | 5.154 | 4.851 | 4.658 | 4.631 | 4.584 | 4.538 | 4.481 |
| Mambular (Mamba SSM) | state-space | 5.506 (2) | — | 5.366 | 4.982 | 4.748 | 4.675 (2) | 4.615 (1) | 4.572 (2) | 4.524 | 4.479 |
| NVIDIA TFM blueprint | tabular-transformer | 5.520 (1) | 5.500 | 5.425 | 5.109 | 4.773 (2) | — | 4.620 | 4.589 | 4.538 | 4.493 (2) |
| Transformer Hawkes | neural-tpp | 5.503 (2) | 5.450 (2) | 5.452 (1) | 5.021 | 4.769 | 4.677 | 4.614 (2) | — | 4.541 | 4.506 |
| TabFormer (TabGPT) | tabular-transformer | 5.521 | 5.518 | 5.515 | 5.492 (2) | 5.372 (1) | 5.241 (2) | 5.233 | 5.186 | 5.146 (2) | 5.106 (2) |
| Language-TPP | llm | 5.516 (2) | 5.508 (1) | — | — | 5.378 | 5.365 (2) | 5.338 (2) | — | — | 5.158 (2) |
| PRAGMA + masked pretraining | hierarchical | 5.457 | 5.409 | 5.348 (2) | — | 4.979 | 4.868 | 4.827 | 4.755 (2) | 4.713 (1) | 4.662 (2) |
| PRAGMA | hierarchical | 5.459 | 5.419 (2) | 5.317 (1) | 5.163 (2) | 4.944 | 4.869 | 4.776 (2) | 4.731 (2) | 4.732 (1) | 4.671 |
| Transaction autoencoder | reconstruction | 5.524 | 5.522 | 5.523 | 5.519 | 5.373 | 5.110 | 4.750 | 4.691 | 4.640 | 4.544 |
| Neural TPP (GRU) | neural-tpp | 5.523 (2) | — | 5.523 | 5.520 | 5.517 | 5.307 (2) | 4.798 (1) | 4.682 (2) | 4.602 | 4.536 |
| Supervised GRU | supervised | 5.524 | 5.523 | 5.523 | 5.520 | 5.517 | 5.205 | 4.797 | 4.699 | 4.603 | 4.537 |
| Marked Markov chain | classic | 5.500 (1) | 5.542 (2) | 5.562 | 5.563 | 5.560 (2) | 5.554 (2) | 5.555 (1) | 5.558 | 5.560 | 5.559 |
| MM-TPP | llm | 5.486 (2) | 5.387 (2) | 5.417 (1) | — | 5.178 (1) | 5.092 | 5.027 (2) | 4.979 (1) | — | — |
| TabFormer (TabBERT) | tabular-transformer | 5.521 | 5.520 | 5.509 (2) | 5.380 (2) | 5.289 (1) | 5.255 | 5.219 | 5.172 | 5.141 (2) | — |
