# Data-scaling exponents

Each curve's error is fitted to `E(N) = a · N^-b`, where `N` is the number
of labelled training sequences. **`b` is the data-scaling exponent: how fast
the error falls as data is added.** A larger `b` means the method is still
converting extra sequences into accuracy; `b` near zero means it has
flattened and more labels will not help.

This ranks architectures by *data efficiency* rather than by score at one
sample size, which is the more relevant question for a foundation model.

Fitted per curve on the points each dataset provides (Amazon Beauty 2014 10, MBD (full) 10, Synthea EHR 12), spanning 64 to 93305 training sequences.
Read `b` as a local slope over the measured range, not an asymptotic
claim. Values marked `*` have
`R² < 0.70` — the power law does not describe that curve well and the
exponent should not be trusted.

Two means are given. **`Mean b (reliable)` is the one to read** — it
averages only the fits that pass the R² gate, with the count in
parentheses, and it is what the rows are sorted by. `Mean b (all)`
includes the flagged fits and is shown so the difference is visible. A
dataset that fails the gate for nearly every method, or shows an
occasional *negative* exponent (error rising with data), is signalling a
task with no learnable signal rather than a set of bad models.

## Classification — fitted on 1 − ROC-AUC

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

## Temporal point process — fitted on 1 − next-type acc.

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

