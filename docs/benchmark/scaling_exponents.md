# Data-scaling exponents

Each curve's error is fitted to `E(N) = a · N^-b`, where `N` is the number
of labelled training sequences. **`b` is the data-scaling exponent: how fast
the error falls as data is added.** A larger `b` means the method is still
converting extra sequences into accuracy; `b` near zero means it has
flattened and more labels will not help.

This ranks architectures by *data efficiency* rather than by score at one
sample size, which is the more relevant question for a foundation model.

Fitted on six points per curve, so read `b` as a local slope over the
measured range, not an asymptotic claim. Values marked `*` have
`R² < 0.70` — the power law does not describe that curve well and the
exponent should not be trusted.

Two means are given. **`Mean b (reliable)` is the one to read** — it
averages only the fits that pass the R² gate, with the count in
parentheses, and it is what the rows are sorted by. `Mean b (all)`
includes the flagged fits and is shown so the difference is visible: on
classification, PaySim fails the gate for nearly every method, and its
occasional *negative* exponent (error rising with data) is the signature
of a task with no learnable signal rather than of a bad model.

## Classification — fitted on 1 − ROC-AUC

| Method | Family | BankSim | PaySim | IBM AML (HI-Small) | MBD-mini | Mean b (all) | Mean b (reliable) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA TFM blueprint | tabular-transformer | 0.162 | 0.012* | 0.130 | 0.115 | 0.105 | **0.136** (3) |
| TabFormer (TabGPT) | tabular-transformer | 0.211 | -0.017* | 0.060 | 0.125 | 0.095 | **0.132** (3) |
| Language-TPP | llm | 0.103 | -0.003* | 0.070 | 0.216 | 0.096 | **0.129** (3) |
| PRAGMA | hierarchical | 0.129 | -0.008* | 0.059* | 0.083* | 0.066 | **0.129** (1) |
| Transformer Hawkes | neural-tpp | 0.184 | 0.011* | 0.070 | 0.065* | 0.083 | **0.127** (2) |
| TabFormer (TabBERT) | tabular-transformer | 0.172 | -0.023* | 0.078 | 0.122 | 0.087 | **0.124** (3) |
| Neural TPP (GRU) | neural-tpp | 0.115 | 0.039* | 0.137 | 0.094 | 0.096 | **0.115** (3) |
| CoLES | contrastive | 0.019* | 0.025* | 0.084 | 0.112 | 0.060 | **0.098** (2) |
| Mambular (Mamba SSM) | state-space | 0.093 | 0.035 | 0.126 | 0.064* | 0.080 | **0.085** (3) |
| PRAGMA + masked pretraining | hierarchical | 0.109* | -0.009* | 0.076 | 0.087* | 0.066 | **0.076** (1) |
| MM-TPP | llm | 0.058* | 0.005* | 0.058 | 0.039* | 0.040 | **0.058** (1) |
| Count + logistic regression | classic | 0.056 | 0.007* | 0.027* | 0.040* | 0.033 | **0.056** (1) |
| TPP-LLM | llm | 0.040 | 0.022* | 0.062 | 0.050* | 0.044 | **0.051** (2) |

## Temporal point process — fitted on 1 − next-type acc.

| Method | Family | BankSim | PaySim | IBM AML (HI-Small) | MBD-mini | Mean b (all) | Mean b (reliable) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TPP-LLM | llm | 0.032 | 0.010* | 0.117 | 0.070 | 0.057 | **0.073** (3) |
| NVIDIA TFM blueprint | tabular-transformer | 0.031 | 0.011* | 0.112 | 0.061 | 0.054 | **0.068** (3) |
| Neural TPP (GRU) | neural-tpp | 0.031 | 0.000 | 0.142 | 0.072 | 0.061 | **0.061** (4) |
| Language-TPP | llm | 0.026 | 0.001* | 0.096 | 0.060 | 0.046 | **0.061** (3) |
| MM-TPP | llm | 0.026 | 0.001* | 0.089 | 0.064 | 0.045 | **0.060** (3) |
| CoLES | contrastive | 0.027 | 0.000 | 0.137 | 0.072 | 0.059 | **0.059** (4) |
| Transformer Hawkes | neural-tpp | 0.017 | 0.017* | 0.107 | 0.048 | 0.047 | **0.057** (3) |
| PRAGMA + masked pretraining | hierarchical | 0.023 | 0.016 | 0.118 | 0.061 | 0.055 | **0.055** (4) |
| Mambular (Mamba SSM) | state-space | 0.027* | 0.016 | 0.102 | 0.044 | 0.047 | **0.054** (3) |
| TabFormer (TabGPT) | tabular-transformer | 0.011 | 0.017* | 0.084 | 0.053 | 0.041 | **0.049** (3) |
| PRAGMA | hierarchical | 0.025 | 0.005 | 0.098 | 0.061 | 0.047 | **0.047** (4) |
| TabFormer (TabBERT) | tabular-transformer | 0.012 | 0.023 | 0.086 | 0.043 | 0.041 | **0.041** (4) |
| Marked Markov chain | classic | 0.001* | 0.000 | 0.000 | 0.012 | 0.003 | **0.004** (3) |

