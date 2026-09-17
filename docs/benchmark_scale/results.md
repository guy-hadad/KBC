# Benchmark results

Every architecture surveyed in the project note, evaluated on every open
benchmark it names, for both tasks. Each cell is the mean over seeds at the
largest training-sample point that completed; the per-point curves behind
these numbers are in the scaling figures.

**Read the `Fidelity` column before citing anything here.** `Implemented`
means the entry is a control with no paper to be faithful to.
`Approximation` means the entry preserves the central comparison but
differs materially from the cited recipe — usually in scale, in the
pretraining corpus, or in the decoding head. The specific divergence for
each method is in `results.csv` and in
[`../research/method_catalog.md`](../research/method_catalog.md). A row
named after a paper is not a reproduction of that paper.

## Task: Binary sequence classification

Positive class is the forward-looking label described in each dataset's `meta.json`. ROC-AUC is the headline metric; accuracy is reported for context because training may be rebalanced while evaluation keeps natural prevalence.

### MBD (full) — 30665 training sequences

| Method | Family | Fidelity | ROC-AUC | Avg. precision | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.731 [0.730, 0.733] | 0.031 [0.030, 0.031] | 0.963 [0.961, 0.965] | 0.523 [0.520, 0.525] |
| PRAGMA | hierarchical | Approximation | 0.730 [0.728, 0.732] | 0.030 [0.029, 0.031] | 0.968 [0.963, 0.972] | 0.520 [0.519, 0.521] |
| Supervised GRU | supervised | Implemented | 0.725 [0.724, 0.727] | 0.028 [0.028, 0.029] | 0.963 [0.962, 0.965] | 0.520 [0.518, 0.522] |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.725 [0.724, 0.727] | 0.028 [0.028, 0.029] | 0.963 [0.962, 0.965] | 0.520 [0.518, 0.522] |
| Transaction MLM | reconstruction | Implemented | 0.720 [0.719, 0.721] | 0.031 [0.030, 0.032] | 0.961 [0.957, 0.963] | 0.521 [0.518, 0.525] |
| Transformer Hawkes | neural-tpp | Approximation | 0.720 [0.716, 0.723] | 0.027 [0.026, 0.027] | 0.969 [0.968, 0.971] | 0.515 [0.511, 0.518] |
| Engineered features + GBDT | classic | Implemented | 0.718 [0.716, 0.719] | 0.029 [0.029, 0.030] | 0.978 [0.977, 0.980] | 0.514 [0.512, 0.516] |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.717 [0.708, 0.722] | 0.028 [0.027, 0.029] | 0.962 [0.960, 0.967] | 0.519 [0.515, 0.522] |
| Autoregressive transaction Transformer | generative | Implemented | 0.717 [0.713, 0.721] | 0.029 [0.028, 0.030] | 0.957 [0.953, 0.961] | 0.519 [0.518, 0.521] |
| Transaction autoencoder | reconstruction | Implemented | 0.716 [0.712, 0.721] | 0.028 [0.027, 0.028] | 0.969 [0.968, 0.969] | 0.517 [0.517, 0.518] |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.715 [0.712, 0.718] | 0.028 [0.028, 0.029] | 0.969 [0.969, 0.970] | 0.520 [0.517, 0.524] |
| TPP-LLM | llm | Approximation | 0.710 [0.708, 0.713] | 0.027 [0.026, 0.027] | 0.982 [0.981, 0.985] | 0.509 [0.506, 0.516] |
| Language-TPP | llm | Approximation | 0.709 [0.708, 0.710] | 0.025 [0.024, 0.026] | 0.979 [0.978, 0.979] | 0.509 [0.505, 0.512] |
| CoLES | contrastive | Approximation | 0.708 [0.702, 0.711] | 0.028 [0.027, 0.029] | 0.965 [0.964, 0.966] | 0.520 [0.517, 0.522] |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.707 [0.707, 0.707] | 0.027 [0.026, 0.027] | 0.969 [0.967, 0.971] | 0.523 [0.519, 0.526] |
| Mambular (Mamba SSM) | state-space | Approximation | 0.704 [0.697, 0.713] | 0.025 [0.024, 0.027] | 0.941 [0.938, 0.945] | 0.511 [0.510, 0.512] |
| Count + logistic regression | classic | Implemented | 0.696 [0.696, 0.696] | 0.027 [0.027, 0.027] | 0.981 [0.981, 0.981] | 0.512 [0.512, 0.512] |
| MM-TPP | llm | Approximation | 0.692 [0.691, 0.693] | 0.023 [0.022, 0.024] | 0.985 [0.984, 0.988] | 0.499 [0.497, 0.504] |

### Synthea EHR — 93305 training sequences

| Method | Family | Fidelity | ROC-AUC | Avg. precision | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Supervised GRU | supervised | Implemented | 0.882 [0.881, 0.883] | 0.367 [0.360, 0.373] | 0.904 [0.902, 0.906] | 0.669 [0.668, 0.669] |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.882 [0.881, 0.883] | 0.367 [0.360, 0.373] | 0.904 [0.902, 0.906] | 0.669 [0.668, 0.669] |
| Transaction MLM | reconstruction | Implemented | 0.881 [0.879, 0.882] | 0.342 [0.323, 0.358] | 0.902 [0.900, 0.905] | 0.666 [0.663, 0.670] |
| Autoregressive transaction Transformer | generative | Implemented | 0.879 [0.879, 0.879] | 0.336 [0.329, 0.341] | 0.900 [0.899, 0.901] | 0.664 [0.664, 0.664] |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.877 [0.877, 0.877] | 0.315 [0.309, 0.328] | 0.898 [0.895, 0.901] | 0.662 [0.661, 0.664] |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.876 [0.875, 0.877] | 0.339 [0.331, 0.344] | 0.902 [0.899, 0.905] | 0.659 [0.657, 0.660] |
| CoLES | contrastive | Approximation | 0.876 [0.876, 0.877] | 0.300 [0.296, 0.307] | 0.899 [0.897, 0.901] | 0.658 [0.656, 0.658] |
| Transaction autoencoder | reconstruction | Implemented | 0.876 [0.875, 0.876] | 0.314 [0.305, 0.327] | 0.899 [0.898, 0.900] | 0.661 [0.659, 0.662] |
| Engineered features + GBDT | classic | Implemented | 0.875 [0.875, 0.875] | 0.300 [0.299, 0.301] | 0.903 [0.903, 0.903] | 0.658 [0.657, 0.658] |
| Mambular (Mamba SSM) | state-space | Approximation | 0.871 [0.869, 0.874] | 0.295 [0.279, 0.315] | 0.892 [0.888, 0.894] | 0.653 [0.652, 0.655] |
| Transformer Hawkes | neural-tpp | Approximation | 0.870 [0.869, 0.871] | 0.291 [0.286, 0.298] | 0.901 [0.899, 0.902] | 0.658 [0.657, 0.659] |
| PRAGMA | hierarchical | Approximation | 0.869 [0.868, 0.870] | 0.285 [0.279, 0.294] | 0.897 [0.895, 0.900] | 0.646 [0.645, 0.647] |
| Count + logistic regression | classic | Implemented | 0.833 [0.833, 0.833] | 0.220 [0.220, 0.220] | 0.910 [0.910, 0.910] | 0.605 [0.605, 0.605] |
| TPP-LLM | llm | Approximation | 0.826 [0.826, 0.827] | 0.223 [0.221, 0.225] | 0.908 [0.907, 0.910] | 0.612 [0.610, 0.614] |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.826 [0.825, 0.827] | 0.275 [0.273, 0.278] | 0.920 [0.917, 0.922] | 0.608 [0.607, 0.609] |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.823 [0.823, 0.824] | 0.266 [0.263, 0.268] | 0.918 [0.917, 0.920] | 0.607 [0.606, 0.608] |
| Language-TPP | llm | Approximation | 0.809 [0.807, 0.810] | 0.194 [0.187, 0.205] | 0.906 [0.900, 0.911] | 0.586 [0.579, 0.592] |

### Amazon Beauty 2014 — 34023 training sequences

| Method | Family | Fidelity | ROC-AUC | Avg. precision | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Autoregressive transaction Transformer | generative | Implemented | 0.708 [0.708, 0.709] | 0.484 [0.482, 0.485] | 0.763 [0.762, 0.765] | 0.585 [0.575, 0.591] |
| Transformer Hawkes | neural-tpp | Approximation | 0.708 [0.708, 0.708] | 0.486 [0.486, 0.486] | 0.764 [0.763, 0.765] | 0.589 [0.583, 0.597] |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.708 [0.708, 0.708] | 0.489 [0.489, 0.490] | 0.766 [0.765, 0.766] | 0.579 [0.569, 0.584] |
| Transaction autoencoder | reconstruction | Implemented | 0.708 [0.707, 0.709] | 0.485 [0.483, 0.487] | 0.762 [0.761, 0.763] | 0.586 [0.582, 0.590] |
| PRAGMA | hierarchical | Approximation | 0.708 [0.707, 0.708] | 0.489 [0.487, 0.490] | 0.765 [0.765, 0.766] | 0.574 [0.572, 0.576] |
| Transaction MLM | reconstruction | Implemented | 0.707 [0.705, 0.708] | 0.484 [0.479, 0.488] | 0.763 [0.763, 0.763] | 0.590 [0.584, 0.596] |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.705 [0.704, 0.707] | 0.475 [0.474, 0.477] | 0.759 [0.758, 0.759] | 0.584 [0.578, 0.589] |
| Engineered features + GBDT | classic | Implemented | 0.705 [0.705, 0.705] | 0.479 [0.478, 0.480] | 0.761 [0.760, 0.762] | 0.572 [0.570, 0.574] |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.705 [0.705, 0.705] | 0.481 [0.480, 0.481] | 0.762 [0.761, 0.763] | 0.590 [0.582, 0.598] |
| Supervised GRU | supervised | Implemented | 0.705 [0.704, 0.705] | 0.480 [0.479, 0.481] | 0.762 [0.761, 0.763] | 0.589 [0.582, 0.598] |
| CoLES | contrastive | Approximation | 0.704 [0.704, 0.705] | 0.480 [0.479, 0.481] | 0.761 [0.760, 0.761] | 0.591 [0.589, 0.592] |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.704 [0.704, 0.705] | 0.474 [0.473, 0.476] | 0.760 [0.759, 0.762] | 0.584 [0.580, 0.588] |
| Mambular (Mamba SSM) | state-space | Approximation | 0.681 [0.680, 0.682] | 0.445 [0.439, 0.451] | 0.745 [0.742, 0.748] | 0.599 [0.596, 0.602] |
| TPP-LLM | llm | Approximation | 0.605 | 0.371 | 0.748 | 0.508 |
| Language-TPP | llm | Approximation | 0.603 [0.601, 0.604] | 0.370 [0.369, 0.370] | 0.748 [0.748, 0.748] | 0.503 [0.498, 0.508] |
| MM-TPP | llm | Approximation | 0.598 [0.598, 0.599] | 0.366 [0.366, 0.367] | 0.750 [0.749, 0.752] | 0.504 [0.497, 0.511] |
| Count + logistic regression | classic | Implemented | 0.589 [0.589, 0.589] | 0.356 [0.356, 0.356] | 0.749 [0.749, 0.749] | 0.471 [0.471, 0.471] |

## Task: Temporal point process (next event)

Scored separately for *what* and *when*, as the neural-TPP review recommends. Time metrics are on `log1p(seconds)`, so **lower is better**.

### MBD (full) — 30665 training sequences

| Method | Family | Fidelity | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) |
| --- | --- | --- | --- | --- | --- | --- |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.586 [0.585, 0.588] | 0.201 [0.198, 0.203] | 1.528 [1.525, 1.530] | 1.150 [1.149, 1.153] |
| Transaction autoencoder | reconstruction | Implemented | 0.586 [0.584, 0.587] | 0.193 [0.191, 0.196] | 1.526 [1.522, 1.528] | 1.156 [1.153, 1.159] |
| CoLES | contrastive | Approximation | 0.586 [0.584, 0.587] | 0.196 [0.195, 0.197] | 1.526 [1.523, 1.527] | 1.156 [1.155, 1.158] |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.586 [0.584, 0.587] | 0.195 [0.192, 0.198] | 1.526 [1.523, 1.528] | 1.157 [1.155, 1.160] |
| Supervised GRU | supervised | Implemented | 0.586 [0.584, 0.587] | 0.195 [0.192, 0.198] | 1.526 [1.523, 1.528] | 1.157 [1.155, 1.160] |
| PRAGMA | hierarchical | Approximation | 0.585 [0.585, 0.586] | 0.202 [0.200, 0.207] | 1.530 [1.526, 1.532] | 1.149 [1.146, 1.151] |
| Transaction MLM | reconstruction | Implemented | 0.585 [0.584, 0.585] | 0.201 [0.200, 0.204] | 1.527 [1.524, 1.529] | 1.152 [1.151, 1.154] |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.584 [0.584, 0.585] | 0.204 [0.202, 0.208] | 1.527 [1.524, 1.529] | 1.154 [1.153, 1.157] |
| Autoregressive transaction Transformer | generative | Implemented | 0.584 [0.584, 0.584] | 0.207 [0.203, 0.212] | 1.527 [1.525, 1.529] | 1.156 [1.154, 1.157] |
| Transformer Hawkes | neural-tpp | Approximation | 0.584 [0.583, 0.585] | 0.203 [0.199, 0.206] | 1.539 [1.535, 1.541] | 1.169 [1.167, 1.171] |
| Mambular (Mamba SSM) | state-space | Approximation | 0.582 [0.582, 0.583] | 0.201 [0.199, 0.204] | 1.527 [1.524, 1.529] | 1.156 [1.153, 1.160] |
| Language-TPP | llm | Approximation | 0.581 [0.581, 0.582] | 0.196 [0.194, 0.199] | 1.739 [1.739, 1.739] | 1.329 [1.329, 1.330] |
| TPP-LLM | llm | Approximation | 0.581 [0.580, 0.581] | 0.201 [0.196, 0.205] | 1.565 [1.564, 1.566] | 1.198 [1.193, 1.206] |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.580 [0.579, 0.580] | 0.198 [0.193, 0.201] | 1.717 [1.712, 1.720] | 1.309 [1.305, 1.311] |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.580 [0.579, 0.580] | 0.199 [0.196, 0.201] | 1.715 [1.710, 1.718] | 1.309 [1.305, 1.312] |
| MM-TPP | llm | Approximation | 0.569 [0.569, 0.570] | 0.175 [0.173, 0.178] | 1.749 [1.746, 1.751] | 1.340 [1.340, 1.340] |
| Marked Markov chain | classic | Implemented | 0.489 [0.489, 0.489] | 0.088 [0.088, 0.088] | 1.857 [1.857, 1.857] | 1.424 [1.424, 1.424] |

### Synthea EHR — 93305 training sequences

| Method | Family | Fidelity | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) |
| --- | --- | --- | --- | --- | --- | --- |
| Transaction autoencoder | reconstruction | Implemented | 0.439 [0.439, 0.439] | 0.189 [0.187, 0.191] | 3.731 [3.729, 3.732] | 2.530 [2.524, 2.539] |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.439 [0.438, 0.439] | 0.186 [0.185, 0.188] | 3.728 [3.728, 3.729] | 2.526 [2.521, 2.530] |
| Supervised GRU | supervised | Implemented | 0.439 [0.438, 0.439] | 0.186 [0.185, 0.188] | 3.728 [3.728, 3.729] | 2.526 [2.521, 2.530] |
| CoLES | contrastive | Approximation | 0.438 [0.437, 0.439] | 0.186 [0.185, 0.187] | 3.732 [3.727, 3.737] | 2.540 [2.531, 2.552] |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.437 [0.436, 0.437] | 0.185 [0.184, 0.187] | 3.711 [3.711, 3.713] | 2.460 [2.454, 2.464] |
| Transaction MLM | reconstruction | Implemented | 0.435 [0.434, 0.437] | 0.194 [0.192, 0.195] | 3.741 [3.740, 3.742] | 2.532 [2.526, 2.538] |
| PRAGMA | hierarchical | Approximation | 0.435 [0.434, 0.436] | 0.185 [0.181, 0.187] | 3.717 [3.714, 3.719] | 2.475 [2.473, 2.476] |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.434 [0.433, 0.435] | 0.193 [0.192, 0.194] | 3.751 [3.747, 3.756] | 2.542 [2.540, 2.544] |
| Transformer Hawkes | neural-tpp | Approximation | 0.434 [0.433, 0.435] | 0.192 [0.190, 0.194] | 3.753 [3.745, 3.757] | 2.551 [2.541, 2.560] |
| Autoregressive transaction Transformer | generative | Implemented | 0.433 [0.432, 0.434] | 0.192 [0.190, 0.194] | 3.750 [3.749, 3.751] | 2.544 [2.542, 2.546] |
| TPP-LLM | llm | Approximation | 0.424 [0.424, 0.424] | 0.190 [0.190, 0.191] | 3.843 [3.839, 3.846] | 2.672 [2.670, 2.675] |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.422 [0.422, 0.422] | 0.188 [0.187, 0.190] | 3.836 [3.834, 3.837] | 2.671 [2.670, 2.672] |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.419 [0.419, 0.420] | 0.184 [0.180, 0.186] | 3.847 [3.844, 3.850] | 2.696 [2.692, 2.700] |
| Language-TPP | llm | Approximation | 0.352 [0.352, 0.353] | 0.122 [0.122, 0.122] | 4.622 [4.617, 4.624] | 3.590 [3.581, 3.597] |
| Marked Markov chain | classic | Implemented | 0.319 [0.319, 0.319] | 0.123 [0.123, 0.123] | 4.749 [4.749, 4.749] | 3.875 [3.875, 3.875] |

### Amazon Beauty 2014 — 34023 training sequences

| Method | Family | Fidelity | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) |
| --- | --- | --- | --- | --- | --- | --- |
| Transaction MLM | reconstruction | Implemented | 0.491 [0.490, 0.493] | 0.326 [0.324, 0.327] | 5.041 [5.039, 5.044] | 4.497 [4.493, 4.505] |
| TPP-LLM | llm | Approximation | 0.491 [0.489, 0.494] | 0.328 [0.327, 0.330] | 5.175 [5.167, 5.181] | 4.744 [4.725, 4.758] |
| CoLES | contrastive | Approximation | 0.490 [0.488, 0.491] | 0.319 [0.310, 0.327] | 5.053 [5.047, 5.061] | 4.530 [4.519, 4.549] |
| Autoregressive transaction Transformer | generative | Implemented | 0.489 [0.488, 0.490] | 0.323 [0.322, 0.323] | 5.045 [5.038, 5.050] | 4.481 [4.474, 4.485] |
| Mambular (Mamba SSM) | state-space | Approximation | 0.489 [0.488, 0.490] | 0.324 [0.321, 0.326] | 5.031 [5.028, 5.033] | 4.479 [4.470, 4.486] |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.489 [0.488, 0.490] | 0.322 [0.322, 0.322] | 5.051 [5.048, 5.055] | 4.493 [4.487, 4.498] |
| Transformer Hawkes | neural-tpp | Approximation | 0.489 [0.487, 0.490] | 0.323 [0.322, 0.323] | 5.049 [5.047, 5.051] | 4.506 [4.501, 4.508] |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.489 [0.488, 0.489] | 0.326 [0.326, 0.327] | 5.435 [5.432, 5.437] | 5.106 [5.103, 5.110] |
| Language-TPP | llm | Approximation | 0.489 [0.488, 0.489] | 0.331 [0.328, 0.334] | 5.456 [5.452, 5.459] | 5.158 [5.153, 5.162] |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.487 [0.487, 0.487] | 0.323 [0.321, 0.324] | 5.212 [5.208, 5.215] | 4.662 [4.659, 4.665] |
| PRAGMA | hierarchical | Approximation | 0.483 [0.482, 0.484] | 0.319 [0.319, 0.320] | 5.208 [5.206, 5.209] | 4.671 [4.665, 4.676] |
| Transaction autoencoder | reconstruction | Implemented | 0.482 [0.481, 0.484] | 0.287 [0.286, 0.288] | 5.063 [5.056, 5.071] | 4.544 [4.521, 4.564] |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.479 [0.477, 0.481] | 0.277 [0.274, 0.281] | 5.061 [5.051, 5.066] | 4.536 [4.529, 4.542] |
| Supervised GRU | supervised | Implemented | 0.479 [0.477, 0.480] | 0.276 [0.274, 0.280] | 5.061 [5.051, 5.066] | 4.537 [4.529, 4.546] |
| Marked Markov chain | classic | Implemented | 0.463 [0.463, 0.463] | 0.274 [0.274, 0.274] | 5.761 [5.761, 5.761] | 5.559 [5.559, 5.559] |

