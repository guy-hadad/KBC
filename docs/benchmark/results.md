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

Positive class is the forward-looking label described in each dataset's `meta.json`. ROC-AUC is the headline metric; accuracy is reported for context because the label is rebalanced but still skewed.

### BankSim — 2048 training sequences

| Method | Family | Fidelity | ROC-AUC | Avg. precision | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.801 | 0.487 | 0.807 | 0.665 |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.801 | 0.483 | 0.830 | 0.670 |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.797 | 0.467 | 0.834 | 0.616 |
| Transformer Hawkes | neural-tpp | Approximation | 0.795 | 0.480 | 0.825 | 0.648 |
| PRAGMA | hierarchical | Approximation | 0.794 | 0.465 | 0.839 | 0.603 |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.787 | 0.463 | 0.839 | 0.599 |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.784 | 0.474 | 0.838 | 0.633 |
| Count + logistic regression | classic | Implemented | 0.757 | 0.462 | 0.846 | 0.610 |
| Mambular (Mamba SSM) | state-space | Approximation | 0.747 | 0.411 | 0.815 | 0.608 |
| CoLES | contrastive | Approximation | 0.743 | 0.435 | 0.843 | 0.595 |
| TPP-LLM | llm | Approximation | 0.652 | 0.289 | 0.824 | 0.519 |
| MM-TPP | llm | Approximation | 0.631 | 0.284 | 0.825 | 0.520 |
| Language-TPP | llm | Approximation | 0.612 | 0.264 | 0.827 | 0.521 |

### PaySim — 2048 training sequences

| Method | Family | Fidelity | ROC-AUC | Avg. precision | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Mambular (Mamba SSM) | state-space | Approximation | 0.546 | 0.263 | 0.768 | 0.484 |
| MM-TPP | llm | Approximation | 0.539 | 0.265 | 0.782 | 0.445 |
| Count + logistic regression | classic | Implemented | 0.532 | 0.258 | 0.781 | 0.439 |
| Language-TPP | llm | Approximation | 0.526 | 0.265 | 0.781 | 0.444 |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.516 | 0.237 | 0.781 | 0.439 |
| PRAGMA | hierarchical | Approximation | 0.514 | 0.252 | 0.781 | 0.439 |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.512 | 0.252 | 0.781 | 0.439 |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.511 | 0.236 | 0.781 | 0.439 |
| Transformer Hawkes | neural-tpp | Approximation | 0.499 | 0.237 | 0.782 | 0.445 |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.489 | 0.225 | 0.781 | 0.439 |
| CoLES | contrastive | Approximation | 0.484 | 0.237 | 0.781 | 0.439 |
| TPP-LLM | llm | Approximation | 0.479 | 0.205 | 0.781 | 0.439 |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.448 | 0.192 | 0.781 | 0.439 |

### IBM AML (HI-Small) — 2048 training sequences

| Method | Family | Fidelity | ROC-AUC | Avg. precision | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Mambular (Mamba SSM) | state-space | Approximation | 0.705 | 0.471 | 0.817 | 0.633 |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.702 | 0.443 | 0.809 | 0.615 |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.701 | 0.399 | 0.809 | 0.599 |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.655 | 0.338 | 0.806 | 0.549 |
| Transformer Hawkes | neural-tpp | Approximation | 0.650 | 0.379 | 0.811 | 0.553 |
| CoLES | contrastive | Approximation | 0.650 | 0.347 | 0.809 | 0.548 |
| PRAGMA | hierarchical | Approximation | 0.642 | 0.330 | 0.802 | 0.528 |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.635 | 0.354 | 0.807 | 0.562 |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.629 | 0.353 | 0.806 | 0.549 |
| Count + logistic regression | classic | Implemented | 0.605 | 0.324 | 0.804 | 0.529 |
| Language-TPP | llm | Approximation | 0.600 | 0.326 | 0.803 | 0.535 |
| TPP-LLM | llm | Approximation | 0.580 | 0.311 | 0.804 | 0.539 |
| MM-TPP | llm | Approximation | 0.575 | 0.310 | 0.807 | 0.534 |

### MBD-mini — 2048 training sequences

| Method | Family | Fidelity | ROC-AUC | Avg. precision | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.717 | 0.371 | 0.781 | 0.528 |
| PRAGMA | hierarchical | Approximation | 0.700 | 0.363 | 0.783 | 0.454 |
| CoLES | contrastive | Approximation | 0.699 | 0.388 | 0.783 | 0.497 |
| Language-TPP | llm | Approximation | 0.699 | 0.359 | 0.785 | 0.445 |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.698 | 0.362 | 0.779 | 0.548 |
| Transformer Hawkes | neural-tpp | Approximation | 0.697 | 0.357 | 0.777 | 0.518 |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.692 | 0.359 | 0.781 | 0.471 |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.688 | 0.371 | 0.780 | 0.513 |
| TPP-LLM | llm | Approximation | 0.684 | 0.374 | 0.786 | 0.524 |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.682 | 0.341 | 0.776 | 0.524 |
| Mambular (Mamba SSM) | state-space | Approximation | 0.679 | 0.339 | 0.755 | 0.566 |
| Count + logistic regression | classic | Implemented | 0.660 | 0.337 | 0.775 | 0.499 |
| MM-TPP | llm | Approximation | 0.609 | 0.272 | 0.783 | 0.439 |

## Task: Temporal point process (next event)

Scored separately for *what* and *when*, as the neural-TPP review recommends. Time metrics are on `log1p(seconds)`, so **lower is better**.

### BankSim — 2048 training sequences

| Method | Family | Fidelity | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) |
| --- | --- | --- | --- | --- | --- | --- |
| Marked Markov chain | classic | Implemented | 0.850 | 0.087 | 1.498 | 0.760 |
| Transformer Hawkes | neural-tpp | Approximation | 0.847 | 0.121 | 1.284 | 0.670 |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.847 | 0.120 | 1.419 | 0.685 |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.846 | 0.115 | 1.377 | 0.736 |
| PRAGMA | hierarchical | Approximation | 0.846 | 0.105 | 1.184 | 0.579 |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.846 | 0.114 | 1.382 | 0.750 |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.845 | 0.100 | 1.159 | 0.562 |
| Mambular (Mamba SSM) | state-space | Approximation | 0.845 | 0.117 | 1.242 | 0.591 |
| TPP-LLM | llm | Approximation | 0.844 | 0.115 | 1.650 | 0.763 |
| CoLES | contrastive | Approximation | 0.843 | 0.092 | 1.319 | 0.655 |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.843 | 0.089 | 1.337 | 0.664 |
| MM-TPP | llm | Approximation | 0.840 | 0.095 | 1.700 | 0.800 |
| Language-TPP | llm | Approximation | 0.840 | 0.101 | 1.696 | 0.795 |

### PaySim — 2048 training sequences

| Method | Family | Fidelity | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) |
| --- | --- | --- | --- | --- | --- | --- |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.546 | 0.209 | 1.606 | 1.282 |
| Marked Markov chain | classic | Implemented | 0.543 | 0.176 | 1.993 | 1.630 |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.542 | 0.209 | 1.590 | 1.272 |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.537 | 0.191 | 1.453 | 1.113 |
| Transformer Hawkes | neural-tpp | Approximation | 0.536 | 0.200 | 1.654 | 1.309 |
| PRAGMA | hierarchical | Approximation | 0.536 | 0.191 | 1.469 | 1.124 |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.532 | 0.181 | 1.581 | 1.206 |
| Language-TPP | llm | Approximation | 0.531 | 0.175 | 1.864 | 1.517 |
| MM-TPP | llm | Approximation | 0.531 | 0.174 | 1.866 | 1.521 |
| CoLES | contrastive | Approximation | 0.531 | 0.173 | 1.620 | 1.231 |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.531 | 0.173 | 1.600 | 1.223 |
| TPP-LLM | llm | Approximation | 0.531 | 0.173 | 1.739 | 1.396 |
| Mambular (Mamba SSM) | state-space | Approximation | 0.528 | 0.198 | 1.573 | 1.199 |

### IBM AML (HI-Small) — 2048 training sequences

| Method | Family | Fidelity | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) |
| --- | --- | --- | --- | --- | --- | --- |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.629 | 0.494 | 2.173 | 1.475 |
| Mambular (Mamba SSM) | state-space | Approximation | 0.628 | 0.532 | 2.092 | 1.400 |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.626 | 0.465 | 2.229 | 1.541 |
| CoLES | contrastive | Approximation | 0.617 | 0.466 | 2.238 | 1.558 |
| TPP-LLM | llm | Approximation | 0.609 | 0.511 | 2.370 | 1.712 |
| Transformer Hawkes | neural-tpp | Approximation | 0.607 | 0.479 | 2.309 | 1.654 |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.605 | 0.505 | 2.303 | 1.644 |
| PRAGMA | hierarchical | Approximation | 0.599 | 0.418 | 2.239 | 1.551 |
| Language-TPP | llm | Approximation | 0.597 | 0.507 | 2.635 | 2.098 |
| MM-TPP | llm | Approximation | 0.594 | 0.503 | 2.661 | 2.111 |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.588 | 0.498 | 2.847 | 2.441 |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.585 | 0.485 | 2.829 | 2.423 |
| Marked Markov chain | classic | Implemented | 0.466 | 0.341 | 2.850 | 2.446 |

### MBD-mini — 2048 training sequences

| Method | Family | Fidelity | Next-type acc. | Next-type macro F1 | Time RMSE (log1p s) | Time MAE (log1p s) |
| --- | --- | --- | --- | --- | --- | --- |
| Language-TPP | llm | Approximation | 0.538 | 0.156 | 1.791 | 1.365 |
| CoLES | contrastive | Approximation | 0.534 | 0.107 | 1.550 | 1.196 |
| Neural TPP (GRU) | neural-tpp | Approximation | 0.533 | 0.102 | 1.557 | 1.201 |
| PRAGMA + masked pretraining | hierarchical | Approximation | 0.532 | 0.124 | 1.556 | 1.186 |
| Mambular (Mamba SSM) | state-space | Approximation | 0.531 | 0.135 | 1.562 | 1.198 |
| PRAGMA | hierarchical | Approximation | 0.531 | 0.132 | 1.556 | 1.186 |
| NVIDIA TFM blueprint | tabular-transformer | Approximation | 0.529 | 0.127 | 1.561 | 1.197 |
| TabFormer (TabGPT) | tabular-transformer | Approximation | 0.529 | 0.127 | 1.783 | 1.368 |
| TPP-LLM | llm | Approximation | 0.528 | 0.112 | 1.622 | 1.233 |
| TabFormer (TabBERT) | tabular-transformer | Approximation | 0.527 | 0.124 | 1.771 | 1.359 |
| Transformer Hawkes | neural-tpp | Approximation | 0.526 | 0.135 | 1.563 | 1.202 |
| MM-TPP | llm | Approximation | 0.518 | 0.115 | 1.800 | 1.372 |
| Marked Markov chain | classic | Implemented | 0.462 | 0.093 | 1.866 | 1.430 |

