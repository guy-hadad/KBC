# Benchmark design

This document records the choices behind the comparison, because most of the
interesting decisions are not in the model code but in how four very different
public datasets were forced into one comparable shape.

For the paper-facing view — what would still have to be true before these
numbers could support a publication claim — see [`research/`](research/), in
particular [`research/experimental_protocol.md`](research/experimental_protocol.md)
and [`research/method_catalog.md`](research/method_catalog.md). This benchmark
is the architecture-screening starting point that programme builds on, and it
uses the same fidelity vocabulary (`Implemented` / `Approximation`) defined in
[`research/README.md`](research/README.md).

## 1. What is being compared

The completed legacy screen covers nine modelling directions, two non-neural
controls, and a second PRAGMA variant:

| Registry key | Paper / system | Family | What is distinctive |
| --- | --- | --- | --- |
| `count-logistic` | — | control | Bag-of-marks + inter-arrival summaries, logistic regression |
| `markov` | — | control | First-order marked transition matrix, per-mark mean log Δt |
| `ntpp-gru` | Shchur et al. 2021 | neural TPP | RNN history encoder, log-normal mixture over Δt |
| `thp` | Transformer-Hawkes family | neural TPP | Causal attention over continuous-time encodings |
| `pragma` | Ostroukhov et al. 2026 | hierarchical | Key/value/time tokens, event encoder → history encoder |
| `pragma-mlm` | Ostroukhov et al. 2026 | hierarchical | Same, with masked-value pretraining first |
| `tabbert` | Padhi et al. 2021 | tabular transformer | Field-pooled events, bidirectional, masked-field pretraining |
| `tabgpt` | Padhi et al. 2021 | tabular transformer | Field-pooled events, causal GPT-2 decoder |
| `nvidia-tfm` | NVIDIA TFM blueprint | tabular transformer | Decoder-only over bucketed tabular tokens, causal-LM pretrained |
| `coles` | Babaev et al. 2022 | contrastive | Sub-sequence InfoNCE pretraining, then a task head |
| `mambular` | Thielmann et al. 2024 | state space | Mamba selective scan instead of attention |
| `tpp-llm` | Liu & Quan 2024 | LLM | Textual marks + continuous temporal embedding, LoRA |
| `language-tpp` | Kong et al. 2026 | LLM | Δt serialised as byte tokens inside the prompt |
| `mm-tpp` | Li et al. 2026 | LLM | Byte-token times + temporal-similarity compression |

The broader paper programme now registers 62 runnable keys. See the generated
[`research/implementation_manifest.md`](research/implementation_manifest.md)
instead of extending this historical 14-row result description by hand.

### Keeping the comparison fair

Everything except the architecture is held fixed: the same splits, the same
event tokenisation, the same optimiser and schedule, the same heads, the same
metrics. Concretely:

* Every non-hierarchical model consumes the **same flat tensor contract**
  (`eventfm/data/flat.py`) and differs only in its sequence mixer and objective.
  PRAGMA keeps its own hierarchical 3-D contract, since that hierarchy *is* the
  method.
* Every TPP model ends in the **same two heads**: a mark classifier and a
  log-normal mixture over `log1p(Δt)`. The mixture is the intensity-free
  decoder recommended by the neural-TPP review; it yields both a proper
  likelihood and a closed-form mean for the point-estimate metrics.
* The three LM entries share one pretrained causal LM
  (`HuggingFaceTB/SmolLM2-135M` by default, override with `KBC_LLM_BASE_MODEL`),
  frozen, with LoRA adapters and trainable heads. They differ **only in how the
  history is serialised**, which is precisely the tokenisation question the
  project note poses.

Where a paper's contribution is a *pretraining objective*, that stage runs on
the training split before the task head is attached: masked-field for TabBERT,
causal-LM for TabGPT / NVIDIA-TFM, masked-value for `pragma-mlm`, sub-sequence
contrastive for CoLES.

## 2. Datasets

All four are the open benchmarks named in section 3.1.5 of the project note.

| Dataset | Source | Entity (one sequence) | Mark | Marks |
| --- | --- | --- | --- | --- |
| BankSim | `github:atavci/fraud-detection-on-banksim-data` | customer | merchant category | 15 |
| PaySim | `hf:theman10/paysim` | destination account | transfer type | 4 |
| IBM AML (HI-Small) | `hf:OsamaMIT/IBM-AML-HI-Small` | receiving account | payment format | 7 |
| MBD-mini | `hf:ai-lab/MBD-mini` | bank client | transaction event type | 54 |

Full MBD is 69 GB; MBD-mini is the official 10 % client subsample with the same
schema, and is what is used here.

### What the legacy conversion produced

| Dataset | Marks | Train | Val | Test | Mean events / seq | Natural pos. rate | Used pos. rate | Feature fields |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BankSim | 15 | 2 644 | 605 | 794 | 82.7 | 16.50 % | 16.50 % | 2 |
| PaySim | 4 | 2 479 | 569 | 777 | 11.5 | 0.41 % | 20 % | 4 |
| IBM AML (HI-Small) | 7 | 3 323 | 806 | 1 051 | 17.5 | 0.56 % | 20 % | 4 |
| MBD-mini | 54 | 3 072 | 693 | 905 | 104.1 | 1.14 % | 20 % | 5 |

BankSim needed no rebalancing — its customers are long-lived enough that 16.5 %
of them see fraud in the held-out tail. The legacy pools for the other three
were rebalanced from well under 2 % up to 20 %. PaySim ends with only four marks because `DEBIT`
never appears on destination accounts that reach eight events.

The two extremes are worth keeping in mind when reading the tables: PaySim has
the fewest marks and the shortest histories (four marks, ~11 events), so its
next-type task is nearly trivial while its classification task is close to
hopeless; MBD-mini has 54 marks over ~104 events, so it is the only benchmark
here where mark prediction is genuinely hard.

### Why the entity is not always the obvious one

PaySim's originating accounts (`nameOrig`) are essentially unique — 6.35 M
distinct originators across 6.36 M rows — so per-originator "histories" would be
single events. Sequences are therefore built on the **destination** account,
which is the side that actually accumulates a stream. IBM AML is grouped the
same way for the same reason. BankSim and MBD-mini group by customer/client
directly.

### Feature encoding

Each adapter reduces its source to a `TransactionTable` and shared code does the
rest (`eventfm/datasets/tabular.py`):

* numeric columns (amounts, balances, FX ratios) → **quantile buckets**, not
  uniform-width bins, because transaction amounts are heavy-tailed and uniform
  bins collapse almost everything into bin 0;
* categorical columns keep their most frequent levels and fold the tail into
  `other`;
* BankSim records only the day and PaySim only the hour of a transaction, so
  rows sharing a bucket are spread evenly inside it — otherwise every
  inter-arrival time inside a bucket would be exactly zero and the TPP time
  target would be degenerate.

## 3. The two tasks

### Classification — forward-looking, not retrospective

The obvious framing for a fraud dataset is "does this history contain fraud?",
but that is a *detection* task: the fraudulent transactions are sitting in the
input. Since the whole project is about **prediction**, the label here is
forward-looking instead:

* **Simulators (BankSim, PaySim, IBM AML)** — the last 30 % of each entity's
  history is held out. The model sees only the earlier events; the label is
  whether a flagged (fraud / laundering) event occurs in that held-out tail.
* **MBD-mini** — the benchmark already ships forward-looking monthly targets.
  History is cut at 2022-09-30 and the label is `target_1` over the reporting
  months that follow, so no observed event can overlap the label window.

The flag column is never an input feature in either case.

Natural entity-level positive rates are low (0.4 %–1.6 %). The current
converter downsamples negatives in **training only**; validation and test retain
natural prevalence. It records natural and used rates separately for every
split. The published 624-cell architecture screen used an older globally
downsampled conversion, so its average precision and operational classification
metrics are not paper-grade natural-prevalence estimates.

### TPP — next event, marked

From each history, prefixes are drawn and the model predicts the next mark and
the next inter-arrival time. Scored separately for *what* and *when*, which the
neural-TPP review specifically recommends over a single likelihood that hides
which half of the problem the model actually solved:

* `next_type_accuracy`, `next_type_macro_f1` — the marks are heavily skewed on
  BankSim and PaySim, so macro F1 is the honest read;
* `delta_log_rmse`, `delta_log_mae` — on `log1p(seconds)`, lower is better;
* `time_nll` — mean negative log-likelihood, where the model defines one.

## 4. Splits and scaling curves

The default datasets assign entities to train / validation / test by a stable
hash of entity id (20 % test, 15 % validation). The four `*_chrono` variants
instead sort whole entities by their final observed timestamp and assign the
newest histories to evaluation. Both policies are entity-disjoint.

Rare-category vocabularies and numeric quantile boundaries are now fit on
retained training histories only and applied unchanged to validation/test.
Prepared split hashes and fitted artifacts are stored in `meta.json`.

A scaling point of size *n* is a **stratified** subsample of the training split
that preserves its label balance — without that, the smallest points would
sometimes contain a single class and the curve would be meaningless. The
vocabulary of feature values is rebuilt from each subsample (only training data
is visible), while the mark list is pinned from the dataset metadata so the
next-event label space is identical at every size.

## 5. Grid and execution

`datasets × tasks × methods × sample sizes` = 624 cells. Cells are independent
and each writes one JSON file, so the grid runs as a SLURM array where a task
that dies costs only its own slice and can be resubmitted without redoing
finished work.

The completed run split as follows:

| Slice | Cells | Resource | Cost |
| --- | ---: | --- | --- |
| 11 non-LLM methods | 480 | 30-way CPU array | ~6 CPU-hours |
| 3 LLM methods | 144 | 12-way RTX 3090 array | ~15 GPU-hours |

All 624 completed with status `ok`.

**One environment gotcha worth recording.** The installed torch is a CUDA 13
build, which dropped Pascal (`sm_61`); on a GTX 1080 every cell dies with
`no kernel image is available`. A node-level `--constraint` does *not* prevent
this — the `*-pheno-*` nodes advertise both `rtx_3090` and `gtx_1080` features,
satisfy the constraint, and then allocate the 1080. The GPU type has to be named
in the gres request itself (`--gres=gpu:rtx_3090:1`). `slurm/run_benchmark_llm_array.sbatch`
also runs a preflight kernel launch so a mis-scheduled job fails in seconds
rather than caching a failure for every cell in its slice.

## 6. Known limitations

Every paper-named entry is labelled `Approximation` unless its recipe has been
validated, and each declares the specific departure. `MethodSpec.__post_init__`
refuses an approximation without a divergence, and tests lock that in, so tables
cannot silently start reading as reproductions.


* Sequence counts after rebalancing are in the low thousands per dataset, which
  caps the scaling x-axis at ~2048 training sequences. The curves show
  sample-scaling direction, not converged scaling laws.
* Pretraining stages run on the same (small) training split as the downstream
  head, so they measure "does this objective help at this scale", not the
  large-corpus foundation-model regime the papers operate in.
* The LM entries use a 135 M-parameter base model with a frozen backbone and
  LoRA (1.8–2.2 M trainable parameters). The original papers use
  billion-parameter models; absolute numbers are therefore not comparable to the
  published ones, only to each other under equal budget. In the completed run
  these three entries occupy three of the bottom four mean ranks — that is a
  statement about the budget they were given here, not about the methods.
* MBD-mini's dialog and geo streams are not used — only the transaction stream —
  so the multimodal aspect of MM-TPP is exercised on time and text, not images.
