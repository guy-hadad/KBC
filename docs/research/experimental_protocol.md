# Paper-grade experimental protocol

This protocol is the contract for results intended for a conference paper. It
extends the current architecture-screening benchmark without invalidating its
existing runs. Deviations must be recorded in run metadata and in the paper.

The executable mapping is in
[`experiment_execution.md`](experiment_execution.md). The current code enforces
train-only tabular preprocessing, training-only rebalancing, natural evaluation
prevalence, hash and chronological whole-entity splits, frozen/adapter/full
adaptation, named ablation variants, source-only leave-one-dataset-out pools,
five-seed headline grids, and provenance-rich per-cell artifacts. Event-level
labels, censored downstream survival endpoints, protected subgroup analyses,
and per-example paired-bootstrap artifacts still depend on defensible dataset
targets and are not implied by registry coverage.

## 1. Research questions and hypotheses

| ID | Research question | Pre-registered hypothesis |
| --- | --- | --- |
| RQ1 | Which pretraining objectives transfer at event, causal-prefix, and customer levels? | Autoregression will be strongest locally, contrastive learning globally, and a controlled hybrid will reduce the trade-off. |
| RQ2 | How should irregular banking time enter a model? | Log-gap plus calendar features will be a strong continuous default; the best discrete tokenizer will depend on interval distribution and periodicity. |
| RQ3 | Does a heterogeneous event encoder help beyond a flat shared encoder? | Type-specific or hierarchical field encoding will help most when event schemas differ materially. |
| RQ4 | Does joint marked time-to-event pretraining add information beyond next-type and next-gap losses? | Joint time/value modeling will improve next-amount and delayed-outcome calibration, especially for amount-sensitive tasks. |
| RQ5 | Do improvements survive transfer, natural class prevalence, and compute matching? | Some gains in the current rebalanced small-data benchmark will shrink; robust gains should persist under frozen probes and untouched test distributions. |

The hypotheses are falsifiable. They should be timestamped with the final task,
metric, and model-selection rules before the headline sweep.

## 2. Evaluation units and tasks

### Event level

- masked field/type reconstruction;
- event category or anomaly classification when labels exist;
- amount/value regression;
- event retrieval, using same semantic type or known paired views as relevance.

### Causal prefix level

- next event type;
- next event time or inter-arrival distribution;
- next numerical mark such as amount;
- horizon-specific imminent fraud/default/churn;
- multi-step rollout on datasets where targets are reliable.

Every prefix task must enforce causal attention and contain no event from the
prediction window.

### Customer/sequence level

- forward-looking fraud/AML outcome;
- churn/default/product propensity where an open label exists;
- lifetime-value or future-spend regression;
- segmentation or customer retrieval, reported as exploratory unless external
  ground truth exists.

Do not average these levels into one score. A main result is a profile of
transfer behavior, not a leaderboard scalar.

## 3. Data and split policy

The current primary datasets remain BankSim, PaySim, IBM AML HI-Small, and
MBD-mini. Their adapter-specific entity and mark definitions remain documented
in [`../benchmark_design.md`](../benchmark_design.md).

For paper runs:

1. **Freeze the test set once.** Entity IDs must never cross splits.
2. **Keep validation and test at natural prevalence.** Rebalancing or negative
   downsampling may be applied to pretraining/training only. Report the natural
   and sampled prevalence separately. Average precision is otherwise not
   comparable to deployment prevalence.
3. **Use forward-looking labels.** Fraud/AML flags in the future window must not
   appear as input fields or determine preprocessing of the visible history.
4. **Fit all artifacts on training only.** This includes quantile buckets,
   normalizers, vocabularies, rare-category thresholds, time-bin ranges, RSQ
   codebooks, imputers, and feature selection.
5. **Add a chronological robustness split.** Train/validate on earlier calendar
   intervals and test later, while preventing entity leakage. This tests drift
   and calendar-token shortcuts.
6. **Add a cross-dataset or cross-schema study.** Pretrain on a subset of banks
   or simulators, then evaluate a target dataset with a new vocabulary/schema.
   Report zero-shot where meaningful and few-shot adaptation at fixed sizes.
7. **Deduplicate before splitting.** Hash normalized event histories and inspect
   near-duplicate entities, synthetic simulator templates, and repeated text.

The five public datasets used by the GEM temporal-tokenization paper may be
used as an external non-banking validation of tokenizer trends, but they must
not replace banking results or be mixed into the banking aggregate.

## 4. Pretraining and adaptation regimes

Every representation method should expose three downstream regimes:

| Regime | Trainable components | What it measures |
| --- | --- | --- |
| Frozen probe | Linear head, small MLP, or LightGBM on frozen representations | Representation quality and ease of use |
| Parameter-efficient | Adapters or LoRA plus task head | Low-cost specialization |
| Full fine-tuning | Entire encoder and head, when feasible | Maximum in-domain adaptation |

Use a separate unlabeled pretraining pool when making foundation-model claims.
Pretraining on the labeled training fold is still valuable, but must be named
"in-domain self-supervision," not large-scale foundation pretraining.

Report event, prefix, and customer representations explicitly. Pooling choices
(last, mean, attention, special user token) are model components and require a
small controlled ablation rather than silent per-method tuning.

## 5. Fair comparison policy

Two complementary tables are required:

- **Controlled table:** common hidden size, context in events, training examples,
  update/FLOP budget, tuning budget, and downstream head. This isolates the
  architectural factor as far as possible.
- **Native-recipe table:** reasonable paper-recommended optimizer, objective,
  decoder, and scale for each family, with measured compute. This avoids
  crippling methods that genuinely require different optimization.

"Same optimizer and schedule for all models" is not sufficient fairness. Equal
hyperparameters can systematically favor one family. Give each method the same
number of validation trials from a pre-declared search space, then lock the
selected configuration before test evaluation.

For LLM tokenizers, run both:

- equal number of historical events, exposing actual token cost; and
- equal token/context budget, exposing effective-history loss.

For pretrained LMs, report results with and without meaningful event text when
possible. Replacing type descriptions with anonymized IDs measures how much of
the gain comes from language priors rather than event-sequence learning.

## 6. Metrics

### Classification and ranking

- ROC-AUC and average precision;
- macro-F1 and balanced accuracy at a validation-selected threshold;
- precision/recall at operational budgets;
- expected calibration error and Brier score;
- subgroup performance where ethically and legally appropriate attributes are
  available and approved.

### Event type and numerical marks

- next-type accuracy, macro-F1, and top-k accuracy;
- amount/value MAE and RMSE on both physical and `log1p` scales;
- quantile/pinball loss if probabilistic numerical predictions are emitted.

### Time and survival

- time negative log-likelihood when a proper density is available;
- MAE/RMSE of the next gap in physical units and on `log1p` scale;
- C-index and integrated Brier score for censored delayed outcomes;
- calibration by prediction horizon;
- invalid-decode rate for generative time tokenizers;
- joint type-time metrics only as supplements to separately reported quality.

### Retrieval and representation

- recall@k, mean reciprocal rank, and normalized discounted cumulative gain;
- linear separability/probe score at each representation level;
- centered kernel alignment or representational similarity only for analysis,
  not as evidence of downstream utility.

### Efficiency

- total and trainable parameters;
- measured training FLOPs or a clearly defined proxy;
- peak accelerator memory and wall-clock training time;
- examples/events/tokens per second at training and inference;
- temporal tokens per event, prompt truncation rate, and effective events seen;
- disk/checkpoint size and energy/carbon estimate when available.

## 7. Statistical protocol

- Use at least five independent seeds for headline cells and three for screening.
- Select checkpoints and thresholds on validation only; evaluate the test set
  once per locked seed/configuration.
- Report mean, standard deviation, and a 95% confidence interval.
- Use paired bootstrap intervals over the same test entities for pairwise metric
  differences; for seed-level conclusions, add a hierarchical bootstrap or a
  paired nonparametric test.
- Correct for the pre-declared family of headline comparisons (for example,
  Holm correction). Label all other comparisons exploratory.
- Report per-dataset results. If an aggregate is useful, use normalized ranks or
  standardized effects with uncertainty; never hide a failed dataset behind a
  raw mean of incompatible metrics.
- Publish all seeds, including failed or divergent runs, with failure reasons.

## 8. Ablation matrix

Run ablations in blocks so the full Cartesian product remains tractable.

### A. Time information under one shared backbone

1. no time;
2. ordinal position only;
3. raw gap;
4. `log1p` gap;
5. gap plus time since sequence start;
6. calendar Fourier features;
7. Time2Vec;
8. functional time representation;
9. continuous-time RoPE;
10. quantile/log buckets;
11. byte tokens;
12. TPP/TTE likelihood;
13. marked TTE.

The GEM tokenizer grid is a nested ablation described in
[`temporal_tokenization.md`](temporal_tokenization.md).

### B. Objective under one shared encoder

- supervised only;
- autoencoding;
- masked fields;
- next event;
- subsequence contrastive;
- time-to-event;
- marked time-to-event;
- pre-registered pairs;
- full candidate multi-objective model.

Match updates and FLOPs. Record per-loss gradient norms and cosine similarity so
negative transfer is diagnosable.

### C. Encoder/backbone under one shared objective

- flat shared field encoder versus hierarchical key/value encoder versus
  event-type-specific encoder;
- GRU, bidirectional attention, causal attention, continuous convolution,
  state-space model, and causal LM;
- short, medium, and long contexts with truncation statistics.

## 9. Scaling studies

Separate three quantities currently entangled by a sample-size sweep:

- **unlabeled pretraining scale** at fixed labeled data;
- **labeled adaptation scale** at fixed checkpoint;
- **model/compute scale** at fixed data.

Use nested subsets, identical validation/test sets, and at least three seeds per
point. Fit a scaling relationship only when there are enough non-saturated
points and report uncertainty on the exponent. Otherwise call the plot a
sample-efficiency curve, not a scaling law.

## 10. Reproducibility and audit artifacts

Each result JSON should contain:

- git commit and dirty-tree indicator;
- full resolved configuration and random seeds;
- dataset version, file hashes, split manifest, and label-window definition;
- fitted preprocessing/tokenizer artifact hashes;
- model source/checkpoint revision and license;
- total/trainable parameters and measured resource use;
- best validation step and selection metric;
- all test metrics plus per-example predictions in a protected artifact store;
- failure status and stack trace when incomplete.

Before submission, release or archive split manifests, preprocessing code,
configuration grids, aggregation code, and a small synthetic end-to-end fixture.
For restricted financial data, release schema-compatible synthetic fixtures and
the exact audit logic even when raw examples cannot be distributed.

## 11. Result validity gates

A table is not paper-ready until:

- all required seeds completed or are explicitly marked;
- the test set has natural prevalence and was not used for selection;
- preprocessing and time tokenizers pass train-only leakage tests;
- methods have comparable tuning and compute accounting;
- confidence intervals and paired differences are generated automatically;
- every method name links to an exact implementation/configuration;
- approximation versus faithful reproduction is visible in the caption;
- the table can be regenerated from immutable per-run artifacts.
