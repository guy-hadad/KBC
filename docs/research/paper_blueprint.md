# Conference-paper blueprint

This is a living argument map, not a promise that the current results already
support the proposed claims.

## Candidate title and one-sentence thesis

**Candidate title:** *What Should a Banking Event Foundation Model Learn? A
Controlled Study of Objectives, Temporal Interfaces, and Transfer Levels*

**Candidate thesis:** Banking event models exhibit a measurable local/global
transfer trade-off, and the most reliable way to reduce it is to combine
heterogeneous field encoding with explicitly controlled generative,
contrastive, and marked-time objectives while adapting the time interface to
the dataset's interval structure.

The thesis is deliberately conditional. If the combined objective does not win
under equal compute, the paper can still make a valuable benchmark finding:
no universal representation exists, and method choice should depend on transfer
level and temporal distribution.

## Proposed contributions

1. **A controlled banking event-representation benchmark** spanning event,
   causal-prefix, and customer tasks, with natural-prevalence evaluation and
   frozen/PEFT/full adaptation.
2. **A factorized comparison** of event encoder, history backbone, pretraining
   objective, and time interface, avoiding comparisons of inseparable systems.
3. **The first banking-focused evaluation of temporal tokenization choices**
   from numeric strings through calendar, byte, uniform-bin, and residual scalar
   quantization, including efficiency and chronological robustness.
4. **A marked time-value objective for transaction streams**, tested through
   controlled additions to masked, autoregressive, and contrastive learning.
5. **Reproducible resources:** versioned adapters, leakage-safe splits,
   configuration manifests, reference-method adapters, and complete run-level
   artifacts.

Claims such as "first," "state of the art," and "foundation model" require a
fresh literature search, substantially larger unlabeled pretraining, and exact
benchmark evidence at submission time. Until then, use "candidate" wording.

## Claim-to-evidence map

| Intended claim | Evidence required | Result that would falsify or weaken it |
| --- | --- | --- |
| Objective choice creates a local/global trade-off | Frozen probes for generative, contrastive, and masked objectives on all three representation levels; paired intervals | One objective dominates consistently or differences vanish under equal compute |
| Hybrid objectives reduce the trade-off | Single-loss, pairwise, and full-objective ablations with matched updates/FLOPs | Gains arise only from more compute, or one task improves by sacrificing another |
| Temporal tokenizer should match data distribution | Training-only distribution diagnostics and the full GEM tokenizer grid across datasets; interaction analysis | A single tokenizer wins robustly or diagnostics do not predict ranking |
| Marked TTE helps banking transfer | Time-only, next-value-only, independent multitask, and joint marked likelihood comparisons | No gain on amount-sensitive or delayed outcomes, poor calibration, or unstable optimization |
| Heterogeneous event encoders improve cross-schema transfer | Flat versus hierarchical versus type-specific encoders on within- and cross-dataset tests | Improvements occur only in-domain or disappear at matched parameter count |
| EventFM is practical | Accuracy/quality versus memory, throughput, parameters, temporal token count, and truncation | Gains require disproportionate compute or unusable context expansion |

## Paper structure

### 1. Introduction

- Banking streams combine heterogeneous structured fields, irregular time,
  continuous values, sparse labels, and several representation levels.
- Existing banking encoders, TPPs, and LLM-based event models optimize different
  slices of this problem, so direct system-to-system comparisons obscure the
  source of gains.
- State the factorized benchmark and the tested thesis.
- List contributions only at the strength supported by final results.

### 2. Related work

Organize by modeling decision, not a chronological list:

- transaction and banking foundation models: PRAGMA, NPPR, TransactionGPT,
  TREASURE, nuFormer, open-banking and multimodal financial models;
- event-sequence representation learning: CoLES, MLEM, joint
  contrastive/generative learning, TabBERT/TabGPT, SOHET;
- continuous-time event models: NHP, THP/SAHP/AttNHP, IFTPP, COTIC, TPP-LLM,
  Language-TPP;
- time-to-event pretraining: MOTOR and marked-TTE/ORA;
- temporal representations and tokenization: Time2Vec, functional time,
  continuous-time RoPE, and Liu et al. (GEM 2026).

End the section with the unfilled gap: no controlled banking study jointly
separates transfer level, learning objective, heterogeneous event encoding, and
temporal interface.

### 3. Problem formulation

Define an entity history as

\[
S_u=\{(t_i,k_i,x_i)\}_{i=1}^{n_u},
\]

where \(t_i\) is time, \(k_i\) is event/schema type, and \(x_i\) contains
categorical, numerical, and optional text fields. Define event representation
\(h_i\), causal prefix state \(z_{u,i}\), and customer representation \(g_u\).
Separate downstream discriminative risk, next-event likelihood, censored
time-to-event risk, and representation-probe evaluation.

### 4. Factorized model and objectives

- field/event encoder;
- time interface;
- history encoder;
- event/prefix/customer readouts;
- masked-field, next-event, contrastive, and marked-TTE losses;
- loss weighting and compute matching.

The proposed model should be presented after the factorization, as one point in
the same design space rather than a special case with hidden advantages.

### 5. Experimental design

Summarize datasets, leakage-safe splits, task levels, adaptation regimes,
baselines, tuning budgets, metrics, seeds, statistical tests, and compute. Point
to the full protocol and appendix for exact configurations.

### 6. Results

Answer research questions in order:

1. Where each baseline transfers well or poorly.
2. How continuous and tokenized time interfaces compare.
3. Whether heterogeneous encoders help.
4. Whether marked-TTE and the combined objective add value.
5. Robustness to data/model scale, chronology, schema transfer, and compute.

### 7. Analysis and limitations

- objective gradient conflict and representation similarity;
- errors by interval quantile, event type frequency, sequence length, and
  calendar regime;
- tokenizer invalid generations and context truncation;
- synthetic versus real-data external validity;
- unavailable industrial scale and restricted labels;
- privacy, fairness, surveillance, and downstream misuse risks.

## Planned main tables

| Table | Contents |
| --- | --- |
| 1. Data/tasks | Dataset sizes before/after training sampling, natural prevalence, sequence/field statistics, label horizons, and split dates. |
| 2. Baselines | Exact method, status (faithful/approximation), objective, time interface, parameters, pretraining data, and compute. |
| 3. Transfer | Frozen-probe results across event, prefix, and customer tasks with confidence intervals. |
| 4. Adaptation | Frozen versus PEFT versus full fine-tuning for representative methods. |
| 5. Time tokenization | All promoted GEM variants: type/time quality, invalid rate, tokens/event, effective history, and throughput. |
| 6. Objective ablation | Single losses, pairs, and full candidate objective at matched updates/FLOPs. |
| 7. Robustness | Chronological and cross-schema transfer, plus data/model scaling. |

Avoid one enormous table combining incompatible tasks. Put complete per-dataset
results and all seeds in the appendix/artifact.

## Planned figures

1. **Design-space diagram:** event encoder → time interface → history backbone →
   event/prefix/customer readouts, with objectives attached to their targets.
2. **Local/global transfer frontier:** next-event performance versus customer
   classification for every objective under equal compute.
3. **Tokenizer/data alignment:** interval-distribution diagnostics versus time
   error and tokens/event for each tokenizer.
4. **Scaling decomposition:** separate curves for unlabeled pretraining data,
   labeled adaptation data, and model compute.
5. **Calibration:** reliability by time horizon and natural-prevalence risk.
6. **Ablation effect plot:** paired effect sizes with confidence intervals,
   rather than only raw scores.

## Analysis that can make the work more than a benchmark

- Test whether simple distribution statistics predict the best time tokenizer;
  evaluate the selector with leave-one-dataset-out validation.
- Measure whether contrastive and generative objectives encode complementary
  subspaces using probes and representational similarity.
- Examine gradient cosine similarity among objectives and relate conflict to
  negative transfer.
- Stratify temporal error by gap quantile and periodic versus non-periodic
  events to explain tokenizer rankings.
- Test whether text-aware LLM gains persist after anonymizing event names.
- Quantify representation stability under schema perturbations, missing fields,
  shifted category vocabularies, and longer histories.

These analyses should be pre-specified where they support a main claim;
post-hoc observations must be labeled exploratory.

## Execution phases and exit criteria

### Phase A: benchmark validity

- natural-prevalence validation/test splits;
- LightGBM and supervised GRU controls;
- five-seed aggregation and paired intervals;
- immutable run metadata and leakage tests.

**Exit:** a small set of current methods produces reproducible, statistically
valid tables from a clean checkout.

### Phase B: transfer benchmark

- event/prefix/customer task APIs;
- frozen, PEFT, and full adaptation;
- NPPR, CMLM+CoLES, and faithful temporal reference baselines.

**Exit:** the local/global trade-off can be measured without relying on one
dataset or one adaptation regime.

### Phase C: temporal tokenization

- pure generative time interface;
- every GEM method and required ablation;
- distribution diagnostics, equal-event/equal-token contexts, and efficiency.

**Exit:** tokenizer conclusions hold on all primary banking datasets and a
chronological split with five-seed uncertainty.

### Phase D: proposed model

- heterogeneous encoder and marked-TTE decoder;
- single/pair/full objective ablations;
- matched compute, gradient analysis, cross-schema transfer.

**Exit:** the final thesis is chosen based on evidence. If the combination does
not dominate, pivot the paper toward the empirical trade-off and selection
guide rather than hiding the negative result.

### Phase E: submission package

- frozen tables/figures regenerated from archived artifacts;
- source/claim audit and updated literature search;
- limitations, ethics, licenses, compute, and reproducibility checklist;
- anonymized repository and appendix reviewed by someone who did not build the
  benchmark.

## Immediate decisions to record before implementation

- target venue and page/appendix constraints;
- what qualifies as the large unlabeled pretraining corpus;
- exact downstream labels available on each public dataset;
- whether external non-banking tokenizer datasets are in scope;
- primary metric per research question;
- compute budget and maximum number of headline cells;
- faithful-reference versus controlled-approximation policy for each baseline;
- release constraints for data, checkpoints, and predictions.

