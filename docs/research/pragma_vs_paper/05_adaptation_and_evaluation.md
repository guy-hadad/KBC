# 05 — Downstream adaptation and evaluation (paper §2.3.5, §3)

[← index](README.md)

## E1. Adaptation mode

**Kind:** Missing · **Impact:** High

**Paper:** There are two modes.

* **Embedding probe**: a frozen backbone with a linear model on top (see [E2](#e2-embedding-probe)).
* **LoRA** on the QKV projections and MLP layers of every encoder layer, which
  trains about 2 to 4 % of the parameters. Default rank 8 and α = 8, with the
  rank swept over {4, 8, 16} on small datasets, trained with **Adam**.

The headline results (Table 2) use LoRA on PRAGMA-L. The paper also reports
that LoRA beats probing at every scale (Table 5) and matches or beats training
from scratch (Table 4).

**Repo:**

* Every PRAGMA result uses **full fine-tuning of all parameters**
  (`TrainingSpec.adaptation = "full"`,
  [`base.py:49`](../../../eventfm/methods/base.py#L49)), with AdamW at lr 3e-4
  for 8 epochs.
* **There is no LoRA for PRAGMA.** The `peft` LoRA dependency is only used by
  the LLM entries.
* The repo's `peft` regime is a *bottleneck adapter*, not LoRA, and it
  **raises an error for PRAGMA**: it requires `backbone.adapter`, which
  `PragmaBackbone` does not have
  ([`base.py:254-259`](../../../eventfm/methods/base.py#L254-L259)). See
  [B3](06_implementation_issues.md#b3-the-peft-regime-crashes-for-pragma).
* PRAGMA is not part of the `adaptation-regimes` suite
  ([`experiments.py:318-337`](../../../eventfm/benchmark/experiments.py#L318-L337)),
  so no frozen or adapter results exist for it.

## E2. Embedding probe

**Kind:** Missing · **Impact:** Medium

| | Paper (§3.1.1) | Repo |
| --- | --- | --- |
| Tokens probed | `[USR]`, the final `[EVT]`, and their concatenation, all from `z_h` | `[USR]` only, in [`extract_embeddings.py:58`](../../../scripts/extract_embeddings.py#L58) (Hydra checkpoint) |
| Pre-processing | Standard scaling, because the model is pre-norm | None |
| Probe | Logistic or linear regression fitted with **L-BFGS**, in minutes | No probe code for PRAGMA. `sklearn` `LogisticRegression` only appears in the count-feature control ([`classic.py:117-133`](../../../eventfm/methods/classic.py#L117-L133)) |
| Benchmark "frozen" regime | – | Trains dropout + a linear head with AdamW for 8 epochs through `Trainer`; no scaling, no L-BFGS, `[USR]` only. Not run for PRAGMA |

The backbone already returns `last_event_embedding`
([`modeling_pragma.py:249-252`](../../../eventfm/models/modeling_pragma.py#L249-L252)),
so the paper's last-`[EVT]` and concatenated variants are easy to add. Only
the TPP head uses it at the moment.

## E3. Task head

**Kind:** Match (minor) · **Impact:** Low

**Paper:** "a classification head". Its form, and the token it reads during
LoRA fine-tuning, are unspecified. Product recommendation is a **multilabel**
task.

**Repo:** `Dropout(0.1) → Linear` on `z_h,0`, the `[USR]` output
([`modeling_pragma.py:354`](../../../eventfm/models/modeling_pragma.py#L354)),
trained with cross-entropy. With `num_labels == 1` it switches to MSE
regression. Multilabel (independent sigmoid) outputs are not supported.

## E4. Scratch vs pretrained comparison

**Kind:** Substitution · **Impact:** Medium

**Paper (Table 4):** PRAGMA-M with LoRA fine-tuning, compared with
full-parameter training from scratch.

**Repo:** `pragma` is the same architecture trained from scratch with full
fine-tuning. `pragma-mlm` adds 3 epochs of in-domain MLM, then also uses full
fine-tuning. The pair is the closest thing to Table 4, with three differences:

1. Both arms update every parameter. The paper's pretrained arm updates only
   2 to 4 %.
2. The pretraining data is the same labelled subset, not a separate corpus
   ([D1](01_data.md#d1-pre-training-corpus)).
3. The scale is about 1 M parameters, not 100 M ([A7](03_architecture.md#a7-model-size-and-shape)).

## E5. Downstream tasks

**Kind:** Scale · **Impact:** High (for comparability)

| Paper task | Formulation | Closest repo analogue |
| --- | --- | --- |
| Credit scoring | P(default within 12 months), binary | none |
| Communication engagement | Opens a re-engagement message, binary, small data | none |
| External fraud | Binary; precision and recall | BankSim, PaySim: "a fraud-flagged event occurs in the held-out tail" |
| Product recommendation | **Multilabel** conversion; mAP | MBD's provided product-propensity target (binary, one target) |
| Recurrent transactions | **Per-transaction** binary (does this payment recur next month?); macro-F1 | none. The repo has no transaction-level targets |
| Lifetime value | Positive gross profit, binary, short histories | none |
| AML (limitation study) | Binary, F0.5; PRAGMA loses to a relational baseline | IBM AML: laundering in the tail of a receiving account. Framed as a normal task, not as a limitation study |
| Uplift | Meta-learner on frozen embeddings; AUUC, SNIPS | none |
| – | – | Synthea: cardiovascular or renal event in the tail. Amazon: negative review in the tail |

The repo also evaluates a next-event task that the paper does not include
([E6](#e6-next-event-tpp-task-not-in-the-paper)).

## E6. Next-event (TPP) task (not in the paper)

**Kind:** Extra

`PragmaForNextEventPrediction`
([`modeling_pragma.py:364-416`](../../../eventfm/models/modeling_pragma.py#L364-L416))
puts two linear heads on `last_event_embedding`:

* a next-event-type classifier, and
* a regression on `log1p(Δt)` with a squared-error loss. The bias is
  initialised to `log1p(6 h)`, and the time loss is weighted by
  `tpp_loss_weight`.

Nothing in the paper corresponds to this. Note that this head differs from the
log-normal mixture used by the other TPP methods, and its reported `time_nll`
is not a true likelihood. See
[B6](06_implementation_issues.md#b6-pragmas-tpp-head-differs-from-the-shared-tpp-decoder).

## E7. Metrics and reporting

**Kind:** Scale · **Impact:** Low

**Paper:** Only relative changes, `x / baseline − 1`, against internal
task-specific production models. Metrics are ROC-AUC, PR-AUC, precision and
recall, mAP, macro-F1, F0.5, AUUC and SNIPS.

**Repo:** Absolute metrics against in-repo controls and baselines:

* classification: AUC, average precision, accuracy, macro-F1;
* TPP: next-type accuracy and macro-F1, and Δ log RMSE and MAE;
* power-law fits `E(N) = a·N^-b` across sample sizes.

Training sets are rebalanced to 20 to 25 % positives while test sets keep
natural prevalence ([D7](01_data.md#d7-filtering-and-pre-processing)). This
affects calibration but not AUC.

## E8. Paper experiments not reproduced

**Kind:** Missing · **Impact:** Medium

| Paper study | Status in this repo |
| --- | --- |
| Model scale S → M → L (Table 3) | Not run. PRAGMA is not in `model-scale-ablation` |
| Pretrained vs scratch (Table 4) | Partly, as `pragma-mlm` vs `pragma` ([E4](#e4-scratch-vs-pretrained-comparison)) |
| LoRA vs embedding probe (Table 5) | Not run. Neither LoRA nor the probe exists for PRAGMA |
| Full vs event-only (Table 6) | **Cannot be run.** Every current run is already event-only ([A3](03_architecture.md#a3-profile-state-encoder)) |
| Uplift with a meta-learner (Table 7) | Not implemented |
| Pre-trained Nemotron text encoder (§3.4.4, Table 8): one frozen text embedding per text value, a trainable projection, and MSE reconstruction of that embedding in MLM | Not implemented. There are no text fields ([T3](02_tokenisation.md#t3-categorical-vs-textual-values)) |
| AML limitation (Table 9): a linear probe on PRAGMA-L vs a network-aware baseline | IBM AML is included, but as a standard fine-tuned task without a relational baseline |
| Checkpoint selection by probe (§3.1.1) | Not implemented |
