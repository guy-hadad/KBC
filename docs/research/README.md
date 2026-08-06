# EventFM research programme

This directory is the paper-facing specification for EventFM. The runnable
benchmark described in [`../benchmark_design.md`](../benchmark_design.md) is the
starting point; the documents here define the broader experiments needed to
support a strong conference submission.

The central question is:

> Which combination of within-event encoding, temporal representation,
> sequence backbone, and self-supervised objective yields banking-event
> representations that transfer across event-, prefix-, and customer-level
> tasks under controlled data and compute budgets?

This framing is intentionally broader than "PRAGMA versus TPP-LLM." Those
systems optimize different capabilities. The paper should separate five axes:

1. **Within-event encoding:** flat fields, hierarchical key/value fields,
   event-type-specific encoders, and text/structured fusion.
2. **History backbone:** recurrent, attention, continuous convolution, state
   space, and pretrained language model.
3. **Learning objective:** reconstruction, autoregression, contrastive learning,
   point-process/time-to-event likelihood, and joint time-value prediction.
4. **Time interface:** continuous features or embeddings, probabilistic heads,
   and the discrete tokenizers studied by Liu et al. (GEM 2026).
5. **Transfer level:** event, causal prefix, and whole-customer representation.

## Documents

| Document | Purpose |
| --- | --- |
| [`method_catalog.md`](method_catalog.md) | Unified inventory of current implementations, required baselines, and related-work-only systems. |
| [`temporal_tokenization.md`](temporal_tokenization.md) | Complete, implementation-oriented specification of every method and ablation in Liu et al. (GEM 2026). |
| [`experimental_protocol.md`](experimental_protocol.md) | Paper-grade datasets, tasks, controls, metrics, statistical tests, and reproducibility rules. |
| [`paper_blueprint.md`](paper_blueprint.md) | Candidate thesis, research questions, claim-to-evidence map, paper outline, and execution phases. |
| [`implementation_manifest.md`](implementation_manifest.md) | Generated inventory of every runnable method, dataset, tokenizer, and experiment suite. |
| [`experiment_execution.md`](experiment_execution.md) | Commands, suite map, result artifacts, validity gates, and execution boundary. |

## Status vocabulary

Every proposed method must use one of these labels. A method is not
"implemented" merely because the repository contains a similarly named small
model.

| Status | Meaning |
| --- | --- |
| **Implemented** | Runnable in the shared benchmark, with a smoke test. This does not by itself imply paper-faithful reproduction. |
| **Approximation** | Runnable implementation that preserves the central comparison but differs materially from the cited recipe. Differences must be disclosed. |
| **Specified** | Paper-facing contract exists, but code and verification are incomplete. |
| **Related work** | Important context whose industrial data, scale, or unavailable details make a fair reproduction inappropriate. |
| **Candidate contribution** | A new EventFM combination; it must never be presented as a published baseline. |

## Minimum evidence before making the main claim

The main paper claim should not be fixed until the following evidence exists:

- at least four banking datasets with untouched natural-prevalence test sets;
- event-, prefix-, and customer-level tasks, rather than one classification
  label and one next-event task alone;
- strong non-neural, supervised, contrastive, generative, and temporal
  baselines;
- frozen probes, parameter-efficient adaptation, and full fine-tuning where
  feasible;
- at least five seeds for headline results, uncertainty intervals, and paired
  comparisons;
- data- and compute-matched comparisons, plus native-recipe results for methods
  whose optimization needs differ;
- independent ablations of the event encoder, time representation, history
  backbone, and loss terms;
- leakage audits showing that splits, vocabularies, quantizers, normalization,
  and label windows use training data only;
- efficiency reporting: trainable/total parameters, temporal tokens per event,
  peak memory, training time, and inference throughput.

## Scope boundary

The current code is a useful architecture-screening benchmark. It is not yet a
foundation-model result: the training sets are small, several named methods are
controlled approximations, and pretraining and downstream training use the same
limited corpus. Paper text and tables must preserve that distinction until a
large unlabeled pretraining stage and genuine transfer evaluation are run.
