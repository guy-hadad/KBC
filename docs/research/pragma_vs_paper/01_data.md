# 01 — Data (paper §2.1, §3.1.3)

[← index](README.md)

## D1. Pre-training corpus

**Kind:** Scale · **Impact:** High

| | Paper | This repo |
| --- | --- | --- |
| Size | 26 M user records, 111 countries, 24 B events, 207 B tokens | 64 to 93 305 labelled training sequences per cell, each ≤64 events at training time |
| Relation to downstream data | A separate unlabelled corpus. Downstream datasets are collected "the same way" but are distinct (§3.1.3) | `pragma-mlm` pretrains on **the same labelled training subset** it fine-tunes on ([`pragma.py:99`](../../../eventfm/methods/pragma.py#L99)) |

The paper's main claim is transfer from a large, broad corpus to many tasks.
The repo's `pragma` vs `pragma-mlm` comparison tests something narrower:
whether an MLM warm-up on the task's own inputs helps later fine-tuning.

The runner can already build a cross-dataset pretraining pool
(`pretrain_datasets`, [`runner.py:345-368`](../../../eventfm/benchmark/runner.py#L345-L368)),
but `Pragma.pretrain` ignores it. See
[B4](06_implementation_issues.md#b4-pragmapretrain-ignores-the-pretraining-pool-and-epoch-overrides).

## D2. Event sources and schemas

**Kind:** Scale · **Impact:** High

**Paper:** Events come from several source types (transactions, in-app
navigation, trading and communications), and each source has its own key set
(for example, `Symbol` only appears on trading events). There are about 60 keys
in total. This heterogeneity is the stated reason for the key–value design.

**Repo:** Each dataset has one source and one fixed schema. Every event carries
the same fields (from `meta.json` of the prepared datasets):

| Dataset | Event fields (besides `event_type`) | Profile fields |
| --- | --- | --- |
| BankSim | `amount`, `merchant` | `age`, `gender` |
| PaySim | `amount`, `balance_delta`, `counterparty_kind`, `dest_balance` | – |
| IBM AML | `amount_paid`, `fx_ratio`, `payment_currency`, `same_bank` | – |
| MBD-mini / MBD | `amount`, `currency`, `dst_type11`, `event_subtype`, `src_type11` | – |
| Synthea | `duration_days` | `gender`, `marital`, `race` |
| Amazon Beauty | `brand`, `price`, `rating`, `rating_value` | – |

Because every event has the same keys, the variable-schema problem the
architecture was built for hardly arises. A flat per-event feature vector
would carry the same information.

## D3. Profile state

**Kind:** Missing · **Impact:** High

**Paper (§2.1.2):** Profile state is a set of key–value pairs (plan, balance
quantile, insurance state, service region, …) timestamped at the evaluation
point (or at the cut-off date during pretraining). It is truncated to ≤200
tokens and encoded by its own Profile State Encoder. The profile ablation
(Table 6) is worth up to +31.8 % PR-AUC on credit scoring and +85.6 % recall
on fraud.

**Repo:**

* Only two adapters produce any profile at all: BankSim (`age`, `gender`) and
  Synthea (`gender`, `marital`, `race`).
* The value is taken from the **first row** of the entity's history
  ([`tabular.py:425`](../../../eventfm/datasets/tabular.py#L425)), not at an
  evaluation point.
* **Profile values never reach the model.** The tokenizer adds them to the
  vocabulary ([`tokenizer.py:146-148`](../../../eventfm/data/tokenizer.py#L146-L148)),
  but `PragmaDataCollator` never emits `profile_key_ids`,
  `profile_value_ids` or `profile_attention_mask`
  ([`collator.py:85-95`](../../../eventfm/data/collator.py#L85-L95)). The
  backbone then falls back to a bare `[USR]` embedding. See
  [A3](03_architecture.md#a3-profile-state-encoder) and
  [B1](06_implementation_issues.md#b1-profile-data-never-reaches-the-model).

## D4. Life-long events

**Kind:** Missing · **Impact:** Medium

**Paper:** The profile is extended with first-occurrence milestones
(`Lifelong: first_topup at 20-11-02 12:09:04`). Each carries its own
timestamp, and the time from that timestamp to the evaluation point is fed to
the profile encoder through RoPE. This recovers information such as account age
that history truncation throws away.

**Repo:** Not implemented. Histories are cut to the most recent 64 events
(see [P10](04_pretraining_and_infrastructure.md#p10-truncation)), and nothing
replaces the milestones that fall outside that window.

## D5. Record and evaluation point

**Kind:** Substitution · **Impact:** Low

**Paper:** A record is a pseudonymised history up to a designated evaluation
point, together with the profile state at that point.

**Repo:** There is no explicit evaluation timestamp.

* For derived labels, the last 25 % of each history (at least one event) is
  held out as the label window, and the model sees the remaining prefix
  ([`tabular.py:58`](../../../eventfm/datasets/tabular.py#L58), `build_sequences`).
* For MBD, the benchmark's own target is used, and history after a fixed
  cutoff is dropped.

In both setups, history time is measured to the **last observed event**,
which matches the paper's `t_e`. The paper also encodes the gap between that
event and the evaluation point, through life-long-event times. The repo loses
that gap completely.

## D6. Time range and timestamp quality

**Kind:** Scale · **Impact:** Medium (through the calendar features)

**Paper:** A deliberately chosen 25-month window (2023–2025) of real event
timestamps.

**Repo:** Each dataset keeps its native span. Several sources have only coarse
time, and the adapters spread events evenly inside each period with
`_within_period_offset`
([`adapters.py:23`](../../../eventfm/datasets/adapters.py#L23)):

| Dataset | Native resolution | Effect on calendar features |
| --- | --- | --- |
| BankSim | day ([L57](../../../eventfm/datasets/adapters.py#L57)) | hour of day is synthetic |
| PaySim | hour ([L90](../../../eventfm/datasets/adapters.py#L90)) | minutes are synthetic |
| Amazon Beauty | day ([L502](../../../eventfm/datasets/adapters.py#L502)) | hour of day is synthetic |
| Synthea | day ([L601](../../../eventfm/datasets/adapters.py#L601)) | hour of day is synthetic |
| IBM AML, MBD | real timestamps | as in the paper |

On four of the seven datasets, the hour-of-day part of PRAGMA's calendar
embedding therefore carries no real signal.

## D7. Filtering and pre-processing

**Kind:** Substitution · **Impact:** Medium

**Paper (§2.1.1, §2.4):** "no additional statistical filtering or
pre-processing, such as outlier removal or vocabulary pruning". Users with zero
events are dropped, and users with more than 6 500 events keep their most
recent ones.

**Repo:** Several filters change the data distribution:

| Filter | Where | Effect |
| --- | --- | --- |
| Minimum 8 events per entity | [`tabular.py:300`](../../../eventfm/datasets/tabular.py#L300) | Short histories are dropped (paper keeps any history with ≥1 event) |
| At most 64 to 128 events per entity at conversion | [`tabular.py:303`](../../../eventfm/datasets/tabular.py#L303), per dataset in [`registry.py`](../../../eventfm/datasets/registry.py) | Further cut to 64 at training time |
| **Vocabulary pruning**: top 64 levels per categorical field, the rest mapped to `other` | [`tabular.py:112-136`](../../../eventfm/datasets/tabular.py#L112-L136) | The paper explicitly avoids this. Event types are capped the same way (Synthea: 65 types → 64 + `other`) |
| Training negatives downsampled to a 20 to 25 % positive rate | [`tabular.py:350`](../../../eventfm/datasets/tabular.py#L350) | Evaluation keeps natural prevalence. Pretraining for `pragma-mlm` also runs on the rebalanced subset |
| Entity caps (`max_entities`, `max_eval_entities`) | [`registry.py`](../../../eventfm/datasets/registry.py) | 40 000 on several datasets |

## D8. Splits

**Kind:** Scale · **Impact:** Low

**Paper:** Fixed folds and splits per downstream task.

**Repo:** A hash-based entity split (65/15/20) or a chronological variant
(`*_chrono`), with nested training subsamples of 64 sequences up to the whole
pool, across several seeds.
