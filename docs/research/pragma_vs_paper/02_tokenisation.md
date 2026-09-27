# 02 — Tokenisation (paper §2.2)

[← index](README.md)

## T1. Keys

**Kind:** Scale · **Impact:** Low

**Paper:** Every semantic type is one token. Event keys and profile keys are
encoded the same way, giving a key vocabulary of about 60 tokens.

**Repo:** Every field is one `key:<field>` token
([`tokenizer.py:263`](../../../eventfm/data/tokenizer.py#L263)), plus
`key:event_type`. That gives 2 to 6 event keys per dataset. Profile keys are
added to the vocabulary but never used
([D3](01_data.md#d3-profile-state)). Keys and values share one namespace and
one embedding table, as in the paper.

## T2. Numerical values

**Kind:** Substitution · **Impact:** Low

**Paper:** Values are mapped to percentile buckets whose boundaries are learned
from training data, with **an extra bucket for zero**, and each bucket is one
token. The number of buckets is unspecified.

**Repo:**

* On the benchmark path, values are cut into 16 quantile buckets (`b_<idx>`),
  fitted only on events in the retained training histories
  ([`tabular.py:79-98`](../../../eventfm/datasets/tabular.py#L79-L98),
  [`tabular.py:402`](../../../eventfm/datasets/tabular.py#L402)). Duplicate edges
  are merged by `np.unique`, so heavily tied columns end up with fewer buckets
  (Synthea `duration_days`: 12, IBM AML `fx_ratio`: 2). Non-finite values
  become `na`.
* **There is no dedicated zero bucket.** Zeros fall into whichever quantile
  bucket the edges put them in. This matters for zero-heavy columns such as
  PaySim `balance_delta` and `dest_balance`.
* On the legacy Hydra/synthetic path, numbers are not bucketed at all.
  `_normalise_value` turns each float into a string with 6 significant digits
  ([`tokenizer.py:157-162`](../../../eventfm/data/tokenizer.py#L157-L162)), so
  every distinct value gets its own token.

## T3. Categorical vs textual values

**Kind:** Missing · **Impact:** Medium

**Paper:** String fields are split by a cardinality threshold. Low-cardinality
fields (plus hand-picked ones such as MCC) become one categorical token.
High-cardinality fields are treated as text and split with a BPE-style
subword tokenizer that has a reserved `[UNK]`. Values therefore take
**one or more** tokens, for example `Description → met, al, plan`.

**Repo:** There is no textual value path. Every string field is categorical:
the 64 most frequent training levels are kept and everything else becomes
`other`
([`tabular.py:112-136`](../../../eventfm/datasets/tabular.py#L112-L136)).
Consequences:

* High-cardinality fields are truncated to their top 64 levels rather than
  subword-tokenised. Amazon `brand` is cut to 64 levels plus `other`. BankSim
  `merchant` has 50 levels, so it happens to fit.
* Free text is either not loaded (Amazon review text and summaries) or used as
  the event *type* rather than as a field value (Synthea SNOMED descriptions).
* **Every field produces exactly one value token.** So multi-token fields,
  key replication for multi-valued fields (§2.3.1), and within-field positions
  0, 1, 2 never occur. This undercuts
  [A2](03_architecture.md#a2-within-field-positional-embedding).

The Nemotron text-encoder extension (§3.4.4) is not implemented either; see
[E8](05_adaptation_and_evaluation.md#e8-paper-experiments-not-reproduced).

## T4. Value vocabulary

**Kind:** Scale · **Impact:** Low

**Paper:** About 28 k value tokens in one vocabulary shared by pretraining and
all tasks.

**Repo:**

* The vocabulary is **rebuilt for every benchmark cell** from that cell's
  training subset plus the pinned event-type list
  ([`runner.py:280-292`](../../../eventfm/benchmark/runner.py#L280-L292)). The
  embedding table therefore changes size with the sample size, and values not
  seen in a small subset map to `[UNK]` at evaluation time
  ([`tokenizer.py:77-78`](../../../eventfm/data/tokenizer.py#L77-L78)).
* Value tokens are namespaced by field (`value:<field>:<value>`,
  [`tokenizer.py:264`](../../../eventfm/data/tokenizer.py#L264)), so `b_3` in
  `amount` and `b_3` in `price` are different tokens. The paper does not say
  whether bucket or categorical tokens are shared across fields. BPE subwords
  certainly are.

## T5. Elapsed-time transform

**Kind:** Substitution · **Impact:** Medium

| | Paper | This repo |
| --- | --- | --- |
| Quantity | Seconds from each event to the most recent event | Same: `event_age_seconds = t_last − t_i` ([`tokenizer.py:292`](../../../eventfm/data/tokenizer.py#L292)) |
| Transform | `8·ln(1 + t/8)`: linear below ~8 s, logarithmic above | `log1p(t / 3600)`: log-hours, linear below ~1 h ([`modeling_pragma.py:80`](../../../eventfm/models/modeling_pragma.py#L80)) |
| Range over 4 years | 0 to ~133 | 0 to ~10.5 |
| Used as | a RoPE coordinate | input to an additive Fourier-feature MLP ([A5](03_architecture.md#a5-time-encoding-in-the-history-encoder)) |

With the repo's transform, events a few seconds or minutes apart are almost
indistinguishable. The paper's transform keeps them apart. The inter-event gap
`delta_seconds` is computed and batched
([`tokenizer.py:289-291`](../../../eventfm/data/tokenizer.py#L289-L291)) but the
PRAGMA backbone discards it. The paper does not use gaps either.

## T6. Calendar features

**Kind:** Match (minor details) · **Impact:** Low

**Paper:** Hour of day, day of week and day of month, turned into sin/cos with
fixed calendar periods, embedded by a two-layer MLP, and applied to event
entries only.

**Repo:** The same six sin/cos features
([`tokenizer.py:165-180`](../../../eventfm/data/tokenizer.py#L165-L180)), with a
`Linear → GELU → Linear` MLP
([`modeling_pragma.py:123-127`](../../../eventfm/models/modeling_pragma.py#L123-L127)),
applied to events only. Small details:

* The timestamps are interpreted in **UTC** (`datetime.utcfromtimestamp`). The
  paper presumably uses local time; it is unspecified.
* Hour includes a minute fraction, and day of month uses a fixed period of 31.
* On BankSim, Amazon and Synthea, the hour of day is synthetic
  ([D6](01_data.md#d6-time-range-and-timestamp-quality)).
