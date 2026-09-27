# 03 — Architecture (paper §2.3–2.3.4, Table 1)

[← index](README.md)

## A1. Layer-normalisation placement

**Kind:** Substitution · **Impact:** Medium

**Paper:** All variants use **pre-norm** Transformer layers (Xiong et al., 2020).
§3.1.1 cites this as the reason the extracted embeddings must be standard-scaled
before probing.

**Repo:** All three encoders are Hugging Face `BertEncoder` stacks
([`modeling_pragma.py:37-52`](../../../eventfm/models/modeling_pragma.py#L37-L52),
[`L120-122`](../../../eventfm/models/modeling_pragma.py#L120-L122)). BERT layers
are **post-norm**: `BertSelfOutput` and `BertOutput` compute
`LayerNorm(x + Dropout(Dense(h)))`. This was checked against the installed
`transformers` 5.13.0. Further differences:

* A BERT-style embedding `LayerNorm` + dropout is applied to the summed event
  token embeddings ([`modeling_pragma.py:196`](../../../eventfm/models/modeling_pragma.py#L196)).
  It is not applied to profile tokens or to the history-encoder inputs.
* Since every block ends in a LayerNorm, the outputs are already normalised.
  The paper's reason for standard-scaling probe inputs does not apply in the
  same way here.
* `layer_norm_eps = 1e-12` (the BERT default) and `normal(0, 0.02)`
  initialisation. The paper does not specify either.

Post-norm stacks are harder to train at the paper's depths (45-layer event
encoder in PRAGMA-L). At the repo's 2-layer depth the difference is small.

## A2. Within-field positional embedding

**Kind:** Substitution · **Impact:** Low in practice, but the opposite of the paper's design

**Paper (Eq. 1):** `x = PosEmb(E(k) + E(v))`, where `PosEmb` is a **static
sine/cosine** encoding and "positions index values *within* a field, not
*across* fields". A single-valued field gets position 0. The three subword
tokens of a `Description` get positions 0, 1, 2.

**Repo:**

* The position embedding is a **learned** `nn.Embedding`
  ([`modeling_pragma.py:116-119`](../../../eventfm/models/modeling_pragma.py#L116-L119)).
* It is indexed by the **field's ordinal within the event**:
  `for pos, (field_name, value) in enumerate(pairs)`
  ([`tokenizer.py:262-267`](../../../eventfm/data/tokenizer.py#L262-L267)).
  With `event_type` first and the other fields sorted alphabetically, position
  *i* always goes with the same key. It is therefore a redundant second key
  embedding, not a within-field position.
* Under the paper's scheme every token in this repo would get position 0,
  since every field is single-valued
  ([T3](02_tokenisation.md#t3-categorical-vs-textual-values)).
* Profile tokens receive no position embedding.

## A3. Profile State Encoder

**Kind:** Missing / Bug · **Impact:** High

**Paper (§2.3.2):** A bidirectional Transformer over `[USR]` + profile tokens.
It has its own **RoPE over `t_a`**: the log-seconds since each life-long event,
or 0 for ordinary profile pairs. This RoPE is kept separate from the
within-field positions. The `[USR]` output becomes `z_a`.

**Repo:** `_encode_profile` exists
([`modeling_pragma.py:139-165`](../../../eventfm/models/modeling_pragma.py#L139-L165)),
but:

1. **It never runs on real inputs.** No collator emits `profile_*` tensors, so
   the method always takes the early-return branch at
   [`L147-148`](../../../eventfm/models/modeling_pragma.py#L147-L148) and returns
   the raw `[USR]` embedding row. `z_a` is therefore **one learned vector shared
   by every user**. A forward/backward check confirmed that 0 of the 16
   profile-encoder parameter tensors receive a gradient. Details in
   [B1](06_implementation_issues.md#b1-profile-data-never-reaches-the-model).
2. Even when given inputs, it has no time coordinate (no `t_a`, no RoPE), no
   position embedding, and no embedding LayerNorm.

The benchmarked model is therefore the paper's **event-only ablation**
(Table 6), plus 132 480 unused parameters.

## A4. Event Encoder

**Kind:** Match · **Impact:** Low

**Paper (§2.3.3):** A bidirectional Transformer applied to each event on its
own, with `[EVT]` prepended. Its token outputs `ẑ_e` feed the MLM head and its
`[EVT]` output is `z'_e`. The calendar MLP output is added: `z_e = z'_e + z_t`.

**Repo:** The same structure
([`modeling_pragma.py:197-218`](../../../eventfm/models/modeling_pragma.py#L197-L218)).
The differences are only in execution:

* Events are processed as a padded `(B·L, W+1)` batch instead of a packed
  varlen buffer ([P9](04_pretraining_and_infrastructure.md#p9-sequence-packing)).
* Padding *events* are still encoded, because their `[EVT]` token is unmasked.
  This wastes compute but does not change results, since the history encoder
  masks them out.
* `[USR]` and `[EVT]` are rows of the shared token table, which is equivalent
  to the paper's "learnable token".

## A5. Time encoding in the History Encoder

**Kind:** Substitution · **Impact:** High

**Paper (§2.3.4):** The input is `z = [z_a : z_e]` with coordinate `t_e`
(log-seconds to the most recent event, 0 at `z_a`), **encoded with RoPE**
inside attention. No other positional signal is mentioned.

**Repo** ([`modeling_pragma.py:227-235`](../../../eventfm/models/modeling_pragma.py#L227-L235)):

```text
history_hidden = [z_a : z_e] + FourierTimeEncoding([0 : age_seconds])
FourierTimeEncoding(s) = MLP([h, sin(h/f_k), cos(h/f_k)]_{k=1..16}),  h = log1p(s/3600)
```

| Aspect | Paper (RoPE) | Repo (additive Fourier MLP) |
| --- | --- | --- |
| Where time enters | Rotation of queries and keys in every attention layer | Added once to the input embeddings |
| What attention sees | Relative time differences between events | Absolute log-age mixed into content |
| `[USR]` at t = 0 | Identity rotation (no change) | A learned, non-zero bias vector is added |
| Scale | `8·ln(1+t/8)` | `log1p(t/3600)` ([T5](02_tokenisation.md#t5-elapsed-time-transform)) |

There is no RoPE anywhere in the PRAGMA code. The Fourier frequencies are also
badly matched to the log-hours input: 12 of the 16 frequencies never exceed
1 radian. See [B2](06_implementation_issues.md#b2-fourier-time-frequencies-do-not-match-the-log-hour-input).
`modeling_flat.py` has a `_continuous_rope` helper, but it rotates the *input
embeddings* rather than Q/K inside attention, so it is not a drop-in
replacement for the paper's RoPE either.

## A6. History Encoder inputs and outputs

**Kind:** Match

`z = [z_a : z_e]`. `z_h,0` is the `[USR]` output and `z_h,1..n` are the `[EVT]`
outputs
([`modeling_pragma.py:243-258`](../../../eventfm/models/modeling_pragma.py#L243-L258)).
The backbone also exposes the last real event, `last_event_embedding`, which
corresponds to the paper's "final [EVT]" probe token.

## A7. Model size and shape

**Kind:** Scale · **Impact:** High (for any scaling conclusion)

| | d_model | d_ffn | Profile / Event / History layers | Heads (head dim) | Params |
| --- | ---: | ---: | --- | --- | ---: |
| Paper PRAGMA-S | 192 | 768 | 1 / 5 / 2 | 3 (64) | 10 M |
| Paper PRAGMA-M | 512 | 2048 | 3 / 16 / 6 | 8 (64) | 100 M |
| Paper PRAGMA-L | 1024 | 4096 | 9 / 45 / 18 | 16 (64) | 1 B |
| **Repo benchmark** (`--hidden-size 128 --layers 2`) | 128 | 256 | 1 / 2 / 2 | 4 (32) | 0.73 M (vocab 200) to 0.96 M (vocab 2 000), of which 132 k are the unused profile encoder |
| Repo `pragma_s_paperish.yaml` (Hydra only) | 192 | 768 | 1 / 5 / 2 | 3 (64) | ≈9.0 M with a 28 k vocab |

Notes:

* The benchmark derives depths from one `num_hidden_layers` value:
  event = `layers // 2 + 1`, history = `layers`, profile = 1
  ([`pragma.py:50-51`](../../../eventfm/methods/pragma.py#L50-L51)). The paper
  makes the **event encoder the deepest block** (a ratio of about 2.5×
  history). The repo makes it the same depth as history.
* The FFN ratio is 2× (`intermediate_size = hidden_size * 2`,
  [`runner.py:404`](../../../eventfm/benchmark/runner.py#L404)). The paper
  uses 4×.
* The only PRAGMA-S-shaped config lives on the legacy Hydra path
  ([`configs/model/pragma_s_paperish.yaml`](../../../configs/model/pragma_s_paperish.yaml)).
  No M or L variant exists, and PRAGMA is not part of the
  `model-scale-ablation` suite
  ([`experiments.py:303-317`](../../../eventfm/benchmark/experiments.py#L303-L317)).
* GELU activations and dropout 0.1 (applied to both attention and hidden
  states) match the paper.
