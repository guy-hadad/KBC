# Temporal tokenization study

This document translates [Liu et al., *Temporal Tokenization Strategies for
Event Sequence Modeling with Large Language Models* (GEM
2026)](https://aclanthology.org/2026.gem-main.5/) into an EventFM experiment
contract. The authors' [reference implementation](https://github.com/CapitalOne-Research/temporal-tokenization)
should be used to resolve implementation details.

The scientific unit of comparison is a **time tokenizer**, not a new history
backbone. Event text, prompt structure, causal-LM objective, base model,
adaptation budget, split, and decoding policy must be fixed while the time
representation changes.

## Shared event format and prediction task

For event sequence

\[
S = \{(t_1,k_1),\ldots,(t_N,k_N)\},
\]

the event type \(k_i\) uses the LLM's text tokenizer. A temporal tokenizer
encodes either absolute time \(t_i\) or relative interval
\(\Delta t_i=t_i-t_{i-1}\). Use the paper's type-before-time template:

```text
<|begin_of_event|><|type_prefix|>{type_tokens}
<|time_prefix|>{time_tokens}<|end_of_event|>
```

The causal LM generates the next event type and time tokens with ordinary
next-token cross-entropy. Report next-type accuracy and macro-F1, time RMSE and
MAE in a shared physical/log unit, invalid-decode rate, and tokens per time
value. Unlike TPP-LLM, the pure-token variants have no continuous temporal side
channel or specialized time head.

### Reported reference recipe

For a close reproduction, start with the paper's shared setup before adding an
EventFM compute-matched condition:

- Llama 3.2 1B with 4-bit quantization;
- QLoRA rank 16 and alpha 32 on all attention projections;
- newly added temporal-token embeddings explicitly trainable;
- five epochs, AdamW, learning rate `1e-3`, cosine schedule, and warmup ratio
  `0.1`;
- per-device batch size 4 and four gradient-accumulation steps (effective batch
  size 16);
- five independent seeds;
- TPP-LLM comparison using the same base model/QLoRA setup but ten epochs;
- malformed or partial time generations decoded to a zero interval for the
  reproduction metric.

The original experiments used A100 GPUs. EventFM should additionally report a
shared number of optimizer updates/tokens and measured compute, because five
versus ten epochs is a native-recipe comparison rather than a compute-matched
one. Preserve the zero-fallback result for reproduction, but also report invalid
decode rate and an error conditional on valid decodes so parser failures are
not hidden inside RMSE.

## Complete main-method grid

The paper reports the following configurations. EventFM should retain each as
an explicit configuration even when a smaller screening grid is used.

| Proposed key | Time input | Encoding and reconstruction | Tokens/time | Fit on training split |
| --- | --- | --- | ---: | --- |
| `time-numeric-p6` | Relative | Fixed six-decimal numeric string, passed through the native subword tokenizer | about 4 | No |
| `time-byte-f32` | Relative | IEEE-754 float32 split deterministically into four byte tokens; add 256 byte symbols | 4 | No |
| `time-calendar-abs-day` | Absolute | Year, month, day calendar tokens | 3 | Vocabulary/range only |
| `time-calendar-abs-second` | Absolute | Year through second calendar tokens | 6 | Vocabulary/range only |
| `time-calendar-rel-day` | Relative | Hierarchical interval components through day resolution | 3 | Vocabulary/range only |
| `time-calendar-rel-second` | Relative | Hierarchical interval components through second resolution | 6 | Vocabulary/range only |
| `time-bin-linear-k256` | Relative | 256 uniform bins between training minimum and maximum in linear space; decode bin centre | 1 | Min/max |
| `time-bin-log-k256` | Relative | Base-10 transform, 256 uniform bins, inverse-transform bin centre | 1 | Positive-value policy and min/max |
| `time-rsq-linear-l1-k256` | Relative | One K-means scalar codebook with 256 centroids | 1 | Codebook |
| `time-rsq-linear-l4-k64` | Relative | Four residual K-means codebooks, 64 centroids per level; decode by summing centroids | 4 | Four codebooks |
| `time-rsq-log-l1-k256` | Relative | One 256-centroid codebook after base-10 transform | 1 | Transform and codebook |
| `time-rsq-log-l4-k64` | Relative | Four 64-centroid residual codebooks after base-10 transform | 4 | Transform and codebooks |
| `tpp-llm-continuous` | Relative | Continuous temporal embedding with TPP-specific type/time heads and loss | n/a | Learned head |

The two calendar families must also support **hour** and **minute** resolutions
for the resolution ablation. The RSQ family must also support two levels
(`128-128`) and three levels (`85-85-86`) so every quantization-level ablation
from the paper is represented.

## Method definitions

### Numeric string

Format the chosen time value to exactly six digits after the decimal point and
let the base tokenizer segment it. It is a necessary naive baseline, but token
count depends on the base vocabulary and numerical magnitude. Persist both the
rendered value and produced token IDs in audit fixtures.

### Byte tokenization

Cast the value to float32 and encode its four bytes using 256 new vocabulary
items. Byte order must be fixed and tested with known values; decoding must be
bit-exact before float conversion. All new embeddings must be trainable during
PEFT. Report endianness and whether the raw or transformed interval is encoded.

### Absolute and relative calendar tokens

Absolute tokenization decomposes a timestamp into Gregorian components.
Relative tokenization decomposes an interval into hierarchical duration
components. Day, hour, minute, and second resolution are separate conditions.
Record timezone, daylight-saving policy, epoch, overflow behavior, and the
handling of intervals outside the available component vocabulary.

For banking data, absolute calendar tokens can capture salary, weekday, and
merchant-cycle effects. They also risk learning collection-period shortcuts;
the chronological generalization split in `experimental_protocol.md` is
therefore mandatory.

### Uniform scale bins

Transform training intervals in either linear or base-10 log space, divide the
training range into 256 equal-width bins, and add one token per bin. Decode to
the bin center and invert the transform. Fit boundaries on the training split
only. Pre-register clipping/overflow tokens for validation and test values
outside the training range, and define the zero-interval transform explicitly
(for example, `log10(1 + delta_seconds)`).

### Residual scalar quantization

Let \(v'=f(v)\). At level 1, choose the nearest centroid
\(c^{(1)}_{q_1}\), then quantize the residual recursively:

\[
r_1=v'-c^{(1)}_{q_1},\qquad
r_j=r_{j-1}-c^{(j)}_{q_j}.
\]

The encoded representation is \((q_1,\ldots,q_L)\), and decoding is

\[
\hat v=f^{-1}\!\left(\sum_{j=1}^{L}c^{(j)}_{q_j}\right).
\]

Fit every K-means level on training residuals only. Persist centroids, random
seed, initialization count, stopping tolerance, empty-cluster behavior, and
transform parameters. Evaluate linear and log spaces for `256`, `128-128`,
`85-85-86`, and `64-64-64-64`, holding the total new-token vocabulary near
256 while allowing the number of generated tokens to vary.

## Required ablations from the paper

| Axis | Conditions | Purpose |
| --- | --- | --- |
| Quantization levels | L1/256, L2/128, L3/85-85-86, L4/64 in linear and log spaces | Separate vocabulary budget from compositional precision. |
| Calendar resolution | day, hour, minute, second for absolute and relative calendars | Match representation granularity to event frequency. |
| Template order | type-time versus time-type | Measure semantic disruption from intervening time tokens. |
| Model size | one small and one larger backbone; the paper used Llama 3.2 1B and 3B | Test whether tokenizer ranking is capacity-dependent. |
| Data size | at least three nested pretraining sizes evaluated on one fixed test set | Test saturation and PEFT bottlenecks. |
| Time space | raw/linear versus a pre-registered log transform | Match smooth, heavy-tailed, spiky, and multimodal interval distributions. |

## EventFM integration and current gap

The existing `language-tpp` entry is an **approximation**, not an implementation
of this full study. It currently serializes `log1p(delta_seconds)` as four byte
tokens, uses a different compact prompt, pools the final hidden state, and
decodes time through the repository's shared log-normal-mixture head. The GEM
paper's pure-token models instead autoregressively generate and parse time
tokens. These conditions must remain separate in tables:

- `language-tpp-head`: current byte-token input plus specialized time head;
- `gem-token-*`: pure causal generation using the tokenizer variants above;
- `tpp-llm-continuous`: continuous-time embedding and TPP-specific head.

Do not silently replace one with another. A useful factorial follow-up can
cross selected tokenizers with both a generative decoder and a probabilistic
time head, but that is an EventFM experiment rather than a reproduction.

## Banking-specific screening protocol

1. Characterize each dataset's interval distribution on the training split:
   zero mass, quantiles, tail index, histogram entropy, modes in linear/log
   space, and hour/day/week periodicity.
2. Run all tokenizers on one small fixed backbone and three seeds. Keep the
   event template, context length in **events**, optimizer budget, and decoding
   policy fixed; also report token length and truncation rate.
3. Promote the best linear/log quantizer, the best calendar variant, byte
   tokens, numeric strings, and the continuous TPP-LLM control to five seeds on
   all datasets.
4. Repeat finalists with a second backbone size and under a chronological test
   split.
5. Test tokenizer-by-objective interactions only after the clean tokenizer
   comparison is complete.

The paper's core result is a hypothesis for EventFM, not a conclusion to copy:
log-space quantizers should help heavy-tailed banking gaps, while absolute
calendar tokens should help strong periodic behavior. The registered study must
allow either hypothesis to fail.

## Reproducibility tests

- round-trip tests for byte, calendar, bin, and RSQ encoders;
- train-only fitting test for every boundary, vocabulary, and codebook;
- deterministic tokenizer fit for a fixed seed;
- explicit test of zero, negative, NaN, infinite, and out-of-range values;
- malformed/partial generation rate and deterministic fallback policy;
- equal-event-context and equal-token-context results, since verbose
  tokenizers otherwise see shorter histories;
- saved prompt examples and decoded predictions for every condition.
