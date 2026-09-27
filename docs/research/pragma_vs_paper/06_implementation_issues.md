# 06 — Implementation issues found during the comparison

[← index](README.md)

These are defects or inconsistencies **inside the repo**. They matter whether
or not the goal is paper fidelity. Each one has the evidence and a proposed
fix.

## B1. Profile data never reaches the model

**Severity:** High · **Related:** [D3](01_data.md#d3-profile-state), [A3](03_architecture.md#a3-profile-state-encoder)

**Evidence**

* `PragmaDataCollator` builds `key_ids`, `value_ids`, `feature_*`,
  `event_*`, `timestamps`, `delta_seconds` and `calendar_features`, plus the
  labels, but **no `profile_*` tensors**
  ([`collator.py:85-95`](../../../eventfm/data/collator.py#L85-L95)).
  `grep -rn profile_key_ids` finds no other producer anywhere in the repo.
* `PragmaBackbone._encode_profile` therefore always returns the bare `[USR]`
  embedding ([`modeling_pragma.py:147-148`](../../../eventfm/models/modeling_pragma.py#L147-L148)).
* A forward/backward check on a batch whose sequences have `profile={"age",
  "gender"}` gave **0 of 16** profile-encoder parameter tensors with a
  gradient.
* The profile keys and values are still added to the vocabulary
  ([`tokenizer.py:146-148`](../../../eventfm/data/tokenizer.py#L146-L148)), so
  they occupy embedding rows that are never trained.

**Consequences**

* Every `pragma` and `pragma-mlm` result is an event-only model. On BankSim
  and Synthea this throws away available demographic information.
* `num_parameters` and `num_trainable_parameters` in the result JSON count
  params with `requires_grad` ([`base.py`](../../../eventfm/methods/base.py),
  `TorchMethod.num_parameters`). They therefore include the **132 480 unused
  profile-encoder parameters**, about 14 to 18 % of the benchmark model. Any
  parameter-matched comparison is skewed by this amount.
* The flat methods ignore `profile` too, so the comparison is still fair
  across methods. What is lost is PRAGMA's distinctive second branch.

**Fix**

1. Add `encode_profile(sequence)` to `EventTokenizer`, returning key/value ids
   and a mask, capped at `max_profile_features`.
2. Pad these in `PragmaDataCollator` and emit `profile_key_ids`,
   `profile_value_ids` and `profile_attention_mask`.
3. For fidelity, add a per-token `profile_age_seconds` input (0 for ordinary
   pairs) and apply RoPE in the profile encoder.
4. Add a test asserting that the profile encoder receives gradients.

## B2. Fourier time frequencies do not match the log-hour input

**Severity:** Medium · **Related:** [A5](03_architecture.md#a5-time-encoding-in-the-history-encoder)

`FourierTimeEncoding` computes `h = log1p(seconds / 3600)`, which lies in
[0, ≈10.5] for up to 4 years. It then divides `h` by 16 frequencies spaced
logarithmically over **[1, 8760]**
([`modeling_pragma.py:71`](../../../eventfm/models/modeling_pragma.py#L71),
[`L80-82`](../../../eventfm/models/modeling_pragma.py#L80-L82)). The value 8760
is the number of hours in a year, which suggests the grid was designed for
raw hours, not log-hours. As a result:

* 12 of the 16 frequencies keep `h / f < 1` radian over the whole range;
* 9 of them keep it below 0.2 radians, where `sin(x) ≈ x` and `cos(x) ≈ 1`.

The MLP still receives `h` itself plus a few useful low-frequency terms, so
the encoder works as a smooth function of log-age. Most of the periodic basis
is wasted, though. The docstring describes it as a "smooth representation of
event age", but that holds mainly because of the raw `h` term.

**Fix:** For paper fidelity, replace this with RoPE over `8·ln(1+t/8)` (see
[A5](03_architecture.md#a5-time-encoding-in-the-history-encoder)). If the
additive encoding is kept as an ablation, space the frequencies over the actual
range of `h`, for example `linspace(0.1, 10)`.

## B3. The `peft` regime crashes for PRAGMA

**Severity:** Medium (latent) · **Related:** [E1](05_adaptation_and_evaluation.md#e1-adaptation-mode)

`TorchMethod.run` requires `backbone.adapter` when `adaptation == "peft"`
([`base.py:254-259`](../../../eventfm/methods/base.py#L254-L259)).
`PragmaBackbone` defines no `adapter`, so any PRAGMA cell with
`regime="peft"` raises `ValueError`. Nothing fails today only because no suite
runs PRAGMA under that regime.

**Fix:** Add LoRA through `peft` (`LoraConfig(r=8, lora_alpha=8,
target_modules=["query", "key", "value", "intermediate.dense",
"output.dense"])` on the three `BertEncoder` stacks), exposed either as its own
`lora` regime or as PRAGMA's `peft`. Then add `pragma-mlm` to
`adaptation-regimes`.

## B4. `Pragma.pretrain` ignores the pretraining pool and epoch overrides

**Severity:** Medium (latent) · **Related:** [D1](01_data.md#d1-pre-training-corpus), [P1](04_pretraining_and_infrastructure.md#p1-pre-training-budget)

`_ObjectivePretrainedMethod.pretrain` honours `extra["pretrain_path"]`,
`extra["pretrain_epochs"]` and `extra["pretrain_max_steps"]`
([`representation_learning.py`](../../../eventfm/methods/representation_learning.py)).
`Pragma.pretrain` does not. It always trains on `context.train_path`
([`pragma.py:99`](../../../eventfm/methods/pragma.py#L99)) for the fixed
`pretrain_epochs = 3.0` ([`pragma.py:94`](../../../eventfm/methods/pragma.py#L94),
[`L111`](../../../eventfm/methods/pragma.py#L111)).

If `pragma-mlm` were added to `cross-schema-transfer`,
`labeled-adaptation-scaling` or `unlabeled-pretraining-scaling`, it would
**silently pretrain in-domain on the labelled subset**. The result file would
still record `pretrain_datasets_resolved`, making it look like transfer.

**Fix:** Copy the `extra` handling from `_ObjectivePretrainedMethod.pretrain`.
The vocabulary side already works, because `build_context` adds the pool to
the vocabulary.

## B5. Label smoothing is not wired into the benchmark path

**Severity:** Low · **Related:** [P3](04_pretraining_and_infrastructure.md#p3-loss)

`Pragma.build_config` ([`pragma.py:40-63`](../../../eventfm/methods/pragma.py#L40-L63))
does not pass `label_smoothing`, so benchmark MLM pretraining uses 0.0. The
Hydra path uses 0.05. The two entry points therefore train different
objectives under the same name.

**Fix:** Set `label_smoothing=float(context.extra.get("pragma_label_smoothing", 0.05))`
in `build_config`. It only affects `PragmaForMaskedEventModeling`.

## B6. PRAGMA's TPP head differs from the shared TPP decoder

**Severity:** Medium (for TPP comparisons) · **Related:** [E6](05_adaptation_and_evaluation.md#e6-next-event-tpp-task-not-in-the-paper)

[`docs/benchmark_design.md`](../../benchmark_design.md) (§1) says:

> Every TPP model ends in the **same two heads**: a mark classifier and a
> log-normal mixture over `log1p(Δt)`.

`PragmaForNextEventPrediction` does not follow this. It uses a single linear
regression with loss `0.5·(pred − target)²`
([`modeling_pragma.py:405-410`](../../../eventfm/models/modeling_pragma.py#L405-L410)).
Two consequences:

* PRAGMA's Δt metrics come from a different decoder than every other method's.
  The gap between them is partly a head difference, not an encoder difference.
* The `time_nll` it reports is a unit-variance squared error without the
  normalising constant. It is **not comparable** with the mixture NLL that the
  other methods report under the same metric name.

**Fix:** Reuse the shared log-normal-mixture head from `modeling_flat.py` on
`last_event_embedding`. Otherwise, drop PRAGMA's `time_nll` from the tables and
say so in `benchmark_design.md`.

## B7. The registry's `divergence` strings understate the gap

**Severity:** Low (documentation)

The generated [`implementation_manifest.md`](../implementation_manifest.md)
takes its text from `MethodSpec.divergence`
([`pragma.py:114-141`](../../../eventfm/methods/pragma.py#L114-L141)):

* `pragma`: "supervised PRAGMA-style hierarchy without foundation-scale pretraining"
* `pragma-mlm`: "masked-value pretraining on the downstream training split, not a separate large unlabelled corpus"

Neither mentions the inactive profile branch, the missing RoPE, post-norm
layers, full fine-tuning instead of LoRA, or the ~1 M-parameter scale.

**Fix:** Add the high-impact items (A1, A3, A5, A7, E1) to both strings and
link to this folder.

## Suggested order of work

If the goal is to move `pragma-mlm` closer to the paper within this repo's
constraints, this order gives the most fidelity per unit of effort:

1. **B1**: wire up the profile. Low effort; restores the second branch.
2. **A1**: switch to pre-norm, either with `nn.TransformerEncoderLayer(norm_first=True)`
   or a small custom layer. This is also the natural place to add RoPE.
3. **A5 / B2**: RoPE over `8·ln(1+t/8)` in the history and profile encoders.
4. **B3 / E1**: LoRA adaptation, and the standard-scaled L-BFGS probe from
   **E2**. `count-logistic` already uses `StandardScaler` +
   `LogisticRegression` (whose default solver is L-BFGS), so it can serve as a
   template ([`classic.py:117-133`](../../../eventfm/methods/classic.py#L117-L133)).
5. **B4**: cross-dataset pretraining, so that `pragma-mlm` actually tests
   transfer.
6. **A2 / T2 / B5**: sinusoidal within-field positions, a zero bucket, and
   label smoothing.

A7, P1, D1, D2 and E5 are scale limits and stay open regardless.
