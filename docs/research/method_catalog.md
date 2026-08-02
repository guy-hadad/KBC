# Method catalog and implementation priorities

This catalog merges the repository's current 14 registry entries, the proposed
banking baseline suite, and all temporal-tokenization variants from Liu et al.
(GEM 2026). It prevents two common paper errors: comparing methods that solve
different problems as if they were interchangeable, and calling a controlled
small implementation a faithful reproduction.

## Current runnable benchmark

These names are already registered. "Approximation" means the method is useful
for controlled screening but paper tables must describe the actual code rather
than inherit every claim made by the cited system.

| Registry key | Family | Current status | Paper-facing interpretation |
| --- | --- | --- | --- |
| `count-logistic` | Engineered control | Implemented | Classification-only bag-of-marks and interval summaries with logistic regression. Add LightGBM/XGBoost for a stronger production floor. |
| `markov` | Marked temporal control | Implemented | TPP-only first-order mark transition and per-mark mean log gap. |
| `ntpp-gru` | Recurrent temporal | Approximation | GRU history plus a log-normal-mixture next-gap head. It is not the continuous-time LSTM/intensity formulation of NHP. |
| `thp` | Attention temporal | Approximation | Causal attention with Fourier time features and a shared intensity-free decoder. It is not a faithful THP/SAHP/AttNHP intensity implementation. |
| `pragma` | Hierarchical encoder | Approximation | Supervised PRAGMA-style hierarchy without foundation-scale pretraining. |
| `pragma-mlm` | Hierarchical MLM | Approximation | Masked-value pretraining on the downstream training split, then task training. |
| `tabbert` | Bidirectional tabular Transformer | Approximation | Field-pooled events with masked-field pretraining. |
| `tabgpt` | Causal tabular Transformer | Approximation | Field-pooled events with a causal decoder. |
| `nvidia-tfm` | Tabular decoder | Approximation | Controlled implementation of the public blueprint, not an industrial checkpoint reproduction. |
| `coles` | Contrastive | Approximation | Same-entity subsequence contrastive pretraining followed by a task head. Audit augmentations and loss against the reference recipe before claiming reproduction. |
| `mambular` | State space | Approximation | Selective state-space history mixer under the shared EventFM event contract. |
| `tpp-llm` | LLM + continuous time | Approximation | Text history, continuous Fourier side channel, LoRA, and shared probabilistic time head. |
| `language-tpp` | LLM + byte time | Approximation | Byte-tokenized `log1p` gaps as input, but time is decoded by the shared probabilistic head rather than autoregressive byte generation. |
| `mm-tpp` | Long-context LLM | Approximation | Byte-time input plus repository-specific temporal-similarity run compression. |

The 624-cell count in the current design is consistent with task support:
classification has `count-logistic` but not `markov`; TPP has `markov` but not
`count-logistic`; the other 12 entries support both tasks. Thus
`4 datasets × 6 sizes × (13 + 13 task-method pairs) = 624`.

## Paper-ready baseline suite

Priority is based on scientific coverage, not chronological novelty.

### Priority 0: controls needed to interpret any result

| Method | Status | Required contract | Why it is indispensable |
| --- | --- | --- | --- |
| Engineered features + LightGBM/XGBoost | Specified | Counts, amount moments/quantiles, recency, frequency, periodicity, merchant/category diversity, and balance features; tune only on validation | A foundation representation should beat or materially simplify a strong tabular production baseline. |
| Supervised GRU/LSTM | Specified | Same event encoder and time inputs as the proposed model, trained only on each downstream label | Isolates the value of pretraining from ordinary supervised sequence learning. |
| Transaction autoencoder | Specified | Categorical CE plus normalized numerical reconstruction; same encoder width/budget | Basic reconstruction control distinct from masked modeling. |
| Transaction MLM | Partly covered | Mask event type and fields with field-appropriate losses | Generic bidirectional representation baseline; compare with `tabbert` and `pragma-mlm`. |
| Autoregressive next-event Transformer/RNN | Partly covered | Predict next type, numerical fields, and gap from causal prefixes | Clean local-state/generative baseline; compare with `tabgpt`. |

### Priority 1: representation-learning objectives

| Method | Status | Distinctive mechanism | Primary evidence level |
| --- | --- | --- | --- |
| [CoLES](https://arxiv.org/abs/2002.08232) | Approximation implemented | Same-customer subsequences are positives; different customers are negatives | Customer/global |
| [NPPR / Foundation Purchasing Model](https://arxiv.org/abs/2401.01641) | Specified | Next-event prediction plus reconstruction of past transactions | Event and prefix |
| [MLEM](https://arxiv.org/abs/2401.15935) | Specified | Generative and contrastive representations treated as complementary modalities | Local/global transfer |
| [CMLM + CoLES](https://arxiv.org/abs/2408.09995) | Specified | Joint masked-event and subsequence-contrastive training | Local/global trade-off |
| [PRAGMA](https://arxiv.org/abs/2604.08649) | Approximation implemented | Key/value field encoder, event/history hierarchy, temporal coordinates and masked fields | Event and customer |
| [SOHET](https://arxiv.org/abs/2606.21356) | Specified | Event-type-specific FT-Transformer encoders; next type, next gap, and future-feature representation | Event, prefix, customer |

NPPR, MLEM, joint CMLM+CoLES, and SOHET should be independent entries. They
cannot be represented by renaming the existing TabGPT, CoLES, or PRAGMA runs;
their objectives and transfer claims are the point of comparison.

### Priority 1: temporal and time-to-event baselines

| Method | Status | Required distinction | Primary evidence |
| --- | --- | --- | --- |
| NHP | Specified | Continuous-time LSTM state and point-process likelihood | Recurrent intensity baseline |
| THP or SAHP | Specified | Transformer history with the paper's temporal encoding and intensity | Attention intensity baseline |
| AttNHP | Specified | Attentive neural Hawkes process under a faithful likelihood | Strong attention TPP baseline |
| IFTPP | Specified | Direct conditional inter-event-time distribution without classical intensity parameterization | Intensity-free control |
| [COTIC](https://arxiv.org/abs/2302.06247) | Specified | Continuous-time convolution plus event-type and return-time heads | Non-recurrent/non-attention temporal baseline |
| [TPP-LLM](https://arxiv.org/abs/2410.02062) | Approximation implemented | Text semantics plus continuous time and specialized temporal prediction | LLM/TPP hybrid |
| [Language-TPP](https://arxiv.org/abs/2502.07139) | Approximation implemented | Reversible byte tokens for continuous intervals with autoregressive decoding in the faithful version | Pure-token LLM |
| [MOTOR](https://arxiv.org/abs/2301.03150) | Specified | Censoring-aware, code-specific time-to-event pretraining | Long-horizon outcomes |
| [ORA / marked TTE](https://arxiv.org/abs/2602.00541) | Specified | Joint future-event time and numerical value distribution | Banking time/amount coupling |

At minimum, a headline temporal comparison needs NHP, AttNHP, IFTPP, COTIC,
and an ORA-style marked-TTE objective in addition to the current simplified GRU
and attention entries. EasyTPP or another verified reference implementation is
preferable to independent reimplementations for intensity-based methods.

### Priority 1: GEM 2026 time tokenizers

All of the following are **Specified** and share one pure causal-LM model:

- six-decimal numeric string;
- float32 byte tokens;
- absolute calendar at day, hour, minute, and second resolution;
- relative calendar at day, hour, minute, and second resolution;
- 256-bin uniform quantization in linear and log spaces;
- residual scalar quantization in linear and log spaces with one, two, three,
  and four levels.

The exact configuration keys and algorithms are in
[`temporal_tokenization.md`](temporal_tokenization.md). They should be reported
as tokenizer conditions, not counted as unrelated foundation models.

## Banking systems for architecture comparison and related work

These systems are important to the narrative. Reproduce them only when the
essential ingredients and data assumptions can be matched; otherwise compare
architectural decisions and label them **Related work**.

| System | Status | Relevance to EventFM |
| --- | --- | --- |
| [TransactionGPT](https://arxiv.org/abs/2511.08939) | Related work / future reproduction | 3D attention over within-transaction, temporal, and modality interactions; causal generation and representation learning. |
| [TREASURE](https://arxiv.org/abs/2511.19693) | Related work / future reproduction | Static/dynamic separation and causal prediction of next fields plus network outcomes. |
| [nuFormer](https://arxiv.org/abs/2507.23267) | Related work / feasible fusion baseline | Tests replacement of engineered features versus fusion with learned transaction representations. |
| [Multimodal financial-event foundation model](https://arxiv.org/abs/2607.09955) | Related work | Early fusion of transactions, digital interactions, and communications, followed by frozen embeddings plus engineered features. |
| [Open Banking Foundational Model](https://arxiv.org/abs/2511.12154) | Related work | Masked modeling of structured attributes and descriptions across institutions; relevant to schema and bank transfer. |
| [LATTE](https://aclanthology.org/2025.emnlp-industry.179/) | Specified if text exists | Contrastive alignment of compact transaction histories with frozen LLM embeddings of behavior summaries. |
| [TabBERT/TabGPT](https://arxiv.org/abs/2011.01843) | Approximation implemented | Generic hierarchical structured-sequence baselines. |
| UniTTab | Related work / future reproduction | Heterogeneous time-dependent tabular records and schema-aware hierarchy. |
| NVIDIA TFM blueprint | Approximation implemented | Public tabular-tokenization/decoder blueprint; disclose deviations and checkpoint availability. |

Publication status and final bibliographic metadata for recent preprints must be
rechecked before submission. A related-work mention is not evidence of
reproduction quality.

## Candidate EventFM contribution

The highest-value new model to test is a controlled composition, not another
undifferentiated architecture:

\[
h_i=\operatorname{FieldEncoder}_{k_i}(x_i),\qquad
z_{1:n}=\operatorname{ContinuousTimeHistoryEncoder}(h_{1:n},t_{1:n}),
\]

with

\[
\mathcal L =
\lambda_{\mathrm{mask}}\mathcal L_{\mathrm{masked\ fields}}+
\lambda_{\mathrm{next}}\mathcal L_{\mathrm{next\ event}}+
\lambda_{\mathrm{tte}}\mathcal L_{\mathrm{marked\ TTE}}+
\lambda_{\mathrm{contrast}}\mathcal L_{\mathrm{subsequence\ contrast}}.
\]

Call this a **candidate contribution** until ablations demonstrate that the
combination improves transfer rather than merely adding losses and compute.
Required ablations are: each individual loss, cumulative additions in a
pre-registered order, equal-update and equal-FLOP controls, and gradient-conflict
diagnostics between objectives.

## Implementation order

1. Add LightGBM and supervised GRU controls; correct evaluation prevalence and
   seed reporting.
2. Make frozen/PEFT/full adaptation and event/prefix/customer probes first-class
   benchmark dimensions.
3. Implement the pure generative temporal-tokenizer interface and all GEM
   variants.
4. Add faithful NHP, AttNHP, IFTPP, and COTIC reference adapters.
5. Add NPPR, joint CMLM+CoLES, and SOHET objectives under shared encoders.
6. Add marked-TTE time/value modeling and then test the candidate combined
   objective.
7. Treat expensive industrial systems as follow-up reproductions only after the
   central factorial study is statistically complete.

