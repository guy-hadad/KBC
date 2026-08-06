# Method catalog and implementation priorities

This catalog organizes the runnable banking baselines and all temporal-
tokenization variants from Liu et al. (GEM 2026). The live, exhaustive registry
table is generated in [`implementation_manifest.md`](implementation_manifest.md).
This document explains why the families matter and which external recipes still
need fidelity validation.

## Current runnable benchmark

The registry now contains 62 runnable keys: controlled classical/supervised
baselines, the original architecture screen, representation-learning and
time-to-event objectives, temporal models, 11 continuous-time ablations, and
20 pure-generative GEM tokenizer conditions. There are 18 implemented controls
and 44 disclosed approximations. The approximation label is mandatory for any
paper-named entry whose corpus, scale, architecture, likelihood, or recipe is
not faithful.

The 624-cell count in the current design is consistent with task support:
classification has `count-logistic` but not `markov`; TPP has `markov` but not
`count-logistic`; the other 12 entries support both tasks. Thus
`4 datasets × 6 sizes × (13 + 13 task-method pairs) = 624`.

## Paper-ready baseline suite

Priority is based on scientific coverage, not chronological novelty.

### Priority 0: controls needed to interpret any result

| Method | Status | Required contract | Why it is indispensable |
| --- | --- | --- | --- |
| Engineered features + histogram GBDT | Implemented control | Counts, amount moments, recency, frequency, periodicity, category diversity, and interval features; tune only on validation | A foundation representation should beat or materially simplify a strong tabular production baseline. |
| Supervised GRU | Implemented control | Same event encoder and time inputs as the proposed model, trained only on each downstream label | Isolates the value of pretraining from ordinary supervised sequence learning. |
| Transaction autoencoder | Implemented control | Masked mark/field reconstruction under the shared encoder width/budget | Basic reconstruction control distinct from masked modeling. |
| Transaction MLM | Implemented control | Bidirectional masked event and field reconstruction | Generic representation baseline; compare with `tabbert` and `pragma-mlm`. |
| Autoregressive next-event Transformer | Implemented control | Predict next type, fields, and gap from causal prefixes | Clean local-state/generative baseline; compare with `tabgpt`. |

### Priority 1: representation-learning objectives

| Method | Status | Distinctive mechanism | Primary evidence level |
| --- | --- | --- | --- |
| [CoLES](https://arxiv.org/abs/2002.08232) | Approximation implemented | Same-customer subsequences are positives; different customers are negatives | Customer/global |
| [NPPR / Foundation Purchasing Model](https://arxiv.org/abs/2401.01641) | Approximation runnable | Next-event prediction plus reconstruction of past transactions | Event and prefix |
| [MLEM](https://arxiv.org/abs/2401.15935) | Approximation runnable | Generative and contrastive representations treated as complementary modalities | Local/global transfer |
| [CMLM + CoLES](https://arxiv.org/abs/2408.09995) | Approximation runnable | Sequential masked-event and subsequence-contrastive training | Local/global trade-off |
| [PRAGMA](https://arxiv.org/abs/2604.08649) | Approximation implemented | Key/value field encoder, event/history hierarchy, temporal coordinates and masked fields | Event and customer |
| [SOHET](https://arxiv.org/abs/2606.21356) | Approximation runnable | Type-conditioned feature gating; next type, next gap, and future-feature representation | Event, prefix, customer |

NPPR, MLEM, joint CMLM+CoLES, and SOHET are independent registry entries. They
are not aliases for TabGPT, CoLES, or PRAGMA; their objectives and transfer
claims are the point of comparison.

### Priority 1: temporal and time-to-event baselines

| Method | Status | Required distinction | Primary evidence |
| --- | --- | --- | --- |
| NHP | Approximation runnable | Decayed continuous-time GRU plus exponential likelihood; not the paper CT-LSTM | Recurrent intensity control |
| THP/SAHP | Approximation runnable | Causal attention and controlled temporal/intensity heads | Attention intensity control |
| AttNHP | Approximation runnable | Causal attentive history with the shared exponential head | Strong attention TPP control |
| IFTPP | Approximation runnable | Direct log-normal-mixture conditional inter-event distribution | Intensity-free control |
| [COTIC](https://arxiv.org/abs/2302.06247) | Approximation runnable | Dilated continuous-time convolution plus shared event/time heads | Non-recurrent/non-attention temporal control |
| [TPP-LLM](https://arxiv.org/abs/2410.02062) | Approximation implemented | Text semantics plus continuous time and specialized temporal prediction | LLM/TPP hybrid |
| [Language-TPP](https://arxiv.org/abs/2502.07139) | Approximation implemented | Reversible byte tokens for continuous intervals with autoregressive decoding in the faithful version | Pure-token LLM |
| [MOTOR](https://arxiv.org/abs/2301.03150) | Approximation runnable | Censoring-aware, code-specific exponential time-to-event pretraining | Long-horizon outcomes |
| [ORA / marked TTE](https://arxiv.org/abs/2602.00541) | Approximation runnable | Joint future-event time and bucketed value conditioning | Banking time/amount coupling |

The headline temporal suite includes NHP, AttNHP, IFTPP, COTIC, and an
ORA-style marked-TTE objective in addition to the original simplified GRU and
attention entries. EasyTPP or another verified reference implementation remains
preferable for a paper-faithful intensity-based reproduction.

### Priority 1: GEM 2026 time tokenizers

All of the following are **runnable approximations** sharing one pure causal-LM
generation interface. The algorithms are implemented; the default small model
and budget differ from the paper's Llama 3.2 QLoRA recipe unless overridden:

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

## Fidelity and execution priorities

1. Validate the approximation implementations against small reference fixtures
   and published trends; promote fidelity status only with evidence.
2. Add verified reference-library adapters for NHP, AttNHP, IFTPP, and COTIC
   alongside, rather than silently replacing, the controlled implementations.
3. Complete the five-seed headline, objective-factorial, GEM, chronological,
   and cross-schema suites under declared compute budgets.
4. Add event-level numerical/retrieval labels only where an open dataset offers
   a defensible target; do not manufacture tasks to fill a table.
5. Treat expensive industrial systems as related work or follow-up
   reproductions until their essential data and recipe assumptions can be met.
