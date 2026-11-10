# ETAM Stage 2: A unified continuous-time and event foundation model



# 4. Proposed Stage 2: ETAM-2 — Dual-Process Predictive Event Modeling

My proposed architecture has four innovations to investigate, but I would implement them incrementally rather than introducing all four simultaneously.

### Proposed architecture

Observed event history

Event types, attributes, timestamps, elapsed silence

Event tokenizer

Marks, attributes, learned semantic factors

Time tokenizer

Calendar, log-gap, age, multiscale temporal state

Event-private mixer

Mark transitions and event dependencies

Time-private mixer

Continuous-time decay and temporal dynamics

Lag-conditioned interaction bridges

Event ↔ Time cross-conditioning, sparse routing and shared causal representation

Time-first decoder

When → What

Event-first decoder

What → When

Coherent joint prediction and transferable user embedding

Next-event likelihood · Future sequences · Classification · Time-to-event prediction

## 4.1 A — Better time representation: Predictive Temporal Encoding

I would replace ETAM's single combined time embedding with three complementary temporal representations.

Let event \\(i\\) have type \\(k_i\\), attributes \\(a_i\\), and timestamp \\(t_i\\). Let \\(\Delta_i=t_i-t\_{i-1}\\).

### Component 1: Calendar time

Represent predictable absolute-time effects using periodic features:

\\[ c_i= \phi\_{\mathrm{calendar}}(t_i). \\]

These include hour-of-day, weekday, month, cyclic time features and learned periodic embeddings.

Rather than using a fixed set of hand-designed frequencies only, I would combine them with a small learned Fourier representation:

\\[ \phi\_{\mathrm{Fourier}}(t) = [\sin(\omega_1 t),\cos(\omega_1 t),\ldots]. \\]

The frequencies should be constrained or initialized across meaningful periods so that learned periodicity does not depend entirely on arbitrary timestamp scales.

Absolute time must remain a separate ablation because ETAM's current classification-label window may overlap its input-history window.

### Component 2: Relative time

Construct an embedding of elapsed time at several resolutions:

\\[ r_i = \phi\_{\mathrm{gap}}(\log(1+\Delta_i)). \\]

I propose a hybrid discrete-continuous representation:

\\[ r_i = E\_{\mathrm{bin}}(b(\Delta_i)) + f\_\theta\bigl(\log(1+\Delta_i)-\mu\_{b(\Delta_i)}\bigr). \\]

Here, \\(b(\Delta_i)\\) is an adaptive quantile bin and the second term models the residual inside that bin.

The benefit is that common temporal regimes can share representations without losing continuous resolution.

A pure binning model is a useful baseline, but I would not make discretized time the only representation: exact event-time likelihood and generation require a continuous component.

### Component 3: Event-dependent temporal memory

This is the more important architectural change.

Elapsed time has meaning relative to which event happened previously, not only relative to the clock.

Introduce a bank of decaying temporal states:

\\[ m_r(t_i^-) = e^{-\Delta_i/\tau_r}m_r(t\_{i-1}^+), \\]

\\[ m_r(t_i^+) = m_r(t_i^-)+V_r e(k_i,a_i), \\]

where \\(\tau_r\\) spans several timescales.

For example, the learned state can distinguish recent activity that persists for minutes, daily behavioral patterns and longer-term changes.

This provides a compact, causal approximation of how past events influence future temporal behavior.

The final temporal representation becomes

\\[ z_i^T = \operatorname{LN} \left( W_c c_i+W_r r_i+ W_m[m_1(t_i^-);\ldots;m_R(t_i^-)] \right). \\]

This design draws on continuous-time state-space models and TPP dynamics, but unlike a full neural ODE it can use inexpensive exponential-state updates.

Important: it should be compared with a learned continuous-time SSM and a neural-ODE alternative. The exponential decay is an inductive bias, not a guaranteed superior model.

## 4.2 B — Event representations that preserve their own predictive structure

The event pathway should retain the structure of the event and its attributes.

I would use

\\[ z_i^E = E\_{k_i} + \sum_f E_f(a\_{if}) + \operatorname{MLP}\_{\mathrm{attr}}(a_i). \\]

For continuous numerical attributes, use a learned scalar encoder or quantile-plus-residual representation instead of treating all values as independent categories.

Then add an event-specific causal mixer:

\\[ h_i^E = F_E(z\_{\le i}^E). \\]

This module learns event transitions independently of time, providing an event-only predictive pathway. The temporal module similarly produces \\(h_i^T\\).

This matters for both performance and interpretation. We can compare:

\\[ p(k\_{i+1}\mid H_i^E),\qquad p(\Delta\_{i+1}\mid H_i^T) \\]

against their jointly conditioned counterparts to quantify what cross-modal interaction contributes.

## 4.3 C — Middle layers: conditional synergy, not unconditional fusion

This is the central architectural hypothesis of Stage 2.

Stage 1 lets both token types communicate through shared causal attention at every layer, with modality-specific projections and FFNs.

Stage 2 should make that interaction more selective.

At a given interaction layer, define

\\[ q_i^{E\to T} = W_q^{ET}h_i^E,\qquad k_j^T=W_k^{ET}h_j^T. \\]

Then include a learned time-lag bias in cross-attention:

\\[ \alpha\_{ij}^{E\to T} = \operatorname{softmax}\_{j\le i} \left( \frac{\langle q_i^{E\to T},k_j^T\rangle}{\sqrt d} + b\_\theta(t_i-t_j,k_i) \right). \\]

The cross-modal update is

\\[ \widetilde h_i^E = h_i^E+ g_i^{ET} \sum\_{j\le i} \alpha\_{ij}^{E\to T}W_v^{ET}h_j^T. \\]

Use a corresponding \\(T\to E\\) interaction in the opposite direction.

The gate should depend on the current event, temporal context and history:

\\[ g_i^{ET} = \sigma\bigl( W_g[h_i^E;h_i^T;\phi(\Delta_i)] \bigr). \\]

This creates interaction-specific specialization instead of forcing every event to use the same amount of temporal information.

A model can learn that time is particularly informative for some event types and less informative for others.

I would place these bridges at only a few intermediate layers, initially layers 2 and 4 of a six-layer model, and compare against every-layer interaction.

### Why I prefer this to simply adding more experts

There are three benefits to test.

First, private paths retain modality-specific information and allow auxiliary unimodal prediction.

Second, cross-attention is activated where the two processes provide complementary information.

Third, the model need not run global attention over twice as many tokens.

An efficient implementation would use one fused slot per observed event in the global causal Transformer, retaining private temporal and event states in lightweight modules. If the global attention operates over \\(n\\) slots rather than \\(2n\\) tokens, its attention-score matrix has approximately one-quarter as many entries as the original interleaved design.

That is an attention-memory comparison, not a fourfold end-to-end speedup guarantee. Private mixers, bridges and the new output heads all add work, so actual latency must be measured.

### A required control

A simple gated-fusion model without modality experts may perform just as well.

Therefore, I would compare:

| Middle architecture                    | Why it is needed                                   |
| -------------------------------------- | -------------------------------------------------- |
| Shared Transformer                     | Tests whether specialization is necessary          |
| Original ETAM deep experts             | Stage 1 baseline                                   |
| Two private pathways + simple gating   | Tests whether extra architecture is necessary      |
| Two pathways + shared expert           | Tests private/shared decomposition                 |
| Two pathways + selective cross-bridges | Main proposal                                      |
| Full cross-attention at every layer    | Tests whether selective interaction actually helps |

I would only retain the full design if it produces a statistically reliable improvement over the simpler variants at matched parameters and computation.

# 5. D — New output modeling: predict both conditionals

I think this is potentially the strongest methodological component.

Currently, ETAM's time-conditioned mark objective is interesting, but the benchmark's next-event prediction does not directly exploit the full conditional structure.

Stage 2 should produce a normalized joint probability distribution over next-event time and event type.

Let \\(H\\) be the observed history, \\(\Delta\\) the next waiting time and \\(k\\) the next event type.

There are two valid factorizations.

TIME-FIRST

## When → What

\\[ p_A(\Delta,k\mid H) = f_A(\Delta\mid H) \pi_A(k\mid\Delta,H) \\]

Predict the waiting-time distribution, then predict the event conditional on that time.

EVENT-FIRST

## What → When

\\[ p_B(\Delta,k\mid H) = \pi_B(k\mid H) f_B(\Delta\mid k,H) \\]

Predict the event distribution, then predict its waiting time conditional on the event type.

Both distributions are valid normalized joint models when their component distributions are normalized.

I propose training both:

\\[ \mathcal L\_{\mathrm{dual}} = \frac{1}{2} \left[ -\log p_A(\Delta,k\mid H) -\log p_B(\Delta,k\mid H) \right]. \\]

Then introduce a consistency objective so that the two factorizations do not learn contradictory joint distributions:

\\[ \mathcal L\_{\mathrm{cons}} = D\_{\mathrm{JS}}\left( p_A(\Delta,k\mid H) \Vert p_B(\Delta,k\mid H) \right). \\]

Because time is continuous, estimate the divergence using numerical integration or Monte Carlo samples. Start with a small consistency weight; excessively constraining the two branches could harm optimization.

For a calibrated joint predictor, a simple inference distribution is the normalized mixture

\\[ p\_{\mathrm{joint}} = \tfrac12 p_A+\tfrac12 p_B. \\]

The key advantage is that all evaluation probabilities can be derived consistently from one joint distribution.

### Predicting the next event without knowing its future time

For the time-first model, the real marginal event probability is

\\[ \boxed{ p_A(k\mid H) = \int_0^\infty \pi_A(k\mid\Delta,H) f_A(\Delta\mid H)\\,d\Delta. } \\]

This matters because providing the true next-event time to the mark predictor would constitute oracle information in ordinary next-event forecasting.

For practical inference, approximate the integral through shared Monte Carlo samples or one-dimensional quadrature.

The event-first branch provides a cheap direct prediction \\(\pi_B(k\mid H)\\).

I would compare the accurate marginal against the cheap branch, and optionally distill the former into the latter for fast serving.

### Predicting when the event will happen

The time-first branch provides \\(f_A(\Delta\mid H)\\).

The event-first branch gives

\\[ p_B(\Delta\mid H) = \sum_k \pi_B(k\mid H)f_B(\Delta\mid k,H). \\]

This yields different, useful representations of temporal uncertainty.

For the continuous distributions, start with an expressive log-normal mixture and a spline-based normalizing flow. Consider a survival/hazard decoder if calibration of event occurrence over time is especially important.

The log-normal mixture is already a strong baseline; the new distribution should only be retained if it improves held-out likelihood, survival calibration and/or generation.

### A particularly useful scientific analysis

For each event type, compare

\\[ p(\Delta\mid k,H) \quad\text{versus}\quad p(\Delta\mid H). \\]

And for event prediction, measure

\\[ \log p(k\mid\Delta,H)-\log p(k\mid H). \\]

This quantifies how much knowing the timing changes the event prediction.

It would provide concrete evidence that the model has learned nontrivial event-time dependencies, rather than simply combining two independent predictors.

These are predictive dependencies, not automatically causal relationships.

# 6. E — Multi-event prediction done right

I would incorporate MTP, but use it selectively.

There are really two distinct objectives:

1. Learn the correct joint distribution of the next event and next waiting time.
2. Learn useful representations and draft predictions for multiple future events.

They should not be conflated.

## 6.1 Categorical Soft-MTP for event types

From the shared history state \\(h_i\\), introduce lightweight future-event heads:

\\[ q\_\ell^K(k\_{i+\ell}\mid H_i), \qquad \ell=2,\ldots,L. \\]

Use a shared categorical output projection with the main next-event predictor.

For each future event, the primary predictor evaluated at its corresponding later observed prefix supplies an aligned teacher distribution during training.

The auxiliary head learns to align with that distribution using Soft-MTP's gradient-routing mechanism.

I would start with \\(L=4\\), and ablate 2, 4 and 8.

The important caveat is that the teacher at a later prefix knows intermediate events that the earlier draft head does not know. The draft head therefore cannot generally reproduce the teacher exactly. Soft-MTP is an optimization strategy for these auxiliary predictions, not a guarantee that the uncertainty of the earlier and later conditionals should coincide.

## 6.2 Continuous-time multi-step prediction

For future time, predict cumulative waiting times or the sequence of inter-event gaps.

For example,

\\[ q\_\ell^T \left( t\_{i+\ell}-t_i \mid H_i \right). \\]

I would test a proper distributional loss for cumulative time, alongside an autoregressive joint likelihood for individual gaps.

One possible auxiliary objective is

\\[ \mathcal L\_{\mathrm{future-time}} = \sum\_{\ell=2}^L w\_\ell \left[ -\log q\_\ell^T(t\_{i+\ell}-t_i\mid H_i) \right]. \\]

For efficiency, these heads can use a small shared decoder conditioned on a learned horizon embedding.

This predicts useful longer-horizon quantities without requiring a separate full autoregressive rollout for every auxiliary target.

However, direct marginal predictions of the \\(\ell\\)-th event time are not a substitute for a consistent sequence generator. They should be auxiliary supervision for the joint model.

## 6.3 Long-horizon simulation

At generation time, roll out the joint decoder:

\\[ (\Delta_1,k_1), (\Delta_2,k_2), \ldots, (\Delta_L,k_L). \\]

Stop at the observation horizon, while retaining the ability to model the absence of another event.

The resulting model can generate full future trajectories and estimate distributions of event counts.

This should be evaluated on sequence-level statistics, not just next-event accuracy.

# 7. F — Latent-tokenization objective for genuinely predictive embeddings

I would make the Stanford-inspired idea an additional pretraining component, not an immediate replacement for the event and time tokenizers.

The immediate goal is to obtain a user embedding that is useful for diverse future tasks.

Define three representations:

\\[ u_E=R_E(H),\qquad u_T=R_T(H),\qquad u_J=R_J(H). \\]

They represent event-specific, time-specific and shared history information.

For a future block \\(Y\\), estimate its incremental predictive information using a generative teacher:

\\[ I\_\phi(H;Y) = \log p\_\phi(Y\mid H) - \log p\_\phi(Y). \\]

Then supervise a shared representation against a representation of the future block:

\\[ \mathcal L\_{\mathrm{repr}} = \left( \langle u_J,R_Y(Y)\rangle - \operatorname{sg}(I\_\phi(H;Y)) \right)^2. \\]

A practical implementation can use local future-event increments instead of entire long future blocks, following the bootstrapping principle in RePR.

The important distinction from ordinary contrastive learning is that the target measures conditional predictive information, estimated through a generative model.

I would also create time-only and event-only versions of this objective.

That would allow a concrete question:

Does the joint representation preserve information that neither private representation captures alone?

For example, test whether the joint embedding provides a reliable gain over concatenating separately trained event-only and time-only embeddings.

### Why I would not introduce vector quantization immediately

The ETAM problem differs from image generation.

Time is continuous, sparse and heavy-tailed, and the exact numerical gap can be important. Aggressive quantization might improve Transformer efficiency but damage timing precision.

Therefore I would evaluate:

| Tokenizer                                       | Role                                   |
| ----------------------------------------------- | -------------------------------------- |
| Original ETAM time encoder                      | Reference                              |
| Adaptive time-bin embedding                     | Tests discrete temporal regimes        |
| Adaptive bins + continuous residual             | Main lightweight candidate             |
| Predictive-information-trained temporal encoder | Tests the Stanford-inspired hypothesis |
| Quantized latent time tokens                    | Optional compression experiment        |

The primary criterion should be downstream accuracy and conditional likelihood per unit of computation, not reconstruction quality alone.

# 8. G — A new pretraining strategy designed to benefit from more data

I would change the pretraining procedure more substantially than the architecture.

## 8.1 Main objective

The primary training objective should remain a proper joint log-likelihood:

\\[ \mathcal L\_{\mathrm{joint}} = -\log p\_{\mathrm{joint}}(\Delta,k\mid H). \\]

The full training objective can be organized as

\\[ \begin{aligned} \mathcal L ={}& \mathcal L\_{\mathrm{joint}} +\lambda_d\mathcal L\_{\mathrm{dual}}\\\ &+\lambda_c\mathcal L\_{\mathrm{cons}} +\lambda_m\mathcal L\_{\mathrm{SoftMTP}}\\\ &+\lambda_h\mathcal L\_{\mathrm{horizon}} +\lambda_r\mathcal L\_{\mathrm{repr}} +\lambda_u\mathcal L\_{\mathrm{unimodal}}. \end{aligned} \\]

I would not turn on every term immediately. The experiment needs a staged curriculum so the gain from each objective can be attributed.

## 8.2 Condition on elapsed silence correctly

This is an important opportunity because ETAM already samples query times inside observed inter-event gaps.

If no new event has happened for \\(s\\) seconds, the residual waiting-time distribution is not generally the original waiting-time distribution.

It satisfies

\\[ P(\Delta>s+v\mid\Delta>s,H) = \frac{S(s+v\mid H)}{S(s\mid H)}. \\]

A correctly conditioned survival decoder can use this information directly.

This is valuable for real forecasting because prediction is frequently requested between observed events.

I would train query-time forecasting using both observed future events and survival information, with a strict causal mask.

## 8.3 Learn from censored histories

At the end of each user's observed history, the dataset may provide no next event before the observation cutoff.

That should not simply be discarded.

Where a valid observation window is known, add the right-censoring likelihood:

\\[ \mathcal L\_{\mathrm{censor}} = -\log S(T\_{\mathrm{cut}}-t_n\mid H_n). \\]

This teaches the model when nothing happens, which is part of the temporal process.

The observation-window assumptions must be checked carefully. A user disappearing from a dataset may not always mean that no event occurred; it can reflect incomplete logging or selection.

## 8.4 Multiscale future objectives

Retain the Stage 1 future-window objective, but replace its broad count-regression formulation with an explicit distribution over future event counts when appropriate.

I would train on horizons such as:

\\[ \mathcal H= \\{1\text{ day},7\text{ days},30\text{ days}\\}. \\]

Predict event-type-specific counts, next-event probabilities and distributional properties of future timestamps.

For count data, compare Poisson and negative-binomial likelihoods, since transaction-count distributions may be overdispersed.

This allows a distinction between short-term event forecasting and stable longer-term user preferences.

## 8.5 How to investigate the non-monotonic data-scaling result

The current files show that the 30% pool beats the full pool in joint validation NLL under fixed optimizer steps. That is an important unresolved result.

I would not simply increase the full-pool epoch count and declare the problem solved.

Instead, run two data-scaling protocols.

Protocol A — fixed compute: compare different numbers of unique training users at equal optimizer steps, batch size, input-token budget and learning-rate schedule.

Protocol B — increasing data and compute: increase unique users while giving every configuration a comparable amount of exposure per user.

Use at least three independent pretraining seeds in each protocol.

Track both event accuracy and time NLL separately.

Additionally, stratify validation performance by event frequency, user activity level and calendar period.

This would help determine whether the non-monotonic result comes from undertraining, user-population heterogeneity, the sampling procedure, overrepresentation of easy examples or a mismatch between training and validation distributions.

Also investigate the reported near-equal training times for the nominal 10-epoch and 20-epoch ETAM configurations. Verify the actual optimizer steps, examples and event-token exposures from logs before interpreting scaling curves.

More data does not mathematically guarantee better finite-sample performance. The goal is to establish reliable positive scaling trends and explain cases where they fail.

# 9. H — Post-training: preserve generality instead of specializing too early

Your Stage 1 frozen representation result suggests the pretrained user embedding may already carry much of the information needed for downstream prediction.

I would therefore prioritize parameter-efficient adaptation and preserve the generative representation.

### For classification

Compare frozen linear probing, a small nonlinear head, lightweight adapters and full fine-tuning.

Use a representation combining the private and shared states:

\\[ u = W_u[u_E;u_T;u_J]. \\]

The downstream classifier should use a strictly prospective history cutoff, with labels beginning after the observed history.

Because positive prevalence is only about 1.14%, the primary metrics should include average precision, Precision\@K, Recall\@K and calibration, in addition to ROC-AUC.

Also compare each adaptation method for multiple labeled-data fractions—e.g. 1%, 5%, 10%, 25% and 100%.

That would test whether Stage 2 genuinely learns a more label-efficient foundation representation.

### For TPP prediction

Keep the generative heads available and compare frozen pretrained heads, head-only fine-tuning, low-rank adapter fine-tuning and full fine-tuning.

For the time decoder, use proper log-likelihood and calibration-aware validation criteria.

I would avoid making a weighted sum of AUC and next-event accuracy the only model-selection criterion for the pretrained model. It can favor a configuration without the best probability estimates or calibration.

### For multitask adaptation

Use shared representations with small task-specific readout heads. Train the backbone conservatively, with separate learning rates for the readouts, interaction bridges and pretrained private encoders.

If gradient conflict between classification and TPP losses becomes measurable, then consider methods such as gradient balancing or projection. I would not add them without evidence of interference.

The target is a model that retains its pretrained forecasting ability while learning new classification labels efficiently.

# 10. Stage 2 experimental plan

I would organize the work into four sequential phases. This avoids ending with a large model whose gains cannot be explained.

| Phase                                   | Experiments                                                                                                                  | What must be demonstrated                                            |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| P0 — Reliable foundation                | Strict prospective labels; matched pretraining compute; matched context and checkpoint selection; multiple pretraining seeds | Stage 1 gains survive fair comparison                                |
| P1 — Temporal and generative modeling   | Adaptive time encoder; multiscale temporal state; time-first, event-first and dual joint decoders                            | Time representation and conditional likelihood improve               |
| P2 — Architecture and synergy           | Shared, private, gated and selectively bridged experts at fixed compute                                                      | Interaction mechanisms improve over simple fusion                    |
| P3 — Predictive representation learning | Hard MTP vs Soft-MTP; latent information objective; data-scaling curves; frozen transfer                                     | Better transfer and long-horizon performance, with favorable scaling |

## The ablations I would prioritize

| Ablation                   | Critical comparison                                                               |
| -------------------------- | --------------------------------------------------------------------------------- |
| Time representation        | Original ETAM vs hybrid quantile-residual vs multiscale temporal state            |
| Cross-modal interaction    | Shared model vs simple gated fusion vs lag-aware bridges                          |
| Joint prediction           | Independent heads vs time-first vs event-first vs bidirectional training          |
| Multi-step prediction      | None vs hard MTP vs Soft-MTP                                                      |
| Representation pretraining | Generative only vs generative + information-gain representation objective         |
| Temporal likelihood        | Log-normal mixture vs spline flow vs survival decoder                             |
| Data scaling               | 30,665 / 52,918 / 158,754 / 529,180 users                                         |
| Efficiency                 | Equal parameter count, equal FLOPs, equal exposure and measured inference latency |

A result should not be counted as evidence for a mechanism if the corresponding ablation changes multiple mechanisms simultaneously.

## Strong baselines to add

The most important additions are not larger language models. They are models whose inductive biases directly challenge Stage 2's claims.

S2P2 challenges the need for Transformer-style temporal processing. FlexTPP challenges the discrete-continuous output design. ITPP challenges event-type separation. Kairos challenges the value of time/event tokenization. Strong conditional TPP models challenge the generative decoder.

All should be rerun under compatible data and evaluation protocols wherever feasible; published scores from different datasets are not direct comparisons.

For transfer and generalization, I would test MBD plus public event datasets spanning different temporal regimes. Candidates include StackOverflow, Retweet/MOOC and a healthcare-event benchmark where permitted. A classification-transfer claim requires datasets with genuine downstream labels, not synthetic labels chosen only because they are convenient.

## Required evaluation beyond Stage 1

The current benchmark emphasizes next-type accuracy, time RMSE/MAE and classification AUC.

For Stage 2 I would add:

| Evaluation                                    | What it measures                                     |
| --------------------------------------------- | ---------------------------------------------------- |
| Joint held-out NLL                            | Whether the model learns the event-time distribution |
| Type NLL and macro-F1                         | Both probabilistic and classification quality        |
| Time predictive intervals and PIT calibration | Whether temporal uncertainty is meaningful           |
| Survival calibration                          | Whether no-event probabilities are reliable          |
| Multi-step event/count forecasting            | Long-horizon predictive ability                      |
| Generated sequence statistics                 | Whether simulated event streams look realistic       |
| Frozen cross-task transfer                    | Foundation-representation quality                    |
| Quality versus inference FLOPs/latency        | Whether the architecture is practically efficient    |

When comparing likelihoods, make sure all models use the same time variable and density units. A likelihood expressed for `log1p(seconds)` cannot be numerically compared with a seconds-density without the appropriate change-of-variables correction.
