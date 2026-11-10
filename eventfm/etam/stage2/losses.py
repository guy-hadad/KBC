"""ETAM-2 training objective (Section G): a proper joint likelihood plus auxiliary terms.

    L = joint + l_d dual + l_c cons + aux_mark + feature + query + survival + censor
        + count + unimodal + l_m MTP + l_h horizon + l_r repr + vq + cls

A term is computed only when its weight is non-zero and the model has the
module it needs, so every ablation of the curriculum is a weight change.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from eventfm.etam.batching import IGNORE_INDEX
from eventfm.etam.stage2.batching import sub_batch
from eventfm.etam.stage2.config import Etam2Weights
from eventfm.etam.stage2.distributions import IndependentJoint, MixtureJoint, js_divergence
from eventfm.etam.stage2.heads import count_log_prob

LOG_2 = math.log(2.0)
CONSISTENCY_POSITIONS = 256


def censor_terms(decoder, states, y_censored) -> Tuple[torch.Tensor, torch.Tensor]:
    """(-log S under the inference distribution, 1/2 sum of the branch -log S) per censored interval."""

    if decoder.kind == "dual":
        first = decoder.branch_a(states).time_log_survival(y_censored)
        second = decoder.branch_b(states).time_log_survival(y_censored)
        mixture = torch.logaddexp(first, second) - LOG_2
        return -mixture, -0.5 * (first + second)
    value = -decoder(states).time_log_survival(y_censored)
    return value, value


def joint_terms(decoder, states, y, k) -> Tuple[torch.Tensor, torch.Tensor]:
    """(-log p_joint, dual term) per example; dual is 1/2 sum of branch NLLs (dual decoder)."""

    if decoder.kind == "dual":
        first = decoder.branch_a(states).log_prob(y, k)
        second = decoder.branch_b(states).log_prob(y, k)
        joint = torch.logaddexp(first, second) - LOG_2
        return -joint, -0.5 * (first + second)
    log_prob = decoder(states).log_prob(y, k)
    return -log_prob, -log_prob


def split_log_prob(joint_dist, y, k) -> Tuple[torch.Tensor, torch.Tensor]:
    """(time part, mark part) of log p(y, k): log f(y) + log pi(k | y) along branch A."""

    dist = joint_dist.first if isinstance(joint_dist, MixtureJoint) else joint_dist
    if hasattr(dist, "time") and not isinstance(dist, IndependentJoint):
        time_part = dist.time.log_prob(y)
        return time_part, dist.log_prob(y, k) - time_part
    if isinstance(dist, IndependentJoint):
        time_part = dist.time.log_prob(y)
        return time_part, dist.log_prob(y, k) - time_part
    mark_part = dist.mark_log_probs().gather(-1, k.long().clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return dist.log_prob(y, k) - mark_part, mark_part


def compute_losses2(model, outputs, batch, weights: Etam2Weights) -> Dict[str, torch.Tensor]:
    terms: Dict[str, torch.Tensor] = {}
    decoder = model.decoder
    hidden = outputs["h"].float()
    next_type, next_y = batch["next_type"], batch["next_log_dt"]
    is_query = batch["is_query"]
    dense = (next_type != IGNORE_INDEX) & ~is_query

    # Right-censoring (G 8.3) enters the same per-interval likelihood as the observed
    # gaps: a full history's last interval contributes -log S(T_cut - t_n), its query
    # slot -log S(T_cut - now). ``weights.censor`` (0 or 1) switches it off or on.
    censored = batch.get("censor_mask")
    use_censor = bool(weights.censor) and censored is not None and bool(censored.any())
    if use_censor:
        censor_last = censor_terms(decoder, hidden[:, -2][censored], batch["censor_log_dt"][censored])
        censor_query = censor_terms(decoder, hidden[:, -1][censored], batch["censor_q_log_dt"][censored])

    if dense.any() and (weights.joint or weights.dual or weights.feature or weights.cons or weights.aux_mark):
        states, y, k = hidden[dense], next_y[dense], next_type[dense]
        if weights.joint or weights.dual:
            joint, dual = joint_terms(decoder, states, y, k)
            if use_censor:
                scale = float(weights.censor)
                count = joint.shape[0] + scale * censor_last[0].shape[0]
                terms["joint"] = (joint.sum() + scale * censor_last[0].sum()) / count
                terms["dual"] = (dual.sum() + scale * censor_last[1].sum()) / count
                terms["censor_nll"] = censor_last[0].mean().detach()
            else:
                terms["joint"] = joint.mean()
                terms["dual"] = dual.mean()
            if not weights.joint:
                terms.pop("joint")
            if not (weights.dual and decoder.kind == "dual"):
                terms.pop("dual")
        if weights.aux_mark and decoder.kind == "time_first":
            terms["aux_mark"] = F.cross_entropy(decoder.cheap_mark_logits(states), k)
        if weights.cons and decoder.kind == "dual":
            count = states.shape[0]
            if count > CONSISTENCY_POSITIONS:
                pick = torch.randperm(count, device=states.device)[:CONSISTENCY_POSITIONS]
                subset = states[pick]
            else:
                subset = states
            table_a = decoder.branch_a(subset).discretized(decoder.grid)
            table_b = decoder.branch_b(subset).discretized(decoder.grid)
            terms["cons"] = js_divergence(table_a, table_b).mean()
        if weights.feature and model.feature_head is not None:
            # Event slots and the query slot (generation samples attributes there).
            query_rows = (next_type[:, -1] != IGNORE_INDEX)
            feature_states = torch.cat([states, hidden[:, -1][query_rows]])
            feature_k = torch.cat([k, next_type[:, -1][query_rows]])
            feature_y = torch.cat([y, next_y[:, -1][query_rows]])
            logits = model.feature_head(feature_states, feature_k, feature_y)
            targets = torch.cat([batch["next_features"][dense], batch["next_features"][:, -1][query_rows]])
            terms["feature"] = torch.stack(
                [
                    F.cross_entropy(head, targets[:, column], ignore_index=IGNORE_INDEX)
                    for column, head in enumerate(logits)
                ]
            ).mean()

    query_has_next = next_type[:, -1] != IGNORE_INDEX
    if weights.query and query_has_next.any():
        states = hidden[:, -1][query_has_next]
        joint, _ = joint_terms(decoder, states, next_y[:, -1][query_has_next], next_type[:, -1][query_has_next])
        if use_censor:
            scale = float(weights.censor)
            terms["query"] = (joint.sum() + scale * censor_query[0].sum()) / (joint.shape[0] + scale * censor_query[0].shape[0])
        else:
            terms["query"] = joint.mean()

    between = batch.get("between_mask")
    if weights.survival and between is not None and between.any():
        dist = decoder(hidden[:, -2][between])
        log_prob = dist.residual_log_prob(
            batch["total_log_dt"][between],
            batch["elapsed_log_dt"][between],
            next_type[:, -1][between],
            next_y[:, -1][between],
        )
        terms["survival"] = -log_prob.mean()


    if weights.count and model.counts is not None:
        params = model.counts(outputs["user"])
        if model.counts.kind == "stage1":
            mask = batch["future_mask"]
            if mask.sum() > 0:
                bce = F.binary_cross_entropy_with_logits(
                    params["multi_hot"], batch["future_types"], reduction="none"
                ).mean(-1)
                squared = (params["log_count"] - batch["future_log_count"]) ** 2
                terms["count"] = ((bce + 0.1 * squared) * mask).sum() / mask.sum()
        else:
            mask = batch["count_mask"]
            if mask.sum() > 0:
                log_prob = count_log_prob(params, batch["counts"]).mean(-1)
                terms["count"] = -(log_prob * mask).sum() / mask.sum()

    if weights.unimodal and getattr(model, "has_unimodal", False) and dense.any():
        y, k = next_y[dense], next_type[dense]
        cross_entropy = F.cross_entropy(model.unimodal_event(outputs["pre_event"][dense].float()), k)
        nll = -model.unimodal_time(outputs["pre_time"][dense].float()).log_prob(y).mean()
        terms["unimodal"] = cross_entropy + nll

    if model.mtp is not None and (weights.mtp or weights.horizon):
        mtp_terms, horizon_terms = multi_event_terms(model, hidden, batch)
        if weights.mtp and mtp_terms:
            terms["mtp"] = torch.stack(mtp_terms).mean()
        if weights.horizon and horizon_terms:
            terms["horizon"] = torch.stack(horizon_terms).mean()

    if weights.repr and model.repr is not None:
        value = repr_term(model, outputs, batch)
        if value is not None:
            terms["repr"] = value

    vq_loss = outputs.get("aux", {}).get("vq_loss")
    if weights.vq and vq_loss is not None:
        terms["vq"] = vq_loss

    if weights.cls:
        labels = batch["labels"]
        labelled = labels >= 0
        if labelled.any():
            terms["cls"] = F.cross_entropy(model.classify(outputs["user"][labelled]), labels[labelled])

    total = None
    for name, value in terms.items():
        if not hasattr(weights, name):
            continue  # logging only
        weighted = getattr(weights, name) * value
        total = weighted if total is None else total + weighted
    terms["loss"] = total if total is not None else hidden.sum() * 0.0
    return terms


def multi_event_terms(model, hidden, batch):
    """Draft mark heads (hard CE or Soft-MTP KL to the aligned teacher) and cumulative-time NLLs.

    Soft-MTP (Teoh 2026): the head for event i+l is trained only by
    KL(sg(q) || p_l), q = the primary next-mark distribution at the later
    prefix i+l-1, so the head and trunk receive g_KL and none of the hard-target
    residual. The primary predictor is untouched by this term.
    """

    decoder = model.decoder
    is_query = batch["is_query"]
    slots = hidden.shape[1]
    drafts = model.mtp.states(hidden)
    teacher = decoder.cheap_mark_logits(hidden).detach() if model.config.mtp_target == "soft" else None
    mtp_terms, horizon_terms = [], []
    slot_index = torch.arange(slots, device=hidden.device).unsqueeze(0)
    for step, state in enumerate(drafts):
        lag = step + 2
        target = batch["mtp_types"][..., lag - 1]
        valid = (target != IGNORE_INDEX) & ~is_query & batch["slot_valid"]
        if teacher is None:
            if valid.any():
                logits = decoder.mark_proj(state[valid])
                mtp_terms.append(F.cross_entropy(logits, target[valid]))
        else:
            shift = lag - 1
            aligned = valid & (slot_index + shift <= slots - 2)
            if aligned.any():
                shifted = torch.cat([teacher[:, shift:], teacher[:, :shift]], dim=1)
                probs = torch.softmax(shifted[aligned], dim=-1)
                log_student = torch.log_softmax(decoder.mark_proj(state[aligned]), dim=-1)
                mtp_terms.append((probs * (torch.log(probs + 1e-12) - log_student)).sum(-1).mean())
        if model.mtp.with_time:
            valid_time = (target != IGNORE_INDEX) & ~is_query & batch["slot_valid"]
            if valid_time.any():
                dist = model.mtp.time_dist(hidden[valid_time], step)
                horizon_terms.append(-dist.log_prob(batch["mtp_log_dt"][..., lag - 1][valid_time]).mean())
    return mtp_terms, horizon_terms


def repr_term(model, outputs, batch):
    """RePR bootstrapped regression <R(H), R(y<=k) - R(y<k)> ~ PMI_k (Huber, delta = 1).

    PMI_k = log p(y_k | y_<k, H) - log p(y_k | y_<k): the first from the main
    forward, the second from a forward over the future block alone (history
    dropped), both under stop-gradient. Joint, event-only and time-only
    versions use the joint, mark and time parts of the log-likelihood.
    """

    rows = batch.get("repr_rows")
    if rows is None or rows.numel() == 0:
        return None
    block = model.config.repr_block
    sub = sub_batch(batch)
    sub_outputs = model(sub)
    decoder = model.decoder
    hidden = outputs["h"].float()[rows]
    sub_hidden = sub_outputs["h"].float()
    slots = hidden.shape[1]
    # Event slots of the block y_1..y_m are slots - 1 - m .. slots - 2 of the main window;
    # y_k (k >= 2) is predicted at the slot holding y_{k-1}.
    main_slots = torch.arange(slots - 1 - block, slots - 2, device=hidden.device)
    sub_slots = torch.arange(0, block - 1, device=hidden.device)
    y = batch["next_log_dt"][rows][:, main_slots]
    k = batch["next_type"][rows][:, main_slots]
    with torch.no_grad():
        main_dist = decoder(hidden[:, main_slots].reshape(-1, hidden.shape[-1]))
        sub_dist = decoder(sub_hidden[:, sub_slots].reshape(-1, hidden.shape[-1]))
        flat_y, flat_k = y.reshape(-1), k.reshape(-1)
        main_time, main_mark = split_log_prob(main_dist, flat_y, flat_k)
        sub_time, sub_mark = split_log_prob(sub_dist, flat_y, flat_k)
    shape = y.shape
    pmi = {
        "joint": ((main_time + main_mark) - (sub_time + sub_mark)).view(shape),
        "event": (main_mark - sub_mark).view(shape),
        "time": (main_time - sub_time).view(shape),
    }
    sources = {
        "joint": (outputs["h"], sub_outputs["h"]),
        "event": (outputs.get("h_event"), sub_outputs.get("h_event")),
        "time": (outputs.get("h_time"), sub_outputs.get("h_time")),
    }
    losses = []
    for name in model.repr.modalities:
        main_states, future_states = sources[name]
        if main_states is None:
            continue
        history = model.repr.history[name](main_states.float()[rows][:, slots - 2 - block])
        future = model.repr.future[name](future_states.float()[:, :block])  # R(y_<=k), k = 1..m
        increments = future[:, 1:] - future[:, :-1]  # k = 2..m
        predicted = (history.unsqueeze(1) * increments).sum(-1)
        losses.append(F.huber_loss(predicted, pmi[name].detach(), delta=1.0))
    return torch.stack(losses).mean() if losses else None
