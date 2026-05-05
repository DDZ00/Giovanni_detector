"""
Binoculars detector (Hans et al. 2024, ICML, arXiv 2401.12070v3).

Detects machine-generated text by comparing two views of the same string
through two language models that share a tokenizer:

    M1 = observer  (e.g. Falcon-7B-Instruct)
    M2 = performer (e.g. Falcon-7B base)

Score:

    B(s) = log PPL_M1(s) / log X-PPL_{M1,M2}(s)

    log PPL_M1(s)   = -1/L · Σ_i log p_M1(token_i | context_<i)
    log X-PPL(s)    = -1/L · Σ_i Σ_v p_M1(v | context_<i) · log p_M2(v | context_<i)

Below a threshold (~0.901 in the paper) → machine-generated.

This module is pure math: it consumes log-distributions already produced by
the scorer layer and returns a Binoculars dict. It does not load models or
touch the GPU. Tests inject synthetic distributions directly.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Stable log-softmax along the last axis."""
    m = logits.max(axis=-1, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def per_token_log_ppl(
    log_distrib_m1: np.ndarray,
    actual_token_ids: np.ndarray,
) -> np.ndarray:
    """Negative log-likelihood of each actual token under M1.

    log_distrib_m1: shape (L, V), already log-softmaxed.
    actual_token_ids: shape (L,), the token observed at each position.
    Returns shape (L,) — non-negative; high = M1 was surprised.
    """
    positions = np.arange(len(actual_token_ids))
    return -log_distrib_m1[positions, actual_token_ids]


def per_token_cross_ppl(
    log_distrib_m1: np.ndarray,
    log_distrib_m2: np.ndarray,
) -> np.ndarray:
    """Per-token cross-entropy H(M1, M2) = -Σ_v p_M1(v) · log p_M2(v).

    Both inputs: shape (L, V), log-softmaxed.
    Returns shape (L,) — non-negative; high = M2 surprised by M1's distribution.
    """
    p_m1 = np.exp(log_distrib_m1)
    return -np.sum(p_m1 * log_distrib_m2, axis=-1)


def compute_binoculars(
    log_distrib_m1: np.ndarray,
    log_distrib_m2: np.ndarray,
    actual_token_ids: np.ndarray,
) -> Dict[str, Any]:
    """Compute the Binoculars score and per-token contributions.

    Returns:
        {
            "score": B,                       # float, ~[0.5, 1.2]
            "log_ppl_m1": float,              # numerator
            "log_x_ppl": float,               # denominator
            "per_token_log_ppl_m1": np.ndarray,  # (L,)
            "per_token_x_ppl": np.ndarray,       # (L,)
        }
    """
    if log_distrib_m1.shape != log_distrib_m2.shape:
        raise ValueError(
            f"Binoculars requires aligned distributions; got {log_distrib_m1.shape} "
            f"vs {log_distrib_m2.shape}. Both models must share a tokenizer."
        )
    if len(actual_token_ids) != log_distrib_m1.shape[0]:
        raise ValueError(
            f"actual_token_ids length {len(actual_token_ids)} does not match "
            f"distribution length {log_distrib_m1.shape[0]}."
        )

    pt_log_ppl = per_token_log_ppl(log_distrib_m1, actual_token_ids)
    pt_x_ppl = per_token_cross_ppl(log_distrib_m1, log_distrib_m2)

    log_ppl_m1 = float(pt_log_ppl.mean())
    log_x_ppl = float(pt_x_ppl.mean())

    # Guard division by ~0 (would only happen if M1==M2 perfectly, which the
    # build_scorer_pair degenerate-pair check already rejects).
    if log_x_ppl < 1e-9:
        score = float("inf")
    else:
        score = log_ppl_m1 / log_x_ppl

    return {
        "score": score,
        "log_ppl_m1": log_ppl_m1,
        "log_x_ppl": log_x_ppl,
        "per_token_log_ppl_m1": pt_log_ppl.astype(np.float32),
        "per_token_x_ppl": pt_x_ppl.astype(np.float32),
    }


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def soft_combine(
    binoculars_score: float,
    tv2: float | None,
    *,
    threshold_binoculars: float,
    threshold_tv2: float,
    binoculars_weight: float,
    binoculars_steepness: float,
    calculus_steepness: float,
) -> Dict[str, float]:
    """Probability-weighted blend of Binoculars and calculus AI-votes.

    p_bino = sigmoid((threshold_binoculars - B) * binoculars_steepness)
        below threshold → AI more likely → p_bino > 0.5
    p_calc = sigmoid((tv2 / threshold_tv2 - 1) * calculus_steepness)
        above threshold → AI more likely → p_calc > 0.5
    p_ai   = w * p_bino + (1 - w) * p_calc

    Returns {p_bino, p_calc, p_ai}. tv2=None drops the calculus term and
    p_ai = p_bino (pure Binoculars verdict, calculus ignored).
    """
    p_bino = float(sigmoid((threshold_binoculars - binoculars_score) * binoculars_steepness))
    if tv2 is None:
        return {"p_bino": p_bino, "p_calc": None, "p_ai": p_bino}
    p_calc = float(sigmoid((tv2 / threshold_tv2 - 1.0) * calculus_steepness))
    p_ai = binoculars_weight * p_bino + (1.0 - binoculars_weight) * p_calc
    return {"p_bino": p_bino, "p_calc": p_calc, "p_ai": float(p_ai)}
