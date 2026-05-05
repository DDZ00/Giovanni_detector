"""
Giovanni v0.4 analyzer — Binoculars (Hans et al. 2024) + calculus tv2.

Pipeline:
    1. Both scorers produce per-token full log-distributions in one pass.
    2. Binoculars score B = log PPL_M1 / log X-PPL_M1,M2.
    3. Calculus signals (tv1, tv2, l2_*, linf_*, mad_*, stv2, slinf2) on the
       base-model actual-token logprobs.
    4. Verdict:
         use_calculus=False → "Synthetic" if B < threshold_binoculars else "Human"
         use_calculus=True  → soft-combine via sigmoid blend; "Synthetic" if p_ai > 0.5

The scorers must share a tokenizer for Binoculars to be meaningful. The caller
(app.py) is responsible for refusing cross-tokenizer pairs upfront.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol

import numpy as np

from config import Config


class FullScorer(Protocol):
    def score_full(self, text: str, *, max_chunk: int) -> dict: ...


def analyze_text(
    text: str,
    scorer_base: FullScorer,
    scorer_instruct: FullScorer,
    *,
    threshold_binoculars: float | None = None,
    threshold_tv2: float | None = None,
    use_calculus: bool | None = None,
    binoculars_weight: float | None = None,
    binoculars_steepness: float | None = None,
    calculus_steepness: float | None = None,
    min_words: int | None = None,
    max_chunk: int | None = None,
    same_tokenizer: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full Binoculars + calculus pipeline on `text`."""
    threshold_binoculars = Config.THRESHOLD_BINOCULARS if threshold_binoculars is None else threshold_binoculars
    threshold_tv2 = Config.THRESHOLD_TV2 if threshold_tv2 is None else threshold_tv2
    use_calculus = Config.USE_CALCULUS if use_calculus is None else use_calculus
    binoculars_weight = Config.BINOCULARS_WEIGHT if binoculars_weight is None else binoculars_weight
    binoculars_steepness = Config.BINOCULARS_STEEPNESS if binoculars_steepness is None else binoculars_steepness
    calculus_steepness = Config.CALCULUS_STEEPNESS if calculus_steepness is None else calculus_steepness
    min_words = Config.MIN_WORDS if min_words is None else min_words
    max_chunk = Config.MAX_CHUNK_TOKENS if max_chunk is None else max_chunk

    word_count = len(text.split())
    if word_count < min_words:
        return {
            "error": f"Text too short ({word_count} words). Minimum: {min_words}",
            "categoria": None,
        }

    # M1 = instruct (observer), M2 = base (performer) — paper's default ordering.
    full_m1 = scorer_instruct.score_full(text, max_chunk=max_chunk)
    full_m2 = scorer_base.score_full(text, max_chunk=max_chunk)

    L = min(len(full_m1["logprobs"]), len(full_m2["logprobs"]))
    if L < 5:
        return {
            "error": "Text produced too few sampled tokens",
            "categoria": None,
        }

    # Tokenizer-mismatch guard: even if app.py allowed it through, the
    # vocab dimensions must match for X-PPL to be meaningful.
    if same_tokenizer and full_m1["log_distrib"].shape[1] != full_m2["log_distrib"].shape[1]:
        return {
            "error": (
                f"Tokenizer mismatch: vocab sizes {full_m1['log_distrib'].shape[1]} "
                f"vs {full_m2['log_distrib'].shape[1]}. Binoculars requires aligned vocab."
            ),
            "categoria": None,
        }

    # Truncate to common length.
    log_distrib_m1 = full_m1["log_distrib"][:L]
    log_distrib_m2 = full_m2["log_distrib"][:L]
    actual_tokens = full_m1["tokens"][:L].astype(np.int64)
    logprobs_base = full_m2["logprobs"][:L]  # for calculus signals on base curve

    # ── Binoculars ────────────────────────────────────────────
    binoculars_score: float | None = None
    binoculars_block: Dict[str, Any] = {"score": None}
    if same_tokenizer:
        from modules.binoculars import compute_binoculars
        b = compute_binoculars(log_distrib_m1, log_distrib_m2, actual_tokens)
        binoculars_score = b["score"]
        binoculars_block = {
            "score": round(b["score"], 6),
            "log_ppl_m1": round(b["log_ppl_m1"], 6),
            "log_x_ppl": round(b["log_x_ppl"], 6),
        }
        if verbose:
            binoculars_block["per_token_log_ppl_m1"] = b["per_token_log_ppl_m1"].tolist()
            binoculars_block["per_token_x_ppl"] = b["per_token_x_ppl"].tolist()

    # ── Calculus signals on the base model's logprob curve ────
    from modules.differential import compute_all as compute_differential
    differential = compute_differential(full_m1["logprobs"][:L], logprobs_base)
    tv2 = differential.get("tv2")

    # ── Soft-combine and verdict ──────────────────────────────
    from modules.binoculars import soft_combine
    if binoculars_score is None:
        # No Binoculars (cross-tokenizer pair): calculus-only fallback.
        # We can still produce a verdict using just tv2 against threshold_tv2.
        from modules.binoculars import sigmoid
        p_calc = float(sigmoid((tv2 / threshold_tv2 - 1.0) * calculus_steepness))
        probabilities = {"p_bino": None, "p_calc": p_calc, "p_ai": p_calc}
        categoria = "Synthetic" if p_calc > 0.5 else "Human"
    elif use_calculus:
        probabilities = soft_combine(
            binoculars_score, tv2,
            threshold_binoculars=threshold_binoculars,
            threshold_tv2=threshold_tv2,
            binoculars_weight=binoculars_weight,
            binoculars_steepness=binoculars_steepness,
            calculus_steepness=calculus_steepness,
        )
        categoria = "Synthetic" if probabilities["p_ai"] > 0.5 else "Human"
    else:
        # Pure Binoculars: paper's binary rule.
        probabilities = {"p_bino": None, "p_calc": None, "p_ai": None}
        categoria = "Synthetic" if binoculars_score < threshold_binoculars else "Human"

    result: Dict[str, Any] = {
        "categoria": categoria,
        "strategy": "binoculars+calculus" if (use_calculus and binoculars_score is not None) else
                    ("binoculars" if binoculars_score is not None else "calculus_only"),
        "binoculars": binoculars_block,
        "differential": {k: round(v, 6) for k, v in differential.items()},
        "probabilities": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in probabilities.items()},
        "thresholds": {
            "binoculars": threshold_binoculars,
            "tv2": threshold_tv2,
        },
        "blend": {
            "binoculars_weight": binoculars_weight,
            "binoculars_steepness": binoculars_steepness,
            "calculus_steepness": calculus_steepness,
        },
        "palabras": word_count,
        "tokens_analizados": int(L),
    }

    if verbose:
        # First-derivative array for the calculus chart (visualizing |l'(t)|).
        l_prime = np.diff(logprobs_base).astype(np.float32)
        result["verbose"] = {
            "logprobs_base": logprobs_base.tolist(),
            "l_prime_base": l_prime.tolist(),
        }

    from modules.diagnose import diagnose
    result["diagnostico"] = diagnose(result)
    return result
