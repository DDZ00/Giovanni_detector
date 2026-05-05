"""
Giovanni v0.4 diagnostic rules — Binoculars + calculus.

Pure function over the result dict produced by analyzer.analyze_text().
First-match wins. English strings (research dashboard).

Categories emitted by analyzer:
  - "Synthetic"
  - "Human"
"""
from __future__ import annotations

from typing import Any, Dict


def diagnose(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"Analysis error: {result['error']}"

    cat = result.get("categoria")
    strat = result.get("strategy")
    bino = (result.get("binoculars") or {}).get("score")
    diff = result.get("differential") or {}
    probs = result.get("probabilities") or {}
    th = result.get("thresholds") or {}

    tv2 = diff.get("tv2")
    p_ai = probs.get("p_ai")

    if strat == "calculus_only":
        return (
            f"Binoculars unavailable on this pair (cross-tokenizer). "
            f"Verdict from calculus alone: tv2={tv2:.1f} vs threshold {th.get('tv2')}."
        )

    if cat == "Synthetic":
        if strat == "binoculars+calculus":
            return (
                f"Synthetic (p_ai={p_ai:.2f}). Binoculars B={bino:.3f} below "
                f"{th.get('binoculars'):.3f}; calculus tv2={tv2:.0f} vs {th.get('tv2'):.0f}."
            )
        return f"Synthetic. Binoculars B={bino:.3f} below threshold {th.get('binoculars'):.3f}."

    if cat == "Human":
        if strat == "binoculars+calculus":
            return (
                f"Human (p_ai={p_ai:.2f}). Binoculars B={bino:.3f} above "
                f"{th.get('binoculars'):.3f}; calculus tv2={tv2:.0f} vs {th.get('tv2'):.0f}."
            )
        return f"Human. Binoculars B={bino:.3f} above threshold {th.get('binoculars'):.3f}."

    return "No diagnosis available."
