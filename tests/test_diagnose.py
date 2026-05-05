"""Diagnose rules — pure-logic tests, no GPU. Researcher-revised 2026-05-01."""
import pytest

from modules.diagnose import diagnose


def _result(categoria: str | None, shadow: float, rg_norm: float, rg_raw: float = 0.5, **extra) -> dict:
    return {
        "categoria": categoria,
        "shadow_score": shadow,
        "roughness_normalized": rg_norm,
        "roughness_raw": rg_raw,
        **extra,
    }


def test_error_short_circuits_other_rules():
    assert diagnose({"error": "too short", "categoria": None}) == "Analysis error"


def test_saturation_label_emits_actionable_message():
    out = diagnose(_result("Inconclusive (humanity saturated)", shadow=0.2, rg_norm=1.0, rg_raw=16.5))
    assert "humanity axis is saturated" in out
    assert "norm_factor" in out
    assert "16.5" in out  # echoes raw humanity so the user knows where to set norm_factor


def test_strong_human_voice_overrides_category_for_human():
    assert (
        diagnose(_result("Human", shadow=0.7, rg_norm=0.8))
        == "Strong human voice: abundant lapses and natural lexical variability."
    )


def test_strong_human_voice_overrides_category_for_mixed():
    # If both axes are extreme, mixed-signals still defers to the strong-voice rule.
    assert (
        diagnose(_result("Mixed signals", shadow=0.7, rg_norm=0.8))
        == "Strong human voice: abundant lapses and natural lexical variability."
    )


def test_synthetic_low_shadow_clean_ai():
    out = diagnose(_result("Synthetic", shadow=0.10, rg_norm=0.2))
    assert "Synthetic pattern" in out
    assert "un-humanised AI" in out


def test_synthetic_with_human_traits():
    out = diagnose(_result("Synthetic", shadow=0.3, rg_norm=0.4))
    assert "Synthetic-leaning" in out
    assert "light human editing" in out


def test_human_clear_voice_high_shadow():
    out = diagnose(_result("Human", shadow=0.6, rg_norm=0.65))
    assert "Clear human voice" in out


def test_human_default_voice():
    out = diagnose(_result("Human", shadow=0.4, rg_norm=0.55))
    assert "Human voice with natural lexical variability" in out


def test_mixed_signals_ambiguous():
    out = diagnose(_result("Mixed signals", shadow=0.4, rg_norm=0.5))
    assert "Mixed signals" in out
    assert "L2 human" in out or "humanised AI" in out


def test_inconclusive():
    out = diagnose(_result("Inconclusive", shadow=0.2, rg_norm=0.3))
    assert "Inconclusive" in out
    assert "Manual review" in out


def test_unknown_categoria_falls_through():
    assert diagnose(_result("Bogus", shadow=0.0, rg_norm=0.0)) == "No diagnosis available."


@pytest.mark.parametrize("shadow,rg_norm", [(0.65, 0.71), (0.66, 0.7)])
def test_strong_voice_rule_strict_inequality_boundary(shadow, rg_norm):
    """Strong-voice rule uses strict > on both terms — boundary equalities should NOT trigger it."""
    out = diagnose(_result("Human", shadow=shadow, rg_norm=rg_norm))
    assert "Strong human voice" not in out
