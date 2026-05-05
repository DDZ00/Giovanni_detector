"""Unit tests for analyzer math + analyze_text orchestration with fake scorers."""
import numpy as np
import pytest

from modules.analyzer import (
    analyze_text,
    categorize,
    compute_fastdetect,
    compute_roughness,
    compute_roughness_windows,
    compute_shadow,
    normalize_roughness,
)


# ── Pure-math tests ─────────────────────────────────────────────
class TestRoughness:
    def test_flat_signal_zero_roughness(self):
        assert compute_roughness(np.array([1.0, 1.0, 1.0, 1.0])) == 0.0

    def test_single_jump_dominates(self):
        # diffs = [0, 0, 1, 0]; max_jump=1, median=0, MAD=0
        assert compute_roughness(np.array([0.0, 0.0, 0.0, 1.0, 1.0])) == pytest.approx(1.0)

    def test_short_input_returns_zero(self):
        assert compute_roughness(np.array([])) == 0.0
        assert compute_roughness(np.array([5.0])) == 0.0

    def test_mad_contributes_half_weight(self):
        # diffs = [1, 1, 1, 1] → max_jump=1, median=1, MAD=0
        # diffs = [-1, 1, -1, 1] → max_jump=1, median=0, MAD=1 → 1 + 0.5 = 1.5
        smooth = compute_roughness(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        zigzag = compute_roughness(np.array([0.0, -1.0, 0.0, -1.0, 0.0]))
        assert smooth == pytest.approx(1.0)
        assert zigzag == pytest.approx(1.5)


class TestRoughnessWindows:
    def test_short_input_one_window(self):
        windows = compute_roughness_windows(np.array([0.0, 1.0, 0.0]), window_size=50)
        assert len(windows) == 1
        assert windows[0][0] == 0

    def test_overlapping_windows_at_half_step(self):
        # 100 points, window_size=50, step=25 → starts at 0, 25, 50
        windows = compute_roughness_windows(np.zeros(100), window_size=50)
        starts = [start for start, _ in windows]
        assert starts == [0, 25, 50]


class TestNormalize:
    def test_caps_at_one(self):
        assert normalize_roughness(5.0, factor=1.2) == 1.0

    def test_below_cap(self):
        assert normalize_roughness(0.6, factor=1.2) == pytest.approx(0.5)


class TestFastDetect:
    def test_identical_logprobs_score_one(self):
        # If both models predict identically, ratio=1 everywhere → score=1
        lp = np.array([-0.1, -0.5, -0.2, -1.0, -0.3])
        assert compute_fastdetect(lp, lp) == pytest.approx(1.0)

    def test_disagreement_drops_score(self):
        lp1 = np.full(10, -0.01)  # very confident (prob ≈ 0.99)
        lp2 = np.full(10, -5.0)   # very uncertain (prob ≈ 0.0067)
        # ratio per position ≈ 0.0067 / 0.99 ≈ 0.0068
        assert compute_fastdetect(lp1, lp2) == pytest.approx(0.0068, abs=1e-3)

    def test_symmetric(self):
        lp1 = np.array([-0.1, -2.0, -0.5])
        lp2 = np.array([-1.5, -0.2, -0.3])
        assert compute_fastdetect(lp1, lp2) == pytest.approx(compute_fastdetect(lp2, lp1))


class TestShadow:
    def test_both_signals_required(self):
        # Pure multiplication — zero either factor and shadow vanishes
        assert compute_shadow(0.0, 0.5) == pytest.approx(0.5)  # (1-0)*0.5
        assert compute_shadow(1.0, 0.9) == pytest.approx(0.0)  # (1-1)*0.9
        assert compute_shadow(0.5, 0.0) == pytest.approx(0.0)  # (1-0.5)*0


class TestCategorize:
    """Researcher-revised 2x2 grid — verify every corner plus saturation."""

    def test_synthetic_high_fd_low_humanity(self):
        assert categorize(fastdetect=0.8, roughness_normalized=0.3) == "Synthetic"

    def test_mixed_signals_high_fd_high_humanity(self):
        # threshold_rg/norm_factor = 1.0/1.2 ≈ 0.833 — need rg_norm > that but < SATURATION_EPS
        assert categorize(fastdetect=0.8, roughness_normalized=0.9) == "Mixed signals"

    def test_human_low_fd_high_humanity(self):
        assert categorize(fastdetect=0.4, roughness_normalized=0.9) == "Human"

    def test_inconclusive_low_fd_low_humanity(self):
        assert categorize(fastdetect=0.4, roughness_normalized=0.3) == "Inconclusive"

    def test_saturation_refuses_verdict(self):
        # Saturated humanity axis → no verdict, regardless of fd
        assert categorize(fastdetect=0.8, roughness_normalized=1.0) == "Inconclusive (humanity saturated)"
        assert categorize(fastdetect=0.4, roughness_normalized=1.0) == "Inconclusive (humanity saturated)"


# ── Orchestration tests with fake scorers ───────────────────────
class FakeScorer:
    """Returns frozen logprobs regardless of input — for deterministic tests."""
    def __init__(self, logprobs: np.ndarray):
        self._logprobs = logprobs

    def logprobs(self, text, *, stride, max_chunk):
        return self._logprobs.copy()


def _long_enough_text(n_words: int = 60) -> str:
    return " ".join(["palabra"] * n_words)


class TestAnalyzeText:
    def test_short_text_returns_error(self):
        result = analyze_text("only three words here", FakeScorer(np.array([])), FakeScorer(np.array([])))
        assert "error" in result
        assert result["categoria"] is None

    def test_too_few_tokens_returns_error(self):
        # Long text but scorers return tiny arrays
        result = analyze_text(
            _long_enough_text(),
            FakeScorer(np.array([-0.1, -0.2])),
            FakeScorer(np.array([-0.1, -0.2])),
        )
        assert "error" in result
        assert "muy pocos tokens" in result["error"]

    def test_happy_path_identical_models_is_synthetic(self):
        # Identical logprobs → fastdetect=1.0; flat → humanity=0 → Synthetic
        # (fd=1.0 > 0.6 high; humanity_norm=0 < 0.833 low)
        flat = np.full(20, -0.5)
        result = analyze_text(_long_enough_text(), FakeScorer(flat), FakeScorer(flat))
        assert result["categoria"] == "Synthetic"
        assert result["fastdetect_score"] == pytest.approx(1.0)
        assert result["roughness_normalized"] == pytest.approx(0.0)
        assert result["shadow_score"] == pytest.approx(0.0)
        assert "diagnostico" in result

    def test_happy_path_disagreement_plus_jumps_is_saturated(self):
        # Humanity is computed on the BASE scorer's logprobs (notebook cell 18),
        # so the synthetic jump must live in the array passed to scorer_base.
        rng = np.random.default_rng(42)
        instruct = rng.uniform(-0.05, -0.01, 30)  # very confident everywhere
        base = np.concatenate([np.full(15, -3.0), np.full(15, -0.05)])  # big mid-text jump
        # Signature is (text, scorer_base, scorer_instruct). Swapping these would
        # silently pass numbers but produce the wrong category — guard against that.
        result = analyze_text(_long_enough_text(), FakeScorer(base), FakeScorer(instruct))
        # raw humanity ≈ 2.95, which / norm_factor=1.2 saturates at 1.0
        assert result["categoria"] == "Inconclusive (humanity saturated)"
        assert result["roughness_raw"] > 1.0  # the synthetic jump dominates

    def test_result_dict_has_notebook_compatible_keys(self):
        flat = np.full(20, -0.5)
        result = analyze_text(_long_enough_text(), FakeScorer(flat), FakeScorer(flat))
        # These keys must match the notebook so JSON reports stay interoperable
        for key in [
            "categoria",
            "fastdetect_score",
            "roughness_raw",
            "roughness_normalized",
            "shadow_score",
            "thresholds",
            "palabras",
            "tokens_analizados",
            "n_ventanas",
            "diagnostico",
        ]:
            assert key in result, f"missing key {key!r}"

    def test_verbose_flag_adds_serialisable_arrays(self):
        flat = np.full(20, -0.5)
        result = analyze_text(_long_enough_text(), FakeScorer(flat), FakeScorer(flat), verbose=True)
        assert isinstance(result["verbose_logprobs1"], list)
        assert isinstance(result["verbose_logprobs2"], list)
        assert all(isinstance(t, tuple) for t in result["verbose_ventanas"])
