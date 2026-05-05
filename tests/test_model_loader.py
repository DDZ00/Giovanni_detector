"""Unit tests for LlamaCppScorer and build_scorer_pair (no CUDA, no real model).

The Llama wheel is never imported: tests inject a fake `_llama` directly into
LlamaCppScorer instances, which short-circuits the lazy `from llama_cpp import Llama`.
"""
import numpy as np
import pytest

from modules.model_loader import LlamaCppScorer, build_scorer_pair


class _FakeLlama:
    """Minimal stand-in: tokenize, reset, eval, .scores — nothing else."""

    def __init__(self, token_seq, logits_per_eval):
        self._token_seq = token_seq
        self._logits = np.asarray(logits_per_eval, dtype=np.float32)
        self.reset_calls = 0
        self.eval_calls = []

    def tokenize(self, raw_bytes, add_bos=True):
        return list(self._token_seq)

    def reset(self):
        self.reset_calls += 1

    def eval(self, chunk):
        self.eval_calls.append(list(chunk))

    @property
    def scores(self):
        return self._logits


# ── LlamaCppScorer lifecycle ────────────────────────────────────
class TestLlamaCppScorerLifecycle:
    def test_init_does_not_load_model(self, tmp_path):
        scorer = LlamaCppScorer(tmp_path / "nonexistent.gguf")
        assert scorer._llama is None

    def test_unload_is_idempotent(self, tmp_path):
        scorer = LlamaCppScorer(tmp_path / "x.gguf")
        scorer.unload()  # never loaded — no-op
        scorer._llama = object()
        scorer.unload()
        assert scorer._llama is None
        scorer.unload()  # second call also fine
        assert scorer._llama is None


# ── LlamaCppScorer.logprobs ─────────────────────────────────────
class TestLlamaCppScorerLogprobs:
    def _build(self, tmp_path, fake):
        scorer = LlamaCppScorer(tmp_path / "x.gguf")
        scorer._llama = fake  # bypass _ensure_loaded → no llama_cpp import
        return scorer

    def test_uniform_logits_give_minus_log_vocab(self, tmp_path):
        # Vocab=4 uniform → log_softmax = -log(4) at every position.
        # Tokens [0,1,2] → 2 logprobs returned (positions 0 and 1).
        fake = _FakeLlama(token_seq=[0, 1, 2], logits_per_eval=np.zeros((3, 4)))
        scorer = self._build(tmp_path, fake)

        out = scorer.logprobs("dummy text")

        assert out.shape == (2,)
        np.testing.assert_allclose(out, [-np.log(4)] * 2, rtol=1e-5)

    def test_short_input_returns_empty(self, tmp_path):
        fake = _FakeLlama(token_seq=[42], logits_per_eval=np.zeros((1, 4)))
        scorer = self._build(tmp_path, fake)
        out = scorer.logprobs("a")
        assert out.shape == (0,)
        assert out.dtype == np.float32

    def test_chunking_concatenates_results(self, tmp_path):
        # 4 tokens, max_chunk=2 → chunks [[t0,t1],[t2,t3]] → 1+1 = 2 logprobs.
        fake = _FakeLlama(token_seq=[0, 1, 2, 3], logits_per_eval=np.zeros((2, 4)))
        scorer = self._build(tmp_path, fake)

        out = scorer.logprobs("xxxx", max_chunk=2)

        assert out.shape == (2,)
        np.testing.assert_allclose(out, [-np.log(4), -np.log(4)], rtol=1e-5)
        # reset() must be called once per chunk so KV cache doesn't leak across chunks.
        assert fake.reset_calls == 2
        assert fake.eval_calls == [[0, 1], [2, 3]]

    def test_targeted_logit_dominates(self, tmp_path):
        # Vocab=3, 2 tokens [t0=0, t1=2]. Position-0 logits make token 2 dominant
        # (logit 10 vs 0 elsewhere) → logprob ≈ 0; far above uniform -log(3).
        logits = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        fake = _FakeLlama(token_seq=[0, 2], logits_per_eval=logits)
        scorer = self._build(tmp_path, fake)

        out = scorer.logprobs("xx")

        assert out.shape == (1,)
        assert out[0] > -0.001  # very close to 0 — model "expected" token 2

    def test_stride_arg_is_accepted_but_ignored(self, tmp_path):
        fake = _FakeLlama(token_seq=[0, 1, 2], logits_per_eval=np.zeros((3, 4)))
        scorer = self._build(tmp_path, fake)
        a = scorer.logprobs("x", stride=1)
        fake2 = _FakeLlama(token_seq=[0, 1, 2], logits_per_eval=np.zeros((3, 4)))
        scorer2 = self._build(tmp_path, fake2)
        b = scorer2.logprobs("x", stride=99)
        np.testing.assert_array_equal(a, b)


# ── build_scorer_pair ───────────────────────────────────────────
class TestBuildScorerPair:
    def test_distinct_paths_pass(self, tmp_path):
        a = tmp_path / "base.gguf"
        b = tmp_path / "instruct.gguf"
        a.write_bytes(b"")
        b.write_bytes(b"")
        base, instruct = build_scorer_pair(base_path=a, instruct_path=b)
        assert base.gguf_path == a
        assert instruct.gguf_path == b

    def test_same_file_refused(self, tmp_path):
        a = tmp_path / "model.gguf"
        a.write_bytes(b"")
        with pytest.raises(RuntimeError, match="degenerate"):
            build_scorer_pair(base_path=a, instruct_path=a)
