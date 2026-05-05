"""
LlamaCppScorer — loads a GGUF model in-process and returns per-token logprobs.

Giovanni owns its model files (mounted at Config.MODELS_DIR). No coupling to
Ollama's storage layout: a path is a path.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class LlamaCppScorer:
    """Scorer protocol implementation backed by a single GGUF model.

    Loaded lazily on first .logprobs() call so import-time cost is zero and
    /ping stays fast. Thread-safe: the underlying Llama instance is not
    re-entrant, so calls are serialized with a per-instance lock.
    """

    def __init__(
        self,
        gguf_path: Path | str,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.gguf_path = Path(gguf_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.verbose = verbose
        self._llama = None  # type: Optional[object]
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._llama is not None:
            return
        # Imported lazily so unit tests that monkey-patch _llama don't need the wheel.
        from llama_cpp import Llama

        if not self.gguf_path.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {self.gguf_path}")

        logger.info("Loading GGUF: %s (n_ctx=%d, gpu_layers=%d)", self.gguf_path, self.n_ctx, self.n_gpu_layers)
        self._llama = Llama(
            model_path=str(self.gguf_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            logits_all=True,  # required: we need logits for every prompt position, not just the last
            seed=self.seed,
            verbose=self.verbose,
        )

    def unload(self):
        """Release VRAM. Idempotent."""
        with self._lock:
            if self._llama is not None:
                logger.info("Unloading GGUF: %s", self.gguf_path)
                self._llama = None  # llama_cpp.Llama.__del__ frees the C-side context

    def tokenize(self, text: str) -> list[int]:
        """Return token IDs (with BOS) for the given text. Loads the model on first call."""
        with self._lock:
            self._ensure_loaded()
            return self._llama.tokenize(text.encode("utf-8"), add_bos=True)

    def logprobs(self, text: str, *, stride: int = 10, max_chunk: int = 2048) -> np.ndarray:
        """Return per-token logprob of each token given its left context.

        Convenience wrapper — equivalent to score_full(text)["logprobs"].
        Kept for callers (calculus signals) that don't need the full distribution.
        """
        del stride
        return self.score_full(text, max_chunk=max_chunk)["logprobs"]

    def score_full(self, text: str, *, max_chunk: int = 2048) -> dict:
        """One-pass scoring: return aligned tokens, per-token actual-token logprobs,
        and the full per-token log-softmax distribution matrix.

        Output keys:
            tokens     — (L,) int64,  the predicted-position tokens (chunk[1:] across chunks)
            logprobs   — (L,) float32, log p_M(actual_token | left_context)
            log_distrib— (L, V) float32, full log-softmax over the vocabulary

        L is total predicted positions across chunks; the first token of each chunk
        is dropped because it has no left context (consistent with the notebook).
        Memory: at Falcon-7B vocab (~65k) and L≈700, log_distrib ≈ 180 MB per call.
        """
        with self._lock:
            self._ensure_loaded()
            llama = self._llama

            tokens: list[int] = llama.tokenize(text.encode("utf-8"), add_bos=True)
            if len(tokens) < 2:
                return {
                    "tokens": np.zeros(0, dtype=np.int64),
                    "logprobs": np.zeros(0, dtype=np.float32),
                    "log_distrib": np.zeros((0, 0), dtype=np.float32),
                }

            chunks = [tokens[i : i + max_chunk] for i in range(0, len(tokens), max_chunk)]
            out_tokens: list[np.ndarray] = []
            out_logprobs: list[np.ndarray] = []
            out_distrib: list[np.ndarray] = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                llama.reset()
                llama.eval(chunk)
                logits = np.asarray(llama.scores[: len(chunk)], dtype=np.float32)
                # log-softmax along vocab axis, stable form
                m = logits.max(axis=-1, keepdims=True)
                shifted = logits - m
                log_distrib_full = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
                # Position i predicts token i+1; drop the last position (nothing to predict).
                log_distrib = log_distrib_full[:-1].astype(np.float32)
                next_token_ids = np.asarray(chunk[1:], dtype=np.int64)
                positions = np.arange(len(chunk) - 1)
                logprobs = log_distrib[positions, next_token_ids]
                out_tokens.append(next_token_ids)
                out_logprobs.append(logprobs.astype(np.float32))
                out_distrib.append(log_distrib)

            if not out_tokens:
                return {
                    "tokens": np.zeros(0, dtype=np.int64),
                    "logprobs": np.zeros(0, dtype=np.float32),
                    "log_distrib": np.zeros((0, 0), dtype=np.float32),
                }
            return {
                "tokens": np.concatenate(out_tokens),
                "logprobs": np.concatenate(out_logprobs),
                "log_distrib": np.concatenate(out_distrib, axis=0),
            }


def build_scorer_pair(
    *,
    base_path: Path | str = None,
    instruct_path: Path | str = None,
    n_ctx: int = 2048,
) -> tuple[LlamaCppScorer, LlamaCppScorer]:
    """Build the (base, instruct) pair from configured GGUF paths.

    Refuses to start if both paths resolve to the same file — that would make
    Binoculars' X-PPL collapse to PPL and the score collapse to 1.0 for every
    input.
    """
    base_path = Path(base_path or Config.MODEL_BASE_PATH)
    instruct_path = Path(instruct_path or Config.MODEL_INSTRUCT_PATH)
    if base_path.resolve() == instruct_path.resolve():
        raise RuntimeError(
            f"Model pair degenerate: base and instruct resolve to the same file "
            f"({base_path}). Binoculars X-PPL would collapse to PPL — refusing to start."
        )
    return (
        LlamaCppScorer(base_path, n_ctx=n_ctx),
        LlamaCppScorer(instruct_path, n_ctx=n_ctx),
    )
