"""Lazy module-level scorer singleton with pair-id and n_ctx awareness.

Why this is its own module: tests monkey-patch `get_scorers` to inject fakes,
and the singleton state needs a stable place to live across requests. App-level
globals would re-init on each gunicorn worker fork; this module-level cache
loads once per worker, on first /analyze call.

Phase 2: get_scorers(n_ctx=...) supports hot context-size changes.
2026-05-01: get_scorers(pair_id=...) supports hot model-pair swaps via the
Config.MODEL_PAIRS preset registry. Either change forces a model reload
(~25 s) because llama-cpp's Llama context and weights are fixed at construction.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

from config import Config
from modules.model_loader import LlamaCppScorer, build_scorer_pair

DEFAULT_N_CTX = 2048

_scorers: Optional[Tuple[LlamaCppScorer, LlamaCppScorer]] = None
_current_pair_id: Optional[str] = None
_current_n_ctx: Optional[int] = None
_lock = threading.Lock()


def _resolve_pair(pair_id: str) -> Tuple[str, str]:
    spec = Config.MODEL_PAIRS.get(pair_id)
    if not spec:
        raise ValueError(
            f"unknown pair_id {pair_id!r}; registered: {sorted(Config.MODEL_PAIRS)}"
        )
    return spec["base"], spec["instruct"]


def get_scorers(
    pair_id: str = None,
    n_ctx: int = DEFAULT_N_CTX,
) -> Tuple[LlamaCppScorer, LlamaCppScorer]:
    """Return the (base, instruct) pair, loading on first call or on (pair_id, n_ctx) change."""
    global _scorers, _current_pair_id, _current_n_ctx
    pair_id = pair_id or Config.DEFAULT_PAIR_ID
    if _scorers is not None and _current_pair_id == pair_id and _current_n_ctx == n_ctx:
        return _scorers
    with _lock:
        if _scorers is not None and _current_pair_id == pair_id and _current_n_ctx == n_ctx:
            return _scorers
        if _scorers is not None:
            for s in _scorers:
                try:
                    s.unload()
                except Exception:
                    pass
        base_path, instruct_path = _resolve_pair(pair_id)
        _scorers = build_scorer_pair(base_path=base_path, instruct_path=instruct_path, n_ctx=n_ctx)
        _current_pair_id = pair_id
        _current_n_ctx = n_ctx
    return _scorers


def reset_scorers() -> None:
    """Test hook: drop the cached pair so the next call rebuilds."""
    global _scorers, _current_pair_id, _current_n_ctx
    _scorers = None
    _current_pair_id = None
    _current_n_ctx = None


def list_pairs():
    """For the UI dropdown: return [{id, label}, ...]."""
    return [{"id": pid, "label": spec["label"]} for pid, spec in Config.MODEL_PAIRS.items()]
