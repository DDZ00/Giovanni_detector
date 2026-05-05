"""Differential-calculus signals on the log-probability curve l(t).

This is the math module behind the paper's central thesis: discrete-calculus
signals on l(t) separate AI-generated from human-generated text. Each function
takes a 1-D logprob array and returns a single scalar.

Convention: all 'first-derivative' signals are computed on np.diff(l). Padding
zeros are never inserted — short inputs return 0.0 instead of throwing.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _diff1(l: np.ndarray) -> np.ndarray:
    return np.diff(l) if len(l) >= 2 else np.zeros(0, dtype=l.dtype)


def _diff2(l: np.ndarray) -> np.ndarray:
    return np.diff(l, n=2) if len(l) >= 3 else np.zeros(0, dtype=l.dtype)


def total_variation(l: np.ndarray) -> float:
    """L¹ norm of l′ — total path length of l."""
    d = _diff1(l)
    return float(np.sum(np.abs(d))) if d.size else 0.0


def l2_energy(l: np.ndarray) -> float:
    """L² norm of l′ — Dirichlet energy proxy."""
    d = _diff1(l)
    return float(np.sqrt(np.sum(d * d))) if d.size else 0.0


def linf_first(l: np.ndarray) -> float:
    """L∞ of l′ — Lipschitz constant proxy. Identical to the analyzer's max_jump term."""
    d = _diff1(l)
    return float(np.max(np.abs(d))) if d.size else 0.0


def linf_second(l: np.ndarray) -> float:
    """L∞ of l″ — peak curvature."""
    d = _diff2(l)
    return float(np.max(np.abs(d))) if d.size else 0.0


def mad_first(l: np.ndarray) -> float:
    """Median absolute deviation of l′ — robust dispersion of the derivative."""
    d = _diff1(l)
    if d.size == 0:
        return 0.0
    med = float(np.median(d))
    return float(np.median(np.abs(d - med)))


def _gaussian_smooth(l: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(l.astype(np.float64), sigma=sigma)
    except ImportError:
        # Fallback: tiny Gaussian via numpy convolution (kernel ±3σ).
        radius = max(1, int(round(3 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return np.convolve(l.astype(np.float64), kernel, mode="same")


def smoothed_tv(l: np.ndarray, sigma: float = 2.0) -> float:
    """Total variation of l after Gaussian smoothing — strips high-freq noise to expose macro shape."""
    if len(l) < 2:
        return 0.0
    return total_variation(_gaussian_smooth(l, sigma))


def smoothed_linf(l: np.ndarray, sigma: float = 2.0) -> float:
    """L∞ of derivative of smoothed l."""
    if len(l) < 2:
        return 0.0
    return linf_first(_gaussian_smooth(l, sigma))


def compute_all(logprobs1: np.ndarray, logprobs2: np.ndarray) -> Dict[str, float]:
    """Compute the 12-element differential vector for a (instruct, base) logprob pair.

    Naming convention: trailing '1' = instruct model, '2' = base model.
    """
    return {
        "tv1": total_variation(logprobs1),
        "tv2": total_variation(logprobs2),
        "l2_1": l2_energy(logprobs1),
        "l2_2": l2_energy(logprobs2),
        "linf1_1": linf_first(logprobs1),
        "linf1_2": linf_first(logprobs2),
        "linf2_1": linf_second(logprobs1),
        "linf2_2": linf_second(logprobs2),
        "mad1": mad_first(logprobs1),
        "mad2": mad_first(logprobs2),
        "stv2": smoothed_tv(logprobs2),
        "slinf2": smoothed_linf(logprobs2),
    }
