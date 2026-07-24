"""Exact tests, multiplicity correction, and deterministic effect estimates."""

from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray


def stable_seed(namespace: str, *items: object) -> int:
    """Return the validated process-independent SHA-256-derived seed."""
    raw = "|".join([namespace, *map(str, items)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def exact_sign_flip_test(values: ArrayLike) -> float:
    """Two-sided exact randomization p-value over all sign assignments.

    Non-finite values are discarded. Ties at the observed absolute mean are
    included, with the same 1e-12 numerical tolerance as the validated audit.
    """
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan")
    observed = abs(float(sample.mean()))
    extreme = 0
    total = 2 ** sample.size
    for signs in product((-1.0, 1.0), repeat=sample.size):
        statistic = abs(float(np.mean(sample * np.asarray(signs))))
        extreme += statistic >= observed - 1e-12
    return extreme / total


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Apply Benjamini-Hochberg FDR correction, preserving NaN positions."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    if ((pv < 0) | (pv > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    q[valid] = restored
    return q


def cohens_d(group_one: ArrayLike, group_zero: ArrayLike) -> float:
    """Cohen's d using the sample-size-weighted pooled sample variance."""
    one = np.asarray(group_one, dtype=float)
    zero = np.asarray(group_zero, dtype=float)
    one, zero = one[np.isfinite(one)], zero[np.isfinite(zero)]
    if len(one) < 2 or len(zero) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(one) - 1) * one.var(ddof=1) + (len(zero) - 1) * zero.var(ddof=1))
        / (len(one) + len(zero) - 2)
    )
    return float("nan") if pooled == 0 or not np.isfinite(pooled) else float((one.mean() - zero.mean()) / pooled)


def participant_bootstrap_cohens_d(
    group_one: ArrayLike,
    group_zero: ArrayLike,
    *,
    n_bootstrap: int = 5000,
    seed: int,
) -> dict[str, float | int | bool]:
    """Stratified participant bootstrap with validated percentile/NA rules."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    one = np.asarray(group_one, dtype=float)
    zero = np.asarray(group_zero, dtype=float)
    one, zero = one[np.isfinite(one)], zero[np.isfinite(zero)]
    observed = cohens_d(one, zero)
    values: list[float] = []
    if len(one) >= 2 and len(zero) >= 2 and np.isfinite(observed):
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            values.append(
                cohens_d(
                    one[rng.integers(0, len(one), len(one))],
                    zero[rng.integers(0, len(zero), len(zero))],
                )
            )
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    low, high = (np.percentile(valid, [2.5, 97.5]) if valid.size else (np.nan, np.nan))
    invalid_count = n_bootstrap - len(valid)
    unavailable = (
        not np.all(np.isfinite([observed, low, high]))
        or np.count_nonzero(one) + np.count_nonzero(zero) < 2
        or invalid_count / n_bootstrap > 0.25
        or np.isclose(low, observed, atol=1e-12, rtol=0)
        or np.isclose(high, observed, atol=1e-12, rtol=0)
    )
    return {
        "cohens_d": float(observed),
        "ci_low": float("nan") if unavailable else float(low),
        "ci_high": float("nan") if unavailable else float(high),
        "interval_unavailable": bool(unavailable),
        "n_valid_bootstrap": int(len(valid)),
        "n_invalid_bootstrap": int(invalid_count),
        "seed": int(seed),
    }
