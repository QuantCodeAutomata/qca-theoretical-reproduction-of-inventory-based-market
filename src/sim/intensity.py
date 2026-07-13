"""
Fill-intensity functions for the Avellaneda-Stoikov market-making model.

The model assumes limit-order fill intensity follows λ(δ) = A·exp(-k·δ),
where δ is the half-spread quoted by the market maker.
"""

from __future__ import annotations

import numpy as np


def lambda_from_delta(
    delta: float | np.ndarray,
    A: float,
    k: float,
) -> float | np.ndarray:
    """Compute the fill intensity λ = A · exp(-k · δ).

    Parameters
    ----------
    delta:
        Half-spread (distance from mid to quoted price).  May be a scalar
        or any numpy array.
    A:
        Baseline arrival rate (events per unit time at δ=0).
    k:
        Sensitivity of arrival rate to the quoted spread.

    Returns
    -------
    float or np.ndarray
        Fill intensity; always positive because of the exponential form.
    """
    return A * np.exp(-k * delta)


def validate_monotonicity(deltas: np.ndarray, A: float, k: float) -> bool:
    """Check that fill intensity is strictly decreasing in δ.

    Parameters
    ----------
    deltas:
        Sorted (ascending) array of half-spread values to evaluate.
    A:
        Baseline arrival rate.
    k:
        Spread sensitivity parameter.

    Returns
    -------
    bool
        True if λ(δ) is strictly decreasing across all supplied δ values.
    """
    lambdas = lambda_from_delta(deltas, A, k)
    return bool(np.all(np.diff(lambdas) < 0))


def poisson_fill_probability(lam: float | np.ndarray, dt: float) -> float | np.ndarray:
    """Exact Poisson probability of at least one fill in interval dt.

    P(N ≥ 1) = 1 - exp(-λ · dt)

    Parameters
    ----------
    lam:
        Fill intensity (events per unit time).
    dt:
        Length of the time step.

    Returns
    -------
    float or np.ndarray
        Probability of at least one fill event in [0, dt].
    """
    return 1.0 - np.exp(-lam * dt)


def bernoulli_fill_probability(lam: float | np.ndarray, dt: float) -> float | np.ndarray:
    """Bernoulli (first-order) approximation of fill probability in interval dt.

    P ≈ λ · dt

    This is the leading-order Taylor expansion of the exact Poisson probability
    and is valid when λ · dt ≪ 1.

    Parameters
    ----------
    lam:
        Fill intensity (events per unit time).
    dt:
        Length of the time step.

    Returns
    -------
    float or np.ndarray
        Approximate probability of a fill in [0, dt].
    """
    return lam * dt
