"""
Optimal quote distances and spread formulas for the Avellaneda-Stoikov
market-making model with exponential order-arrival intensity.

Given an exponential arrival rate λ(δ) = A·exp(-k·δ), the first-order
conditions of the HJB equation yield closed-form optimal spreads.

Notation used throughout:
  δ^a : ask spread (distance from mid-price to posted ask)
  δ^b : bid spread (distance from mid-price to posted bid)
  p^a : posted ask price  = s + δ^a
  p^b : posted bid price  = s - δ^b
  r   : average reservation price  = s - q·γ·σ²·(T-t)
  h   : half-spread = (γσ²τ)/2 + (1/γ)ln(1+γ/k)
  k   : order-book liquidity / arrival-rate decay parameter
  γ   : absolute risk-aversion coefficient

Reference: Avellaneda & Stoikov (2008), Quantitative Finance 8(3), 217-224.
"""

from typing import Tuple

import numpy as np
import sympy as sp

from src.theory.frozen_inventory import average_reservation_price


# ---------------------------------------------------------------------------
# Spread formulas
# ---------------------------------------------------------------------------


def half_spread(
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> float:
    """Compute the symmetric half-spread h(t).

    h(t) = (γ·σ²·(T-t))/2 + (1/γ)·ln(1 + γ/k)

    The half-spread is the distance from the average reservation price r
    to each of the optimal quotes:  p^a = r + h,  p^b = r - h.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient (γ > 0).
    sigma : float
        Mid-price volatility (σ > 0).
    T : float
        Terminal horizon.
    t : float
        Current time (t <= T).
    k : float
        Arrival-rate decay parameter (k > 0).

    Returns
    -------
    float
        Half-spread h(t).
    """
    tau = T - t
    return (gamma * sigma**2 * tau) / 2.0 + (1.0 / gamma) * np.log(1.0 + gamma / k)


def total_spread(
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> float:
    """Compute the total quoted spread δ^a + δ^b.

    δ^a + δ^b = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)

    Note this equals 2·h(t) and is independent of inventory q.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.
    t : float
        Current time.
    k : float
        Arrival-rate decay parameter.

    Returns
    -------
    float
        Total quoted spread.
    """
    tau = T - t
    return gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)


def delta_ask(
    q: float,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> float:
    """Compute the optimal ask distance from the mid-price.

    δ^a = (1/γ)·ln(1 + γ/k) + ((1 - 2q)·γ·σ²·(T-t))/2

    The inventory-dependent term shifts δ^a: positive inventory (long)
    tightens the ask (market maker wants to sell), negative inventory widens it.

    Parameters
    ----------
    q : float
        Current inventory.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.
    t : float
        Current time.
    k : float
        Arrival-rate decay parameter.

    Returns
    -------
    float
        Optimal ask spread δ^a.
    """
    tau = T - t
    return (1.0 / gamma) * np.log(1.0 + gamma / k) + ((1.0 - 2.0 * q) * gamma * sigma**2 * tau) / 2.0


def delta_bid(
    q: float,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> float:
    """Compute the optimal bid distance from the mid-price.

    δ^b = (1/γ)·ln(1 + γ/k) + ((1 + 2q)·γ·σ²·(T-t))/2

    Positive inventory widens δ^b (market maker is reluctant to buy more),
    negative inventory tightens it.

    Parameters
    ----------
    q : float
        Current inventory.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.
    t : float
        Current time.
    k : float
        Arrival-rate decay parameter.

    Returns
    -------
    float
        Optimal bid spread δ^b.
    """
    tau = T - t
    return (1.0 / gamma) * np.log(1.0 + gamma / k) + ((1.0 + 2.0 * q) * gamma * sigma**2 * tau) / 2.0


def quoted_ask(
    s: float,
    q: float,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> float:
    """Compute the posted ask price p^a.

    p^a = s + δ^a

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.
    t : float
        Current time.
    k : float
        Arrival-rate decay parameter.

    Returns
    -------
    float
        Posted ask price p^a.
    """
    return s + delta_ask(q, gamma, sigma, T, t, k)


def quoted_bid(
    s: float,
    q: float,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> float:
    """Compute the posted bid price p^b.

    p^b = s - δ^b

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.
    t : float
        Current time.
    k : float
        Arrival-rate decay parameter.

    Returns
    -------
    float
        Posted bid price p^b.
    """
    return s - delta_bid(q, gamma, sigma, T, t, k)


def verify_quote_equivalence(
    s: float,
    q: float,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
    tol: float = 1e-10,
) -> bool:
    """Verify that p^a = r + h and p^b = r - h.

    The average reservation price r and half-spread h decompose the
    optimal quotes symmetrically around r:
      p^a = r + h
      p^b = r - h

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.
    t : float
        Current time.
    k : float
        Arrival-rate decay parameter.
    tol : float, optional
        Absolute tolerance for np.isclose.

    Returns
    -------
    bool
        True if both p^a = r + h and p^b = r - h hold within tolerance.
    """
    r = average_reservation_price(s, q, t, T, gamma, sigma)
    h = half_spread(gamma, sigma, T, t, k)
    pa = quoted_ask(s, q, gamma, sigma, T, t, k)
    pb = quoted_bid(s, q, gamma, sigma, T, t, k)
    ask_ok = bool(np.isclose(pa, r + h, atol=tol))
    bid_ok = bool(np.isclose(pb, r - h, atol=tol))
    return ask_ok and bid_ok


def constant_spread_component(gamma: float, k: float) -> float:
    """Return the inventory-independent part of the total spread.

    This is the component of the optimal spread that does not depend on
    inventory q or time-to-maturity τ:

        (2/γ)·ln(1 + γ/k)

    It arises solely from the arrival-rate structure and persists even at
    the terminal time T (where the inventory-risk component vanishes).

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    k : float
        Arrival-rate decay parameter.

    Returns
    -------
    float
        Constant spread component (2/γ)·ln(1 + γ/k) > 0.
    """
    return (2.0 / gamma) * np.log(1.0 + gamma / k)


# ---------------------------------------------------------------------------
# Symbolic FOC analysis
# ---------------------------------------------------------------------------


def foc_general_symbolic() -> sp.Expr:
    """Return the general first-order condition for the bid spread symbolically.

    For a general intensity λ^b(δ^b), the FOC of the HJB optimization
    gives the implicit equation:

      δ^b - (1/γ)·ln(1 - γ·λ^b(δ^b) / (dλ^b/dδ^b)) = s - r^b

    where r^b is the bid reservation price.  The left-hand side encodes
    the trade-off between earning more per trade (larger δ^b) and
    attracting fewer executions (λ^b falls with δ^b).

    Returns
    -------
    sympy.Expr
        The left-hand side of the FOC:
        δ^b - (1/γ)·ln(1 - γ·λ^b / (dλ^b/dδ^b))
        expressed as a sympy Equality (lhs = s - r^b).
    """
    gam, delta_b, s_sym, r_b = sp.symbols("gamma delta^b s r^b", positive=True)
    lam_b = sp.Function("lambda^b")(delta_b)
    dlam_b = sp.diff(lam_b, delta_b)

    lhs = delta_b - (1 / gam) * sp.log(1 - gam * lam_b / dlam_b)
    rhs = s_sym - r_b

    foc_eq = sp.Eq(lhs, rhs)

    print("=" * 60)
    print("GENERAL FOC (BID SIDE)")
    print("=" * 60)
    print(f"\nFOC: {foc_eq}")
    print("\nInterpretation:")
    print("  delta^b - (1/gamma)*ln(1 - gamma*lambda/(dlambda/ddelta)) = s - r^b")
    print("  The optimal spread balances marginal gain vs. arrival loss.")
    print("=" * 60)

    return foc_eq


def foc_exponential_simplification() -> sp.Expr:
    """Simplify the FOC for exponential intensity λ(δ) = A·exp(-k·δ).

    With λ^b(δ^b) = A·exp(-k·δ^b):
      dλ^b/dδ^b = -k·A·exp(-k·δ^b)
      λ^b / (dλ^b/dδ^b) = A·exp(-k·δ^b) / (-k·A·exp(-k·δ^b)) = -1/k

    Substituting into the general FOC:
      δ^b - (1/γ)·ln(1 - γ·(-1/k)) = s - r^b
      δ^b - (1/γ)·ln(1 + γ/k) = s - r^b
      δ^b = (s - r^b) + (1/γ)·ln(1 + γ/k)

    Returns
    -------
    sympy.Expr
        The simplified ratio λ/(dλ/dδ) = -1/k, as a sympy expression.
        Also prints the fully simplified FOC.
    """
    gam, k, delta_b, A, s_sym, r_b = sp.symbols(
        "gamma k delta^b A s r^b", positive=True
    )

    lam = A * sp.exp(-k * delta_b)
    dlam = sp.diff(lam, delta_b)
    ratio = sp.simplify(lam / dlam)   # should be -1/k

    # Simplified FOC
    lhs_simplified = delta_b - (1 / gam) * sp.log(1 - gam * ratio)
    lhs_expanded = sp.simplify(lhs_simplified)
    foc_simplified = sp.Eq(lhs_expanded, s_sym - r_b)

    # Explicit optimal delta^b
    opt_delta_b = sp.solve(foc_simplified, delta_b)[0]

    print("=" * 60)
    print("EXPONENTIAL INTENSITY FOC SIMPLIFICATION")
    print("=" * 60)
    print(f"\nlambda(delta) = A * exp(-k * delta)")
    print(f"  lambda / (d_lambda/d_delta) = {ratio}")
    print(f"\nSimplified FOC: {foc_simplified}")
    print(f"\nOptimal delta^b = {opt_delta_b}")
    print("=" * 60)

    return ratio
