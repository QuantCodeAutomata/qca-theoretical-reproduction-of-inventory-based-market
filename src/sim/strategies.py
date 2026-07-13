"""
Quoting strategies for the Avellaneda-Stoikov market-making model.

Two strategies are implemented:
- **Inventory-aware**: The reservation price adjusts for the current inventory,
  so the market maker skews quotes to reduce position risk.
- **Symmetric**: Quotes are centred at the mid-price regardless of inventory,
  replicating a naive market-maker benchmark.
"""

from __future__ import annotations

import numpy as np


def half_spread(
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> float:
    """Compute the optimal half-spread h from the Avellaneda-Stoikov formula.

    h = ½ · γ · σ² · (T - t) + (1/γ) · ln(1 + γ/k)

    The first term accounts for inventory risk over the remaining horizon;
    the second is a constant that depends only on risk aversion and market
    impact.

    Parameters
    ----------
    gamma:
        Absolute risk-aversion coefficient (γ > 0).
    sigma:
        Mid-price volatility per unit time.
    T:
        Terminal time (end of trading horizon).
    t:
        Current time (0 ≤ t ≤ T).
    k:
        Fill-intensity sensitivity parameter.

    Returns
    -------
    float
        Optimal half-spread h > 0.
    """
    inventory_risk_term = 0.5 * gamma * sigma**2 * (T - t)
    constant_term = np.log(1.0 + gamma / k) / gamma
    return inventory_risk_term + constant_term


def inventory_center(
    S: float,
    q: float,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
) -> float:
    """Compute the reservation (indifference) price r.

    r = S - q · γ · σ² · (T - t)

    A positive inventory (q > 0) lowers r below the mid-price S, causing the
    market maker to skew quotes downward to encourage selling.

    Parameters
    ----------
    S:
        Current mid-price.
    q:
        Current inventory (signed; positive = long).
    gamma:
        Absolute risk-aversion coefficient.
    sigma:
        Mid-price volatility per unit time.
    T:
        Terminal time.
    t:
        Current time.

    Returns
    -------
    float
        Reservation price r.
    """
    return S - q * gamma * sigma**2 * (T - t)


def inventory_quotes(
    S: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> dict:
    """Compute inventory-aware bid and ask quotes.

    The ask and bid are placed symmetrically around the reservation price r
    at distance h (the optimal half-spread):

        p_ask = r + h,  p_bid = r - h
        δ_ask = p_ask - S,  δ_bid = S - p_bid

    Parameters
    ----------
    S:
        Current mid-price.
    q:
        Current inventory.
    t:
        Current time.
    T:
        Terminal time.
    gamma:
        Absolute risk-aversion coefficient.
    sigma:
        Mid-price volatility.
    k:
        Fill-intensity sensitivity parameter.

    Returns
    -------
    dict
        Keys: ``r``, ``h``, ``p_ask``, ``p_bid``, ``delta_ask``, ``delta_bid``.
    """
    r = inventory_center(S, q, gamma, sigma, T, t)
    h = half_spread(gamma, sigma, T, t, k)
    p_ask = r + h
    p_bid = r - h
    delta_ask = p_ask - S
    delta_bid = S - p_bid
    return {
        'r': r,
        'h': h,
        'p_ask': p_ask,
        'p_bid': p_bid,
        'delta_ask': delta_ask,
        'delta_bid': delta_bid,
    }


def symmetric_quotes(
    S: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> dict:
    """Compute symmetric (inventory-blind) bid and ask quotes.

    Both quotes are placed equidistant from the mid-price S:

        p_ask = S + h,  p_bid = S - h
        δ_ask = δ_bid = h

    Parameters
    ----------
    S:
        Current mid-price.
    t:
        Current time.
    T:
        Terminal time.
    gamma:
        Absolute risk-aversion coefficient.
    sigma:
        Mid-price volatility.
    k:
        Fill-intensity sensitivity parameter.

    Returns
    -------
    dict
        Keys: ``p_ask``, ``p_bid``, ``delta_ask``, ``delta_bid``.
    """
    h = half_spread(gamma, sigma, T, t, k)
    p_ask = S + h
    p_bid = S - h
    return {
        'p_ask': p_ask,
        'p_bid': p_bid,
        'delta_ask': h,
        'delta_bid': h,
    }


def constant_spread_component(gamma: float, k: float) -> float:
    """Return the time-invariant part of the total quoted spread.

    spread_constant = (2/γ) · ln(1 + γ/k)

    This equals the full quoted spread (2h) at t = T (i.e. when the
    inventory-risk term vanishes) and serves as the representative spread
    figure reported in the paper's comparison table.

    Parameters
    ----------
    gamma:
        Absolute risk-aversion coefficient.
    k:
        Fill-intensity sensitivity parameter.

    Returns
    -------
    float
        Constant component of the total spread.
    """
    return (2.0 / gamma) * np.log(1.0 + gamma / k)
