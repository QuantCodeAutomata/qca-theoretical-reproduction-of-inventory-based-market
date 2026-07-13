"""
Frozen-inventory value function and reservation prices for the
Avellaneda-Stoikov market-making model.

In the "frozen-inventory" approximation the mid-price still follows
Brownian motion but inventory q is treated as a fixed parameter when
solving the HJB equation.  The resulting value function has an explicit
closed-form and directly yields the reservation (indifference) prices.

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a
limit order book", Quantitative Finance 8(3), 217-224.
"""

from typing import Dict, Tuple

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Numerical functions
# ---------------------------------------------------------------------------


def frozen_inventory_value(
    x: float,
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Compute the frozen-inventory value function.

    v(x, s, q, t) = -exp(-γx) · exp(-γqs) · exp(γ²q²σ²(T-t)/2)

    Parameters
    ----------
    x : float
        Cash wealth of the market maker.
    s : float
        Current mid-price of the asset.
    q : float
        Inventory (signed number of shares held).
    t : float
        Current time.
    T : float
        Terminal horizon time (T >= t).
    gamma : float
        Absolute risk-aversion coefficient (γ > 0).
    sigma : float
        Volatility of the mid-price process (σ > 0).

    Returns
    -------
    float
        The value of the frozen-inventory utility function at (x, s, q, t).
        Always strictly negative.
    """
    tau = T - t
    log_neg_v = (
        -gamma * x
        - gamma * q * s
        + (gamma**2 * q**2 * sigma**2 * tau) / 2.0
    )
    return -np.exp(log_neg_v)


def reservation_ask(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Compute the ask reservation (indifference) price.

    r^a = s + ((1 - 2q) · γ · σ² · (T-t)) / 2

    This is the minimum ask price at which the market maker is indifferent
    between selling one unit now versus not, i.e. it solves
    v(x + r^a, s, q-1, t) = v(x, s, q, t).

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    T : float
        Terminal horizon.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.

    Returns
    -------
    float
        Ask reservation price r^a.
    """
    tau = T - t
    return s + ((1.0 - 2.0 * q) * gamma * sigma**2 * tau) / 2.0


def reservation_bid(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Compute the bid reservation (indifference) price.

    r^b = s + ((-1 - 2q) · γ · σ² · (T-t)) / 2

    This is the maximum bid price at which the market maker is indifferent
    between buying one unit now versus not, i.e. it solves
    v(x - r^b, s, q+1, t) = v(x, s, q, t).

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    T : float
        Terminal horizon.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.

    Returns
    -------
    float
        Bid reservation price r^b.
    """
    tau = T - t
    return s + ((-1.0 - 2.0 * q) * gamma * sigma**2 * tau) / 2.0


def average_reservation_price(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Compute the average (mid) reservation price.

    r = (r^a + r^b) / 2 = s - q · γ · σ² · (T-t)

    This is the certainty-equivalent mid-price adjusted for inventory risk.
    Positive inventory lowers r below the market mid-price s.

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    T : float
        Terminal horizon.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.

    Returns
    -------
    float
        Average reservation price r.
    """
    tau = T - t
    return s - q * gamma * sigma**2 * tau


def verify_utility_indifference(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    tol: float = 1e-10,
) -> Tuple[bool, bool]:
    """Verify the utility-indifference conditions for reservation prices.

    Checks in log-space (to avoid floating-point overflow) that:
      - r^a solves  v(x + r^a, s, q-1, t) = v(x, s, q, t)
      - r^b solves  v(x - r^b, s, q+1, t) = v(x, s, q, t)

    Since v is always negative, the check uses
    log(-v(·)) = -γx - γqs + γ²q²σ²τ/2.
    The result is independent of x, so x = 0 is used internally.

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    T : float
        Terminal horizon.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    tol : float, optional
        Absolute tolerance for np.isclose.  Defaults to 1e-10.

    Returns
    -------
    (bool_ask, bool_bid) : Tuple[bool, bool]
        Each element is True if the corresponding indifference condition holds.
    """
    tau = T - t
    r_a = reservation_ask(s, q, t, T, gamma, sigma)
    r_b = reservation_bid(s, q, t, T, gamma, sigma)

    def log_neg_v(x: float, inv: float) -> float:
        return (
            -gamma * x
            - gamma * inv * s
            + (gamma**2 * inv**2 * sigma**2 * tau) / 2.0
        )

    # Ask: v(0 + r^a, s, q-1, t) == v(0, s, q, t)
    lhs_ask = log_neg_v(r_a, q - 1.0)
    rhs_ask = log_neg_v(0.0, q)
    bool_ask = bool(np.isclose(lhs_ask, rhs_ask, atol=tol))

    # Bid: v(0 - r^b, s, q+1, t) == v(0, s, q, t)
    lhs_bid = log_neg_v(-r_b, q + 1.0)
    rhs_bid = log_neg_v(0.0, q)
    bool_bid = bool(np.isclose(lhs_bid, rhs_bid, atol=tol))

    return bool_ask, bool_bid


# ---------------------------------------------------------------------------
# Symbolic derivation
# ---------------------------------------------------------------------------


def symbolic_derivation() -> Dict[str, sp.Expr]:
    """Symbolically derive reservation prices from the frozen-inventory function.

    Uses sympy to:
      1. Define the frozen-inventory value function v(x, s, q, t).
      2. Set up the ask-indifference condition
         v(x + r^a, s, q-1, t) = v(x, s, q, t) and solve for r^a.
      3. Set up the bid-indifference condition
         v(x - r^b, s, q+1, t) = v(x, s, q, t) and solve for r^b.
      4. Derive the average reservation price r = (r^a + r^b)/2.

    All intermediate steps are printed as a derivation summary.

    Returns
    -------
    dict
        Keys: 'v', 'r_ask', 'r_bid', 'r_avg',
              'indiff_ask_lhs', 'indiff_ask_rhs',
              'indiff_bid_lhs', 'indiff_bid_rhs'
        Values: sympy expressions.
    """
    x, s, q, t, T, gam, sig, r_a, r_b, tau = sp.symbols(
        "x s q t T gamma sigma r^a r^b tau", real=True, positive=True
    )
    # τ = T - t
    tau_expr = T - t

    def v_sym(cash, inv):
        return (
            -sp.exp(-gam * cash)
            * sp.exp(-gam * inv * s)
            * sp.exp((gam**2 * inv**2 * sig**2 * tau_expr) / 2)
        )

    v_base = v_sym(x, q)

    # --- Ask indifference ---
    v_ask = v_sym(x + r_a, q - 1)
    ask_ratio = sp.simplify(v_ask / v_base)
    # condition: ask_ratio == 1  →  solve for r^a
    ask_eq = sp.Eq(sp.log(-v_ask), sp.log(-v_base))
    ask_lhs_log = sp.expand(-gam * (x + r_a) - gam * (q - 1) * s
                            + (gam**2 * (q - 1)**2 * sig**2 * tau_expr) / 2)
    ask_rhs_log = sp.expand(-gam * x - gam * q * s
                            + (gam**2 * q**2 * sig**2 * tau_expr) / 2)
    r_ask_sol = sp.solve(sp.Eq(ask_lhs_log, ask_rhs_log), r_a)[0]
    r_ask_simplified = sp.simplify(r_ask_sol)

    # --- Bid indifference ---
    v_bid = v_sym(x - r_b, q + 1)
    bid_lhs_log = sp.expand(-gam * (x - r_b) - gam * (q + 1) * s
                            + (gam**2 * (q + 1)**2 * sig**2 * tau_expr) / 2)
    bid_rhs_log = ask_rhs_log
    r_bid_sol = sp.solve(sp.Eq(bid_lhs_log, bid_rhs_log), r_b)[0]
    r_bid_simplified = sp.simplify(r_bid_sol)

    # --- Average reservation price ---
    r_avg = sp.simplify((r_ask_simplified + r_bid_simplified) / 2)

    results = {
        "v": v_base,
        "r_ask": r_ask_simplified,
        "r_bid": r_bid_simplified,
        "r_avg": r_avg,
        "indiff_ask_lhs": ask_lhs_log,
        "indiff_ask_rhs": ask_rhs_log,
        "indiff_bid_lhs": bid_lhs_log,
        "indiff_bid_rhs": bid_rhs_log,
    }

    print("=" * 60)
    print("FROZEN-INVENTORY SYMBOLIC DERIVATION")
    print("=" * 60)
    print(f"\nValue function:  v(x,s,q,t) = {v_base}\n")
    print("Ask indifference condition  log(-v(x+r^a,s,q-1,t)) = log(-v(x,s,q,t))")
    print(f"  Solved r^a = {r_ask_simplified}\n")
    print("Bid indifference condition  log(-v(x-r^b,s,q+1,t)) = log(-v(x,s,q,t))")
    print(f"  Solved r^b = {r_bid_simplified}\n")
    print(f"Average reservation price  r = {r_avg}")
    print("=" * 60)

    return results
