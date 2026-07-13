"""
Hamilton-Jacobi-Bellman (HJB) equation analysis for the Avellaneda-Stoikov
market-making model.

The market maker solves the optimal-control problem

  max_{δ^a,δ^b} E[-exp(-γ · W_T)]

subject to dS_t = σ dB_t and jump processes for inventory.

The HJB equation for the value function u(x, s, q, t) is

  ∂u/∂t + (1/2)σ²∂²u/∂s²
  + max_{δ^a} λ^a(δ^a)[u(x+s+δ^a, s, q-1, t) - u(x,s,q,t)]
  + max_{δ^b} λ^b(δ^b)[u(x-s+δ^b, s, q+1, t) - u(x,s,q,t)] = 0

with terminal condition u(x, s, q, T) = -exp(-γ(x + qs)).

The exponential ansatz u = -exp(-γ(x + θ(s,q,t))) reduces the problem
to a PDE for the performance index θ.  A quadratic approximation in q
yields an explicit solution consistent with the frozen-inventory prices.

Reference: Avellaneda & Stoikov (2008), Quantitative Finance 8(3), 217-224.
"""

from typing import Callable, Dict

import numpy as np
import sympy as sp

from src.theory.frozen_inventory import reservation_ask, reservation_bid


# ---------------------------------------------------------------------------
# HJB description
# ---------------------------------------------------------------------------


def describe_hjb_terms() -> Dict[str, str]:
    """Return a human-readable description of each term in the HJB equation.

    The HJB PDE for u(x, s, q, t) is broken into four components:
      - time_derivative  : ∂u/∂t
      - diffusion        : (1/2)σ² ∂²u/∂s²   (Brownian price uncertainty)
      - ask_jump         : λ^a(δ^a)[u(x+s+δ^a,s,q-1,t) - u(x,s,q,t)]
      - bid_jump         : λ^b(δ^b)[u(x-s+δ^b,s,q+1,t) - u(x,s,q,t)]

    A readable summary is also printed to stdout.

    Returns
    -------
    dict
        Mapping from term name (str) to LaTeX-style description (str).
    """
    terms = {
        "time_derivative": "du/dt",
        "diffusion": "(1/2) * sigma^2 * d^2u/ds^2",
        "ask_jump": (
            "lambda^a(delta^a) * "
            "[u(x + s + delta^a, s, q-1, t) - u(x, s, q, t)]"
        ),
        "bid_jump": (
            "lambda^b(delta^b) * "
            "[u(x - s + delta^b, s, q+1, t) - u(x, s, q, t)]"
        ),
        "terminal_condition": "u(x, s, q, T) = -exp(-gamma * (x + q*s))",
        "full_hjb": (
            "du/dt + (1/2)*sigma^2*d^2u/ds^2 "
            "+ max_{delta^a} lambda^a*(u_jump_ask) "
            "+ max_{delta^b} lambda^b*(u_jump_bid) = 0"
        ),
    }

    print("=" * 60)
    print("HJB EQUATION TERMS")
    print("=" * 60)
    for name, desc in terms.items():
        print(f"  {name:20s}: {desc}")
    print()
    print("Interpretation:")
    print("  time_derivative : rate of change of value over time")
    print("  diffusion       : value change from Brownian price fluctuations")
    print("  ask_jump        : expected value gain from an ask execution")
    print("  bid_jump        : expected value gain from a bid execution")
    print("  terminal_cond   : CARA utility of terminal wealth x + q*s")
    print("=" * 60)

    return terms


# ---------------------------------------------------------------------------
# Theta function helpers
# ---------------------------------------------------------------------------


def theta_terminal_condition(s: float, q: float) -> float:
    """Return the terminal value of the performance index θ.

    From the terminal condition u(x,s,q,T) = -exp(-γ(x+qs)) and the
    ansatz u = -exp(-γ(x+θ(s,q,t))), we immediately get:

        θ(s, q, T) = q · s

    Parameters
    ----------
    s : float
        Mid-price at maturity.
    q : float
        Inventory at maturity.

    Returns
    -------
    float
        Terminal performance index value q*s.
    """
    return float(q * s)


def reservation_from_theta(
    theta_func: Callable[[float, float, float], float],
    s: float,
    q: float,
    t: float,
) -> Dict[str, float]:
    """Derive reservation prices from the performance index θ.

    Given the ansatz u = -exp(-γ(x + θ)), the utility-indifference
    conditions yield:

        r^b = θ(s, q+1, t) - θ(s, q, t)   (bid reservation price)
        r^a = θ(s, q, t) - θ(s, q-1, t)   (ask reservation price)

    Parameters
    ----------
    theta_func : callable
        A function theta_func(s, q, t) -> float representing θ.
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.

    Returns
    -------
    dict
        Keys 'r_bid' and 'r_ask' with the corresponding reservation prices.
    """
    theta_q = theta_func(s, q, t)
    r_bid = theta_func(s, q + 1.0, t) - theta_q
    r_ask = theta_q - theta_func(s, q - 1.0, t)
    return {"r_bid": r_bid, "r_ask": r_ask}


def approximate_theta(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Evaluate the quadratic approximate performance index θ.

    The quadratic-in-q ansatz is:

        θ(s, q, t) ≈ q·s - (1/2)·γ·σ²·(T-t)·q²

    This approximation is exact for the diffusion part of the HJB PDE
    (the non-linear term (∂θ/∂s)² equals q², which is already quadratic,
    so second-order cross-terms vanish), and yields reservation prices
    consistent with the frozen-inventory solution.

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
        Approximate value of θ(s, q, t).
    """
    tau = T - t
    return float(q * s - 0.5 * gamma * sigma**2 * tau * q**2)


def verify_reservation_from_theta(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> Dict[str, float]:
    """Verify that θ-derived reservation prices match the frozen-inventory formulas.

    Computes r^a and r^b from both:
      (a) `reservation_from_theta` applied to `approximate_theta`, and
      (b) the direct `reservation_ask` / `reservation_bid` formulas.

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
    dict
        Keys: 'r_ask_theta', 'r_bid_theta', 'r_ask_formula', 'r_bid_formula',
              'ask_match', 'bid_match'.
    """
    theta_func = lambda sv, qv, tv: approximate_theta(sv, qv, tv, T, gamma, sigma)

    from_theta = reservation_from_theta(theta_func, s, q, t)
    r_ask_formula = reservation_ask(s, q, t, T, gamma, sigma)
    r_bid_formula = reservation_bid(s, q, t, T, gamma, sigma)

    return {
        "r_ask_theta": from_theta["r_ask"],
        "r_bid_theta": from_theta["r_bid"],
        "r_ask_formula": r_ask_formula,
        "r_bid_formula": r_bid_formula,
        "ask_match": bool(np.isclose(from_theta["r_ask"], r_ask_formula)),
        "bid_match": bool(np.isclose(from_theta["r_bid"], r_bid_formula)),
    }


# ---------------------------------------------------------------------------
# Symbolic HJB ansatz
# ---------------------------------------------------------------------------


def symbolic_hjb_ansatz() -> Dict[str, sp.Expr]:
    """Apply the exponential ansatz to the HJB and derive the PDE for θ.

    Substitutes u = -exp(-γ(x + θ(s,q,t))) into the HJB, divides by u,
    and collects terms.  The approximate quadratic θ is then substituted
    to verify that the diffusion terms cancel exactly.

    Returns
    -------
    dict
        Keys:
          'ansatz'           : expression for u in terms of theta symbols
          'pde_theta'        : PDE for theta after ansatz substitution
          'diffusion_term'   : (1/2)σ²(-γθ_ss + γ²θ_s²) coefficient
          'time_term'        : -γ θ_t  (time-derivative coefficient)
          'approx_theta'     : symbolic quadratic approximation
          'diffusion_cancel' : sympy expression showing diffusion cancellation
    """
    gam, sig, x_sym, s_sym, q_sym, t_sym, T_sym = sp.symbols(
        "gamma sigma x s q t T", real=True, positive=True
    )
    tau_sym = T_sym - t_sym

    # Symbolic theta as a generic function and its approximate form
    theta_fn = sp.Function("theta")(s_sym, q_sym, t_sym)
    theta_approx = q_sym * s_sym - sp.Rational(1, 2) * gam * sig**2 * tau_sym * q_sym**2

    # Ansatz: u = -exp(-gamma*(x + theta))
    u_ansatz = -sp.exp(-gam * (x_sym + theta_fn))

    # Partial derivatives of u with the ansatz
    # u_t / u = -gamma * theta_t  (dividing by u, note u < 0 cancels sign)
    theta_t = sp.Symbol("theta_t")   # ∂θ/∂t (placeholder)
    theta_s = sp.Symbol("theta_s")   # ∂θ/∂s
    theta_ss = sp.Symbol("theta_ss") # ∂²θ/∂s²

    # After dividing HJB by u (negative, so careful with direction):
    # du/dt / u = -(-gamma*theta_t) = gamma*theta_t  ... wait no
    # u = -exp(-gamma*(x+theta))
    # du/dt = -exp(-gamma*(x+theta)) * (-gamma) * theta_t = gamma * theta_t * (-exp(...))
    # du/dt = gamma * theta_t * u
    # du/dt / u = gamma * theta_t   → dividing HJB by u gives +gamma*theta_t for time term
    # BUT HJB is du/dt + diffusion + jumps = 0
    # dividing by u:  gamma*theta_t + (diffusion/u) + (jumps/u) = 0
    # time coefficient after /u: gamma * theta_t
    # But convention: we write as -gamma*d(theta)/dt because d/dt of exponent = +gamma*theta_t
    # Let me redo carefully:
    # d/dt[-exp(-gamma*(x+theta))] = gamma * theta_t * exp(-gamma*(x+theta))
    #                               = -gamma * theta_t * u
    # So du/dt = -gamma * theta_t * u
    # du/dt / u = -gamma * theta_t
    time_coeff = -gam * theta_t

    # d²u/ds²: u = -exp(-gamma*f), f=x+theta
    # du/ds = gamma*theta_s * u  (same sign argument as above, u < 0)
    # d²u/ds² = gamma*theta_ss * u + gamma*theta_s * du/ds
    #          = gamma*theta_ss * u + gamma*theta_s * gamma*theta_s * u
    #          = u * (gamma*theta_ss + gamma^2 * theta_s^2)
    # Wait: u = -exp(-gamma*f)
    # du/ds = -exp(-gamma*f) * (-gamma) * theta_s = gamma*theta_s * exp(-gamma*f) = -gamma*theta_s * u
    # d²u/ds² = -gamma*theta_ss * u + (-gamma*theta_s) * du/ds
    #          = -gamma*theta_ss * u + (-gamma*theta_s)*(-gamma*theta_s * u)
    #          = u*(-gamma*theta_ss + gamma^2 * theta_s^2)
    # (1/2)*sigma^2 * d²u/ds^2 / u = (1/2)*sigma^2 * (-gamma*theta_ss + gamma^2 * theta_s^2)
    diffusion_coeff = sp.Rational(1, 2) * sig**2 * (-gam * theta_ss + gam**2 * theta_s**2)

    # Full PDE for theta (omitting jump terms which require discrete q):
    pde_theta = sp.Eq(time_coeff + diffusion_coeff, sp.Symbol("jump_terms"))

    # --- Verify diffusion cancellation for approximate theta ---
    # For theta_approx = q*s - (1/2)*gamma*sigma^2*tau*q^2:
    #   theta_s = q  →  theta_ss = 0
    #   theta_t = d/dt[-(1/2)*gamma*sigma^2*(T-t)*q^2] = (1/2)*gamma*sigma^2*q^2
    theta_t_approx = sp.diff(theta_approx, t_sym)   # = (1/2)*gamma*sigma^2*q^2
    theta_s_approx = sp.diff(theta_approx, s_sym)   # = q
    theta_ss_approx = sp.diff(theta_approx, s_sym, 2)  # = 0

    time_val = (-gam * theta_t_approx).subs([
        (gam, gam), (sig, sig), (q_sym, q_sym)
    ])
    diff_val = (sp.Rational(1, 2) * sig**2 * (
        -gam * theta_ss_approx + gam**2 * theta_s_approx**2
    ))
    diffusion_cancel = sp.simplify(time_val + diff_val)

    results = {
        "ansatz": u_ansatz,
        "pde_theta": pde_theta,
        "diffusion_term": diffusion_coeff,
        "time_term": time_coeff,
        "approx_theta": theta_approx,
        "diffusion_cancel": diffusion_cancel,
    }

    print("=" * 60)
    print("SYMBOLIC HJB ANSATZ")
    print("=" * 60)
    print(f"\nAnsatz:  u = {u_ansatz}")
    print(f"\nPDE for theta (time + diffusion = jump_terms):")
    print(f"  time term        : {time_coeff}")
    print(f"  diffusion coeff  : {diffusion_coeff}")
    print(f"\nApproximate theta : {theta_approx}")
    print(f"  theta_t          : {theta_t_approx}")
    print(f"  theta_s          : {theta_s_approx}")
    print(f"  theta_ss         : {theta_ss_approx}")
    print(f"\nDiffusion cancellation check (time + diffusion = 0?):")
    print(f"  Result           : {diffusion_cancel}  (0 means exact cancellation)")
    print("=" * 60)

    return results
