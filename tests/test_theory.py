"""
Pytest test suite for the Avellaneda-Stoikov theory modules.

Covers:
  - frozen_inventory: value function, reservation prices, indifference
  - hjb: theta functions, theta-derived reservation prices
  - quotes: spread formulas, quote prices, FOC

Run with:
    pytest tests/test_theory.py -v
"""

import numpy as np
import pytest
import sympy as sp

from src.theory.frozen_inventory import (
    average_reservation_price,
    frozen_inventory_value,
    reservation_ask,
    reservation_bid,
    verify_utility_indifference,
)
from src.theory.hjb import (
    approximate_theta,
    reservation_from_theta,
    theta_terminal_condition,
    verify_reservation_from_theta,
)
from src.theory.quotes import (
    constant_spread_component,
    delta_ask,
    delta_bid,
    foc_exponential_simplification,
    half_spread,
    quoted_ask,
    quoted_bid,
    total_spread,
    verify_quote_equivalence,
)


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

BASE_S = 100.0
BASE_SIGMA = 2.0
BASE_T = 1.0
BASE_GAMMA = 0.1
BASE_K = 1.5


# ---------------------------------------------------------------------------
# 1. frozen_inventory: value function at terminal time
# ---------------------------------------------------------------------------


def test_frozen_value_function_terminal():
    """At t=T the time-dependent exponent vanishes and v simplifies.

    v(x, s, q, T) = -exp(-γx) · exp(-γqs) · exp(0)
                  = -exp(-γ(x + qs))
    """
    x, s, q = 50.0, BASE_S, 3.0
    gamma, sigma = BASE_GAMMA, BASE_SIGMA
    T = BASE_T
    # At t=T
    v = frozen_inventory_value(x, s, q, T, T, gamma, sigma)
    expected = -np.exp(-gamma * (x + q * s))
    assert np.isclose(v, expected), f"Expected {expected}, got {v}"


# ---------------------------------------------------------------------------
# 2. Reservation ask at q=0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("t", [0.0, 0.5, 0.9])
def test_reservation_ask_q0(gamma, t):
    """At q=0 the ask reservation price equals s + γσ²(T-t)/2."""
    s, sigma, T = BASE_S, BASE_SIGMA, BASE_T
    r_a = reservation_ask(s, 0.0, t, T, gamma, sigma)
    expected = s + gamma * sigma**2 * (T - t) / 2.0
    assert np.isclose(r_a, expected), f"gamma={gamma}, t={t}: got {r_a}, expected {expected}"


# ---------------------------------------------------------------------------
# 3. Reservation bid at q=0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("t", [0.0, 0.5, 0.9])
def test_reservation_bid_q0(gamma, t):
    """At q=0 the bid reservation price equals s - γσ²(T-t)/2."""
    s, sigma, T = BASE_S, BASE_SIGMA, BASE_T
    r_b = reservation_bid(s, 0.0, t, T, gamma, sigma)
    expected = s - gamma * sigma**2 * (T - t) / 2.0
    assert np.isclose(r_b, expected), f"gamma={gamma}, t={t}: got {r_b}, expected {expected}"


# ---------------------------------------------------------------------------
# 4. Average reservation price at q=0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_average_reservation_q0(gamma):
    """At q=0 the average reservation price equals the mid-price s."""
    s, sigma, T, t = BASE_S, BASE_SIGMA, BASE_T, 0.5
    r = average_reservation_price(s, 0.0, t, T, gamma, sigma)
    assert np.isclose(r, s), f"gamma={gamma}: got {r}, expected {s}"


# ---------------------------------------------------------------------------
# 5. Utility indifference – ask side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [-3.0, 0.0, 3.0])
def test_utility_indifference_ask(q):
    """v(x + r^a, s, q-1, t) == v(x, s, q, t) in log-space."""
    s, sigma, T, t = BASE_S, BASE_SIGMA, BASE_T, 0.5
    gamma = BASE_GAMMA
    bool_ask, _ = verify_utility_indifference(s, q, t, T, gamma, sigma)
    assert bool_ask, f"Ask indifference failed at q={q}"


# ---------------------------------------------------------------------------
# 6. Utility indifference – bid side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [-3.0, 0.0, 3.0])
def test_utility_indifference_bid(q):
    """v(x - r^b, s, q+1, t) == v(x, s, q, t) in log-space."""
    s, sigma, T, t = BASE_S, BASE_SIGMA, BASE_T, 0.5
    gamma = BASE_GAMMA
    _, bool_bid = verify_utility_indifference(s, q, t, T, gamma, sigma)
    assert bool_bid, f"Bid indifference failed at q={q}"


# ---------------------------------------------------------------------------
# 7. Average reservation price converges to s as t → T
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [-5.0, 0.0, 5.0])
def test_r_converges_to_s_at_maturity(q):
    """r(s, q, t) → s as t → T for any finite q."""
    s, sigma, T, gamma = BASE_S, BASE_SIGMA, BASE_T, BASE_GAMMA
    r = average_reservation_price(s, q, T, T, gamma, sigma)
    assert np.isclose(r, s), f"q={q}: r={r} did not converge to s={s}"


# ---------------------------------------------------------------------------
# 8. Inventory correction sign
# ---------------------------------------------------------------------------


def test_inventory_correction_sign():
    """Positive inventory lowers the reservation price below mid (r < s)."""
    s, sigma, T, t, gamma = BASE_S, BASE_SIGMA, BASE_T, 0.0, BASE_GAMMA
    for q in [1.0, 3.0, 5.0]:
        r = average_reservation_price(s, q, t, T, gamma, sigma)
        assert r < s, f"q={q}: r={r} should be below s={s}"
    # Negative inventory raises r above s
    for q in [-1.0, -3.0, -5.0]:
        r = average_reservation_price(s, q, t, T, gamma, sigma)
        assert r > s, f"q={q}: r={r} should be above s={s}"


# ---------------------------------------------------------------------------
# 9. delta_ask positivity for small |q|
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [-1.0, 0.0, 1.0])
def test_delta_ask_positive(q):
    """delta^a > 0 for small inventory and reasonable parameters."""
    gamma, sigma, T, t, k = 0.1, 2.0, 1.0, 0.5, 1.5
    da = delta_ask(q, gamma, sigma, T, t, k)
    assert da > 0.0, f"q={q}: delta_ask={da} should be positive"


# ---------------------------------------------------------------------------
# 10. Total spread positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("t", [0.0, 0.5, 0.99])
def test_spread_positivity(gamma, t):
    """Total spread delta^a + delta^b is strictly positive."""
    sigma, T, k = BASE_SIGMA, BASE_T, BASE_K
    ts = total_spread(gamma, sigma, T, t, k)
    assert ts > 0.0, f"gamma={gamma}, t={t}: total_spread={ts}"


# ---------------------------------------------------------------------------
# 11. Quote equivalence: p^a = r + h and p^b = r - h
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [-5.0, 0.0, 5.0])
@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_quote_equivalence(q, gamma):
    """Verify p^a = r + h and p^b = r - h."""
    s, sigma, T, t, k = BASE_S, BASE_SIGMA, BASE_T, 0.5, BASE_K
    ok = verify_quote_equivalence(s, q, gamma, sigma, T, t, k)
    assert ok, f"Quote equivalence failed: q={q}, gamma={gamma}"


# ---------------------------------------------------------------------------
# 12. FOC exponential: lambda/lambda' = -1/k
# ---------------------------------------------------------------------------


def test_foc_exponential():
    """For lambda = A*exp(-k*delta), lambda/(dlambda/ddelta) = -1/k symbolically."""
    ratio = foc_exponential_simplification()
    k_sym = sp.Symbol("k", positive=True)
    # ratio should be -1/k
    expected = -1 / k_sym
    diff = sp.simplify(ratio - expected)
    assert diff == 0, f"FOC ratio: got {ratio}, expected -1/k"


# ---------------------------------------------------------------------------
# 13. Constant spread component positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma,k", [(0.01, 1.5), (0.1, 1.5), (0.5, 2.0)])
def test_constant_spread(gamma, k):
    """Constant spread component (2/gamma)*ln(1+gamma/k) is strictly positive."""
    cs = constant_spread_component(gamma, k)
    assert cs > 0.0, f"gamma={gamma}, k={k}: constant_spread={cs}"


# ---------------------------------------------------------------------------
# 14. Half-spread at t = T
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_half_spread_at_maturity(gamma):
    """At t=T the time-risk component vanishes: h = (1/gamma)*ln(1+gamma/k)."""
    sigma, T, k = BASE_SIGMA, BASE_T, BASE_K
    h = half_spread(gamma, sigma, T, T, k)
    expected = (1.0 / gamma) * np.log(1.0 + gamma / k)
    assert np.isclose(h, expected), f"gamma={gamma}: h={h}, expected={expected}"


# ---------------------------------------------------------------------------
# 15. Theta terminal condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q,s", [(0.0, 100.0), (3.0, 50.0), (-2.0, 200.0)])
def test_theta_terminal(q, s):
    """theta(s, q, T) = q * s."""
    result = theta_terminal_condition(s, q)
    expected = q * s
    assert np.isclose(result, expected), f"q={q}, s={s}: theta_T={result}, expected={expected}"


# ---------------------------------------------------------------------------
# 16. Reservation prices from theta match direct formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [-3.0, 0.0, 3.0])
@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_reservation_from_theta(q, gamma):
    """Theta-derived r^a and r^b match frozen-inventory formulas."""
    s, sigma, T, t = BASE_S, BASE_SIGMA, BASE_T, 0.5
    result = verify_reservation_from_theta(s, q, t, T, gamma, sigma)
    assert result["ask_match"], (
        f"Ask mismatch: q={q}, gamma={gamma}: "
        f"theta={result['r_ask_theta']:.6f}, formula={result['r_ask_formula']:.6f}"
    )
    assert result["bid_match"], (
        f"Bid mismatch: q={q}, gamma={gamma}: "
        f"theta={result['r_bid_theta']:.6f}, formula={result['r_bid_formula']:.6f}"
    )
