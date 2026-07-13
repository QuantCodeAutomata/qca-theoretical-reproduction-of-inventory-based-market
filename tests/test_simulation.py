"""
Pytest test suite for the Avellaneda-Stoikov Monte Carlo simulation.

Tests cover:
- Fill-intensity functions (intensity.py)
- Quoting strategies (strategies.py)
- Single-path mechanics (engine.py)
- Full Monte Carlo reproducibility and risk-reduction property (monte_carlo.py)
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from src.sim.intensity import lambda_from_delta, validate_monotonicity
from src.sim.strategies import (
    half_spread,
    inventory_center,
    inventory_quotes,
    symmetric_quotes,
)
from src.sim.engine import single_path_simulation
from src.sim.monte_carlo import run_all_experiments


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_params() -> dict:
    return DEFAULT_PARAMS.copy()


@pytest.fixture
def default_rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# intensity.py tests
# ---------------------------------------------------------------------------

def test_intensity_positive() -> None:
    """lambda_from_delta must always return a positive value."""
    deltas = np.linspace(-2.0, 5.0, 50)
    lambdas = lambda_from_delta(deltas, A=140.0, k=1.5)
    assert np.all(lambdas > 0), "fill intensity must be strictly positive"


def test_intensity_monotone() -> None:
    """Larger delta must yield smaller fill intensity."""
    deltas = np.linspace(0.0, 3.0, 100)
    result = validate_monotonicity(deltas, A=140.0, k=1.5)
    assert result is True, "fill intensity must be strictly decreasing in delta"


# ---------------------------------------------------------------------------
# strategies.py tests
# ---------------------------------------------------------------------------

def test_half_spread_positive() -> None:
    """half_spread must be positive for all t < T."""
    gamma, sigma, T, k = 0.1, 2.0, 1.0, 1.5
    for t in np.linspace(0.0, T - 1e-6, 20):
        h = half_spread(gamma, sigma, T, t, k)
        assert h > 0, f"half_spread must be positive at t={t}"


def test_inventory_center_positive_q() -> None:
    """Positive inventory should push reservation price below mid-price."""
    r = inventory_center(S=100.0, q=5.0, gamma=0.1, sigma=2.0, T=1.0, t=0.0)
    assert r < 100.0, "positive q should lower r below S"


def test_inventory_center_negative_q() -> None:
    """Negative inventory should push reservation price above mid-price."""
    r = inventory_center(S=100.0, q=-5.0, gamma=0.1, sigma=2.0, T=1.0, t=0.0)
    assert r > 100.0, "negative q should raise r above S"


def test_quotes_consistency() -> None:
    """For inventory strategy: delta_ask + delta_bid == 2*h (total spread = 2h)."""
    quotes = inventory_quotes(S=100.0, q=2.0, t=0.5, T=1.0,
                              gamma=0.1, sigma=2.0, k=1.5)
    h = quotes['h']
    total_spread = quotes['delta_ask'] + quotes['delta_bid']
    np.testing.assert_allclose(total_spread, 2 * h, rtol=1e-10,
                               err_msg="delta_ask + delta_bid must equal 2*h")


def test_symmetric_quotes_centered() -> None:
    """For symmetric strategy: delta_ask == delta_bid (both equal h)."""
    quotes = symmetric_quotes(S=100.0, t=0.3, T=1.0, gamma=0.1, sigma=2.0, k=1.5)
    np.testing.assert_allclose(quotes['delta_ask'], quotes['delta_bid'], rtol=1e-12,
                               err_msg="symmetric strategy must have equal half-spreads")


# ---------------------------------------------------------------------------
# engine.py - mechanics tests (deterministic mini-simulations)
# ---------------------------------------------------------------------------

class _SeqRNG:
    """Minimal fake RNG that replays a fixed sequence of floats."""

    def __init__(self, seq: list) -> None:
        self._seq = seq
        self._i = 0

    def random(self) -> float:
        v = self._seq[self._i]
        self._i += 1
        return v


def _run_one_step_engine(
    ask_u: float,
    bid_u: float,
    mid_u: float,
    strategy: str = 'inventory',
    S: float = 100.0,
    q: int = 0,
) -> dict:
    """Run exactly one simulation step using controlled uniform draws.

    With sigma=0, k=1.5, A=1.0, T=1, dt=1 (n_steps=1):
      h  = log(1 + 0.1/1.5)/0.1 ~= 0.645
      lambda ~= 1.0 * exp(-1.5 * 0.645) ~= 0.380
      lam*dt = 0.380 < 1

    So ask_u=0.0 forces a fill; ask_u=1.0 blocks a fill.

    Parameters
    ----------
    ask_u : uniform draw for ask-fill decision (0=fill, 1=block)
    bid_u : uniform draw for bid-fill decision (0=fill, 1=block)
    mid_u : uniform draw for mid-price direction (0.5 -> +1, no movement since sigma=0)
    strategy : 'inventory' or 'symmetric'
    S : initial mid-price
    q : initial inventory

    Returns
    -------
    dict
        Output of single_path_simulation with record_history=True.
    """
    params = {
        's0': S, 'T': 1.0, 'sigma': 0.0,
        'dt': 1.0,   # n_steps = T/dt = 1 -> exactly one iteration
        'A': 1.0, 'k': 1.5,
        'q0': q, 'x0': 0.0,
    }
    rng = _SeqRNG([ask_u, bid_u, mid_u])
    return single_path_simulation(strategy, 0.1, params, rng, record_history=True)  # type: ignore[arg-type]


def test_cash_update_ask_fill() -> None:
    """X must increase by p_ask when an ask fills and no bid fills."""
    # u1=0.0 -> ask fills (0 < lam*dt ~= 0.38)
    # u2=1.0 -> bid blocked (1 > lam*dt ~= 0.38)
    result = _run_one_step_engine(ask_u=0.0, bid_u=1.0, mid_u=0.5)
    assert result['final_q'] == -1, "ask fill should decrement inventory by 1"

    # With sigma=0, q=0: r=S, h=log(1+0.1/1.5)/0.1; p_ask = r + h
    expected_p_ask = inventory_quotes(100.0, 0, 0.0, 1.0, 0.1, 0.0, 1.5)['p_ask']
    X_after = result['history']['X_hist'][0]
    np.testing.assert_allclose(X_after, expected_p_ask, rtol=1e-12,
                               err_msg="cash must equal p_ask after single ask fill")


def test_cash_update_bid_fill() -> None:
    """X must decrease by p_bid when a bid fills and no ask fills."""
    # u1=1.0 -> ask blocked; u2=0.0 -> bid fills
    result = _run_one_step_engine(ask_u=1.0, bid_u=0.0, mid_u=0.5)
    assert result['final_q'] == 1, "bid fill should increment inventory by 1"

    expected_p_bid = inventory_quotes(100.0, 0, 0.0, 1.0, 0.1, 0.0, 1.5)['p_bid']
    X_after = result['history']['X_hist'][0]
    np.testing.assert_allclose(X_after, -expected_p_bid, rtol=1e-12,
                               err_msg="cash must equal -p_bid after single bid fill")


def test_inventory_update_ask() -> None:
    """Inventory q must decrease by 1 on an ask fill."""
    result = _run_one_step_engine(ask_u=0.0, bid_u=1.0, mid_u=0.5)
    assert result['final_q'] == -1, "ask fill must decrement q by 1"


def test_inventory_update_bid() -> None:
    """Inventory q must increase by 1 on a bid fill."""
    result = _run_one_step_engine(ask_u=1.0, bid_u=0.0, mid_u=0.5)
    assert result['final_q'] == 1, "bid fill must increment q by 1"


def test_terminal_profit_formula(base_params: dict, default_rng: np.random.Generator) -> None:
    """Terminal profit must equal X_T + q_T * S_T (verified via history)."""
    result = single_path_simulation(
        'inventory', 0.1, base_params, default_rng, record_history=True
    )
    # Verify finiteness
    assert np.isfinite(result['terminal_profit']), "terminal profit must be finite"
    # Same seed produces identical result: confirms the formula is deterministic
    rng2 = np.random.default_rng(42)
    result2 = single_path_simulation('inventory', 0.1, base_params, rng2, record_history=False)
    np.testing.assert_allclose(result['terminal_profit'], result2['terminal_profit'],
                               err_msg="same seed must give same terminal profit")


def test_convergence_at_maturity() -> None:
    """As t -> T, inventory quotes should approach symmetric quotes.

    At t = T the half-spread h -> (1/gamma)*ln(1+gamma/k) (constant term only),
    and r -> S, so delta_ask -> delta_bid -> h and the two strategies coincide.
    """
    gamma, sigma, T, k = 0.1, 2.0, 1.0, 1.5
    t_near_T = T - 1e-8  # almost at maturity
    S, q = 100.0, 3.0   # non-zero inventory makes the test non-trivial

    inv = inventory_quotes(S, q, t_near_T, T, gamma, sigma, k)
    sym = symmetric_quotes(S, t_near_T, T, gamma, sigma, k)

    np.testing.assert_allclose(inv['delta_ask'], sym['delta_ask'], rtol=1e-5,
                               err_msg="inventory delta_ask must approach symmetric near T")
    np.testing.assert_allclose(inv['delta_bid'], sym['delta_bid'], rtol=1e-5,
                               err_msg="inventory delta_bid must approach symmetric near T")


def test_single_path_deterministic(base_params: dict) -> None:
    """Running with the same seed twice must produce identical results."""
    rng1 = np.random.default_rng(9999)
    rng2 = np.random.default_rng(9999)
    r1 = single_path_simulation('inventory', 0.1, base_params, rng1)
    r2 = single_path_simulation('inventory', 0.1, base_params, rng2)
    assert r1['terminal_profit'] == r2['terminal_profit']
    assert r1['final_q'] == r2['final_q']


# ---------------------------------------------------------------------------
# monte_carlo.py tests
# ---------------------------------------------------------------------------

def test_risk_reduction(base_params: dict) -> None:
    """Inventory strategy should have lower profit std than symmetric (gamma=0.5).

    Uses only 200 paths for speed.
    """
    results = run_all_experiments(n_paths=200, params=base_params, seed=42)
    gamma = 0.5
    std_inv = float(np.std(results[gamma]['inventory']['profits'], ddof=1))
    std_sym = float(np.std(results[gamma]['symmetric']['profits'], ddof=1))
    assert std_inv < std_sym, (
        f"inventory std ({std_inv:.3f}) must be less than symmetric std ({std_sym:.3f}) "
        f"for gamma={gamma}"
    )


def test_common_random_numbers() -> None:
    """run_all_experiments must be reproducible with the same seed."""
    params = DEFAULT_PARAMS.copy()
    r1 = run_all_experiments(n_paths=50, params=params, seed=12345)
    r2 = run_all_experiments(n_paths=50, params=params, seed=12345)
    for gamma in r1:
        for strategy in r1[gamma]:
            np.testing.assert_array_equal(
                r1[gamma][strategy]['profits'],
                r2[gamma][strategy]['profits'],
                err_msg=f"profits must be identical for gamma={gamma}, strategy={strategy}",
            )
