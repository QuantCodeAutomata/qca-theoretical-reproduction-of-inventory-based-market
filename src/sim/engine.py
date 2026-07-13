"""
Single-path Monte Carlo engine for the Avellaneda-Stoikov market-making model.

Each simulation step:
1. Computes bid/ask quotes from the chosen strategy.
2. Determines fills via Bernoulli draws (u < λ·dt).
3. Updates inventory q and cash X accordingly.
4. Advances the mid-price by a symmetric random walk step.

Terminal P&L = X_T + q_T · S_T  (mark-to-market at the final mid-price).
"""

from __future__ import annotations

import numpy as np

from src.sim.intensity import lambda_from_delta
from src.sim.strategies import inventory_quotes, symmetric_quotes


def single_path_simulation(
    strategy: str,
    gamma: float,
    params: dict,
    rng: np.random.Generator,
    record_history: bool = False,
) -> dict:
    """Simulate one path of the Avellaneda-Stoikov market-making model.

    Parameters
    ----------
    strategy:
        ``'inventory'`` for the reservation-price strategy or
        ``'symmetric'`` for the inventory-blind benchmark.
    gamma:
        Absolute risk-aversion coefficient.
    params:
        Simulation parameters.  Expected keys:

        * ``s0``  – initial mid-price (default 100)
        * ``T``   – terminal time (default 1)
        * ``sigma`` – mid-price volatility (default 2)
        * ``dt``  – time step (default 0.005)
        * ``A``   – baseline fill-intensity (default 140)
        * ``k``   – fill-intensity sensitivity (default 1.5)
        * ``q0``  – initial inventory (default 0)
        * ``x0``  – initial cash (default 0)

    rng:
        NumPy random-number generator (``np.random.Generator`` instance).
    record_history:
        If ``True``, arrays of state variables are stored at every step and
        returned under the ``'history'`` key.

    Returns
    -------
    dict
        * ``terminal_profit`` – float, X_T + q_T · S_T
        * ``final_q``         – int, inventory position at T
        * ``diagnostics``     – dict with keys:

          - ``neg_delta_a``        – steps where δ_ask < 0
          - ``neg_delta_b``        – steps where δ_bid < 0
          - ``prob_gt1_ask``       – steps where λ_a·dt > 1
          - ``prob_gt1_bid``       – steps where λ_b·dt > 1
          - ``simultaneous_fills`` – steps where both sides filled

        * ``history``         – dict (only when ``record_history=True``) with
          arrays: ``times``, ``S_hist``, ``r_hist`` (``None`` for symmetric),
          ``p_ask_hist``, ``p_bid_hist``, ``q_hist``, ``X_hist``,
          ``lambda_a_hist``, ``lambda_b_hist``.
    """
    s0: float = params['s0']
    T: float = params['T']
    sigma: float = params['sigma']
    dt: float = params['dt']
    A: float = params['A']
    k: float = params['k']
    q0: int = params['q0']
    x0: float = params['x0']

    n_steps: int = int(T / dt)
    sqrt_dt: float = np.sqrt(dt)

    S: float = s0
    q: int = q0
    X: float = x0

    # Diagnostic counters
    neg_delta_a: int = 0
    neg_delta_b: int = 0
    prob_gt1_ask: int = 0
    prob_gt1_bid: int = 0
    simultaneous_fills: int = 0

    # History storage (pre-allocated for efficiency)
    if record_history:
        times = np.empty(n_steps)
        S_hist = np.empty(n_steps)
        r_hist: np.ndarray | None = np.empty(n_steps) if strategy == 'inventory' else None
        p_ask_hist = np.empty(n_steps)
        p_bid_hist = np.empty(n_steps)
        q_hist = np.empty(n_steps)
        X_hist = np.empty(n_steps)
        lambda_a_hist = np.empty(n_steps)
        lambda_b_hist = np.empty(n_steps)

    for i in range(n_steps):
        t: float = i * dt

        # --- Step a: compute quotes ---
        if strategy == 'inventory':
            quotes = inventory_quotes(S, q, t, T, gamma, sigma, k)
            r_val: float = quotes['r']
            delta_ask: float = quotes['delta_ask']
            delta_bid: float = quotes['delta_bid']
            p_ask: float = quotes['p_ask']
            p_bid: float = quotes['p_bid']
        else:  # symmetric
            quotes = symmetric_quotes(S, t, T, gamma, sigma, k)
            r_val = float('nan')
            delta_ask = quotes['delta_ask']
            delta_bid = quotes['delta_bid']
            p_ask = quotes['p_ask']
            p_bid = quotes['p_bid']

        # --- Step b: fill intensities ---
        lambda_a: float = lambda_from_delta(delta_ask, A, k)
        lambda_b: float = lambda_from_delta(delta_bid, A, k)

        # --- Diagnostic checks ---
        if delta_ask < 0:
            neg_delta_a += 1
        if delta_bid < 0:
            neg_delta_b += 1
        if lambda_a * dt > 1:
            prob_gt1_ask += 1
        if lambda_b * dt > 1:
            prob_gt1_bid += 1

        # --- Step c: draw two uniforms ---
        u1: float = rng.random()
        u2: float = rng.random()

        # --- Step d: Bernoulli fill decisions ---
        ask_fill: bool = u1 < lambda_a * dt
        bid_fill: bool = u2 < lambda_b * dt

        if ask_fill and bid_fill:
            simultaneous_fills += 1

        # --- Steps e & f: update inventory and cash ---
        if ask_fill:
            q -= 1
            X += p_ask
        if bid_fill:
            q += 1
            X -= p_bid

        # --- Record history before mid-price update ---
        if record_history:
            times[i] = t
            S_hist[i] = S
            if strategy == 'inventory':
                r_hist[i] = r_val  # type: ignore[index]
            p_ask_hist[i] = p_ask
            p_bid_hist[i] = p_bid
            q_hist[i] = q
            X_hist[i] = X
            lambda_a_hist[i] = lambda_a
            lambda_b_hist[i] = lambda_b

        # --- Step g: advance mid-price ---
        sign: float = 1.0 if rng.random() < 0.5 else -1.0
        S += sign * sigma * sqrt_dt

    terminal_profit: float = X + q * S

    result: dict = {
        'terminal_profit': terminal_profit,
        'final_q': q,
        'diagnostics': {
            'neg_delta_a': neg_delta_a,
            'neg_delta_b': neg_delta_b,
            'prob_gt1_ask': prob_gt1_ask,
            'prob_gt1_bid': prob_gt1_bid,
            'simultaneous_fills': simultaneous_fills,
        },
    }

    if record_history:
        result['history'] = {
            'times': times,
            'S_hist': S_hist,
            'r_hist': r_hist,
            'p_ask_hist': p_ask_hist,
            'p_bid_hist': p_bid_hist,
            'q_hist': q_hist,
            'X_hist': X_hist,
            'lambda_a_hist': lambda_a_hist,
            'lambda_b_hist': lambda_b_hist,
        }

    return result
