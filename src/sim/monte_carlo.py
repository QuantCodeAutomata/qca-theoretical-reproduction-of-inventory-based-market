"""
Monte Carlo orchestration for the Avellaneda-Stoikov market-making model.

Key design choices
------------------
* **Common Random Numbers (CRN)**: Within each gamma, the inventory and
  symmetric strategies are driven by the *same* sequence of random numbers.
  This reduces noise in the strategy comparison and is the standard approach
  used in the Avellaneda-Stoikov paper.
* Seeds for different gamma values are offset from the base seed so that
  different gamma experiments are independent of each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sim.engine import single_path_simulation
from src.sim.strategies import constant_spread_component


def run_monte_carlo(
    strategy: str,
    gamma: float,
    n_paths: int,
    params: dict,
    seed: int = 12345,
) -> dict:
    """Run n_paths simulations for a single (strategy, gamma) combination.

    Parameters
    ----------
    strategy:
        ``'inventory'`` or ``'symmetric'``.
    gamma:
        Absolute risk-aversion coefficient.
    n_paths:
        Number of independent Monte Carlo paths.
    params:
        Simulation parameter dict (see :func:`~src.sim.engine.single_path_simulation`).
    seed:
        Seed for the NumPy random-number generator.

    Returns
    -------
    dict
        * ``profits``     – np.ndarray of shape (n_paths,), terminal P&L per path
        * ``final_qs``    – np.ndarray of shape (n_paths,), final inventory per path
        * ``diagnostics`` – dict of aggregated diagnostic counts (summed over paths)
        * ``strategy``    – str, echoed from input
        * ``gamma``       – float, echoed from input
    """
    rng = np.random.default_rng(seed)

    profits = np.empty(n_paths)
    final_qs = np.empty(n_paths)

    agg_diag: dict = {
        'neg_delta_a': 0,
        'neg_delta_b': 0,
        'prob_gt1_ask': 0,
        'prob_gt1_bid': 0,
        'simultaneous_fills': 0,
    }

    for i in range(n_paths):
        result = single_path_simulation(strategy, gamma, params, rng, record_history=False)
        profits[i] = result['terminal_profit']
        final_qs[i] = result['final_q']
        for key in agg_diag:
            agg_diag[key] += result['diagnostics'][key]

    return {
        'profits': profits,
        'final_qs': final_qs,
        'diagnostics': agg_diag,
        'strategy': strategy,
        'gamma': gamma,
    }


def run_all_experiments(
    n_paths: int = 1000,
    params: dict = None,
    seed: int = 12345,
) -> dict:
    """Run all (gamma, strategy) combinations using Common Random Numbers.

    For each gamma, inventory and symmetric strategies share the same seed so
    that their path-by-path comparison is variance-reduced.  Different gamma
    values use seeds offset by their index so they remain independent.

    Parameters
    ----------
    n_paths:
        Number of Monte Carlo paths per (gamma, strategy) pair.
    params:
        Simulation parameter dict.  Defaults to :data:`~config.defaults.DEFAULT_PARAMS`
        if ``None``.
    seed:
        Base seed.  Gamma-specific seeds are ``seed + gamma_index``.

    Returns
    -------
    dict
        Nested dict ``results[gamma][strategy]`` where each leaf is the output
        of :func:`run_monte_carlo`.
    """
    if params is None:
        from config.defaults import DEFAULT_PARAMS
        params = DEFAULT_PARAMS

    from config.defaults import GAMMA_VALUES, STRATEGIES

    results: dict = {}

    for gamma_idx, gamma in enumerate(GAMMA_VALUES):
        gamma_seed = seed + gamma_idx
        results[gamma] = {}
        for strategy in STRATEGIES:
            results[gamma][strategy] = run_monte_carlo(
                strategy=strategy,
                gamma=gamma,
                n_paths=n_paths,
                params=params,
                seed=gamma_seed,
            )

    return results


def compute_summary_stats(results: dict) -> pd.DataFrame:
    """Build a summary DataFrame from the Monte Carlo results.

    Parameters
    ----------
    results:
        Nested dict as returned by :func:`run_all_experiments`.

    Returns
    -------
    pd.DataFrame
        Columns: ``gamma``, ``strategy``, ``mean_profit``, ``std_profit``,
        ``mean_q``, ``std_q``, ``spread``.

        * ``std_profit`` and ``std_q`` use ``ddof=1`` (sample standard deviation).
        * ``spread`` is the constant component :func:`~src.sim.strategies.constant_spread_component`.
    """
    rows = []
    for gamma, strat_dict in results.items():
        # k=1.5 matches DEFAULT_PARAMS; constant_spread_component gives the
        # paper's "spread" column: (2/γ)·ln(1 + γ/k).
        spread_val = constant_spread_component(gamma, 1.5)
        for strategy, mc_out in strat_dict.items():
            profits = mc_out['profits']
            final_qs = mc_out['final_qs']
            rows.append({
                'gamma': gamma,
                'strategy': strategy,
                'mean_profit': float(np.mean(profits)),
                'std_profit': float(np.std(profits, ddof=1)),
                'mean_q': float(np.mean(final_qs)),
                'std_q': float(np.std(final_qs, ddof=1)),
                'spread': spread_val,
            })

    df = pd.DataFrame(rows, columns=['gamma', 'strategy', 'mean_profit', 'std_profit',
                                     'mean_q', 'std_q', 'spread'])
    return df
