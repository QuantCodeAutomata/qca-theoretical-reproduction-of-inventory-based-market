"""
Textual reporting and strategy-comparison utilities.

Functions here consume the raw Monte Carlo results dict and the summary
DataFrame produced by :func:`~src.sim.monte_carlo.compute_summary_stats`
and turn them into human-readable strings or summary statistics.
"""

from __future__ import annotations

import pandas as pd


def format_results_table(df: pd.DataFrame) -> str:
    """Format the summary DataFrame as a nicely aligned plain-text table.

    Parameters
    ----------
    df:
        Summary DataFrame as returned by
        :func:`~src.sim.monte_carlo.compute_summary_stats`.

    Returns
    -------
    str
        Multi-line string with column headers and right-aligned numeric columns.
    """
    header = (
        f"{'gamma':>6}  {'strategy':>10}  {'spread':>7}  "
        f"{'mean_profit':>12}  {'std_profit':>10}  "
        f"{'mean_q':>8}  {'std_q':>7}"
    )
    separator = "-" * len(header)
    lines = [header, separator]

    for _, row in df.iterrows():
        line = (
            f"{row['gamma']:>6.3f}  {row['strategy']:>10}  "
            f"{row['spread']:>7.3f}  "
            f"{row['mean_profit']:>12.4f}  {row['std_profit']:>10.4f}  "
            f"{row['mean_q']:>8.4f}  {row['std_q']:>7.4f}"
        )
        lines.append(line)

    return "\n".join(lines)


def compute_diagnostics_report(results: dict) -> str:
    """Build a text report of all diagnostic counters across every run.

    The report groups output by (gamma, strategy) and shows per-path
    averages for each diagnostic counter.

    Parameters
    ----------
    results:
        Nested dict ``results[gamma][strategy]`` as returned by
        :func:`~src.sim.monte_carlo.run_all_experiments`.

    Returns
    -------
    str
        Multi-line diagnostic report.
    """
    lines = ["=" * 70, "DIAGNOSTIC REPORT", "=" * 70]

    for gamma, strat_dict in results.items():
        lines.append(f"\ngamma = {gamma}")
        lines.append("-" * 40)
        for strategy, mc_out in strat_dict.items():
            diag = mc_out['diagnostics']
            n_paths = len(mc_out['profits'])
            lines.append(f"  Strategy: {strategy}")
            lines.append(f"    neg_delta_a        (total / per path): "
                         f"{diag['neg_delta_a']:>8} / {diag['neg_delta_a'] / n_paths:>8.3f}")
            lines.append(f"    neg_delta_b        (total / per path): "
                         f"{diag['neg_delta_b']:>8} / {diag['neg_delta_b'] / n_paths:>8.3f}")
            lines.append(f"    prob_gt1_ask       (total / per path): "
                         f"{diag['prob_gt1_ask']:>8} / {diag['prob_gt1_ask'] / n_paths:>8.3f}")
            lines.append(f"    prob_gt1_bid       (total / per path): "
                         f"{diag['prob_gt1_bid']:>8} / {diag['prob_gt1_bid'] / n_paths:>8.3f}")
            lines.append(f"    simultaneous_fills (total / per path): "
                         f"{diag['simultaneous_fills']:>8} / "
                         f"{diag['simultaneous_fills'] / n_paths:>8.3f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def compare_strategies(results: dict) -> dict:
    """Compute risk-reduction ratios of inventory vs symmetric strategy.

    For each gamma, returns the ratio ``inventory_std / symmetric_std`` for
    both terminal profit and final inventory.  Values below 1.0 indicate
    that the inventory strategy successfully reduces variability.

    Parameters
    ----------
    results:
        Nested dict ``results[gamma][strategy]`` as returned by
        :func:`~src.sim.monte_carlo.run_all_experiments`.

    Returns
    -------
    dict
        Keys ``'risk_reduction_profit_std'`` and ``'risk_reduction_q_std'``,
        each mapping ``gamma -> ratio``.  A ratio < 1 means inventory strategy
        has lower standard deviation than symmetric.
    """
    import numpy as np

    risk_reduction_profit: dict = {}
    risk_reduction_q: dict = {}

    for gamma, strat_dict in results.items():
        inv = strat_dict['inventory']
        sym = strat_dict['symmetric']

        std_profit_inv = float(np.std(inv['profits'], ddof=1))
        std_profit_sym = float(np.std(sym['profits'], ddof=1))
        std_q_inv = float(np.std(inv['final_qs'], ddof=1))
        std_q_sym = float(np.std(sym['final_qs'], ddof=1))

        risk_reduction_profit[gamma] = std_profit_inv / std_profit_sym
        risk_reduction_q[gamma] = std_q_inv / std_q_sym

    return {
        'risk_reduction_profit_std': risk_reduction_profit,
        'risk_reduction_q_std': risk_reduction_q,
    }
