"""
Visualisation utilities for the Avellaneda-Stoikov Monte Carlo results.

All functions save figures to disk and return nothing.  Matplotlib is used
throughout; seaborn is applied for styling where available.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', palette='muted')
    _SEABORN = True
except ImportError:
    _SEABORN = False


def plot_single_path(
    history_inv: dict,
    history_sym: dict,
    gamma: float,
    save_path: str,
) -> None:
    """Plot mid-price, reservation price, and bid/ask quotes for one path.

    Two subplots are produced side-by-side:
    - Left:  inventory strategy (mid, reservation price, bid, ask).
    - Right: symmetric strategy (mid, bid, ask).

    Parameters
    ----------
    history_inv:
        ``history`` dict from :func:`~src.sim.engine.single_path_simulation`
        with ``strategy='inventory'``.
    history_sym:
        ``history`` dict from :func:`~src.sim.engine.single_path_simulation`
        with ``strategy='symmetric'``.
    gamma:
        Risk-aversion coefficient (used in the figure title).
    save_path:
        Full file path (including extension) where the figure is saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Single-Path Quotes  (γ = {gamma})", fontsize=14)

    for ax, hist, title in [
        (axes[0], history_inv, 'Inventory Strategy'),
        (axes[1], history_sym, 'Symmetric Strategy'),
    ]:
        times = hist['times']
        ax.plot(times, hist['S_hist'], color='black', linewidth=1.2, label='Mid-price S')
        ax.plot(times, hist['p_ask_hist'], color='tab:red', linewidth=0.9,
                linestyle='--', label='Ask quote')
        ax.plot(times, hist['p_bid_hist'], color='tab:blue', linewidth=0.9,
                linestyle='--', label='Bid quote')
        if hist.get('r_hist') is not None:
            ax.plot(times, hist['r_hist'], color='tab:green', linewidth=1.0,
                    linestyle=':', label='Reservation price r')
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_profit_histograms(
    results: dict,
    gamma: float,
    save_path: str,
) -> None:
    """Overlapping terminal-profit histograms for inventory vs symmetric.

    Parameters
    ----------
    results:
        Nested dict ``results[gamma][strategy]`` from
        :func:`~src.sim.monte_carlo.run_all_experiments`.
    gamma:
        Risk-aversion coefficient whose results to plot.
    save_path:
        Full file path where the figure is saved.
    """
    inv_profits = results[gamma]['inventory']['profits']
    sym_profits = results[gamma]['symmetric']['profits']

    fig, ax = plt.subplots(figsize=(9, 5))

    all_profits = np.concatenate([inv_profits, sym_profits])
    bins = np.linspace(all_profits.min(), all_profits.max(), 50)

    ax.hist(inv_profits, bins=bins, alpha=0.55, color='tab:blue',
            label=f"Inventory  (μ={np.mean(inv_profits):.2f}, σ={np.std(inv_profits, ddof=1):.2f})")
    ax.hist(sym_profits, bins=bins, alpha=0.55, color='tab:orange',
            label=f"Symmetric  (μ={np.mean(sym_profits):.2f}, σ={np.std(sym_profits, ddof=1):.2f})")

    ax.axvline(np.mean(inv_profits), color='tab:blue', linestyle='--', linewidth=1.5)
    ax.axvline(np.mean(sym_profits), color='tab:orange', linestyle='--', linewidth=1.5)

    ax.set_title(f"Terminal Profit Distribution  (γ = {gamma})")
    ax.set_xlabel('Terminal Profit  X_T + q_T · S_T')
    ax.set_ylabel('Frequency')
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_summary_comparison(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart comparing mean profit and profit std for all (gamma, strategy).

    Two grouped bar charts are shown side-by-side: one for mean profit and
    one for the standard deviation of profit.

    Parameters
    ----------
    df:
        Summary DataFrame from :func:`~src.sim.monte_carlo.compute_summary_stats`.
    save_path:
        Full file path where the figure is saved.
    """
    gammas = df['gamma'].unique()
    strategies = df['strategy'].unique()
    x = np.arange(len(gammas))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Strategy Comparison Across Risk-Aversion Levels", fontsize=13)

    colours = {'inventory': 'tab:blue', 'symmetric': 'tab:orange'}

    for metric, ax, ylabel, title in [
        ('mean_profit', axes[0], 'Mean Terminal Profit', 'Mean Profit'),
        ('std_profit',  axes[1], 'Std Dev of Terminal Profit', 'Profit Std Dev'),
    ]:
        for s_idx, strat in enumerate(strategies):
            sub = df[df['strategy'] == strat]
            vals = [sub[sub['gamma'] == g][metric].values[0] for g in gammas]
            offset = (s_idx - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=strat,
                          color=colours.get(strat, f'C{s_idx}'))
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'γ={g}' for g in gammas])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_all(
    results: dict,
    df: pd.DataFrame,
    params: dict,
    output_dir: str,
) -> None:
    """Generate and save all plots for all gamma values.

    Plots produced
    --------------
    * ``single_path_gamma_<gamma>.png`` – per-gamma single-path quote plot.
    * ``profit_hist_gamma_<gamma>.png`` – per-gamma profit histogram.
    * ``summary_comparison.png``        – overall bar-chart comparison.

    Representative paths for the single-path plots are generated fresh
    (using a fixed seed) to avoid storing large history arrays in the
    results dict.

    Parameters
    ----------
    results:
        Nested dict from :func:`~src.sim.monte_carlo.run_all_experiments`.
    df:
        Summary DataFrame from :func:`~src.sim.monte_carlo.compute_summary_stats`.
    params:
        Simulation parameter dict.
    output_dir:
        Directory where all figures are saved (created if absent).
    """
    from src.sim.engine import single_path_simulation

    os.makedirs(output_dir, exist_ok=True)

    gammas = list(results.keys())
    for gamma_idx, gamma in enumerate(gammas):
        # --- Single-path plot ---
        path_seed = 99999 + gamma_idx
        rng_inv = np.random.default_rng(path_seed)
        rng_sym = np.random.default_rng(path_seed)

        res_inv = single_path_simulation('inventory', gamma, params, rng_inv,
                                         record_history=True)
        res_sym = single_path_simulation('symmetric', gamma, params, rng_sym,
                                         record_history=True)

        sp_path = os.path.join(output_dir, f'single_path_gamma_{gamma}.png')
        plot_single_path(res_inv['history'], res_sym['history'], gamma, sp_path)

        # --- Profit histogram ---
        ph_path = os.path.join(output_dir, f'profit_hist_gamma_{gamma}.png')
        plot_profit_histograms(results, gamma, ph_path)

    # --- Summary comparison bar chart ---
    sc_path = os.path.join(output_dir, 'summary_comparison.png')
    plot_summary_comparison(df, sc_path)
