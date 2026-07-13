"""
Experiment 2 – Inventory-aware market making (Avellaneda-Stoikov).

Run this script from the project root:

    python exp/run_simulation.py

It will:
1. Run the full Monte Carlo experiment (1 000 paths per strategy/gamma).
2. Generate and save diagnostic plots to results/.
3. Print the summary statistics table.
4. Print the diagnostics report.
5. Validate the key qualitative claims from the paper.
6. Save results/simulation_results.csv.
7. Print a side-by-side comparison with paper values.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the path when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from config.defaults import DEFAULT_PARAMS, GAMMA_VALUES, N_PATHS, SEED, PAPER_RESULTS
from src.sim.engine import single_path_simulation
from src.sim.monte_carlo import run_all_experiments, compute_summary_stats
from src.analysis.summary import format_results_table, compute_diagnostics_report, compare_strategies
from src.analysis.plots import plot_all

RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Full Monte Carlo experiment
    # ------------------------------------------------------------------
    print("Running Monte Carlo experiments …")
    results = run_all_experiments(n_paths=N_PATHS, params=DEFAULT_PARAMS, seed=SEED)
    print("Done.\n")

    # ------------------------------------------------------------------
    # 2. Representative single-path histories for gamma=0.1
    # ------------------------------------------------------------------
    rep_gamma = 0.1
    rep_seed = 77777
    rng_inv = np.random.default_rng(rep_seed)
    rng_sym = np.random.default_rng(rep_seed)  # same seed → CRN

    res_inv_path = single_path_simulation(
        'inventory', rep_gamma, DEFAULT_PARAMS, rng_inv, record_history=True
    )
    res_sym_path = single_path_simulation(
        'symmetric', rep_gamma, DEFAULT_PARAMS, rng_sym, record_history=True
    )

    history_inv = res_inv_path['history']
    history_sym = res_sym_path['history']

    # ------------------------------------------------------------------
    # 3. Generate all plots
    # ------------------------------------------------------------------
    print("Generating plots …")
    df = compute_summary_stats(results)
    plot_all(results, df, DEFAULT_PARAMS, RESULTS_DIR)
    print(f"Plots saved to {RESULTS_DIR}\n")

    # ------------------------------------------------------------------
    # 4. Summary stats table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(format_results_table(df))
    print()

    # ------------------------------------------------------------------
    # 5. Diagnostics report
    # ------------------------------------------------------------------
    print(compute_diagnostics_report(results))

    # ------------------------------------------------------------------
    # 6. Qualitative validation assertions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("QUALITATIVE VALIDATIONS")
    print("=" * 70)

    comparison = compare_strategies(results)
    all_passed = True

    for gamma in GAMMA_VALUES:
        inv_profits = results[gamma]['inventory']['profits']
        sym_profits = results[gamma]['symmetric']['profits']
        inv_qs = results[gamma]['inventory']['final_qs']
        sym_qs = results[gamma]['symmetric']['final_qs']

        std_p_inv = float(np.std(inv_profits, ddof=1))
        std_p_sym = float(np.std(sym_profits, ddof=1))
        std_q_inv = float(np.std(inv_qs, ddof=1))
        std_q_sym = float(np.std(sym_qs, ddof=1))

        ok_profit = std_p_inv < std_p_sym
        ok_q = std_q_inv < std_q_sym
        status_p = "PASS" if ok_profit else "FAIL"
        status_q = "PASS" if ok_q else "FAIL"

        print(f"gamma={gamma:>4}  std_profit: inv={std_p_inv:.3f} < sym={std_p_sym:.3f}  [{status_p}]")
        print(f"gamma={gamma:>4}  std_q:      inv={std_q_inv:.3f} < sym={std_q_sym:.3f}  [{status_q}]")

        if not ok_profit or not ok_q:
            all_passed = False

    # gamma=0.01 vs gamma=0.5: smaller difference for gamma=0.01
    diff_01 = comparison['risk_reduction_profit_std'][0.01]
    diff_05 = comparison['risk_reduction_profit_std'][0.5]
    # gamma=0.5 has stronger inventory skewing → larger risk reduction
    ok_diff = diff_05 < diff_01  # smaller ratio = more risk reduction for gamma=0.5
    status_diff = "PASS" if ok_diff else "FAIL"
    print(
        f"\ngamma=0.5 has stronger risk reduction than gamma=0.01 in profit std "
        f"(ratio {diff_05:.3f} < {diff_01:.3f})  [{status_diff}]"
    )
    if not ok_diff:
        all_passed = False

    print(f"\nOverall validation: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    # ------------------------------------------------------------------
    # 7. Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, 'simulation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to {csv_path}")

    # ------------------------------------------------------------------
    # 8. Comparison with paper values
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER VALUES")
    print("=" * 70)
    header = (
        f"{'gamma':>5}  {'strategy':>10}  {'metric':>14}  "
        f"{'simulated':>10}  {'paper':>10}  {'diff%':>7}"
    )
    print(header)
    print("-" * len(header))

    for gamma in GAMMA_VALUES:
        for strategy in ['inventory', 'symmetric']:
            sim_row = df[(df['gamma'] == gamma) & (df['strategy'] == strategy)].iloc[0]
            paper = PAPER_RESULTS[gamma][strategy]
            for metric in ['mean_profit', 'std_profit', 'mean_q', 'std_q', 'spread']:
                sim_val = sim_row[metric]
                paper_val = paper[metric]
                diff_pct = (sim_val - paper_val) / abs(paper_val) * 100 if paper_val != 0 else float('nan')
                print(
                    f"{gamma:>5}  {strategy:>10}  {metric:>14}  "
                    f"{sim_val:>10.4f}  {paper_val:>10.4f}  {diff_pct:>7.2f}%"
                )


if __name__ == '__main__':
    main()
