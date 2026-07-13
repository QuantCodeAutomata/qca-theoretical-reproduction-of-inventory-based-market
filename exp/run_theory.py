"""
Exp 1 – Theoretical reproduction of inventory-based market-making formulas.

This script validates and visualises the Avellaneda-Stoikov closed-form
results for an inventory-aware market maker with CARA utility and
exponential order-arrival intensity.

Outputs (written to /workspace/project/results/):
  theory_reservation_vs_q.png
  theory_halfspread_vs_ttm.png
  theory_quotes_vs_q.png
  theory_results.csv

Usage:
    python exp/run_theory.py
"""

import os
import sys
from itertools import product

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make project root importable regardless of working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.theory.frozen_inventory import (
    average_reservation_price,
    reservation_ask,
    reservation_bid,
    symbolic_derivation,
    verify_utility_indifference,
)
from src.theory.hjb import (
    approximate_theta,
    describe_hjb_terms,
    reservation_from_theta,
    symbolic_hjb_ansatz,
    verify_reservation_from_theta,
)
from src.theory.quotes import (
    constant_spread_component,
    delta_ask,
    delta_bid,
    foc_exponential_simplification,
    foc_general_symbolic,
    half_spread,
    quoted_ask,
    quoted_bid,
    total_spread,
    verify_quote_equivalence,
)

RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Section 1 – Symbolic derivations
# ---------------------------------------------------------------------------


def run_symbolic_derivations() -> None:
    print("\n" + "#" * 70)
    print("# SECTION 1: SYMBOLIC DERIVATIONS")
    print("#" * 70)

    print("\n--- 1a. Frozen-inventory value function ---")
    sym_fi = symbolic_derivation()

    print("\n--- 1b. HJB equation terms ---")
    describe_hjb_terms()

    print("\n--- 1c. HJB exponential ansatz ---")
    symbolic_hjb_ansatz()

    print("\n--- 1d. General FOC (bid side) ---")
    foc_general_symbolic()

    print("\n--- 1e. Exponential intensity FOC simplification ---")
    foc_exponential_simplification()


# ---------------------------------------------------------------------------
# Section 2 – Numerical validation table
# ---------------------------------------------------------------------------

S = 100.0
SIGMA = 2.0
T = 1.0
K = 1.5
GAMMAS = [0.01, 0.1, 0.5]
QS = [-5, 0, 5]
TS = [0.0, 0.5, 0.99]


def build_results_table() -> pd.DataFrame:
    rows = []
    for gamma, q, t in product(GAMMAS, QS, TS):
        r_a = reservation_ask(S, q, t, T, gamma, SIGMA)
        r_b = reservation_bid(S, q, t, T, gamma, SIGMA)
        r = average_reservation_price(S, q, t, T, gamma, SIGMA)
        da = delta_ask(q, gamma, SIGMA, T, t, K)
        db = delta_bid(q, gamma, SIGMA, T, t, K)
        h = half_spread(gamma, SIGMA, T, t, K)
        pa = quoted_ask(S, q, gamma, SIGMA, T, t, K)
        pb = quoted_bid(S, q, gamma, SIGMA, T, t, K)
        spread = total_spread(gamma, SIGMA, T, t, K)
        csc = constant_spread_component(gamma, K)
        ba, bb = verify_utility_indifference(S, q, t, T, gamma, SIGMA)
        eq = verify_quote_equivalence(S, q, gamma, SIGMA, T, t, K)
        rows.append({
            "gamma": gamma,
            "q": q,
            "t": t,
            "tau": T - t,
            "r_ask": r_a,
            "r_bid": r_b,
            "r_avg": r,
            "delta_ask": da,
            "delta_bid": db,
            "half_spread": h,
            "quoted_ask": pa,
            "quoted_bid": pb,
            "total_spread": spread,
            "const_spread": csc,
            "indiff_ask_ok": ba,
            "indiff_bid_ok": bb,
            "quote_equiv_ok": eq,
        })
    return pd.DataFrame(rows)


def run_numerical_validation() -> pd.DataFrame:
    print("\n" + "#" * 70)
    print("# SECTION 2: NUMERICAL VALIDATION")
    print("#" * 70)

    df = build_results_table()

    print(f"\nResults table: {len(df)} rows (gamma × q × t combinations)")
    print(df[[
        "gamma", "q", "t", "r_avg", "delta_ask", "delta_bid", "half_spread"
    ]].to_string(index=False, float_format="{:.4f}".format))

    # Verify all sanity checks pass
    assert df["indiff_ask_ok"].all(), "Utility indifference (ask) failed for some params"
    assert df["indiff_bid_ok"].all(), "Utility indifference (bid) failed for some params"
    assert df["quote_equiv_ok"].all(), "Quote equivalence (p^a=r+h) failed for some params"
    print("\n✓ All utility-indifference checks pass")
    print("✓ All quote-equivalence checks pass")

    return df


# ---------------------------------------------------------------------------
# Section 3 – Plots
# ---------------------------------------------------------------------------


def plot_reservation_vs_q(df: pd.DataFrame) -> None:
    """Plot average reservation price r vs. inventory q for different gamma at t=0."""
    print("\n--- Plot: reservation price vs q ---")
    fig, ax = plt.subplots(figsize=(8, 5))

    q_fine = np.linspace(-10, 10, 200)
    colors = ["steelblue", "darkorange", "seagreen"]

    for gamma, color in zip(GAMMAS, colors):
        r_vals = [average_reservation_price(S, q, t=0.0, T=T, gamma=gamma, sigma=SIGMA)
                  for q in q_fine]
        ax.plot(q_fine, r_vals, color=color, lw=2,
                label=f"γ = {gamma}")

    ax.axhline(S, color="black", linestyle="--", lw=1, label="Mid-price s")
    ax.set_xlabel("Inventory q", fontsize=12)
    ax.set_ylabel("Reservation price r", fontsize=12)
    ax.set_title("Average Reservation Price vs Inventory\n(s=100, σ=2, T=1, t=0)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _save(fig, "theory_reservation_vs_q.png")


def plot_halfspread_vs_ttm() -> None:
    """Plot half-spread h vs. time-to-maturity τ = T-t for different gamma."""
    print("\n--- Plot: half-spread vs time-to-maturity ---")
    fig, ax = plt.subplots(figsize=(8, 5))

    tau_vals = np.linspace(0.0, T, 300)
    colors = ["steelblue", "darkorange", "seagreen"]

    for gamma, color in zip(GAMMAS, colors):
        h_vals = [half_spread(gamma, SIGMA, T, T - tau, K) for tau in tau_vals]
        ax.plot(tau_vals, h_vals, color=color, lw=2, label=f"γ = {gamma}")
        # Mark the constant component at τ=0
        h0 = (1.0 / gamma) * np.log(1.0 + gamma / K)
        ax.axhline(h0, color=color, linestyle=":", lw=1, alpha=0.6)

    ax.set_xlabel("Time to maturity  τ = T − t", fontsize=12)
    ax.set_ylabel("Half-spread  h(t)", fontsize=12)
    ax.set_title(
        "Half-Spread vs Time to Maturity\n"
        "(s=100, σ=2, T=1, k=1.5; dotted = constant component)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _save(fig, "theory_halfspread_vs_ttm.png")


def plot_quotes_vs_q() -> None:
    """Plot delta^a and delta^b vs. inventory q at t=0.5."""
    print("\n--- Plot: quote distances vs q ---")
    fig, ax = plt.subplots(figsize=(8, 5))

    q_fine = np.linspace(-10, 10, 200)
    t_mid = 0.5
    gamma = 0.1

    da_vals = [delta_ask(q, gamma, SIGMA, T, t_mid, K) for q in q_fine]
    db_vals = [delta_bid(q, gamma, SIGMA, T, t_mid, K) for q in q_fine]

    ax.plot(q_fine, da_vals, color="steelblue", lw=2, label="δ^a (ask distance)")
    ax.plot(q_fine, db_vals, color="darkorange", lw=2, label="δ^b (bid distance)")
    ax.axhline(0, color="black", linestyle="--", lw=0.8)

    ax.set_xlabel("Inventory q", fontsize=12)
    ax.set_ylabel("Quote distance from mid (δ)", fontsize=12)
    ax.set_title(
        "Optimal Quote Distances vs Inventory\n"
        f"(γ={gamma}, σ=2, T=1, t=0.5, k=1.5)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _save(fig, "theory_quotes_vs_q.png")


# ---------------------------------------------------------------------------
# Section 4 – Limit checks
# ---------------------------------------------------------------------------


def run_limit_checks() -> None:
    print("\n" + "#" * 70)
    print("# SECTION 4: LIMIT / CONVERGENCE CHECKS")
    print("#" * 70)

    q, gamma = 3.0, 0.1

    # (a) As t → T, r → s
    t_near_T = T - 1e-9
    r_near_T = average_reservation_price(S, q, t_near_T, T, gamma, SIGMA)
    print(f"\n(a) r as t→T  (q={q}): r={r_near_T:.8f},  s={S:.8f}")
    assert np.isclose(r_near_T, S, atol=1e-5), "r should converge to s as t→T"
    print("    ✓ r → s as t → T")

    # (b) As gamma → 0, r → s
    tiny_gammas = [0.1, 0.01, 0.001, 0.0001]
    r_vals = [average_reservation_price(S, q, 0.0, T, g, SIGMA) for g in tiny_gammas]
    print(f"\n(b) r as γ→0 (q={q}, t=0):")
    for g, r in zip(tiny_gammas, r_vals):
        print(f"    γ={g:.4f}  →  r={r:.6f}")
    assert np.isclose(r_vals[-1], S, atol=1e-3), "r should → s as gamma → 0"
    print("    ✓ r → s as γ → 0")

    # (c) Utility indifference holds at extreme q values
    print(f"\n(c) Utility indifference at extreme q:")
    for q_ext in [-10.0, 0.0, 10.0]:
        ba, bb = verify_utility_indifference(S, q_ext, 0.5, T, gamma, SIGMA)
        print(f"    q={q_ext:5.0f}: ask_ok={ba}, bid_ok={bb}")
        assert ba and bb, f"Indifference failed at q={q_ext}"
    print("    ✓ Indifference holds at extreme inventories")

    # (d) Theta-derived prices match formulas
    print(f"\n(d) Theta-derived prices vs. direct formulas:")
    for q_chk, gamma_chk in [(0.0, 0.1), (3.0, 0.1), (-3.0, 0.5)]:
        res = verify_reservation_from_theta(S, q_chk, 0.5, T, gamma_chk, SIGMA)
        print(
            f"    q={q_chk:+.0f}, γ={gamma_chk}: "
            f"r^a theta={res['r_ask_theta']:.4f} formula={res['r_ask_formula']:.4f}  "
            f"ask_match={res['ask_match']}  bid_match={res['bid_match']}"
        )
        assert res["ask_match"] and res["bid_match"]
    print("    ✓ Theta-derived prices match direct formulas")


# ---------------------------------------------------------------------------
# Section 5 – Save results
# ---------------------------------------------------------------------------


def save_results(df: pd.DataFrame) -> None:
    path = os.path.join(RESULTS_DIR, "theory_results.csv")
    df.to_csv(path, index=False, float_format="%.8f")
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Section 6 – Derivation summary
# ---------------------------------------------------------------------------


def print_derivation_summary() -> None:
    print("\n" + "#" * 70)
    print("# DERIVATION SUMMARY")
    print("#" * 70)
    summary = """
Avellaneda-Stoikov Market Making – Key Formulas
================================================

1. FROZEN-INVENTORY VALUE FUNCTION
   v(x,s,q,t) = -exp(-γx) · exp(-γqs) · exp(γ²q²σ²(T-t)/2)

2. RESERVATION PRICES (utility-indifference)
   r^a  =  s + [(1 − 2q)·γ·σ²·(T−t)] / 2
   r^b  =  s + [(−1 − 2q)·γ·σ²·(T−t)] / 2
   r    =  (r^a + r^b)/2  =  s − q·γ·σ²·(T−t)

3. APPROXIMATE PERFORMANCE INDEX (quadratic ansatz)
   θ(s,q,t) ≈ q·s − ½·γ·σ²·(T−t)·q²
   → r^b = θ(s,q+1,t) − θ(s,q,t)   (recover reservation bid)
   → r^a = θ(s,q,t) − θ(s,q−1,t)   (recover reservation ask)

4. OPTIMAL SPREADS (exponential intensity λ = A·exp(−kδ))
   δ^a = (1/γ)·ln(1+γ/k) + [(1−2q)·γ·σ²·(T−t)] / 2
   δ^b = (1/γ)·ln(1+γ/k) + [(1+2q)·γ·σ²·(T−t)] / 2

5. HALF-SPREAD AND QUOTE DECOMPOSITION
   h    = (γ·σ²·(T−t))/2 + (1/γ)·ln(1+γ/k)
   p^a  = r + h          p^b  = r − h
   Total spread = γ·σ²·(T−t) + (2/γ)·ln(1+γ/k)

6. FOC SIMPLIFICATION (exponential intensity)
   λ / (dλ/dδ) = −1/k
   → δ^b = (s − r^b) + (1/γ)·ln(1+γ/k)
"""
    print(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  EXP 1 – Avellaneda-Stoikov Theory Reproduction")
    print("=" * 70)

    # 1. Symbolic derivations
    run_symbolic_derivations()

    # 2. Numerical validation table
    df = run_numerical_validation()

    # 3. Plots
    print("\n" + "#" * 70)
    print("# SECTION 3: PLOTS")
    print("#" * 70)
    plot_reservation_vs_q(df)
    plot_halfspread_vs_ttm()
    plot_quotes_vs_q()

    # 4. Limit / convergence checks
    run_limit_checks()

    # 5. Save CSV
    save_results(df)

    # 6. Summary
    print_derivation_summary()

    print("\n" + "=" * 70)
    print("  ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
