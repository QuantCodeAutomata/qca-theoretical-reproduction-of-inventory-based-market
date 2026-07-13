# Project Agent Memory – Avellaneda-Stoikov Market Making

## Project Overview
Inventory-aware market-making theory and simulation based on the Avellaneda-Stoikov (2008) model.
Reference: "High-frequency trading in a limit order book", Quantitative Finance 8(3), 217–224.

## Directory Layout
```
src/
  theory/          # Closed-form formulas (Exp 1)
    frozen_inventory.py   – value function, reservation prices, indifference
    hjb.py                – HJB PDE, theta ansatz, approximate solution
    quotes.py             – spread formulas, FOC, quote prices
  sim/             # Simulation engine (existing)
  analysis/        # Analysis utilities (existing)
exp/
  run_theory.py    – standalone Exp 1 script, saves plots+CSV to results/
tests/
  test_theory.py   – 72 pytest tests, all passing
results/           – generated plots and CSVs (not in version control)
```

## Key Formulas (Exp 1 / Theory)

### Frozen-inventory value function
`v(x,s,q,t) = -exp(-γx)·exp(-γqs)·exp(γ²q²σ²(T-t)/2)`

### Reservation prices
- `r^a = s + ((1-2q)·γ·σ²·(T-t))/2`
- `r^b = s + ((-1-2q)·γ·σ²·(T-t))/2`
- `r   = s - q·γ·σ²·(T-t)`

### Quadratic theta ansatz
`θ(s,q,t) ≈ q·s - (1/2)·γ·σ²·(T-t)·q²`
- `r^b = θ(s,q+1,t) - θ(s,q,t)`
- `r^a = θ(s,q,t) - θ(s,q-1,t)`
- Diffusion terms cancel exactly for this quadratic form.

### Optimal spreads (exponential intensity λ=A·exp(-kδ))
- `δ^a = (1/γ)·ln(1+γ/k) + ((1-2q)·γ·σ²·(T-t))/2`
- `δ^b = (1/γ)·ln(1+γ/k) + ((1+2q)·γ·σ²·(T-t))/2`
- `h   = (γ·σ²·(T-t))/2 + (1/γ)·ln(1+γ/k)`
- `p^a = r + h`,  `p^b = r - h`

### FOC simplification
`λ/(dλ/dδ) = -1/k`  →  `δ^b = (s-r^b) + (1/γ)·ln(1+γ/k)`

## Testing
- Run: `pytest tests/test_theory.py -v`  (72 tests, all pass in ~1s)
- Uses log-space comparisons for utility indifference (avoids exp overflow)
- Parametrize decorators used for gamma, q, t sweeps

## Conventions
- `tau = T - t` (time to maturity) used throughout
- `verify_utility_indifference` uses x=0 internally and compares log(-v)
- `symbolic_derivation()`, `symbolic_hjb_ansatz()`, `foc_*()` print summaries and return dicts/expressions
- All plots use `matplotlib.use("Agg")` (non-interactive backend)
- Results saved to `/workspace/project/results/`

## Dependencies (all pre-installed)
numpy 1.26.4, sympy 1.14.0, pandas 2.2.2, matplotlib 3.10.9, pytest 9.0.3

---

## Exp 2 – Monte Carlo Simulation

### Added Files
```
src/sim/intensity.py   – lambda_from_delta, validate_monotonicity, fill probabilities
src/sim/strategies.py  – half_spread, inventory_center, inventory_quotes, symmetric_quotes,
                          constant_spread_component
src/sim/engine.py      – single_path_simulation (core Bernoulli-step loop)
src/sim/monte_carlo.py – run_monte_carlo, run_all_experiments (CRN), compute_summary_stats
src/analysis/summary.py – format_results_table, compute_diagnostics_report, compare_strategies
src/analysis/plots.py  – plot_single_path, plot_profit_histograms, plot_summary_comparison, plot_all
config/defaults.py     – DEFAULT_PARAMS, GAMMA_VALUES, PAPER_RESULTS, N_PATHS, SEED
exp/run_simulation.py  – full Exp 2 runner (saves plots + CSV to results/)
tests/test_simulation.py – 16 pytest tests for Exp 2 (all pass)
```

### Simulation Loop (engine.py)
At each of n_steps=200 steps:
1. Compute quotes (inventory or symmetric)
2. lambda_a = A*exp(-k*delta_ask); lambda_b = A*exp(-k*delta_bid)
3. Draw u1, u2 ~ Uniform; ask_fill = u1 < lambda_a*dt; bid_fill = u2 < lambda_b*dt
4. If ask_fill: q -= 1, X += p_ask; if bid_fill: q += 1, X -= p_bid
5. Draw u3 ~ Uniform; S += (+1 if u3 < 0.5 else -1) * sigma * sqrt(dt)
Terminal: profit = X + q * S

### Common Random Numbers
`run_all_experiments` uses `seed + gamma_index` per gamma so that both strategies
at the same gamma share the seed → variance-reduced strategy comparison.

### Test Patterns (test_simulation.py)
- `_SeqRNG`: fake RNG replaying a fixed float list (controls fills deterministically)
- One-step tests: sigma=0, dt=1, T=1, A=1, k=1.5 → lam·dt ≈ 0.38 < 1
  - u=0.0 forces fill (0 < 0.38); u=1.0 blocks fill (1 > 0.38)
- `test_risk_reduction`: 200 paths for speed; gamma=0.5 shows clearest separation

### Qualitative Results
For all gamma: std_profit(inventory) < std_profit(symmetric) ✓
For all gamma: std_q(inventory)      < std_q(symmetric)      ✓
gamma=0.5 shows stronger inventory skewing (larger risk reduction ratio)
