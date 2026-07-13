# Inventory-Based Market-Making: Theoretical Reproduction & Monte Carlo Simulation

Reproduction of the Avellaneda-Stoikov (2008) inventory-aware market-making model, including:
- **Exp 1**: Symbolic and numerical validation of all closed-form formulas
- **Exp 2**: Monte Carlo simulation comparing inventory vs. symmetric quoting strategies

## Repository Structure

```
├── data/                    # Data directory (synthetic; no external data needed)
├── src/
│   ├── theory/
│   │   ├── frozen_inventory.py   # Value function, reservation prices, indifference
│   │   ├── hjb.py                # HJB equation, theta ansatz, approximate solution
│   │   └── quotes.py             # Spread formulas, FOC, quote prices
│   ├── sim/
│   │   ├── intensity.py          # Execution intensity functions
│   │   ├── strategies.py         # Inventory and symmetric quote computation
│   │   ├── engine.py             # Single-path simulation loop
│   │   └── monte_carlo.py        # Batch Monte Carlo runner
│   └── analysis/
│       ├── summary.py            # Summary statistics and diagnostics
│       └── plots.py              # All visualization functions
├── exp/
│   ├── run_theory.py             # Run Exp 1 (theoretical reproduction)
│   └── run_simulation.py         # Run Exp 2 (Monte Carlo simulation)
├── config/
│   └── defaults.py               # Parameter defaults and paper reference values
├── tests/
│   ├── test_theory.py            # 72 pytest tests for Exp 1
│   └── test_simulation.py        # 16 pytest tests for Exp 2
├── results/
│   ├── RESULTS.md                # Summary of all experiment findings
│   ├── theory_results.csv        # Tabulated theoretical formula evaluations
│   └── simulation_results.csv   # Monte Carlo summary statistics
└── AGENTS.md                     # Repository knowledge base
```

## Model Overview

The Avellaneda-Stoikov model describes an inventory-aware market maker who quotes bid and ask
prices in a continuous-time environment. The mid-price follows arithmetic Brownian motion and
execution intensities depend exponentially on quote distances.

### Key Formulas

**Frozen-inventory value function** (arithmetic BM, exponential utility):
```
v(x,s,q,t) = -exp(-γx) · exp(-γqs) · exp(γ²q²σ²(T-t)/2)
```

**Reservation prices** (utility-indifference):
```
r^a = s + [(1-2q)·γ·σ²·(T-t)] / 2
r^b = s + [(-1-2q)·γ·σ²·(T-t)] / 2
r   = s - q·γ·σ²·(T-t)           (average / indifference price)
```

**Optimal quotes** (λ(δ) = A·exp(-kδ)):
```
δ^a = (1/γ)·ln(1+γ/k) + [(1-2q)·γ·σ²·(T-t)] / 2
δ^b = (1/γ)·ln(1+γ/k) + [(1+2q)·γ·σ²·(T-t)] / 2
h   = (γ·σ²·(T-t))/2 + (1/γ)·ln(1+γ/k)   (half-spread)
p^a = r + h,  p^b = r - h
```

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Initial mid-price s₀ | 100 |
| Horizon T | 1 |
| Volatility σ | 2 |
| Time step dt | 0.005 (200 steps) |
| Intensity A | 140 |
| Intensity k | 1.5 |
| Initial inventory q₀ | 0 |
| Risk-aversion γ | {0.01, 0.1, 0.5} |
| Monte Carlo paths | 1000 |

## Installation & Usage

All required libraries are pre-installed in the environment. Standard scientific Python stack.

```bash
# Run Exp 1: Theoretical reproduction
python exp/run_theory.py

# Run Exp 2: Monte Carlo simulation
python exp/run_simulation.py

# Run all tests
pytest tests/ -v
```

## Results Summary

### Exp 1 – Theory
All 16 theoretical identities verified symbolically and numerically:
- Utility-indifference conditions hold to floating-point tolerance
- Quote equivalence p^a = r+h, p^b = r-h confirmed
- Limit checks: r→s as t→T; r→s as γ→0

### Exp 2 – Monte Carlo (1000 paths per setting)

| γ | Strategy | Spread | Mean Profit | Std Profit | Mean q | Std q |
|---|----------|--------|-------------|------------|--------|-------|
| 0.10 | Inventory | 1.291 | 64.85 | 6.73 | -0.06 | 2.99 |
| 0.10 | Symmetric | 1.291 | 67.74 | 13.05 | 0.13 | 8.39 |
| 0.01 | Inventory | 1.329 | 68.48 | 9.01 | -0.26 | 5.29 |
| 0.01 | Symmetric | 1.329 | 68.42 | 14.22 | -0.67 | 8.95 |
| 0.50 | Inventory | 1.151 | 48.40 | 5.89 | 0.05 | 1.98 |
| 0.50 | Symmetric | 1.151 | 58.56 | 11.65 | 0.50 | 7.01 |

**Key finding**: Inventory strategy consistently delivers lower P&L standard deviation and
lower final inventory standard deviation across all γ values. The effect is strongest at γ=0.5.

## Reference

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*.
Quantitative Finance, 8(3), 217–224.
