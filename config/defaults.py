"""Default configuration and paper reference values for the Avellaneda-Stoikov simulation."""

DEFAULT_PARAMS = {
    's0': 100.0,
    'T': 1.0,
    'sigma': 2.0,
    'dt': 0.005,
    'A': 140.0,
    'k': 1.5,
    'q0': 0,
    'x0': 0.0,
}

GAMMA_VALUES = [0.1, 0.01, 0.5]

N_PATHS = 1000

SEED = 12345

STRATEGIES = ['inventory', 'symmetric']

# Paper reported values for comparison (Avellaneda & Stoikov, 2008 – Table 1)
PAPER_RESULTS = {
    0.1: {
        'inventory': {
            'spread': 1.29,
            'mean_profit': 62.94,
            'std_profit': 5.89,
            'mean_q': 0.10,
            'std_q': 2.80,
        },
        'symmetric': {
            'spread': 1.29,
            'mean_profit': 67.21,
            'std_profit': 13.43,
            'mean_q': -0.018,
            'std_q': 8.66,
        },
    },
    0.01: {
        'inventory': {
            'spread': 1.33,
            'mean_profit': 66.78,
            'std_profit': 8.76,
            'mean_q': -0.02,
            'std_q': 4.70,
        },
        'symmetric': {
            'spread': 1.33,
            'mean_profit': 67.36,
            'std_profit': 13.40,
            'mean_q': -0.31,
            'std_q': 8.65,
        },
    },
    0.5: {
        'inventory': {
            'spread': 1.15,
            'mean_profit': 33.92,
            'std_profit': 4.72,
            'mean_q': -0.02,
            'std_q': 1.88,
        },
        'symmetric': {
            'spread': 1.15,
            'mean_profit': 66.20,
            'std_profit': 14.53,
            'mean_q': 0.25,
            'std_q': 9.06,
        },
    },
}
