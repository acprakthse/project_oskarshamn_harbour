"""
config/settings.py
==================
Global project settings: file paths, solver configuration, column names.
Edit this file when deploying to a new project / site.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "input_data"
RESULTS_DIR  = BASE_DIR / "results"

HELIOSCOPE_CSV      = DATA_DIR / "helioscope_8760.csv"
LOAD_CSV            = DATA_DIR / "load_8760.csv"
DAY_AHEAD_PRICE_CSV = DATA_DIR / "day_ahead_price.csv"

# =============================================================================
# CSV column names  (edit if your export uses different headers)
# =============================================================================

HS_DATETIME_COL = "datetime"
HS_POWER_COL    = "ac_power_kw"      # kW per hour from HelioScope

LOAD_DATETIME_COL = "datetime"
LOAD_KW_COL       = "load_kw"

# =============================================================================
# Solver
# =============================================================================

SOLVER_NAME  = "gurobi"              # "gurobi" | "cbc" | "glpk"
SOLVER_FALLBACK = "cbc"              # used if primary solver not available

SOLVER_OPTIONS = {
    "gurobi": {
        "MIPGap":    0.005,          # 0.5 % optimality gap
        "TimeLimit": 3600,           # seconds
        "Threads":   0,              # 0 = use all cores
        "OutputFlag": 1,
    },
    "cbc": {
        "ratioGap": 0.005,
        "sec":      3600,
    },
    "glpk": {
        "mipgap": 0.005,
        "tmlim":  3600,
    },
}

# =============================================================================
# Simulation
# =============================================================================

TIMESTEP_H = 1.0        # hours per timestep (8760 × 1h = 1 year)
N_HOURS    = 8760
