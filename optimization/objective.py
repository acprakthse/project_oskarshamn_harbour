"""
optimization/objective.py
==========================
Pyomo objective function for the BESS MILP.

Objective: minimise total annual electricity cost
  = Σ_t [ price[t] × P_imp[t] − export_rate × P_exp[t] ] × dt
  + Σ_m [ demand_charge × D[m] ]

This file is intentionally isolated so that switching objectives
(e.g. maximise self-consumption, maximise NPV) means editing
only this file, not the model builder.

Usage:
    from optimization.objective import total_cost
    model.obj = Objective(rule=total_cost, sense=minimize)
"""

from pyomo.environ import minimize
from config.tariff import Tariff


def total_cost(m):
    """
    Annual electricity cost objective.

    Components
    ----------
    energy_cost  : Σ_t  price[t] × P_imp[t] × dt
    export_credit: Σ_t  export_rate × P_exp[t] × dt  (subtracted)
    demand_cost  : Σ_m  demand_charge × D[m]
                   where D[m] = max hourly grid import in month m

    All variables and params are defined on model m by model_builder.py.
    """
    dt = m.dt

    energy_cost = sum(
        m.price[t] * m.P_imp[t] * dt
        for t in m.T
    )

    export_credit = sum(
        Tariff.EXPORT_RATE * m.P_exp[t] * dt
        for t in m.T
    )

    demand_cost = sum(
        Tariff.DEMAND_CHG * m.D[mo]
        for mo in m.M
    )

    return energy_cost - export_credit + demand_cost


# =============================================================================
# Alternative objectives  (swap in by changing model_builder import)
# =============================================================================

def maximise_self_consumption(m):
    """
    Maximise self-consumption ratio (SCR).

    Minimise grid export — proxy for maximising direct PV use.
    Note: combine with a cost floor constraint to keep the solution feasible.
    """
    dt = m.dt
    return sum(m.P_exp[t] * dt for t in m.T)


def minimise_peak_demand(m):
    """
    Minimise monthly peak demand (peak shaving only — ignores energy cost).

    Useful for sites where demand charge dominates the bill.
    """
    return sum(m.D[mo] for mo in m.M)
