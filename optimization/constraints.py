"""
optimization/constraints.py
============================
Pyomo constraint rule functions for the BESS MILP.

Each function takes (model, index) and returns a Pyomo expression.
They are registered onto the model in model_builder.py via Constraint().

Constraints implemented:
  1. power_balance        — energy balance at every timestep
  2. soc_dynamics         — SoC state transition with charge/discharge losses
  3. soc_lower_bound      — SoC ≥ soc_min × E_cap  (protect cells)
  4. soc_upper_bound      — SoC ≤ soc_max × E_cap
  5. soc_periodicity      — SoC at end of year ≈ SoC at start
  6. charge_power_limit   — P_c ≤ P_c_max × u  (binary dispatch mode)
  7. discharge_power_limit— P_d ≤ P_d_max × (1 − u)
  8. monthly_peak_demand  — D[m] ≥ P_imp[t]  for all t in month m
                            (added as ConstraintList in model_builder)

All functions expect the ConcreteModel to have the attributes defined in
model_builder.build_milp_model().
"""

from pyomo.environ import Constraint


# =============================================================================
# 1. Power balance
# =============================================================================

def power_balance(m, t):
    return (
        m.pv[t] + m.P_d[t] + m.P_imp[t]
        ==
        m.load_kw[t] + m.P_c[t] + m.P_exp[t]
    )


# =============================================================================
# 2. SoC dynamics
# =============================================================================

def soc_dynamics(m, t):
    """
    SoC[t] = SoC[t-1]  +  eta_c × P_c[t] × dt  −  P_d[t] / eta_d × dt

    At t=0 the previous SoC is the initial condition SoC_init.
    """
    dt = m.dt
    if t == 0:
        return (
            m.SoC[0]
            == m.soc_init_kwh
            + m.eta_c * m.P_c[0] * dt
            - m.P_d[0] / m.eta_d * dt
        )
    return (
        m.SoC[t]
        == m.SoC[t - 1]
        + m.eta_c * m.P_c[t] * dt
        - m.P_d[t] / m.eta_d * dt
    )


# =============================================================================
# 3 & 4. SoC bounds
# =============================================================================

def soc_lower_bound(m, t):
    """SoC must stay above the minimum depth-of-discharge limit."""
    return m.SoC[t] >= m.soc_min_kwh


def soc_upper_bound(m, t):
    """SoC must not exceed the maximum charge level."""
    return m.SoC[t] <= m.soc_max_kwh


# =============================================================================
# 5. SoC periodicity (annual closure)
# =============================================================================

def soc_periodicity(m):
    """
    Ensure the battery does not finish the year with significantly less
    energy than it started.  Prevents the optimizer from draining the
    battery in December at the expense of future periods.

    Tolerance: ±5 % of initial SoC.
    """
    T_last = max(m.T)
    return m.SoC[T_last] >= m.soc_init_kwh * 0.95


# =============================================================================
# 6 & 7. Binary charge/discharge mutex
# =============================================================================

def charge_power_limit(m, t):
    """
    Charge power ≤ P_c_max when u[t]=1 (charging mode).
    When u[t]=0 (discharging mode) this forces P_c[t]=0.
    """
    return m.P_c[t] <= m.P_c_max * m.u[t]


def discharge_power_limit(m, t):
    """
    Discharge power ≤ P_d_max when u[t]=0 (discharging mode).
    When u[t]=1 (charging mode) this forces P_d[t]=0.
    """
    return m.P_d[t] <= m.P_d_max * (1 - m.u[t])


# =============================================================================
# 8. Monthly peak demand  (added as ConstraintList — see model_builder.py)
# =============================================================================
# This constraint is not a simple rule function because it indexes over
# two sets (months × hours-in-month).  It is added directly in
# model_builder.build_milp_model() using a ConstraintList.
#
# Logic:
#   for each month m:
#     for each hour t in month m:
#       D[m] >= P_imp[t]
#
# D[m] is then multiplied by the demand charge in the objective.
