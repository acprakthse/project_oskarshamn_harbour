"""
optimization/model_builder.py
==============================
Assemble the complete Pyomo MILP model from data + battery parameters.

build_milp_model() wires together:
  - Sets   (T, M)
  - Params (pv, load, price, battery constants)
  - Vars   (P_c, P_d, SoC, P_imp, P_exp, u, D)
  - Constraints (from constraints.py)
  - Objective   (from objective.py)

Usage:
    from optimization.model_builder import build_milp_model
    model, month_map = build_milp_model(df, bat)
"""

import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, ConstraintList,
    Objective, NonNegativeReals, Binary, minimize,
)

from battery.lfp_model import LFPBattery
from config.settings   import TIMESTEP_H
from optimization.constraints import (
    power_balance,
    soc_dynamics,
    soc_lower_bound,
    soc_upper_bound,
    soc_periodicity,
    charge_power_limit,
    discharge_power_limit,
)
from optimization.objective import total_cost


# =============================================================================

def build_milp_model(
    df:  pd.DataFrame,
    bat: LFPBattery,
) -> tuple[ConcreteModel, dict]:
    """
    Build and return a Pyomo ConcreteModel for the BESS MILP.

    Parameters
    ----------
    df  : pd.DataFrame  — columns: datetime, pv_kw, load_kw, price
                          length T (typically 8760)
    bat : LFPBattery    — battery parameters (fixed sizing)

    Returns
    -------
    model       : pyomo.ConcreteModel   — unsolved model
    month_map   : dict[int, list[int]]  — {month: [t_indices]}
                  used by results/extractor.py for demand-charge accounting
    """
    bat.validate()

    T       = len(df)
    months  = pd.to_datetime(df["datetime"]).dt.month.values
    month_list = sorted(set(months))

    month_map: dict[int, list[int]] = {m: [] for m in month_list}
    for t, mo in enumerate(months):
        month_map[mo].append(t)

    mdl = ConcreteModel(name="BESS_MILP")

    # ── Sets ──────────────────────────────────────────────────────────────────
    mdl.T = Set(initialize=range(T), ordered=True)
    mdl.M = Set(initialize=month_list, ordered=True)

    # ── Scalar params ──────────────────────────────────────────────────────────
    mdl.dt          = Param(initialize=TIMESTEP_H)
    mdl.eta_c       = Param(initialize=bat.eta_c)
    mdl.eta_d       = Param(initialize=bat.eta_d)
    mdl.E_cap       = Param(initialize=bat.E_cap_kwh)
    mdl.soc_min_kwh = Param(initialize=bat.soc_min_kwh)
    mdl.soc_max_kwh = Param(initialize=bat.soc_max_kwh)
    mdl.soc_init_kwh= Param(initialize=bat.soc_init_kwh)
    mdl.P_c_max     = Param(initialize=bat.P_charge_max)
    mdl.P_d_max     = Param(initialize=bat.P_discharge_max)

    # ── Time-series params ────────────────────────────────────────────────────
    mdl.pv    = Param(mdl.T, initialize=dict(enumerate(df["pv_kw"].values)))
    mdl.load_kw = Param(mdl.T, initialize=dict(enumerate(df["load_kw"].values)))
    mdl.price = Param(mdl.T, initialize=dict(enumerate(df["price"].values)))

    # ── Decision variables ────────────────────────────────────────────────────
    mdl.P_c   = Var(mdl.T, domain=NonNegativeReals, bounds=(0, bat.P_charge_max))
    mdl.P_d   = Var(mdl.T, domain=NonNegativeReals, bounds=(0, bat.P_discharge_max))
    mdl.SoC   = Var(mdl.T, domain=NonNegativeReals,
                    bounds=(bat.soc_min_kwh, bat.soc_max_kwh))
    mdl.P_imp = Var(mdl.T, domain=NonNegativeReals)
    mdl.P_exp = Var(mdl.T, domain=NonNegativeReals)
    mdl.u     = Var(mdl.T, domain=Binary)          # 1=charging, 0=discharging
    mdl.D     = Var(mdl.M, domain=NonNegativeReals) # monthly peak import [kW]

    # ── Constraints ───────────────────────────────────────────────────────────
    mdl.c_power_balance   = Constraint(mdl.T, rule=power_balance)
    mdl.c_soc_dynamics    = Constraint(mdl.T, rule=soc_dynamics)
    mdl.c_soc_lb          = Constraint(mdl.T, rule=soc_lower_bound)
    mdl.c_soc_ub          = Constraint(mdl.T, rule=soc_upper_bound)
    mdl.c_soc_period      = Constraint(rule=soc_periodicity)
    mdl.c_charge_lim      = Constraint(mdl.T, rule=charge_power_limit)
    mdl.c_discharge_lim   = Constraint(mdl.T, rule=discharge_power_limit)

    # Monthly peak demand tracking  (ConstraintList — two-index loop)
    mdl.c_demand = ConstraintList()
    for mo, hours in month_map.items():
        for t in hours:
            mdl.c_demand.add(mdl.D[mo] >= mdl.P_imp[t])

    # ── Objective ─────────────────────────────────────────────────────────────
    mdl.obj = Objective(rule=total_cost, sense=minimize)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"[model_builder] MILP built | "
        f"T={T}  vars={mdl.nvariables():,}  "
        f"constraints={mdl.nconstraints():,}"
    )

    return mdl, month_map
