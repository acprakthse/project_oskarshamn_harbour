"""
results/extractor.py
====================
Extract optimised variable values from a solved Pyomo model into a
pandas DataFrame.

Usage:
    from results.extractor import extract_dispatch
    res = extract_dispatch(model, df, bat)
"""

import numpy  as np
import pandas as pd
from pyomo.environ import value

from battery.lfp_model import LFPBattery
from config.settings   import TIMESTEP_H


def extract_dispatch(
    model,
    df:  pd.DataFrame,
    bat: LFPBattery,
) -> pd.DataFrame:
    """
    Pull hourly dispatch variables from the solved model.

    Parameters
    ----------
    model : pyomo.ConcreteModel  — solved MILP model
    df    : pd.DataFrame         — input data (datetime, pv_kw, load_kw, price)
    bat   : LFPBattery           — battery parameters (for SoC normalisation)

    Returns
    -------
    pd.DataFrame  with columns:
        datetime, pv_kw, load_kw, price
        P_c_kw      — battery charge power   [kW]
        P_d_kw      — battery discharge power[kW]
        soc_kwh     — state of charge        [kWh]
        soc_pct     — state of charge        [%]
        P_imp_kw    — grid import            [kW]
        P_exp_kw    — grid export            [kW]
        net_load_kw — load minus PV          [kW]  (positive = deficit)
        u           — binary: 1=charging, 0=discharging
    """
    T = len(df)

    P_c   = np.array([value(model.P_c[t])   for t in range(T)])
    P_d   = np.array([value(model.P_d[t])   for t in range(T)])
    SoC   = np.array([value(model.SoC[t])   for t in range(T)])
    P_imp = np.array([value(model.P_imp[t]) for t in range(T)])
    P_exp = np.array([value(model.P_exp[t]) for t in range(T)])
    u_bin = np.array([value(model.u[t])     for t in range(T)])

    res = df.copy()
    res["P_c_kw"]     = P_c
    res["P_d_kw"]     = P_d
    res["soc_kwh"]    = SoC
    res["soc_pct"]    = SoC / bat.E_cap_kwh * 100.0
    res["P_imp_kw"]   = P_imp
    res["P_exp_kw"]   = P_exp
    res["net_load_kw"]= res["load_kw"] - res["pv_kw"]
    res["u"]          = u_bin.round().astype(int)

    return res


def extract_monthly_demand(model) -> pd.Series:
    """
    Extract the monthly peak grid import values D[m].

    Returns
    -------
    pd.Series  index=month (1–12), values=peak import [kW]
    """
    return pd.Series(
        {mo: value(model.D[mo]) for mo in model.M},
        name="peak_import_kw",
    )
