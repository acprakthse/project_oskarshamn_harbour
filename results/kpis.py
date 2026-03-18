"""
results/kpis.py
===============
Compute KPIs from the dispatch DataFrame produced by extractor.py.

All KPIs are returned as a flat dict (easily written to CSV or JSON).

KPIs computed:
  Energy     : total PV, load, import, export, charge, discharge  [MWh]
  Performance: self-consumption ratio (SCR), self-sufficiency (SSR)
  Battery    : equivalent full cycles, estimated life, year-end SoH
  Financial  : baseline cost, optimised cost, annual savings
  Demand     : peak monthly import, demand charge paid

Usage:
    from results.kpis import compute_kpis
    kpis = compute_kpis(res, model, bat, project_years=20)
"""

import numpy  as np
import pandas as pd
from pyomo.environ import value

from battery.lfp_model   import LFPBattery
from battery.degradation import (
    equivalent_full_cycles,
    state_of_health,
    years_to_eol,
)
from config.tariff   import Tariff
from config.settings import TIMESTEP_H


def compute_kpis(
    res:           pd.DataFrame,
    model,
    bat:           LFPBattery,
    project_years: float = 20.0,
) -> dict:
    """
    Compute annual KPIs.

    Parameters
    ----------
    res           : pd.DataFrame  — dispatch results from extractor
    model         : pyomo.ConcreteModel  — solved model (for objective value)
    bat           : LFPBattery
    project_years : float  — used for life / SoH projection

    Returns
    -------
    dict  — all KPIs as floats (ready for CSV / JSON export)
    """
    dt = TIMESTEP_H

    # ── Energy [MWh] ──────────────────────────────────────────────────────────
    pv_MWh       = res["pv_kw"].sum()    * dt / 1000
    load_MWh     = res["load_kw"].sum()  * dt / 1000
    import_MWh   = res["P_imp_kw"].sum() * dt / 1000
    export_MWh   = res["P_exp_kw"].sum() * dt / 1000
    charge_MWh   = res["P_c_kw"].sum()   * dt / 1000
    discharge_MWh= res["P_d_kw"].sum()   * dt / 1000

    # ── Self-consumption ratio (SCR) ─────────────────────────────────────────
    # PV directly consumed = PV − export
    pv_consumed  = (res["pv_kw"] - res["P_exp_kw"]).clip(lower=0)
    scr = pv_consumed.sum() / max(res["pv_kw"].sum(), 1e-6) * 100

    # ── Self-sufficiency ratio (SSR) ─────────────────────────────────────────
    # load met locally = load − import
    ssr = (1 - res["P_imp_kw"].sum() / max(res["load_kw"].sum(), 1e-6)) * 100

    # ── Battery metrics ───────────────────────────────────────────────────────
    efc  = equivalent_full_cycles(res["soc_kwh"].values, bat.E_usable)
    soh  = state_of_health(1.0, efc,
                            bat.calendar_deg_per_yr,
                            bat.cycle_deg_per_cycle)
    life = years_to_eol(bat.calendar_deg_per_yr,
                         bat.cycle_deg_per_cycle,
                         cycles_per_year=efc)

    # ── Demand ────────────────────────────────────────────────────────────────
    monthly_peaks = pd.Series({mo: value(model.D[mo]) for mo in model.M})
    peak_demand_kw = monthly_peaks.max()
    demand_cost    = monthly_peaks.sum() * Tariff.DEMAND_CHG

    # ── Financial ─────────────────────────────────────────────────────────────
    baseline_cost   = (res["price"] * res["load_kw"] * dt).sum()
    optimised_cost  = value(model.obj)
    annual_savings  = baseline_cost - optimised_cost
    savings_pct     = annual_savings / max(baseline_cost, 1e-6) * 100
    lifetime_savings= annual_savings * project_years   # simple, no discounting

    return {
        # Energy
        "pv_MWh":              round(pv_MWh,       2),
        "load_MWh":            round(load_MWh,      2),
        "import_MWh":          round(import_MWh,    2),
        "export_MWh":          round(export_MWh,    2),
        "charge_MWh":          round(charge_MWh,    2),
        "discharge_MWh":       round(discharge_MWh, 2),
        # Performance
        "SCR_pct":             round(scr, 2),
        "SSR_pct":             round(ssr, 2),
        # Battery
        "equiv_full_cycles":   round(efc,  1),
        "year1_SoH_pct":       round(soh * 100, 2),
        "estimated_life_yr":   round(life, 1),
        # Demand
        "peak_demand_kw":      round(peak_demand_kw, 1),
        "annual_demand_cost":  round(demand_cost,    2),
        # Financial
        "baseline_cost":       round(baseline_cost,  2),
        "optimised_cost":      round(optimised_cost, 2),
        "annual_savings":      round(annual_savings, 2),
        "savings_pct":         round(savings_pct,    2),
        "lifetime_savings":    round(lifetime_savings, 2),
        # Battery config (for GA traceability)
        "E_cap_kwh":           bat.E_cap_kwh,
        "P_max_kw":            bat.P_max_kw,
    }


def print_kpis(kpis: dict) -> None:
    """Pretty-print the KPI dictionary."""
    print("\n" + "=" * 55)
    print("  ANNUAL KPIs")
    print("=" * 55)
    print(f"  PV generation       : {kpis['pv_MWh']:>9.1f} MWh")
    print(f"  Site load           : {kpis['load_MWh']:>9.1f} MWh")
    print(f"  Grid import         : {kpis['import_MWh']:>9.1f} MWh")
    print(f"  Grid export         : {kpis['export_MWh']:>9.1f} MWh")
    print(f"  Battery charged     : {kpis['charge_MWh']:>9.1f} MWh")
    print(f"  Battery discharged  : {kpis['discharge_MWh']:>9.1f} MWh")
    print(f"  Self-consumption    : {kpis['SCR_pct']:>9.1f} %")
    print(f"  Self-sufficiency    : {kpis['SSR_pct']:>9.1f} %")
    print(f"  Equiv. full cycles  : {kpis['equiv_full_cycles']:>9.0f}")
    print(f"  Year-1 SoH          : {kpis['year1_SoH_pct']:>9.2f} %")
    print(f"  Est. battery life   : {kpis['estimated_life_yr']:>9.1f} yr")
    print(f"  Peak demand         : {kpis['peak_demand_kw']:>9.1f} kW")
    print(f"  Baseline cost       : {kpis['baseline_cost']:>14,.0f}")
    print(f"  Optimised cost      : {kpis['optimised_cost']:>14,.0f}")
    print(f"  Annual savings      : {kpis['annual_savings']:>14,.0f}  "
          f"({kpis['savings_pct']:.1f} %)")
    print(f"  Lifetime savings    : {kpis['lifetime_savings']:>14,.0f}")
    print("=" * 55)
