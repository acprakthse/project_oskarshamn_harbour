"""
=============================================================================
BESS MILP Optimizer  —  Pyomo + Gurobi
=============================================================================
System   : 863 kW rooftop PV  +  LFP Battery Energy Storage System
Solver   : Gurobi (via Pyomo)
Timestep : Hourly, 8760 steps (1 year)
Purpose  : Minimise electricity bill  (peak-demand charge + ToU energy cost)
           Battery capacity is FIXED here — use GA wrapper for sizing.

Inputs (CSV columns expected from HelioScope export)
  - datetime  : ISO8601 timestamp  e.g. 2024-01-01 00:00:00
  - ac_power_kw : AC output kW  (or ac_energy_kwh — see LOAD_COLUMN note)

Load profile CSV columns expected
  - datetime
  - load_kw

Tariff (edit TARIFF section below)
  - Peak / off-peak energy rate  (THB/kWh  or any currency)
  - Demand charge               (THB/kW of monthly peak)

LFP Battery parameters (edit BATTERY section below)
  - E_cap_kwh   : usable energy capacity  [kWh]
  - P_max_kw    : max charge / discharge power  [kW]
  - eta_c       : charge efficiency
  - eta_d       : discharge efficiency
  - soc_min     : minimum SoC  (as fraction 0–1)
  - soc_max     : maximum SoC  (as fraction 0–1)
  - soc_init    : initial SoC
  - c_rate_max  : maximum C-rate  (used to derive P_max if not set directly)
  - calendar_degradation : annual capacity fade  [fraction/yr] — informational

=============================================================================
"""

import os
import sys
import math
import warnings
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from   pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Objective,
    NonNegativeReals, Binary, value, SolverFactory, minimize
)

warnings.filterwarnings("ignore")


# =============================================================================
# 0.  FILE PATHS  — edit these
# =============================================================================

HELIOSCOPE_CSV = "../data/helioscope_8760.csv"   # HelioScope export
LOAD_CSV       = "../data/load_8760.csv"          # Site load profile
RESULTS_DIR    = "../data/results"               # Output folder

# Column names in the CSVs
HS_DATETIME_COL  = "timestamp"
HS_POWER_COL     = "grid_power"         # kW per hour  (= kWh for 1-h step)
LOAD_DATETIME_COL = "timestamp"
LOAD_KW_COL      = "total_load_kW"


# =============================================================================
# 1.  TARIFF  — Thailand PEA MVT (example — adjust to your utility)
# =============================================================================

class Tariff:
    """
    Time-of-Use energy tariff + monthly peak demand charge.
    Extend with real block/seasonal tariffs as needed.
    """
    PEAK_HOURS   = list(range(9, 22))     # 09:00–21:59  (on-peak)
    ENERGY_PEAK  = 5.80                   # THB / kWh  on-peak
    ENERGY_OFFPK = 2.60                   # THB / kWh  off-peak
    DEMAND_CHG   = 220.0                  # THB / kW   of monthly max import
    EXPORT_RATE  = 0.0                    # THB / kWh  (0 = no net-metering)

    @staticmethod
    def energy_price(hour: int) -> float:
        if hour in Tariff.PEAK_HOURS:
            return Tariff.ENERGY_PEAK
        return Tariff.ENERGY_OFFPK

    @staticmethod
    def price_vector(timestamps: pd.DatetimeIndex) -> np.ndarray:
        return np.array([Tariff.energy_price(ts.hour) for ts in timestamps])


# =============================================================================
# 2.  LFP BATTERY MODEL  — parameters & constraints
# =============================================================================

class LFPBattery:
    """
    LFP (LiFePO4) battery model parameters.

    Physical characteristics of LFP vs other chemistries:
      - Flat discharge curve  (~3.2 V nominal)
      - Lower energy density  (~120–160 Wh/kg)
      - Excellent cycle life  (3000–6000 cycles to 80% DoD)
      - Low temperature sensitivity compared to NMC
      - Safer — no thermal runaway risk
      - Efficiency (round-trip): 92–96%

    Degradation modelling (informational, not in LP):
      Capacity fade ~ 0.5–1% per year calendar + cycle-dependent.
      For GA sizing step, apply fade factor to E_cap for year-N analysis.
    """

    # ── Core sizing  (FIXED for this run; GA will sweep these) ──────────────
    E_cap_kwh   : float = 1726.0    # Usable capacity  [kWh]
                                    # ~2h at 863 kW  (2C discharge)
    P_max_kw    : float = 863.0     # Max charge/discharge power [kW]
                                    # = 1C for 1726 kWh pack

    # ── Efficiency ───────────────────────────────────────────────────────────
    eta_c       : float = 0.97      # Charge efficiency    (one-way)
    eta_d       : float = 0.97      # Discharge efficiency (one-way)
    # Round-trip = eta_c × eta_d = 0.94

    # ── State-of-charge bounds ────────────────────────────────────────────────
    soc_min     : float = 0.10      # 10%  — protect against deep discharge
    soc_max     : float = 0.95      # 95%  — LFP can tolerate near-full
    soc_init    : float = 0.50      # Starting SoC

    # ── C-rate limits ────────────────────────────────────────────────────────
    c_rate_charge    : float = 0.5  # Max charge C-rate   (0.5C = 2h full charge)
    c_rate_discharge : float = 1.0  # Max discharge C-rate (1C  = 1h full discharge)

    # ── Calendar degradation (for reporting / GA) ────────────────────────────
    calendar_deg_per_yr : float = 0.008   # 0.8 % / year
    cycle_deg_per_cycle : float = 0.00003 # per equivalent full cycle

    # ── Derived limits ────────────────────────────────────────────────────────
    @property
    def P_charge_max(self) -> float:
        return min(self.P_max_kw, self.c_rate_charge * self.E_cap_kwh)

    @property
    def P_discharge_max(self) -> float:
        return min(self.P_max_kw, self.c_rate_discharge * self.E_cap_kwh)

    @property
    def E_usable(self) -> float:
        return self.E_cap_kwh * (self.soc_max - self.soc_min)

    def summarise(self):
        print("─" * 50)
        print("LFP Battery Configuration")
        print(f"  Capacity (usable)  : {self.E_cap_kwh:.0f} kWh")
        print(f"  Max charge power   : {self.P_charge_max:.0f} kW  ({self.c_rate_charge}C)")
        print(f"  Max discharge power: {self.P_discharge_max:.0f} kW  ({self.c_rate_discharge}C)")
        print(f"  Round-trip eff.    : {self.eta_c*self.eta_d*100:.1f}%")
        print(f"  SoC window         : {self.soc_min*100:.0f}% – {self.soc_max*100:.0f}%")
        print(f"  Calendar fade/yr   : {self.calendar_deg_per_yr*100:.1f}%")
        print("─" * 50)


# =============================================================================
# 3.  DATA LOADER
# =============================================================================

def load_data(hs_path: str, load_path: str) -> pd.DataFrame:
    """
    Load and align HelioScope 8760 + load profile.
    Returns a single DataFrame with columns:
        datetime, pv_kw, load_kw, price
    """
    # ── HelioScope ────────────────────────────────────────────────────────────
    hs = pd.read_csv(hs_path, parse_dates=[HS_DATETIME_COL])
    hs = hs.rename(columns={HS_DATETIME_COL: "datetime",
                             HS_POWER_COL:    "pv_kw"})
    hs = hs.set_index("datetime").sort_index()

    # ── Load profile ─────────────────────────────────────────────────────────
    ld = pd.read_csv(load_path, parse_dates=[LOAD_DATETIME_COL])
    ld = ld.rename(columns={LOAD_DATETIME_COL: "datetime",
                             LOAD_KW_COL:       "load_kw"})
    ld = ld.set_index("datetime").sort_index()

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = hs[["pv_kw"]].join(ld[["load_kw"]], how="inner")
    df = df.dropna()

    if len(df) < 8760:
        raise ValueError(
            f"Merged dataset has only {len(df)} rows — expected 8760. "
            "Check datetime alignment between CSVs."
        )
    df = df.iloc[:8760]   # ensure exactly one year

    # ── Tariff price vector ────────────────────────────────────────────────────
    df["price"] = Tariff.price_vector(df.index)

    print(f"Data loaded: {len(df)} timesteps  "
          f"| PV peak: {df.pv_kw.max():.1f} kW  "
          f"| Load peak: {df.load_kw.max():.1f} kW")
    return df.reset_index()


def make_demo_data(n: int = 8760) -> pd.DataFrame:
    """
    Generate synthetic 8760 data for standalone testing
    when real CSVs are not available.
    """
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    # Solar: zero at night, bell curve noon peak with seasonal variation
    hour  = idx.hour
    doy   = idx.dayofyear
    solar_mask  = (hour >= 6) & (hour <= 18)
    base_solar  = np.where(solar_mask,
                           np.sin(np.pi * (hour - 6) / 12) ** 1.5, 0.0)
    seasonal    = 0.7 + 0.3 * np.cos(2 * np.pi * (doy - 172) / 365)
    pv_kw       = (base_solar * seasonal * 863
                   * (1 + 0.05 * np.random.randn(n))).clip(0)

    # Load: business hours profile
    load_base   = np.where((hour >= 8) & (hour <= 20), 600, 250)
    load_kw     = (load_base + 50 * np.random.randn(n)).clip(100)

    price       = Tariff.price_vector(idx)

    df = pd.DataFrame({"datetime": idx,
                       "pv_kw":    pv_kw,
                       "load_kw":  load_kw,
                       "price":    price})
    return df


# =============================================================================
# 4.  MILP MODEL  (Pyomo)
# =============================================================================

def build_milp_model(df: pd.DataFrame, bat: LFPBattery) -> ConcreteModel:
    """
    Build the MILP Pyomo model.

    Decision variables (per hour t):
        P_c[t]   : charge power      [kW]   ≥ 0
        P_d[t]   : discharge power   [kW]   ≥ 0
        SoC[t]   : state of charge   [kWh]
        P_imp[t] : grid import       [kW]   ≥ 0
        P_exp[t] : grid export       [kW]   ≥ 0
        u[t]     : binary — 1=charging, 0=discharging  (prevents simultaneous)
        D[m]     : monthly peak demand [kW]             (for demand charge)

    Objective:
        Minimise  Σ_t [ price[t] × P_imp[t] − export_rate × P_exp[t] ]
                + Σ_m [ demand_charge × D[m] ]

    Constraints:
        (1)  Power balance    : pv + P_d + P_imp = load + P_c + P_exp
        (2)  SoC dynamics     : SoC[t] = SoC[t-1] + eta_c×P_c[t] − P_d[t]/eta_d
        (3)  SoC bounds       : E_cap×soc_min ≤ SoC[t] ≤ E_cap×soc_max
        (4)  SoC periodicity  : SoC[8759] = SoC[0]   (close the year)
        (5)  Charge limit     : P_c[t] ≤ P_charge_max × u[t]
        (6)  Discharge limit  : P_d[t] ≤ P_discharge_max × (1 − u[t])
        (7)  Monthly peak     : D[m] ≥ P_imp[t]  ∀t in month m
    """
    T    = len(df)
    dt   = 1.0           # timestep [h]
    mdl  = ConcreteModel(name="BESS_MILP")

    # ── Sets ──────────────────────────────────────────────────────────────────
    mdl.T = Set(initialize=range(T))

    months      = df["datetime"].dt.month.values
    month_list  = sorted(set(months))
    mdl.M       = Set(initialize=month_list)

    # ── Parameters ────────────────────────────────────────────────────────────
    mdl.pv      = Param(mdl.T, initialize=dict(enumerate(df["pv_kw"].values)))
    mdl.load    = Param(mdl.T, initialize=dict(enumerate(df["load_kw"].values)))
    mdl.price   = Param(mdl.T, initialize=dict(enumerate(df["price"].values)))

    mdl.eta_c       = Param(initialize=bat.eta_c)
    mdl.eta_d       = Param(initialize=bat.eta_d)
    mdl.E_cap       = Param(initialize=bat.E_cap_kwh)
    mdl.soc_min_kwh = Param(initialize=bat.soc_min * bat.E_cap_kwh)
    mdl.soc_max_kwh = Param(initialize=bat.soc_max * bat.E_cap_kwh)
    mdl.soc_init_kwh= Param(initialize=bat.soc_init * bat.E_cap_kwh)
    mdl.P_c_max     = Param(initialize=bat.P_charge_max)
    mdl.P_d_max     = Param(initialize=bat.P_discharge_max)

    # ── Variables ─────────────────────────────────────────────────────────────
    mdl.P_c   = Var(mdl.T, domain=NonNegativeReals)
    mdl.P_d   = Var(mdl.T, domain=NonNegativeReals)
    mdl.SoC   = Var(mdl.T, domain=NonNegativeReals)
    mdl.P_imp = Var(mdl.T, domain=NonNegativeReals)
    mdl.P_exp = Var(mdl.T, domain=NonNegativeReals)
    mdl.u     = Var(mdl.T, domain=Binary)             # 1=charging
    mdl.D     = Var(mdl.M, domain=NonNegativeReals)   # monthly peak demand

    # ── Constraints ───────────────────────────────────────────────────────────

    # (1) Power balance
    def power_balance(m, t):
        return (m.pv[t] + m.P_d[t] + m.P_imp[t]
                == m.load[t] + m.P_c[t] + m.P_exp[t])
    mdl.c_power = Constraint(mdl.T, rule=power_balance)

    # (2) SoC dynamics
    def soc_dynamics(m, t):
        if t == 0:
            return m.SoC[0] == (m.soc_init_kwh
                                 + m.eta_c * m.P_c[0] * dt
                                 - m.P_d[0] / m.eta_d * dt)
        return (m.SoC[t] == m.SoC[t-1]
                + m.eta_c * m.P_c[t] * dt
                - m.P_d[t] / m.eta_d * dt)
    mdl.c_soc = Constraint(mdl.T, rule=soc_dynamics)

    # (3) SoC bounds
    def soc_lb(m, t):  return m.SoC[t] >= m.soc_min_kwh
    def soc_ub(m, t):  return m.SoC[t] <= m.soc_max_kwh
    mdl.c_soc_lb = Constraint(mdl.T, rule=soc_lb)
    mdl.c_soc_ub = Constraint(mdl.T, rule=soc_ub)

    # (4) SoC periodicity (annual closure)
    mdl.c_soc_close = Constraint(
        expr=mdl.SoC[T-1] >= mdl.soc_init_kwh * 0.95   # ±5% tolerance
    )

    # (5) Charge power limit (binary)
    def charge_lim(m, t): return m.P_c[t] <= m.P_c_max * m.u[t]
    mdl.c_charge_lim = Constraint(mdl.T, rule=charge_lim)

    # (6) Discharge power limit (binary)
    def discharge_lim(m, t): return m.P_d[t] <= m.P_d_max * (1 - m.u[t])
    mdl.c_discharge_lim = Constraint(mdl.T, rule=discharge_lim)

    # (7) Monthly peak demand tracking
    month_to_hours = {m: [] for m in month_list}
    for t, mo in enumerate(months):
        month_to_hours[mo].append(t)

    def demand_track(m, mo):
        return [m.D[mo] >= m.P_imp[t] for t in month_to_hours[mo]]

    # Pyomo constraint list for monthly demand
    from pyomo.environ import ConstraintList
    mdl.c_demand = ConstraintList()
    for mo in month_list:
        for t in month_to_hours[mo]:
            mdl.c_demand.add(mdl.D[mo] >= mdl.P_imp[t])

    # ── Objective ─────────────────────────────────────────────────────────────
    def total_cost(m):
        energy_cost = sum(
            m.price[t] * m.P_imp[t] * dt
            - Tariff.EXPORT_RATE * m.P_exp[t] * dt
            for t in m.T
        )
        demand_cost = sum(
            Tariff.DEMAND_CHG * m.D[mo] for mo in m.M
        )
        return energy_cost + demand_cost

    mdl.obj = Objective(rule=total_cost, sense=minimize)

    return mdl, month_to_hours


# =============================================================================
# 5.  SOLVER
# =============================================================================

def solve_model(mdl: ConcreteModel,
                mipgap: float = 0.005,
                timelimit: int = 3600) -> dict:
    """
    Solve with Gurobi. Falls back to CBC if Gurobi is unavailable.
    Returns solver result object.
    """
    solver = SolverFactory("gurobi")
    if not solver.available():
        print("Gurobi not found — falling back to CBC (open-source).")
        print("Install Gurobi or run:  pip install gurobipy")
        solver = SolverFactory("cbc")
        if not solver.available():
            raise RuntimeError(
                "No MILP solver found. Install Gurobi or CBC:\n"
                "  Gurobi: https://www.gurobi.com/downloads/\n"
                "  CBC:    pip install cylp  (Linux/Mac) or use conda"
            )
        timelimit_opt = {"sec": timelimit}
        mipgap_opt    = {"ratioGap": mipgap}
        options = {**timelimit_opt, **mipgap_opt}
    else:
        options = {
            "MIPGap":   mipgap,
            "TimeLimit": timelimit,
            "Threads":   0,           # use all available cores
            "OutputFlag": 1,
        }

    print(f"\nSolving MILP with {solver.name.upper()}  "
          f"(MIPGap={mipgap*100:.1f}%, TimeLimit={timelimit}s) …")

    result = solver.solve(mdl, options=options, tee=True)
    return result


# =============================================================================
# 6.  RESULTS EXTRACTION
# =============================================================================

def extract_results(mdl: ConcreteModel,
                    df: pd.DataFrame,
                    bat: LFPBattery,
                    month_to_hours: dict) -> pd.DataFrame:
    """
    Pull optimised variables into a results DataFrame and compute KPIs.
    """
    T = len(df)

    P_c   = np.array([value(mdl.P_c[t])   for t in range(T)])
    P_d   = np.array([value(mdl.P_d[t])   for t in range(T)])
    SoC   = np.array([value(mdl.SoC[t])   for t in range(T)])
    P_imp = np.array([value(mdl.P_imp[t]) for t in range(T)])
    P_exp = np.array([value(mdl.P_exp[t]) for t in range(T)])

    res = df.copy()
    res["P_c_kw"]    = P_c
    res["P_d_kw"]    = P_d
    res["soc_kwh"]   = SoC
    res["soc_pct"]   = SoC / bat.E_cap_kwh * 100
    res["P_imp_kw"]  = P_imp
    res["P_exp_kw"]  = P_exp
    res["net_load"]  = res["load_kw"] - res["pv_kw"]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    dt = 1.0   # hours

    # Annual energy
    total_pv_kwh      = df["pv_kw"].sum() * dt
    total_load_kwh    = df["load_kw"].sum() * dt
    total_import_kwh  = P_imp.sum() * dt
    total_export_kwh  = P_exp.sum() * dt
    total_charge_kwh  = P_c.sum() * dt
    total_dischg_kwh  = P_d.sum() * dt

    # Self-consumption ratio  (SCR) = PV directly consumed / total PV
    pv_to_load  = np.minimum(df["pv_kw"].values, df["load_kw"].values)
    scr = pv_to_load.sum() / max(total_pv_kwh, 1) * 100

    # Self-sufficiency ratio  (SSR) = locally-met load / total load
    ssr = (total_load_kwh - total_import_kwh) / max(total_load_kwh, 1) * 100

    # Battery throughput & cycles
    equiv_cycles = total_charge_kwh / max(bat.E_cap_kwh, 1)

    # Financial
    baseline_cost = (df["price"] * df["load_kw"] * dt).sum()  # no PV/BESS
    optimised_cost = value(mdl.obj)
    savings        = baseline_cost - optimised_cost

    monthly_demand = {mo: value(mdl.D[mo]) for mo in mdl.M}
    peak_demand_kw = max(monthly_demand.values())

    kpis = {
        "total_pv_MWh":          total_pv_kwh / 1000,
        "total_load_MWh":        total_load_kwh / 1000,
        "total_import_MWh":      total_import_kwh / 1000,
        "total_export_MWh":      total_export_kwh / 1000,
        "battery_charge_MWh":    total_charge_kwh / 1000,
        "battery_discharge_MWh": total_dischg_kwh / 1000,
        "equiv_full_cycles":     equiv_cycles,
        "SCR_pct":               scr,
        "SSR_pct":               ssr,
        "peak_demand_kw":        peak_demand_kw,
        "baseline_cost":         baseline_cost,
        "optimised_cost":        optimised_cost,
        "annual_savings":        savings,
        "savings_pct":           savings / max(baseline_cost, 1) * 100,
    }

    print("\n" + "=" * 55)
    print("  OPTIMISATION RESULTS  —  ANNUAL KPIs")
    print("=" * 55)
    print(f"  PV generation         : {kpis['total_pv_MWh']:>8.1f} MWh")
    print(f"  Site load             : {kpis['total_load_MWh']:>8.1f} MWh")
    print(f"  Grid import           : {kpis['total_import_MWh']:>8.1f} MWh")
    print(f"  Grid export           : {kpis['total_export_MWh']:>8.1f} MWh")
    print(f"  Battery throughput    : {kpis['battery_charge_MWh']:>8.1f} MWh charged")
    print(f"  Equiv. full cycles    : {kpis['equiv_full_cycles']:>8.0f}")
    print(f"  Self-consumption (SCR): {kpis['SCR_pct']:>8.1f} %")
    print(f"  Self-sufficiency (SSR): {kpis['SSR_pct']:>8.1f} %")
    print(f"  Peak demand (monthly) : {kpis['peak_demand_kw']:>8.1f} kW")
    print(f"  Baseline cost         : {kpis['baseline_cost']:>12,.0f}")
    print(f"  Optimised cost        : {kpis['optimised_cost']:>12,.0f}")
    print(f"  Annual savings        : {kpis['annual_savings']:>12,.0f}  "
          f"({kpis['savings_pct']:.1f}%)")
    print("=" * 55)

    return res, kpis


# =============================================================================
# 7.  VISUALISATION
# =============================================================================

def plot_results(res: pd.DataFrame,
                 bat: LFPBattery,
                 kpis: dict,
                 out_dir: str = "results"):

    os.makedirs(out_dir, exist_ok=True)

    # ── A. Sample week — dispatch chart ───────────────────────────────────────
    sample_start = res[res["datetime"].dt.month == 7].index[0]
    week = res.iloc[sample_start: sample_start + 168]   # 168 h = 7 days

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.fill_between(week["datetime"], week["pv_kw"],
                    alpha=0.6, color="#EF9F27", label="PV generation")
    ax.fill_between(week["datetime"], week["load_kw"],
                    alpha=0.3, color="#378ADD", label="Site load")
    ax.plot(week["datetime"], week["P_imp_kw"],
            color="#E24B4A", lw=1.2, label="Grid import")
    ax.plot(week["datetime"], week["P_exp_kw"],
            color="#1D9E75", lw=1.0, ls="--", label="Grid export")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Sample week — power dispatch")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.bar(week["datetime"], week["P_c_kw"],
           width=0.04, color="#534AB7", alpha=0.8, label="Charge")
    ax.bar(week["datetime"], -week["P_d_kw"],
           width=0.04, color="#D85A30", alpha=0.8, label="Discharge")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("BESS Power [kW]")
    ax.set_title("Battery charge / discharge")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(week["datetime"], week["soc_pct"],
            color="#0F6E56", lw=1.5, label="SoC %")
    ax.axhline(bat.soc_min * 100, color="red",   ls=":", lw=1, label=f"SoC min {bat.soc_min*100:.0f}%")
    ax.axhline(bat.soc_max * 100, color="orange", ls=":", lw=1, label=f"SoC max {bat.soc_max*100:.0f}%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("State of Charge [%]")
    ax.set_title("LFP Battery SoC")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

    plt.tight_layout()
    fig.savefig(f"{out_dir}/dispatch_week.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/dispatch_week.png")

    # ── B. Annual SoC heatmap ─────────────────────────────────────────────────
    soc_2d = res["soc_pct"].values[:8760].reshape(365, 24)
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(soc_2d.T, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Hour of day")
    ax.set_title("Annual LFP SoC heatmap  [%]")
    plt.colorbar(im, ax=ax, label="SoC [%]")
    plt.tight_layout()
    fig.savefig(f"{out_dir}/soc_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/soc_heatmap.png")

    # ── C. Monthly summary bar chart ──────────────────────────────────────────
    res["month"] = res["datetime"].dt.month
    monthly = res.groupby("month").agg(
        pv_MWh   = ("pv_kw",    lambda x: x.sum() / 1000),
        load_MWh = ("load_kw",  lambda x: x.sum() / 1000),
        imp_MWh  = ("P_imp_kw", lambda x: x.sum() / 1000),
        exp_MWh  = ("P_exp_kw", lambda x: x.sum() / 1000),
    ).reset_index()

    months_lbl = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(12)
    w = 0.2
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w,   monthly["pv_MWh"],   w, label="PV",          color="#EF9F27", alpha=0.85)
    ax.bar(x,       monthly["load_MWh"], w, label="Load",         color="#378ADD", alpha=0.85)
    ax.bar(x + w,   monthly["imp_MWh"],  w, label="Grid import",  color="#E24B4A", alpha=0.85)
    ax.bar(x + 2*w, monthly["exp_MWh"],  w, label="Grid export",  color="#1D9E75", alpha=0.85)
    ax.set_xticks(x + w/2)
    ax.set_xticklabels(months_lbl)
    ax.set_ylabel("Energy [MWh]")
    ax.set_title("Monthly energy summary — PV + BESS optimised")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/monthly_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/monthly_summary.png")


# =============================================================================
# 8.  EXPORT RESULTS
# =============================================================================

def export_results(res: pd.DataFrame, kpis: dict, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    # Full 8760 results
    csv_path = f"{out_dir}/bess_dispatch_8760.csv"
    res.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"  Saved: {csv_path}")
    # KPI summary
    kpi_df = pd.DataFrame([kpis])
    kpi_df.to_csv(f"{out_dir}/kpis.csv", index=False, float_format="%.2f")
    print(f"  Saved: {out_dir}/kpis.csv")


# =============================================================================
# 9.  MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BESS MILP Optimizer — Pyomo + Gurobi")
    parser.add_argument("--helioscope", default=HELIOSCOPE_CSV)
    parser.add_argument("--load",       default=LOAD_CSV)
    parser.add_argument("--outdir",     default=RESULTS_DIR)
    parser.add_argument("--mipgap",     type=float, default=0.005)
    parser.add_argument("--timelimit",  type=int,   default=3600)
    parser.add_argument("--demo",       action="store_true",
                        help="Run with synthetic data (no CSVs needed)")
    args = parser.parse_args()

    print("=" * 55)
    print("  BESS MILP Optimizer  |  Pyomo + Gurobi  |  LFP")
    print("=" * 55)

    # ── Battery ───────────────────────────────────────────────────────────────
    bat = LFPBattery()
    bat.summarise()

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.demo or (not os.path.exists(args.helioscope)
                     or not os.path.exists(args.load)):
        print("\nUsing synthetic demo data (pass --helioscope and --load for real data)")
        df = make_demo_data()
    else:
        df = load_data(args.helioscope, args.load)

    # ── Build model ───────────────────────────────────────────────────────────
    print("\nBuilding MILP model …")
    mdl, month_to_hours = build_milp_model(df, bat)
    print(f"  Variables   : {mdl.nvariables():,}")
    print(f"  Constraints : {mdl.nconstraints():,}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    result = solve_model(mdl, mipgap=args.mipgap, timelimit=args.timelimit)

    # ── Extract & report ──────────────────────────────────────────────────────
    res, kpis = extract_results(mdl, df, bat, month_to_hours)

    # ── Plots & export ────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_results(res, bat, kpis, out_dir=args.outdir)
    export_results(res, kpis, out_dir=args.outdir)

    print(f"\nDone.  All outputs in: {os.path.abspath(args.outdir)}/")
    print("\nNext step: wrap this in a GA to optimise E_cap_kwh and P_max_kw.")


if __name__ == "__main__":
    main()