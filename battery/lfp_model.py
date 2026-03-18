"""
battery/lfp_model.py
====================
LFP (LiFePO4) battery physical model — parameters and derived properties.

LFP characteristics vs other chemistries:
  - Flat discharge curve       (~3.2 V nominal cell voltage)
  - Lower energy density       (~120–160 Wh/kg)
  - Excellent cycle life       (3 000–6 000 cycles to 80 % DoD)
  - Low temperature sensitivity vs NMC/NCA
  - Intrinsically safe         (no thermal runaway)
  - Round-trip efficiency      ~92–96 %

This class is a plain dataclass — no Pyomo, no pandas.
The GA wrapper (ga/fitness.py) will instantiate this with different
E_cap_kwh / P_max_kw values to sweep the sizing space.

Usage:
    from battery.lfp_model import LFPBattery
    bat = LFPBattery(E_cap_kwh=1726, P_max_kw=863)
    bat.summarise()
"""

from dataclasses import dataclass, field


@dataclass
class LFPBattery:
    """
    Parameters
    ----------
    E_cap_kwh        : Usable energy capacity  [kWh]
    P_max_kw         : Nameplate power limit   [kW]  (inverter-side)
    eta_c            : One-way charge efficiency
    eta_d            : One-way discharge efficiency
    soc_min          : Minimum SoC as fraction of E_cap  (protects cells)
    soc_max          : Maximum SoC as fraction of E_cap  (LFP: up to 0.95)
    soc_init         : Initial SoC fraction at t=0
    c_rate_charge    : Maximum charge C-rate
    c_rate_discharge : Maximum discharge C-rate
    """

    # ── Sizing (fixed for MILP run; swept by GA) ──────────────────────────────
    E_cap_kwh        : float = 1_726.0   # 2 h at 863 kW  →  ~2C discharge
    P_max_kw         : float = 863.0     # 1C  for 1726 kWh pack

    # ── Efficiency ────────────────────────────────────────────────────────────
    eta_c            : float = 0.97      # charge   (one-way)
    eta_d            : float = 0.97      # discharge (one-way)
    # Round-trip = eta_c × eta_d ≈ 0.941

    # ── SoC window ───────────────────────────────────────────────────────────
    soc_min          : float = 0.10      # 10 %  — avoids deep discharge
    soc_max          : float = 0.95      # 95 %  — LFP tolerates near-full
    soc_init         : float = 0.50      # 50 %  starting point

    # ── C-rate limits ─────────────────────────────────────────────────────────
    c_rate_charge    : float = 0.50      # 0.5C → 2 h full charge
    c_rate_discharge : float = 1.00      # 1.0C → 1 h full discharge

    # ── Degradation parameters (used by degradation.py and GA fitness) ────────
    calendar_deg_per_yr  : float = 0.008    # 0.8 % / year  calendar fade
    cycle_deg_per_cycle  : float = 0.00003  # per equivalent full cycle

    # ── Temperature derating (informational — HelioScope handles PV side) ─────
    temp_derating_per_degC : float = 0.003  # 0.3 % per °C above 25 °C

    # =========================================================================
    # Derived properties
    # =========================================================================

    @property
    def P_charge_max(self) -> float:
        """Max charge power [kW] — lower of nameplate and C-rate limit."""
        return min(self.P_max_kw, self.c_rate_charge * self.E_cap_kwh)

    @property
    def P_discharge_max(self) -> float:
        """Max discharge power [kW] — lower of nameplate and C-rate limit."""
        return min(self.P_max_kw, self.c_rate_discharge * self.E_cap_kwh)

    @property
    def E_usable(self) -> float:
        """Usable energy within SoC window [kWh]."""
        return self.E_cap_kwh * (self.soc_max - self.soc_min)

    @property
    def soc_init_kwh(self) -> float:
        return self.soc_init * self.E_cap_kwh

    @property
    def soc_min_kwh(self) -> float:
        return self.soc_min * self.E_cap_kwh

    @property
    def soc_max_kwh(self) -> float:
        return self.soc_max * self.E_cap_kwh

    @property
    def round_trip_efficiency(self) -> float:
        return self.eta_c * self.eta_d

    @property
    def duration_h(self) -> float:
        """Nominal duration at max discharge power [hours]."""
        return self.E_usable / max(self.P_discharge_max, 1e-6)

    # =========================================================================

    def validate(self) -> None:
        """Raise ValueError if parameters are physically inconsistent."""
        assert 0 < self.soc_min < self.soc_max <= 1.0, \
            "SoC window invalid: must have 0 < soc_min < soc_max ≤ 1"
        assert 0.0 < self.soc_init <= 1.0, \
            "soc_init must be in (0, 1]"
        assert 0.5 <= self.eta_c <= 1.0 and 0.5 <= self.eta_d <= 1.0, \
            "Efficiency values must be between 0.5 and 1.0"
        assert self.E_cap_kwh > 0 and self.P_max_kw > 0, \
            "E_cap_kwh and P_max_kw must be positive"

    def summarise(self) -> None:
        print("─" * 50)
        print("LFP Battery — Configuration")
        print(f"  Capacity (usable)    : {self.E_cap_kwh:>8.1f} kWh")
        print(f"  Usable energy        : {self.E_usable:>8.1f} kWh  "
              f"(SoC {self.soc_min*100:.0f}%–{self.soc_max*100:.0f}%)")
        print(f"  Max charge power     : {self.P_charge_max:>8.1f} kW  "
              f"({self.c_rate_charge}C)")
        print(f"  Max discharge power  : {self.P_discharge_max:>8.1f} kW  "
              f"({self.c_rate_discharge}C)")
        print(f"  Nominal duration     : {self.duration_h:>8.2f} h")
        print(f"  Round-trip efficiency: {self.round_trip_efficiency*100:>8.1f} %")
        print(f"  Calendar fade / yr   : {self.calendar_deg_per_yr*100:>8.2f} %")
        print(f"  Cycle fade / cycle   : {self.cycle_deg_per_cycle*100:.4f} %")
        print("─" * 50)
