"""
battery/degradation.py
======================
LFP battery degradation model.

Two mechanisms modelled:
  1. Calendar degradation  — capacity fade purely from time + temperature
  2. Cycle degradation     — capacity fade from charge/discharge throughput
                             estimated via equivalent full cycle (EFC) count

Rainflow cycle counting (simplified half-cycle method) is used to convert
an arbitrary SoC time-series into an EFC count for cycle-life estimation.

These functions are used by:
  - results/kpis.py         — to report end-of-year residual capacity
  - ga/fitness.py           — to penalise high-cycle solutions over project life

Usage:
    from battery.degradation import calendar_fade, cycle_fade, equivalent_full_cycles
"""

import numpy as np
from typing import Sequence


# =============================================================================
# Calendar degradation
# =============================================================================

def calendar_fade(
    years: float,
    annual_rate: float = 0.008,
    arrhenius_factor: float = 1.0,
) -> float:
    """
    Estimate fractional capacity remaining after `years` of calendar ageing.

    Uses a square-root-of-time model consistent with LFP SEI growth kinetics:
        fade = annual_rate × sqrt(years) × arrhenius_factor

    Parameters
    ----------
    years            : float  — elapsed time in years
    annual_rate      : float  — fraction lost per year at reference temperature
                                (default 0.008 = 0.8 %/yr for LFP at ~25 °C)
    arrhenius_factor : float  — temperature multiplier
                                (1.0 = 25 °C, ~1.5 at 35 °C, ~2.5 at 45 °C)

    Returns
    -------
    float : fraction of original capacity remaining  (e.g. 0.92 = 92 %)
    """
    fade = annual_rate * np.sqrt(years) * arrhenius_factor
    return max(0.0, 1.0 - fade)


# =============================================================================
# Cycle degradation
# =============================================================================

def equivalent_full_cycles(soc_series: Sequence[float],
                            e_cap_kwh: float) -> float:
    """
    Compute equivalent full cycles (EFC) from a SoC time series [kWh].

    EFC = total energy discharged / usable capacity

    Parameters
    ----------
    soc_series : array-like  — SoC in kWh at each timestep (length T)
    e_cap_kwh  : float       — usable capacity [kWh]

    Returns
    -------
    float : equivalent full cycles
    """
    soc = np.asarray(soc_series, dtype=float)
    delta = np.diff(soc)
    discharged = -delta[delta < 0].sum()   # only negative deltas = discharge
    return discharged / max(e_cap_kwh, 1e-6)


def cycle_fade(
    efc: float,
    fade_per_cycle: float = 0.00003,
) -> float:
    """
    Capacity remaining after `efc` equivalent full cycles.

        remaining = max(0, 1 - fade_per_cycle × efc)

    Parameters
    ----------
    efc            : float — equivalent full cycles accumulated
    fade_per_cycle : float — fraction of capacity lost per EFC
                             (default 0.003 % / cycle → 3000 cycles to 91 % SOH)

    Returns
    -------
    float : fraction of original capacity remaining
    """
    return max(0.0, 1.0 - fade_per_cycle * efc)


# =============================================================================
# Combined state-of-health
# =============================================================================

def state_of_health(
    years: float,
    efc: float,
    annual_cal_rate: float = 0.008,
    fade_per_cycle: float  = 0.00003,
    arrhenius_factor: float = 1.0,
) -> float:
    """
    Combined SoH = min(calendar_fade, cycle_fade).

    LFP degradation is dominated by whichever mechanism is worse.

    Parameters
    ----------
    years            : float  — project age [years]
    efc              : float  — accumulated equivalent full cycles
    annual_cal_rate  : float  — calendar rate
    fade_per_cycle   : float  — cycle rate
    arrhenius_factor : float  — temperature multiplier for calendar fade

    Returns
    -------
    float : state of health  (1.0 = new, 0.8 = end-of-life threshold)
    """
    cal = calendar_fade(years, annual_cal_rate, arrhenius_factor)
    cyc = cycle_fade(efc, fade_per_cycle)
    return min(cal, cyc)


def years_to_eol(
    annual_cal_rate: float = 0.008,
    fade_per_cycle: float  = 0.00003,
    cycles_per_year: float = 365.0,
    eol_soh: float = 0.80,
    arrhenius_factor: float = 1.0,
    max_years: float = 25.0,
) -> float:
    """
    Estimate project life in years until SoH drops below `eol_soh`.

    Iterates year by year to find the crossover point.

    Parameters
    ----------
    annual_cal_rate  : float  — calendar degradation rate
    fade_per_cycle   : float  — cycle degradation rate
    cycles_per_year  : float  — annual EFC  (e.g. 365 = 1 cycle/day)
    eol_soh          : float  — end-of-life SoH threshold (default 0.80)
    arrhenius_factor : float  — temperature multiplier
    max_years        : float  — search ceiling

    Returns
    -------
    float : estimated life in years
    """
    for y in np.arange(0.5, max_years + 0.5, 0.5):
        efc = y * cycles_per_year
        soh = state_of_health(y, efc, annual_cal_rate,
                               fade_per_cycle, arrhenius_factor)
        if soh <= eol_soh:
            return float(y)
    return float(max_years)


def degradation_report(
    years: float,
    efc: float,
    e_cap_kwh: float,
    annual_cal_rate: float = 0.008,
    fade_per_cycle: float  = 0.00003,
    arrhenius_factor: float = 1.0,
) -> dict:
    """Return a summary dict of degradation metrics for reporting."""
    cal  = calendar_fade(years, annual_cal_rate, arrhenius_factor)
    cyc  = cycle_fade(efc, fade_per_cycle)
    soh  = min(cal, cyc)
    eol  = years_to_eol(annual_cal_rate, fade_per_cycle,
                         efc / max(years, 1e-6), arrhenius_factor=arrhenius_factor)
    return {
        "years":                    years,
        "efc":                      efc,
        "calendar_remaining_frac":  cal,
        "cycle_remaining_frac":     cyc,
        "state_of_health":          soh,
        "residual_capacity_kwh":    soh * e_cap_kwh,
        "estimated_life_years":     eol,
    }
