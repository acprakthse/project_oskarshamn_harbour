"""
config/tariff.py
================
Electricity tariff definition.

Default: Thai PEA Medium Voltage Tariff (MVT) — adjust for your utility.
Extend with seasonal blocks or TOU schedules as needed.

Usage:
    from config.tariff import Tariff
    price_thb = Tariff.energy_price(hour=14)   # on-peak hour
    vec = Tariff.price_vector(timestamps)       # numpy array, shape (T,)
"""

import numpy as np
import pandas as pd


class Tariff:
    """
    Time-of-Use (ToU) energy tariff with monthly peak demand charge.

    Attributes
    ----------
    PEAK_HOURS   : list[int]  — hours (0–23) considered on-peak
    ENERGY_PEAK  : float      — on-peak energy rate  [currency/kWh]
    ENERGY_OFFPK : float      — off-peak energy rate [currency/kWh]
    DEMAND_CHG   : float      — monthly demand charge [currency/kW]
    EXPORT_RATE  : float      — feed-in / net-metering rate [currency/kWh]
                               set to 0.0 if no export credit
    """

    # ── PEA MVT Thailand (example) ────────────────────────────────────────────
    PEAK_HOURS   : list = list(range(9, 22))   # 09:00 – 21:59
    ENERGY_PEAK  : float = 5.80                # THB / kWh  (on-peak)
    ENERGY_OFFPK : float = 2.60                # THB / kWh  (off-peak)
    DEMAND_CHG   : float = 220.0               # THB / kW   (monthly peak import)
    EXPORT_RATE  : float = 0.0                 # THB / kWh  (0 = no net-metering)

    # ── Optional: weekend / holiday off-peak override ─────────────────────────
    WEEKEND_ALL_OFFPEAK : bool = True          # Sat/Sun always off-peak

    # =========================================================================

    @classmethod
    def energy_price(cls, hour: int, weekday: int = 0) -> float:
        """
        Return energy price for a given hour and weekday (0=Mon … 6=Sun).

        Parameters
        ----------
        hour    : int  0–23
        weekday : int  0–6  (from datetime.weekday())
        """
        if cls.WEEKEND_ALL_OFFPEAK and weekday >= 5:
            return cls.ENERGY_OFFPK
        if hour in cls.PEAK_HOURS:
            return cls.ENERGY_PEAK
        return cls.ENERGY_OFFPK

    @classmethod
    def price_vector(cls, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Build an array of hourly energy prices aligned to timestamps.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex  length T

        Returns
        -------
        np.ndarray  shape (T,)  dtype float64
        """
        return np.array([
            cls.energy_price(ts.hour, ts.weekday())
            for ts in timestamps
        ])

    @classmethod
    def summarise(cls) -> None:
        print("─" * 45)
        print("Tariff configuration")
        print(f"  On-peak hours    : {cls.PEAK_HOURS[0]:02d}:00 – {cls.PEAK_HOURS[-1]+1:02d}:00")
        print(f"  Energy on-peak   : {cls.ENERGY_PEAK:.2f} / kWh")
        print(f"  Energy off-peak  : {cls.ENERGY_OFFPK:.2f} / kWh")
        print(f"  Demand charge    : {cls.DEMAND_CHG:.2f} / kW / month")
        print(f"  Export rate      : {cls.EXPORT_RATE:.2f} / kWh")
        print(f"  Weekend off-peak : {cls.WEEKEND_ALL_OFFPEAK}")
        print("─" * 45)
