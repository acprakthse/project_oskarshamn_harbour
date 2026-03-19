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
    def load_day_ahead_prices(
        cls,
        timestamps: pd.DatetimeIndex,
        csv_path=None,
    ) -> np.ndarray:
        """
        Load real day-ahead electricity prices from CSV and align to timestamps.

        The CSV must have columns:
            Datetime            — e.g. "1/1/2025 0:00"
            Price (EUR/MWhe)    — price in EUR per MWh

        Prices are converted EUR/MWhe → EUR/kWh (÷ 1000).
        Alignment is by (month, day, hour) so year mismatches are handled.
        If a timestamp has no match in the CSV, falls back to Tariff.price_vector().

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
        csv_path   : Path or str — defaults to settings.DAY_AHEAD_PRICE_CSV

        Returns
        -------
        np.ndarray  shape (T,)  dtype float64  [EUR/kWh]
        """
        from config.settings import DAY_AHEAD_PRICE_CSV
        if csv_path is None:
            csv_path = DAY_AHEAD_PRICE_CSV

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        dt_col    = "Datetime"
        price_col = "Price (EUR/MWhe)"
        df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=False)

        # Build lookup: (month, day, hour) → price in EUR/kWh
        lookup = {
            (row[dt_col].month, row[dt_col].day, row[dt_col].hour): row[price_col] / 1000.0
            for _, row in df.iterrows()
        }

        fallback = cls.price_vector(timestamps)
        prices = np.array([
            lookup.get((ts.month, ts.day, ts.hour), fallback[i])
            for i, ts in enumerate(timestamps)
        ])
        return prices

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
