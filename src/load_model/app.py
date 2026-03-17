"""
app.py
======
Oskarshamn Port — Hourly Electricity Demand Model 2025
Master Thesis Demo | Streamlit Interactive Dashboard

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit demandlib pandas numpy matplotlib holidays openpyxl
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import streamlit as st
import holidays

try:
    import demandlib.bdew as bdew
    DEMANDLIB_OK = True
except ImportError:
    DEMANDLIB_OK = False

try:
    import pvlib
    PVLIB_OK = True
except ImportError:
    PVLIB_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# TMY WEATHER PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_epw(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse an EnergyPlus Weather (EPW) file into an 8760-row DataFrame.
    Columns extracted: temp_air [°C], ghi, dni, dhi [W/m²],
    relative_humidity [%], wind_speed [m/s], timestamp.
    """
    lines = file_bytes.decode("utf-8", errors="replace").splitlines()
    data_lines = []
    for line in lines[8:]:
        parts = line.strip().split(",")
        if len(parts) >= 35:
            try:
                int(parts[0])
                data_lines.append(parts)
            except ValueError:
                continue
    if len(data_lines) < 8760:
        raise ValueError(
            f"EPW file contains only {len(data_lines)} data rows (need 8760). "
            "Please upload a complete annual EPW file."
        )
    data_lines = data_lines[:8760]
    records = []
    for parts in data_lines:
        try:
            records.append({
                "month":               int(parts[1]),
                "day":                 int(parts[2]),
                "hour":                int(parts[3]) - 1,
                "temp_air":            float(parts[6]),
                "dew_point":           float(parts[7]),
                "relative_humidity":   float(parts[8]),
                "ghi":                 float(parts[13]),
                "dni":                 float(parts[14]),
                "dhi":                 float(parts[15]),
                "wind_speed":          float(parts[21]),
            })
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(records)
    try:
        df["timestamp"] = pd.to_datetime({
            "year": 2025, "month": df["month"],
            "day": df["day"], "hour": df["hour"],
        })
    except Exception:
        df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.iloc[:8760].reset_index(drop=True)
    for col in ["ghi", "dni", "dhi"]:
        df[col] = df[col].clip(lower=0)
    return df


def parse_tmy_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse a generic TMY CSV. Auto-detects PVGIS, SAM, and plain CSV formats.
    Maps common column names to: temp_air, ghi, dni, dhi,
    relative_humidity, wind_speed, timestamp.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    buf = io.StringIO(text)

    # Skip comment/header rows
    try:
        raw = pd.read_csv(buf, comment="#", skip_blank_lines=True)
        if len(raw) < 100:
            raise ValueError
    except Exception:
        buf.seek(0)
        skip = 0
        for i, line in enumerate(text.splitlines()):
            parts = line.split(",")
            if len(parts) > 3:
                try:
                    float(parts[0])
                    skip = i
                    break
                except ValueError:
                    continue
        buf.seek(0)
        raw = pd.read_csv(buf, skiprows=skip)

    raw.columns = [str(c).strip() for c in raw.columns]

    COLUMN_MAP = {
        "temp_air": [
            "T2m", "temperature", "temp_air", "dry_bulb_temp",
            "Dry Bulb Temperature", "Temperature", "TEMP", "T_air",
            "DB Temperature", "Tamb",
        ],
        "ghi": [
            "G(h)", "GHI", "ghi", "GlobalHorizontal",
            "Global Horizontal Radiation", "Global Horiz",
            "Solar Radiation", "RADIATION", "Irradiance",
        ],
        "dni": [
            "Gb(n)", "DNI", "dni", "DirectNormal",
            "Direct Normal Radiation", "Direct Normal",
        ],
        "dhi": [
            "Gd(h)", "DHI", "dhi", "DiffuseHorizontal",
            "Diffuse Horizontal Radiation", "Diffuse Horiz",
        ],
        "relative_humidity": [
            "RH", "relative_humidity", "Rhum",
            "Relative Humidity", "RH%", "Humidity",
        ],
        "wind_speed": [
            "WS10m", "wind_speed", "WindSpeed", "FF",
            "Wind Speed", "WINDSPEED", "WS",
        ],
    }

    out = pd.DataFrame()
    for std, candidates in COLUMN_MAP.items():
        for cand in candidates:
            if cand in raw.columns:
                out[std] = pd.to_numeric(raw[cand], errors="coerce")
                break
        if std not in out.columns:
            out[std] = np.nan

    # Timestamp detection
    if "time" in raw.columns:
        try:
            out["timestamp"] = pd.to_datetime(raw["time"], errors="coerce")
        except Exception:
            out["timestamp"] = pd.date_range("2025-01-01", periods=len(raw), freq="h")
    elif "Date (MM/DD/YYYY)" in raw.columns and "Time (HH:MM)" in raw.columns:
        try:
            out["timestamp"] = pd.to_datetime(
                raw["Date (MM/DD/YYYY)"].astype(str) + " " + raw["Time (HH:MM)"].astype(str),
                errors="coerce",
            )
        except Exception:
            out["timestamp"] = pd.date_range("2025-01-01", periods=len(raw), freq="h")
    else:
        out["timestamp"] = pd.date_range("2025-01-01", periods=len(raw), freq="h")

    out = out.iloc[:8760].reset_index(drop=True)
    if len(out) < 8760:
        pad = 8760 - len(out)
        last = out.iloc[-1]
        fill = pd.DataFrame([last.to_dict()] * pad)
        fill["timestamp"] = pd.date_range(
            out["timestamp"].iloc[-1] + pd.Timedelta(hours=1), periods=pad, freq="h"
        )
        out = pd.concat([out, fill], ignore_index=True)

    for col in ["ghi", "dni", "dhi"]:
        out[col] = out[col].clip(lower=0)
    return out


def fetch_pvgis_tmy(latitude: float, longitude: float, year: int = 2025) -> pd.DataFrame:
    """
    Fetch TMY from PVGIS REST API (requires internet + pvlib).
    """
    if not PVLIB_OK:
        raise ImportError("pvlib not installed. Run: pip install pvlib")
    tmy_data, _, _, _ = pvlib.iotools.get_pvgis_tmy(
        latitude=latitude,
        longitude=longitude,
        outputformat="json",
        startyear=2005,
        endyear=2020,
    )
    tmy_data = tmy_data.reset_index()
    col_map = {
        "T2m": "temp_air", "G(h)": "ghi", "Gb(n)": "dni",
        "Gd(h)": "dhi", "RH": "relative_humidity", "WS10m": "wind_speed",
    }
    tmy_data = tmy_data.rename(columns={k: v for k, v in col_map.items() if k in tmy_data.columns})
    if "time(UTC)" in tmy_data.columns:
        tmy_data = tmy_data.rename(columns={"time(UTC)": "timestamp"})
    if "timestamp" not in tmy_data.columns:
        tmy_data["timestamp"] = pd.date_range(f"{year}-01-01", periods=len(tmy_data), freq="h")
    else:
        try:
            tmy_data["timestamp"] = pd.to_datetime(tmy_data["timestamp"]).apply(
                lambda t: t.replace(year=year)
            )
        except Exception:
            tmy_data["timestamp"] = pd.date_range(f"{year}-01-01", periods=len(tmy_data), freq="h")
    return tmy_data.iloc[:8760].reset_index(drop=True)


def tmy_summary(tmy_df: pd.DataFrame) -> dict:
    """Compute display statistics from a parsed TMY DataFrame."""
    t = tmy_df.get("temp_air", pd.Series(dtype=float)).dropna()
    g = tmy_df.get("ghi", pd.Series(dtype=float)).dropna()
    return {
        "t_mean":            round(float(t.mean()), 1)  if len(t) else 0,
        "t_min":             round(float(t.min()),  1)  if len(t) else 0,
        "t_max":             round(float(t.max()),  1)  if len(t) else 0,
        "annual_ghi_kWh_m2": round(float(g.sum() / 1000), 1) if len(g) else 0,
        "n_hours_below_0":   int((t < 0).sum())  if len(t) else 0,
        "n_hours_above_25":  int((t > 25).sum()) if len(t) else 0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oskarshamn Port — Demand Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #0a0a0a;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    line-height: 1.1;
}

.main-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 300;
    color: #666;
    margin-top: 4px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 2rem 0 0.5rem 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #e8e8e8;
}

.kpi-container {
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}

.stMetric {
    background: transparent !important;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem !important;
    font-weight: 600;
    color: #0a0a0a;
}

[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.75rem !important;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem !important;
}

.info-box {
    background: #f0f4ff;
    border-left: 3px solid #3b5bdb;
    padding: 0.75rem 1rem;
    border-radius: 0 4px 4px 0;
    font-size: 0.85rem;
    color: #2d3a8c;
    margin: 0.5rem 0;
}

.warn-box {
    background: #fffbe6;
    border-left: 3px solid #f59f00;
    padding: 0.75rem 1rem;
    border-radius: 0 4px 4px 0;
    font-size: 0.85rem;
    color: #664d00;
    margin: 0.5rem 0;
}

div[data-testid="stSidebar"] {
    background: #0a0a0a;
}

div[data-testid="stSidebar"] * {
    color: #e8e8e8 !important;
}

div[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #3b5bdb !important;
}

div[data-testid="stSidebar"] label {
    color: #aaa !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button {
    background: #0a0a0a;
    color: #fff;
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 0.6rem 2rem;
    text-transform: uppercase;
    transition: background 0.15s;
}

.stButton > button:hover {
    background: #3b5bdb;
    color: #fff;
}

.stDownloadButton > button {
    background: transparent;
    color: #3b5bdb;
    border: 1.5px solid #3b5bdb;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 0.4rem 1rem;
    text-transform: uppercase;
}

.stDownloadButton > button:hover {
    background: #3b5bdb;
    color: #fff;
}

.stDataFrame {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
}

div[data-testid="stExpander"] {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_HVAC = pd.DataFrame([
    {"parameter": "heating_setpoint_degC",         "value": 21.0},
    {"parameter": "cooling_setpoint_degC",          "value": 24.0},
    {"parameter": "heating_sensitivity_kW_per_degC","value": 8.0},
    {"parameter": "cooling_sensitivity_kW_per_degC","value": 5.0},
    {"parameter": "hvac_day_multiplier",            "value": 1.0},
    {"parameter": "hvac_night_multiplier",          "value": 0.6},
])

DEFAULT_EQUIPMENT = pd.DataFrame([
    {"load_type": "cranes",          "weekday_day_kW": 120.0, "weekday_night_kW": 20.0,  "weekend_day_kW": 60.0,  "weekend_night_kW": 10.0,  "holiday_kW": 5.0},
    {"load_type": "forklifts",       "weekday_day_kW": 45.0,  "weekday_night_kW": 10.0,  "weekend_day_kW": 20.0,  "weekend_night_kW": 5.0,   "holiday_kW": 3.0},
    {"load_type": "conveyors",       "weekday_day_kW": 35.0,  "weekday_night_kW": 8.0,   "weekend_day_kW": 15.0,  "weekend_night_kW": 4.0,   "holiday_kW": 2.0},
    {"load_type": "pumps",           "weekday_day_kW": 22.0,  "weekday_night_kW": 18.0,  "weekend_day_kW": 20.0,  "weekend_night_kW": 16.0,  "holiday_kW": 15.0},
    {"load_type": "reefer_support",  "weekday_day_kW": 80.0,  "weekday_night_kW": 80.0,  "weekend_day_kW": 80.0,  "weekend_night_kW": 80.0,  "holiday_kW": 80.0},
    {"load_type": "lighting",        "weekday_day_kW": 18.0,  "weekday_night_kW": 6.0,   "weekend_day_kW": 12.0,  "weekend_night_kW": 6.0,   "holiday_kW": 6.0},
    {"load_type": "office_it",       "weekday_day_kW": 14.0,  "weekday_night_kW": 2.0,   "weekend_day_kW": 3.0,   "weekend_night_kW": 1.0,   "holiday_kW": 1.0},
])

DEFAULT_SHORE = pd.DataFrame([
    {"vessel_name": "Birka Carrier",    "start": "2025-01-06 08:00", "end": "2025-01-06 16:00", "power_kW": 850.0},
    {"vessel_name": "Viking Glory",     "start": "2025-02-14 06:00", "end": "2025-02-14 20:00", "power_kW": 1200.0},
    {"vessel_name": "Gotland Ferry",    "start": "2025-03-22 10:00", "end": "2025-03-23 06:00", "power_kW": 600.0},
    {"vessel_name": "Baltic Carrier",   "start": "2025-05-10 14:00", "end": "2025-05-11 08:00", "power_kW": 450.0},
    {"vessel_name": "Stena Scandica",   "start": "2025-06-18 07:00", "end": "2025-06-18 18:00", "power_kW": 950.0},
    {"vessel_name": "RoRo Express",     "start": "2025-07-04 12:00", "end": "2025-07-05 04:00", "power_kW": 700.0},
    {"vessel_name": "Nordic Breeze",    "start": "2025-08-20 09:00", "end": "2025-08-20 21:00", "power_kW": 500.0},
    {"vessel_name": "Gotland Ferry",    "start": "2025-09-15 05:00", "end": "2025-09-16 07:00", "power_kW": 600.0},
    {"vessel_name": "Arctic Carrier",   "start": "2025-10-28 11:00", "end": "2025-10-29 09:00", "power_kW": 800.0},
    {"vessel_name": "Baltic Express",   "start": "2025-12-20 16:00", "end": "2025-12-21 08:00", "power_kW": 750.0},
])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_time_index(year: int = 2025) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", periods=8760, freq="h")


def swedish_holidays(year: int = 2025) -> set:
    se = holidays.Sweden(years=year)
    return set(se.keys())


def synthetic_temperature(idx: pd.DatetimeIndex,
                           annual_mean: float,
                           annual_amplitude: float,
                           daily_amplitude: float) -> np.ndarray:
    """
    T(t) = mean - amp_annual * cos(2π(doy-15)/365) + amp_daily * sin(2π*hod/24)
    Peak summer ~mid-July, peak daily ~14:00.
    """
    doy = idx.day_of_year.values
    hod = idx.hour.values
    seasonal = -annual_amplitude * np.cos(2 * np.pi * (doy - 15) / 365)
    diurnal   =  daily_amplitude  * np.sin(2 * np.pi * (hod - 6)  / 24)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.2, len(idx))
    return annual_mean + seasonal + diurnal + noise


def base_load_profile(idx: pd.DatetimeIndex,
                       annual_kwh: float,
                       hol_set: set,
                       use_demandlib: bool) -> np.ndarray:
    """
    Generate industrial base load profile.
    Uses demandlib G3 (industrial) if available; falls back to synthetic.
    """
    if use_demandlib and DEMANDLIB_OK:
        try:
            year = idx[0].year
            e_slp = bdew.ElecSlp(year, holidays=hol_set)
            df_slp = e_slp.get_profile({"g3": annual_kwh})
            df_slp = df_slp.resample("h").mean()
            df_slp = df_slp.reindex(idx, method="nearest", tolerance="1h")
            profile = df_slp["g3"].values
            profile = np.nan_to_num(profile, nan=annual_kwh / 8760)
            # Normalise to exact annual total
            profile = profile / (profile.sum() + 1e-9) * annual_kwh
            return profile
        except Exception:
            pass

    # Synthetic fallback: flat with weekday/weekend/night variation
    hod = idx.hour.values
    dow = idx.dayofweek.values
    dates = idx.normalize()
    is_holiday = np.array([d.date() in hol_set for d in dates])
    is_day     = (hod >= 6) & (hod < 22)
    is_weekday = dow < 5

    mult = np.ones(len(idx))
    mult[is_weekday  &  is_day  & ~is_holiday] = 1.15
    mult[is_weekday  & ~is_day  & ~is_holiday] = 0.75
    mult[~is_weekday &  is_day  & ~is_holiday] = 0.90
    mult[~is_weekday & ~is_day  & ~is_holiday] = 0.60
    mult[is_holiday] = 0.50

    base = annual_kwh / 8760
    profile = base * mult
    profile = profile / profile.sum() * annual_kwh
    return profile


def hvac_profile(idx: pd.DatetimeIndex,
                  temp: np.ndarray,
                  hvac_params: dict,
                  hol_set: set) -> tuple:
    """Return (heating_kw, cooling_kw) arrays."""
    t_heat = hvac_params["heating_setpoint_degC"]
    t_cool = hvac_params["cooling_setpoint_degC"]
    k_heat = hvac_params["heating_sensitivity_kW_per_degC"]
    k_cool = hvac_params["cooling_sensitivity_kW_per_degC"]
    day_m  = hvac_params["hvac_day_multiplier"]
    night_m= hvac_params["hvac_night_multiplier"]

    hod  = idx.hour.values
    dates = idx.normalize()
    is_holiday = np.array([d.date() in hol_set for d in dates])
    is_day = (hod >= 6) & (hod < 22)

    time_mult = np.where(is_day, day_m, night_m)
    time_mult[is_holiday] *= 0.5

    heating_kw = np.maximum(t_heat - temp, 0) * k_heat * time_mult
    cooling_kw = np.maximum(temp - t_cool, 0) * k_cool * time_mult

    return heating_kw, cooling_kw


def equipment_profile(idx: pd.DatetimeIndex,
                       equip_df: pd.DataFrame,
                       hol_set: set) -> np.ndarray:
    """Sum equipment loads using time/day/holiday logic."""
    hod  = idx.hour.values
    dow  = idx.dayofweek.values
    dates = idx.normalize()
    is_holiday = np.array([d.date() in hol_set for d in dates])
    is_day     = (hod >= 6) & (hod < 22)
    is_weekday = dow < 5

    total = np.zeros(len(idx))
    for _, row in equip_df.iterrows():
        load = np.zeros(len(idx))
        load[is_holiday]                                  = row["holiday_kW"]
        load[~is_holiday & is_weekday  & is_day]          = row["weekday_day_kW"]
        load[~is_holiday & is_weekday  & ~is_day]         = row["weekday_night_kW"]
        load[~is_holiday & ~is_weekday & is_day]          = row["weekend_day_kW"]
        load[~is_holiday & ~is_weekday & ~is_day]         = row["weekend_night_kW"]
        total += load
    return total


def shore_power_profile(idx: pd.DatetimeIndex,
                         shore_df: pd.DataFrame) -> np.ndarray:
    """Convert vessel schedule to hourly kW array."""
    load = np.zeros(len(idx))
    ts = pd.Series(load, index=idx)

    for _, row in shore_df.iterrows():
        try:
            start = pd.Timestamp(row["start"])
            end   = pd.Timestamp(row["end"])
            power = float(row["power_kW"])
            mask  = (ts.index >= start) & (ts.index < end)
            ts[mask] += power
        except Exception:
            continue
    return ts.values


def run_model(idx, annual_base_kwh, hvac_params, equip_df, shore_df,
              hol_set, t_mean, t_amp_annual, t_amp_daily, use_demandlib,
              tmy_df=None):
    """
    Assemble full demand model. Returns (df, kpis).

    Parameters
    ----------
    tmy_df : pd.DataFrame or None
        If provided, uses real TMY temperature (temp_air column) instead
        of the synthetic sinusoidal model. Must have 8760 rows aligned
        to idx or contain a 'timestamp' column for alignment.
    """

    # ── Temperature source ──────────────────────────────────────────────────
    temp_source = "synthetic"
    if tmy_df is not None and "temp_air" in tmy_df.columns:
        t_series = tmy_df["temp_air"].values
        if len(t_series) == 8760:
            temp = np.nan_to_num(t_series.astype(float),
                                  nan=float(np.nanmean(t_series)))
            temp_source = "TMY file"
        else:
            # Align by timestamp if available
            if "timestamp" in tmy_df.columns:
                tmy_indexed = tmy_df.set_index("timestamp")["temp_air"]
                tmy_reindexed = tmy_indexed.reindex(idx, method="nearest",
                                                     tolerance="2h")
                if tmy_reindexed.notna().sum() > 4000:
                    temp = tmy_reindexed.fillna(method="ffill").fillna(method="bfill").values
                    temp_source = "TMY file (aligned)"
                else:
                    temp = synthetic_temperature(idx, t_mean, t_amp_annual, t_amp_daily)
            else:
                temp = synthetic_temperature(idx, t_mean, t_amp_annual, t_amp_daily)
    else:
        temp = synthetic_temperature(idx, t_mean, t_amp_annual, t_amp_daily)

    # ── GHI source (stored in output if available) ──────────────────────────
    ghi = None
    if tmy_df is not None and "ghi" in tmy_df.columns:
        g = tmy_df["ghi"].values
        if len(g) == 8760:
            ghi = np.nan_to_num(g.astype(float), nan=0.0).clip(min=0)

    base_kw           = base_load_profile(idx, annual_base_kwh, hol_set, use_demandlib)
    heat_kw, cool_kw  = hvac_profile(idx, temp, hvac_params, hol_set)
    equip_kw          = equipment_profile(idx, equip_df, hol_set)
    shore_kw          = shore_power_profile(idx, shore_df)

    total_kw = base_kw + heat_kw + cool_kw + equip_kw + shore_kw

    df = pd.DataFrame({
        "timestamp":        idx,
        "temperature_degC": temp,
        "base_load_kW":     base_kw,
        "hvac_heating_kW":  heat_kw,
        "hvac_cooling_kW":  cool_kw,
        "equipment_kW":     equip_kw,
        "shore_power_kW":   shore_kw,
        "total_load_kW":    total_kw,
    })
    if ghi is not None:
        df["ghi_W_m2"] = ghi
    df["total_load_kWh"] = df["total_load_kW"] * 1.0

    kpis = {
        "annual_total_MWh":  total_kw.sum() / 1000,
        "peak_load_kW":      total_kw.max(),
        "avg_load_kW":       total_kw.mean(),
        "base_MWh":          base_kw.sum() / 1000,
        "hvac_MWh":          (heat_kw + cool_kw).sum() / 1000,
        "equipment_MWh":     equip_kw.sum() / 1000,
        "shore_MWh":         shore_kw.sum() / 1000,
        "load_factor_pct":   total_kw.mean() / (total_kw.max() + 1e-9) * 100,
        "temp_source":       temp_source,
    }
    return df, kpis


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "base":       "#1c3d6e",
    "heating":    "#c0392b",
    "cooling":    "#2980b9",
    "equipment":  "#27ae60",
    "shore":      "#8e44ad",
    "total":      "#2d2d2d",
    "temp":       "#e67e22",
}

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, loc="left",
                 fontfamily="monospace")
    ax.set_xlabel(xlabel, fontsize=8, color="#666")
    ax.set_ylabel(ylabel, fontsize=8, color="#666")
    ax.tick_params(labelsize=7, colors="#666")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e0e0e0")
    ax.spines["bottom"].set_color("#e0e0e0")
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.5, linestyle="--")


def fig_annual_load(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    facecolor="white")
    fig.subplots_adjust(hspace=0.08)

    # Stacked area
    ts = df["timestamp"]
    ax1.stackplot(ts,
                  df["base_load_kW"],
                  df["hvac_heating_kW"],
                  df["hvac_cooling_kW"],
                  df["equipment_kW"],
                  df["shore_power_kW"],
                  labels=["Base load", "HVAC heating", "HVAC cooling",
                          "Equipment", "Shore power"],
                  colors=[PALETTE["base"], PALETTE["heating"], PALETTE["cooling"],
                          PALETTE["equipment"], PALETTE["shore"]],
                  alpha=0.85)
    ax1.plot(ts, df["total_load_kW"], color=PALETTE["total"],
             linewidth=0.4, alpha=0.5)
    style_ax(ax1, title="Annual load profile — Oskarshamn port 2025",
             ylabel="Power [kW]")
    ax1.legend(fontsize=7, loc="upper right", framealpha=0.9,
               ncol=5, columnspacing=1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.set_xlim(ts.iloc[0], ts.iloc[-1])
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Temperature strip
    ax2.fill_between(ts, df["temperature_degC"],
                     color=PALETTE["temp"], alpha=0.3)
    ax2.plot(ts, df["temperature_degC"], color=PALETTE["temp"],
             linewidth=0.5, alpha=0.8)
    ax2.axhline(0, color="#aaa", linewidth=0.5, linestyle="--")
    style_ax(ax2, ylabel="Temp [°C]")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.set_xlim(ts.iloc[0], ts.iloc[-1])

    fig.patch.set_facecolor("white")
    return fig


def fig_daily_profile(df):
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")

    components = {
        "Base load":    "base_load_kW",
        "HVAC heating": "hvac_heating_kW",
        "HVAC cooling": "hvac_cooling_kW",
        "Equipment":    "equipment_kW",
        "Shore power":  "shore_power_kW",
    }
    colors = [PALETTE["base"], PALETTE["heating"], PALETTE["cooling"],
              PALETTE["equipment"], PALETTE["shore"]]

    df_h = df.copy()
    df_h["hour"] = df_h["timestamp"].dt.hour

    bottoms = np.zeros(24)
    for (label, col), color in zip(components.items(), colors):
        vals = df_h.groupby("hour")[col].mean().values
        ax.bar(np.arange(24), vals, bottom=bottoms, label=label,
               color=color, alpha=0.85, width=0.8)
        bottoms += vals

    # Total line
    total_avg = df_h.groupby("hour")["total_load_kW"].mean().values
    ax.plot(np.arange(24), total_avg, color=PALETTE["total"],
            marker="o", markersize=3, linewidth=1.5, label="Total (mean)", zorder=5)

    style_ax(ax, title="Average daily load profile — all days 2025",
             xlabel="Hour of day", ylabel="Average power [kW]")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=3)
    fig.tight_layout()
    fig.patch.set_facecolor("white")
    return fig


def fig_weekly_profile(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

    df_w = df.copy()
    df_w["hour"]    = df_w["timestamp"].dt.hour
    df_w["weekday"] = df_w["timestamp"].dt.dayofweek < 5

    labels = ["Weekday", "Weekend"]
    for ax, is_wd, label in zip(axes, [True, False], labels):
        sub = df_w[df_w["weekday"] == is_wd]
        bottoms = np.zeros(24)
        for col, color, name in [
            ("base_load_kW",    PALETTE["base"],      "Base"),
            ("hvac_heating_kW", PALETTE["heating"],   "Heating"),
            ("hvac_cooling_kW", PALETTE["cooling"],   "Cooling"),
            ("equipment_kW",    PALETTE["equipment"], "Equipment"),
            ("shore_power_kW",  PALETTE["shore"],     "Shore"),
        ]:
            vals = sub.groupby("hour")[col].mean().values
            ax.bar(np.arange(24), vals, bottom=bottoms, color=color,
                   alpha=0.85, width=0.8, label=name)
            bottoms += vals

        total_avg = sub.groupby("hour")["total_load_kW"].mean().values
        ax.plot(np.arange(24), total_avg, color=PALETTE["total"],
                marker="o", markersize=3, linewidth=1.5, label="Total")

        style_ax(ax, title=f"{label} average profile",
                 xlabel="Hour of day", ylabel="Average power [kW]")
        ax.set_xticks(np.arange(0, 24, 3))
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.patch.set_facecolor("white")
    return fig


def fig_monthly_contribution(df):
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor="white")

    df_m = df.copy()
    df_m["month"] = df_m["timestamp"].dt.month

    months = range(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(12)
    w = 0.65

    components = [
        ("base_load_kW",    PALETTE["base"],      "Base load"),
        ("hvac_heating_kW", PALETTE["heating"],   "HVAC heating"),
        ("hvac_cooling_kW", PALETTE["cooling"],   "HVAC cooling"),
        ("equipment_kW",    PALETTE["equipment"], "Equipment"),
        ("shore_power_kW",  PALETTE["shore"],     "Shore power"),
    ]

    bottoms = np.zeros(12)
    for col, color, label in components:
        vals = np.array([df_m[df_m["month"] == m][col].sum() / 1000
                         for m in months])
        ax.bar(x, vals, bottom=bottoms, width=w, color=color,
               alpha=0.85, label=label)
        bottoms += vals

    style_ax(ax, title="Monthly energy breakdown by component [MWh]",
             xlabel="Month", ylabel="Energy [MWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=5)
    fig.tight_layout()
    fig.patch.set_facecolor("white")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSV DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def fig_tmy_overview(tmy_df: pd.DataFrame) -> plt.Figure:
    """
    4-panel TMY overview: temperature time series, GHI time series,
    monthly mean temperature, monthly GHI total.
    """
    has_temp = "temp_air" in tmy_df.columns and tmy_df["temp_air"].notna().sum() > 100
    has_ghi  = "ghi"      in tmy_df.columns and tmy_df["ghi"].notna().sum() > 100

    fig, axes = plt.subplots(2, 2, figsize=(13, 6), facecolor="white")
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    idx_ts = pd.date_range("2025-01-01", periods=8760, freq="h")

    # Panel 1 — Temperature time series
    ax = axes[0, 0]
    if has_temp:
        t = tmy_df["temp_air"].values
        ax.fill_between(idx_ts, t, alpha=0.25, color=PALETTE["heating"])
        ax.plot(idx_ts, t, color=PALETTE["heating"], linewidth=0.5, alpha=0.8)
        ax.axhline(0, color="#aaa", linewidth=0.5, linestyle="--")
    style_ax(ax, title="Outdoor temperature [°C]", ylabel="°C")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # Panel 2 — GHI time series
    ax = axes[0, 1]
    if has_ghi:
        g = tmy_df["ghi"].values
        ax.fill_between(idx_ts, g, alpha=0.25, color=PALETTE["temp"])
        ax.plot(idx_ts, g, color=PALETTE["temp"], linewidth=0.3, alpha=0.7)
    style_ax(ax, title="Global horizontal irradiance [W/m²]", ylabel="W/m²")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # Panel 3 — Monthly mean temperature
    ax = axes[1, 0]
    month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    if has_temp:
        t_series = pd.Series(tmy_df["temp_air"].values, index=idx_ts)
        monthly_t = t_series.groupby(t_series.index.month).mean().values
        colors_t = [PALETTE["cooling"] if v > 0 else PALETTE["heating"] for v in monthly_t]
        bars = ax.bar(range(1, 13), monthly_t, color=colors_t, alpha=0.85, width=0.7)
        ax.axhline(0, color="#aaa", linewidth=0.5, linestyle="--")
        ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=2)
    style_ax(ax, title="Monthly mean temperature [°C]", ylabel="°C")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, fontsize=8)

    # Panel 4 — Monthly GHI total
    ax = axes[1, 1]
    if has_ghi:
        g_series = pd.Series(tmy_df["ghi"].values, index=idx_ts)
        monthly_g = g_series.groupby(g_series.index.month).sum().values / 1000
        bars = ax.bar(range(1, 13), monthly_g, color=PALETTE["temp"], alpha=0.85, width=0.7)
        ax.bar_label(bars, fmt="%.0f", fontsize=6.5, padding=2)
    style_ax(ax, title="Monthly GHI total [kWh/m²]", ylabel="kWh/m²")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, fontsize=8)

    fig.patch.set_facecolor("white")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

if "hvac_df" not in st.session_state:
    st.session_state["hvac_df"] = DEFAULT_HVAC.copy()
if "equip_df" not in st.session_state:
    st.session_state["equip_df"] = DEFAULT_EQUIPMENT.copy()
if "shore_df" not in st.session_state:
    st.session_state["shore_df"] = DEFAULT_SHORE.copy()
if "results" not in st.session_state:
    st.session_state["results"] = None
if "tmy_df" not in st.session_state:
    st.session_state["tmy_df"] = None
if "tmy_source_label" not in st.session_state:
    st.session_state["tmy_source_label"] = None


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

col_t, col_badge = st.columns([5, 1])
with col_t:
    st.markdown('<p class="main-title">⚡ Oskarshamn Port</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Hourly Electricity Demand Model — 2025</p>',
                unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Master Thesis Demo · SE3 Grid Zone")

st.markdown("""
<div class="info-box">
This tool models the full hourly electricity demand of Oskarshamn harbour terminal 
for 2025, combining four components: <strong>industrial base load</strong> (demandlib G3 profile), 
<strong>HVAC demand</strong> (temperature-driven heat pump model), 
<strong>port equipment</strong> (schedule-based), and 
<strong>shore power</strong> (vessel berthing events). 
Adjust parameters in the sidebar and tables below, then click <em>Run Model</em>.
</div>
""", unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙ Global Parameters")
    st.markdown("---")

    st.markdown("**Base load**")
    annual_base_kwh = st.number_input(
        "Annual base demand [kWh]",
        min_value=100_000, max_value=50_000_000,
        value=3_500_000, step=100_000,
        help="Annual industrial base electricity demand (excl. HVAC, equipment, shore power)"
    )

    use_demandlib = st.checkbox(
        "Use demandlib G3 profile",
        value=True,
        help="If disabled, uses a simplified synthetic profile"
    )
    if not DEMANDLIB_OK:
        st.markdown('<div class="warn-box">demandlib not installed — using synthetic profile</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Temperature source**")
    temp_source_choice = st.radio(
        "Temperature data",
        ["Synthetic (parametric)", "TMY file (see section 00)"],
        index=0,
        help="Use a real TMY file to drive HVAC loads, or the synthetic sinusoidal model",
        label_visibility="collapsed",
    )
    use_tmy = (temp_source_choice == "TMY file (see section 00)")

    if use_tmy:
        if st.session_state["tmy_df"] is not None:
            st.markdown(
                f'<div style="background:#1a3a1a;color:#7ddd7d;padding:6px 10px;'
                f'border-radius:3px;font-size:0.75rem;font-family:monospace;">'
                f'TMY loaded: {st.session_state["tmy_source_label"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#3a2a00;color:#ffcc44;padding:6px 10px;'
                'border-radius:3px;font-size:0.75rem;font-family:monospace;">'
                'No TMY loaded — upload in section 00 below</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("**Synthetic temperature (fallback)**")
    t_mean = st.slider("Annual mean temperature [°C]", -5.0, 15.0, 6.5, 0.5)
    t_amp_annual = st.slider("Seasonal amplitude [°C]", 5.0, 20.0, 12.0, 0.5)
    t_amp_daily  = st.slider("Daily amplitude [°C]", 0.0, 8.0, 3.5, 0.5)

    st.markdown("---")
    st.markdown("**Operational hours**")
    day_start = st.slider("Day-shift start", 0, 12, 6)
    day_end   = st.slider("Day-shift end",   12, 24, 22)

    st.markdown("---")
    st.markdown("**Year**")
    year = st.selectbox("Simulation year", [2025, 2026, 2027], index=0)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT TABLES
# ─────────────────────────────────────────────────────────────────────────────

# ── 00 — TMY Weather Data ─────────────────────────────────────────────────────
st.markdown('<p class="section-header">00 — TMY Weather Data (optional — improves HVAC accuracy)</p>',
            unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Upload a real Typical Meteorological Year (TMY) file to replace the synthetic temperature model.
When a TMY is loaded, the <strong>outdoor temperature</strong> drives HVAC heating and cooling loads directly.
The GHI column is also stored in the output for PV potential analysis.
Supported formats: <strong>EPW</strong> (EnergyPlus), <strong>CSV</strong> (PVGIS, SAM, or plain hourly CSV).
</div>
""", unsafe_allow_html=True)

tmy_col1, tmy_col2, tmy_col3 = st.columns([2, 1, 1])

with tmy_col1:
    tmy_method = st.radio(
        "TMY source method",
        ["Upload file (EPW or CSV)", "Fetch from PVGIS API"],
        horizontal=True,
        label_visibility="collapsed",
    )

with tmy_col2:
    if st.button("Clear TMY data", use_container_width=True):
        st.session_state["tmy_df"] = None
        st.session_state["tmy_source_label"] = None
        st.rerun()

with tmy_col3:
    # Download a CSV template showing expected column names
    template_tmy = pd.DataFrame({
        "temp_air": [5.2, 4.8, 4.1],
        "ghi":      [0.0, 12.5, 45.3],
        "dni":      [0.0, 8.1, 30.2],
        "dhi":      [0.0, 4.4, 15.1],
        "relative_humidity": [82.0, 80.0, 76.0],
        "wind_speed": [3.2, 4.1, 2.8],
    })
    st.download_button(
        "↓ CSV template",
        data=to_csv_bytes(template_tmy),
        file_name="tmy_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

if tmy_method == "Upload file (EPW or CSV)":
    uploaded_tmy = st.file_uploader(
        "Upload TMY file",
        type=["epw", "csv", "txt"],
        key="tmy_upload",
        help="EPW: EnergyPlus Weather format (e.g. from EnergyPlus.net or Climate.OneBuilding.Org). "
             "CSV: PVGIS TMY export, SAM weather file, or plain hourly CSV with temp_air column.",
        label_visibility="collapsed",
    )
    if uploaded_tmy is not None:
        with st.spinner("Parsing weather file..."):
            try:
                file_bytes = uploaded_tmy.read()
                fname = uploaded_tmy.name.lower()
                if fname.endswith(".epw"):
                    tmy_parsed = parse_epw(file_bytes)
                    label = f"EPW · {uploaded_tmy.name}"
                else:
                    tmy_parsed = parse_tmy_csv(file_bytes)
                    label = f"CSV · {uploaded_tmy.name}"

                st.session_state["tmy_df"] = tmy_parsed
                st.session_state["tmy_source_label"] = label
                st.success(
                    f"Loaded {label} — {len(tmy_parsed)} hourly rows · "
                    f"T range: {tmy_parsed['temp_air'].min():.1f}°C to "
                    f"{tmy_parsed['temp_air'].max():.1f}°C"
                )
            except Exception as e:
                st.error(f"Could not parse file: {e}")
                import traceback; st.code(traceback.format_exc())

else:  # PVGIS API fetch
    st.markdown("""
    <div class="warn-box">
    Requires internet connection and <code>pip install pvlib</code>.
    Fetches TMY from the EU PVGIS database (SARAH-2, 2005–2020 average).
    </div>
    """, unsafe_allow_html=True)

    pv_col1, pv_col2, pv_col3 = st.columns([1, 1, 1])
    with pv_col1:
        pvgis_lat = st.number_input("Latitude", value=57.26, format="%.4f",
                                     help="Oskarshamn: 57.2636")
    with pv_col2:
        pvgis_lon = st.number_input("Longitude", value=16.45, format="%.4f",
                                     help="Oskarshamn: 16.4483")
    with pv_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Fetch from PVGIS", use_container_width=True):
            with st.spinner("Fetching TMY from PVGIS API..."):
                try:
                    tmy_fetched = fetch_pvgis_tmy(pvgis_lat, pvgis_lon, year=year)
                    label = f"PVGIS API · lat={pvgis_lat:.2f} lon={pvgis_lon:.2f}"
                    st.session_state["tmy_df"] = tmy_fetched
                    st.session_state["tmy_source_label"] = label
                    st.success(
                        f"Fetched {label} — "
                        f"T range: {tmy_fetched['temp_air'].min():.1f}°C to "
                        f"{tmy_fetched['temp_air'].max():.1f}°C"
                    )
                except Exception as e:
                    st.error(f"PVGIS fetch failed: {e}")

# ── TMY preview when loaded ───────────────────────────────────────────────────
if st.session_state["tmy_df"] is not None:
    tmy_df_loaded = st.session_state["tmy_df"]
    s = tmy_summary(tmy_df_loaded)

    st.markdown("**TMY statistics**")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Mean temp", f"{s['t_mean']} °C")
    mc2.metric("Min temp",  f"{s['t_min']} °C")
    mc3.metric("Max temp",  f"{s['t_max']} °C")
    mc4.metric("Annual GHI", f"{s['annual_ghi_kWh_m2']} kWh/m²")
    mc5.metric("Hours < 0°C", f"{s['n_hours_below_0']} h")

    with st.expander("TMY overview charts", expanded=True):
        fig_tmy = fig_tmy_overview(tmy_df_loaded)
        st.pyplot(fig_tmy, use_container_width=True)
        plt.close(fig_tmy)

    with st.expander("TMY raw data preview (first 48 hours)", expanded=False):
        preview_cols = [c for c in ["timestamp","temp_air","ghi","dni","dhi",
                                     "relative_humidity","wind_speed"]
                        if c in tmy_df_loaded.columns]
        st.dataframe(
            tmy_df_loaded[preview_cols].head(48).style.format(
                {c: "{:.2f}" for c in preview_cols if c != "timestamp"}
            ),
            use_container_width=True,
            height=280,
        )

    col_tmy_dl1, col_tmy_dl2 = st.columns([1, 4])
    with col_tmy_dl1:
        st.download_button(
            "↓ Download parsed TMY (CSV)",
            data=to_csv_bytes(tmy_df_loaded),
            file_name="tmy_parsed.csv",
            mime="text/csv",
        )

st.divider()

# ── Equipment ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">01 — HVAC Load Schedule</p>', unsafe_allow_html=True)

col_upload_hvac, col_dl_hvac = st.columns([2, 1])
with col_upload_hvac:
    uploaded_hvac = st.file_uploader(
        "Upload HVAC CSV (optional — replaces table)",
        type="csv", key="hvac_upload", label_visibility="collapsed"
    )
    if uploaded_hvac:
        try:
            st.session_state["hvac_df"] = pd.read_csv(uploaded_hvac)
            st.success("HVAC table replaced from CSV")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

with col_dl_hvac:
    st.download_button(
        "↓ Download template",
        data=to_csv_bytes(DEFAULT_HVAC),
        file_name="hvac_params_template.csv",
        mime="text/csv",
    )

hvac_edited = st.data_editor(
    st.session_state["hvac_df"],
    use_container_width=True,
    num_rows="fixed",
    key="hvac_editor",
    column_config={
        "parameter": st.column_config.TextColumn("Parameter", disabled=True),
        "value":     st.column_config.NumberColumn("Value", format="%.2f", step=0.1),
    }
)
st.session_state["hvac_df"] = hvac_edited


# ── Equipment ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">02 — Equipment Load Schedule</p>', unsafe_allow_html=True)

col_upload_eq, col_dl_eq = st.columns([2, 1])
with col_upload_eq:
    uploaded_eq = st.file_uploader(
        "Upload equipment CSV (optional)",
        type="csv", key="equip_upload", label_visibility="collapsed"
    )
    if uploaded_eq:
        try:
            st.session_state["equip_df"] = pd.read_csv(uploaded_eq)
            st.success("Equipment table replaced from CSV")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

with col_dl_eq:
    st.download_button(
        "↓ Download template",
        data=to_csv_bytes(DEFAULT_EQUIPMENT),
        file_name="equipment_schedule_template.csv",
        mime="text/csv",
    )

equip_edited = st.data_editor(
    st.session_state["equip_df"],
    use_container_width=True,
    num_rows="dynamic",
    key="equip_editor",
    column_config={
        "load_type":         st.column_config.TextColumn("Load type"),
        "weekday_day_kW":    st.column_config.NumberColumn("Weekday day [kW]",   format="%.1f"),
        "weekday_night_kW":  st.column_config.NumberColumn("Weekday night [kW]", format="%.1f"),
        "weekend_day_kW":    st.column_config.NumberColumn("Weekend day [kW]",   format="%.1f"),
        "weekend_night_kW":  st.column_config.NumberColumn("Weekend night [kW]", format="%.1f"),
        "holiday_kW":        st.column_config.NumberColumn("Holiday [kW]",       format="%.1f"),
    }
)
st.session_state["equip_df"] = equip_edited


# ── Shore power ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">03 — Shore Power Vessel Schedule</p>', unsafe_allow_html=True)

col_upload_sp, col_dl_sp = st.columns([2, 1])
with col_upload_sp:
    uploaded_sp = st.file_uploader(
        "Upload shore power CSV (optional)",
        type="csv", key="shore_upload", label_visibility="collapsed"
    )
    if uploaded_sp:
        try:
            st.session_state["shore_df"] = pd.read_csv(uploaded_sp)
            st.success("Shore power table replaced from CSV")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

with col_dl_sp:
    st.download_button(
        "↓ Download template",
        data=to_csv_bytes(DEFAULT_SHORE),
        file_name="shore_power_template.csv",
        mime="text/csv",
    )

shore_edited = st.data_editor(
    st.session_state["shore_df"],
    use_container_width=True,
    num_rows="dynamic",
    key="shore_editor",
    column_config={
        "vessel_name": st.column_config.TextColumn("Vessel name"),
        "start":       st.column_config.TextColumn("Start (YYYY-MM-DD HH:MM)"),
        "end":         st.column_config.TextColumn("End   (YYYY-MM-DD HH:MM)"),
        "power_kW":    st.column_config.NumberColumn("Power [kW]", format="%.0f"),
    }
)
st.session_state["shore_df"] = shore_edited


# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
col_run, _ = st.columns([1, 4])
with col_run:
    run_clicked = st.button("▶  RUN MODEL", use_container_width=True)

if run_clicked:
    with st.spinner("Computing hourly demand model..."):
        try:
            idx = make_time_index(year)
            hol_set = swedish_holidays(year)

            # Parse HVAC params from editable table
            hvac_params = {}
            for _, row in st.session_state["hvac_df"].iterrows():
                hvac_params[row["parameter"]] = float(row["value"])

            # Decide which TMY to pass
            tmy_to_use = None
            if use_tmy and st.session_state["tmy_df"] is not None:
                tmy_to_use = st.session_state["tmy_df"]

            df_result, kpis = run_model(
                idx=idx,
                annual_base_kwh=annual_base_kwh,
                hvac_params=hvac_params,
                equip_df=st.session_state["equip_df"],
                shore_df=st.session_state["shore_df"],
                hol_set=hol_set,
                t_mean=t_mean,
                t_amp_annual=t_amp_annual,
                t_amp_daily=t_amp_daily,
                use_demandlib=use_demandlib,
                tmy_df=tmy_to_use,
            )
            st.session_state["results"] = (df_result, kpis)
            temp_label = kpis.get("temp_source", "synthetic")
            st.success(f"Model run complete. Temperature source: **{temp_label}**")
        except Exception as e:
            st.error(f"Model error: {e}")
            import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["results"] is not None:
    df_result, kpis = st.session_state["results"]

    st.markdown('<p class="section-header">04 — Key Performance Indicators</p>',
                unsafe_allow_html=True)

    temp_src = kpis.get("temp_source", "synthetic")
    if "TMY" in temp_src or "file" in temp_src.lower():
        badge_color = "#1a3a1a"; badge_text_color = "#7ddd7d"
    else:
        badge_color = "#1a1a3a"; badge_text_color = "#9999ff"
    st.markdown(
        f'<span style="background:{badge_color};color:{badge_text_color};'
        f'padding:3px 10px;border-radius:3px;font-size:0.72rem;font-family:monospace;">'
        f'Temperature source: {temp_src}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Annual demand",
                  f"{kpis['annual_total_MWh']:,.1f} MWh",
                  delta=f"Load factor {kpis['load_factor_pct']:.1f}%")
    with c2:
        st.metric("Peak load",
                  f"{kpis['peak_load_kW']:,.0f} kW",
                  delta=f"Avg {kpis['avg_load_kW']:,.0f} kW")
    with c3:
        st.metric("HVAC demand",
                  f"{kpis['hvac_MWh']:,.1f} MWh",
                  delta=f"{kpis['hvac_MWh']/kpis['annual_total_MWh']*100:.1f}% of total")
    with c4:
        st.metric("Shore power",
                  f"{kpis['shore_MWh']:,.1f} MWh",
                  delta=f"{kpis['shore_MWh']/kpis['annual_total_MWh']*100:.1f}% of total")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Base load demand",
                  f"{kpis['base_MWh']:,.1f} MWh",
                  delta=f"{kpis['base_MWh']/kpis['annual_total_MWh']*100:.1f}% of total")
    with c6:
        st.metric("Equipment demand",
                  f"{kpis['equipment_MWh']:,.1f} MWh",
                  delta=f"{kpis['equipment_MWh']/kpis['annual_total_MWh']*100:.1f}% of total")
    with c7:
        avg_day = df_result.groupby(df_result["timestamp"].dt.date)["total_load_kW"].sum().mean()
        st.metric("Avg daily energy",
                  f"{avg_day/1000:.2f} MWh/day",
                  delta=f"{avg_day:.0f} kWh/day")
    with c8:
        max_month = df_result.groupby(df_result["timestamp"].dt.month)["total_load_kW"].sum().idxmax()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        peak_month_mwh = df_result.groupby(df_result["timestamp"].dt.month)["total_load_kW"].sum().max()/1000
        st.metric("Peak demand month",
                  f"{month_names[max_month]}",
                  delta=f"{peak_month_mwh:,.1f} MWh that month")

    # ── CHARTS ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">05 — Load Profile Charts</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Annual profile",
        "⏰ Daily average",
        "📅 Weekday vs Weekend",
        "📊 Monthly breakdown",
        "🌡 Weather (TMY)",
    ])

    with tab1:
        fig1 = fig_annual_load(df_result)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with tab2:
        fig2 = fig_daily_profile(df_result)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with tab3:
        fig3 = fig_weekly_profile(df_result)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    with tab4:
        fig4 = fig_monthly_contribution(df_result)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    with tab5:
        if st.session_state["tmy_df"] is not None:
            fig5 = fig_tmy_overview(st.session_state["tmy_df"])
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)
        else:
            st.markdown("""
            <div class="warn-box">
            No TMY file loaded. Upload a weather file in <strong>section 00</strong> above
            to see real temperature and irradiance charts here.
            </div>
            """, unsafe_allow_html=True)
            fig5 = fig_tmy_overview(pd.DataFrame({
                "temp_air": synthetic_temperature(
                    make_time_index(year), t_mean, t_amp_annual, t_amp_daily
                )
            }))
            st.caption("Showing synthetic temperature profile (no TMY loaded)")
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)

    # ── ENERGY SUMMARY TABLE ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">06 — Annual Energy Summary</p>',
                unsafe_allow_html=True)

    summary_df = pd.DataFrame([
        {"Component": "Industrial base load", "Annual energy [MWh]": round(kpis["base_MWh"], 1),
         "Share [%]": round(kpis["base_MWh"]/kpis["annual_total_MWh"]*100, 1)},
        {"Component": "HVAC (heating + cooling)", "Annual energy [MWh]": round(kpis["hvac_MWh"], 1),
         "Share [%]": round(kpis["hvac_MWh"]/kpis["annual_total_MWh"]*100, 1)},
        {"Component": "Port equipment", "Annual energy [MWh]": round(kpis["equipment_MWh"], 1),
         "Share [%]": round(kpis["equipment_MWh"]/kpis["annual_total_MWh"]*100, 1)},
        {"Component": "Shore power", "Annual energy [MWh]": round(kpis["shore_MWh"], 1),
         "Share [%]": round(kpis["shore_MWh"]/kpis["annual_total_MWh"]*100, 1)},
        {"Component": "TOTAL", "Annual energy [MWh]": round(kpis["annual_total_MWh"], 1),
         "Share [%]": 100.0},
    ])

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── HOURLY DATA PREVIEW ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">07 — Hourly Data (preview)</p>',
                unsafe_allow_html=True)

    with st.expander("Show hourly data table (first 168 hours — Week 1)", expanded=False):
        st.dataframe(
            df_result.head(168).style.format({
                "temperature_degC": "{:.1f}",
                "base_load_kW":     "{:.1f}",
                "hvac_heating_kW":  "{:.1f}",
                "hvac_cooling_kW":  "{:.1f}",
                "equipment_kW":     "{:.1f}",
                "shore_power_kW":   "{:.1f}",
                "total_load_kW":    "{:.1f}",
                "total_load_kWh":   "{:.1f}",
            }),
            use_container_width=True,
            height=350,
        )

    # ── MONTHLY SUMMARY TABLE ─────────────────────────────────────────────────
    with st.expander("Show monthly summary table", expanded=False):
        df_result["month"] = df_result["timestamp"].dt.month
        monthly = df_result.groupby("month").agg(
            base_MWh    =("base_load_kW",    lambda x: x.sum()/1000),
            hvac_MWh    =("hvac_heating_kW", lambda x: (x + df_result.loc[x.index,"hvac_cooling_kW"]).sum()/1000),
            equipment_MWh=("equipment_kW",   lambda x: x.sum()/1000),
            shore_MWh   =("shore_power_kW",  lambda x: x.sum()/1000),
            total_MWh   =("total_load_kW",   lambda x: x.sum()/1000),
            peak_kW     =("total_load_kW",   "max"),
        ).round(1)
        monthly.index = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        st.dataframe(monthly, use_container_width=True)
        df_result.drop(columns=["month"], inplace=True)

    # ── DOWNLOADS ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">08 — Export Results</p>',
                unsafe_allow_html=True)

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.download_button(
            "↓ Download hourly data (CSV)",
            data=to_csv_bytes(df_result),
            file_name=f"oskarshamn_hourly_demand_{year}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_d2:
        st.download_button(
            "↓ Download summary (CSV)",
            data=to_csv_bytes(summary_df),
            file_name=f"oskarshamn_summary_{year}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_d3:
        # Monthly export
        df_result["month"] = df_result["timestamp"].dt.month
        monthly_exp = df_result.groupby("month").agg(
            base_MWh    =("base_load_kW",    lambda x: round(x.sum()/1000, 2)),
            hvac_MWh    =("hvac_heating_kW", lambda x: round((x + df_result.loc[x.index,"hvac_cooling_kW"]).sum()/1000, 2)),
            equipment_MWh=("equipment_kW",   lambda x: round(x.sum()/1000, 2)),
            shore_MWh   =("shore_power_kW",  lambda x: round(x.sum()/1000, 2)),
            total_MWh   =("total_load_kW",   lambda x: round(x.sum()/1000, 2)),
            peak_kW     =("total_load_kW",   "max"),
        ).reset_index()
        df_result.drop(columns=["month"], inplace=True)
        st.download_button(
            "↓ Download monthly summary (CSV)",
            data=to_csv_bytes(monthly_exp),
            file_name=f"oskarshamn_monthly_{year}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="warn-box">
    No results yet — configure parameters in the sidebar and tables above, then click <strong>▶ RUN MODEL</strong>.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#bbb;font-size:0.75rem;font-family:monospace;
            border-top:1px solid #e8e8e8;padding-top:1rem;">
    Oskarshamn Port Demand Model · Master Thesis 2025 · 
    Built with Streamlit · demandlib · pandas · numpy · matplotlib
</div>
""", unsafe_allow_html=True)
