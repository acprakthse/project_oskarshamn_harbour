import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays

from demandlib.particular_profiles import IndustrialLoadProfile

# ============================================================
# 1. USER INPUTS
# ============================================================
year = 2025
freq = "60min"   # hourly resolution

# -----------------------------
# Annual base demand [kWh/year]
# This is the "background" port demand:
# offices, lighting, base services, etc.
# -----------------------------
annual_base_demand_kwh = 3_500_000

# -----------------------------
# HVAC parameters
# Simple temperature-based HVAC model
# -----------------------------
hvac = {
    "enabled": True,
    "heating_setpoint_degC": 17,
    "cooling_setpoint_degC": 22,
    "heating_sensitivity_kW_per_degC": 18,   # stronger heating in winter
    "cooling_sensitivity_kW_per_degC": 8,    # weaker cooling in summer
    "hvac_day_multiplier": 1.15,             # more HVAC activity daytime
    "hvac_night_multiplier": 0.85
}

# -----------------------------
# Port equipment parameters
# Add regular equipment schedule
# Example: cranes, forklifts, conveyors, pumps, handling systems
# -----------------------------
equipment = {
    "enabled": True,
    "weekday_day_kW": 700,
    "weekday_night_kW": 250,
    "weekend_day_kW": 350,
    "weekend_night_kW": 150,
    "holiday_kW": 100,
    "active_day_start": 6,
    "active_day_end": 22
}

# -----------------------------
# Shore power parameters
# Ships connect to port and consume power while berthed
# Define arrivals manually
# start and end timestamps must match the selected year
# -----------------------------
shore_power = {
    "enabled": True,
    "vessels": [
        {
            "name": "Vessel_1",
            "start": f"{year}-01-10 08:00:00",
            "end":   f"{year}-01-11 06:00:00",
            "power_kW": 1800
        },
        {
            "name": "Vessel_2",
            "start": f"{year}-02-05 14:00:00",
            "end":   f"{year}-02-06 10:00:00",
            "power_kW": 2200
        },
        {
            "name": "Vessel_3",
            "start": f"{year}-03-18 07:00:00",
            "end":   f"{year}-03-19 20:00:00",
            "power_kW": 1500
        },
        {
            "name": "Vessel_4",
            "start": f"{year}-05-03 09:00:00",
            "end":   f"{year}-05-04 08:00:00",
            "power_kW": 2500
        },
        {
            "name": "Vessel_5",
            "start": f"{year}-08-21 12:00:00",
            "end":   f"{year}-08-22 18:00:00",
            "power_kW": 2000
        }
    ]
}

# ============================================================
# 2. TIME INDEX AND HOLIDAYS
# ============================================================
dt_index = pd.date_range(
    start=f"{year}-01-01 00:00:00",
    end=f"{year+1}-01-01 00:00:00",
    freq=freq,
    inclusive="left"
)

se_holidays = holidays.Sweden(years=year)
holiday_dict = {day: name for day, name in se_holidays.items()}

# ============================================================
# 3. BASE PORT LOAD USING DEMANDLIB
# ============================================================
ilp = IndustrialLoadProfile(
    dt_index=dt_index,
    holidays=holiday_dict
)

# Synthetic industrial profile for background/base demand
base_profile_kwh_per_timestep = ilp.simple_profile(
    annual_demand=annual_base_demand_kwh,
    am=dt.time(6, 0, 0),
    pm=dt.time(22, 0, 0),
    profile_factors={
        "week": {
            "day": 1.00,
            "night": 0.65
        },
        "weekend": {
            "day": 0.75,
            "night": 0.55
        },
        "holiday": {
            "day": 0.45,
            "night": 0.45
        },
    }
)

df = pd.DataFrame(index=dt_index)
df["base_kWh"] = base_profile_kwh_per_timestep

# Since freq is hourly, kWh per timestep = average kW over that hour
df["base_kW"] = df["base_kWh"]

# ============================================================
# 4. SYNTHETIC OUTDOOR TEMPERATURE FOR HVAC
# ============================================================
# Simple synthetic yearly temperature profile for Oskarshamn-like climate
# You can later replace this with real SMHI hourly temperature data

day_of_year = df.index.dayofyear.values
hour_of_day = df.index.hour.values

# Annual sinusoidal temperature + small daily cycle
temp_outdoor = (
    8
    + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    + 2 * np.sin(2 * np.pi * (hour_of_day - 14) / 24.0)
)

df["temp_degC"] = temp_outdoor

# ============================================================
# 5. HVAC LOAD MODEL
# ============================================================
df["hvac_kW"] = 0.0

if hvac["enabled"]:
    heating_load = np.maximum(
        hvac["heating_setpoint_degC"] - df["temp_degC"], 0
    ) * hvac["heating_sensitivity_kW_per_degC"]

    cooling_load = np.maximum(
        df["temp_degC"] - hvac["cooling_setpoint_degC"], 0
    ) * hvac["cooling_sensitivity_kW_per_degC"]

    hvac_raw = heating_load + cooling_load

    is_daytime = (df.index.hour >= 6) & (df.index.hour < 22)
    hvac_multiplier = np.where(
        is_daytime,
        hvac["hvac_day_multiplier"],
        hvac["hvac_night_multiplier"]
    )

    df["hvac_kW"] = hvac_raw * hvac_multiplier

# ============================================================
# 6. EQUIPMENT LOAD MODEL
# ============================================================
df["equipment_kW"] = 0.0

if equipment["enabled"]:
    is_weekend = df.index.weekday >= 5
    is_holiday = pd.Series(df.index.date, index=df.index).isin(holiday_dict.keys()).values
    is_day = (
        (df.index.hour >= equipment["active_day_start"]) &
        (df.index.hour < equipment["active_day_end"])
    )

    eq_load = np.zeros(len(df))

    # Holiday
    eq_load[is_holiday] = equipment["holiday_kW"]

    # Weekdays (non-holiday)
    weekday_mask = (~is_weekend) & (~is_holiday)
    eq_load[weekday_mask & is_day] = equipment["weekday_day_kW"]
    eq_load[weekday_mask & (~is_day)] = equipment["weekday_night_kW"]

    # Weekends (non-holiday)
    weekend_mask = is_weekend & (~is_holiday)
    eq_load[weekend_mask & is_day] = equipment["weekend_day_kW"]
    eq_load[weekend_mask & (~is_day)] = equipment["weekend_night_kW"]

    df["equipment_kW"] = eq_load

# ============================================================
# 7. SHORE POWER LOAD MODEL
# ============================================================
df["shore_power_kW"] = 0.0

if shore_power["enabled"]:
    for vessel in shore_power["vessels"]:
        start = pd.Timestamp(vessel["start"])
        end = pd.Timestamp(vessel["end"])
        power = vessel["power_kW"]

        mask = (df.index >= start) & (df.index < end)
        df.loc[mask, "shore_power_kW"] += power

# ============================================================
# 8. TOTAL LOAD
# ============================================================
df["total_load_kW"] = (
    df["base_kW"] +
    df["hvac_kW"] +
    df["equipment_kW"] +
    df["shore_power_kW"]
)

# For hourly frequency, kWh per hour equals kW average over hour
df["total_load_kWh"] = df["total_load_kW"]

# ============================================================
# 9. RESULTS SUMMARY
# ============================================================
annual_total_kwh = df["total_load_kWh"].sum()
peak_load_kw = df["total_load_kW"].max()
avg_load_kw = df["total_load_kW"].mean()

print("========== Oskarshamn Port Demand Model ==========")
print(f"Year: {year}")
print(f"Base annual demand input [kWh/year]: {annual_base_demand_kwh:,.0f}")
print(f"Total generated annual demand [kWh/year]: {annual_total_kwh:,.0f}")
print(f"Peak demand [kW]: {peak_load_kw:,.2f}")
print(f"Average demand [kW]: {avg_load_kw:,.2f}")
print("==================================================")

component_summary = pd.DataFrame({
    "Annual Energy [kWh]": [
        df["base_kWh"].sum(),
        df["hvac_kW"].sum(),
        df["equipment_kW"].sum(),
        df["shore_power_kW"].sum(),
        df["total_load_kWh"].sum()
    ]
}, index=[
    "Base load",
    "HVAC",
    "Equipment",
    "Shore power",
    "Total"
])

print("\nAnnual component summary:")
print(component_summary.round(2))

# ============================================================
# 10. EXPORT CSV
# ============================================================
df.to_csv("oskarshamn_port_hourly_demand_model.csv")
component_summary.to_csv("oskarshamn_port_component_summary.csv")

# ============================================================
# 11. PLOTS
# ============================================================

# --- Annual hourly total load
plt.figure(figsize=(14, 5))
plt.plot(df.index, df["total_load_kW"], label="Total load")
plt.title("Annual Hourly Electricity Demand - Oskarshamn Port")
plt.xlabel("Time")
plt.ylabel("Load [kW]")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Average daily profile by hour
avg_daily = df["total_load_kW"].groupby(df.index.hour).mean()

plt.figure(figsize=(8, 4))
plt.plot(avg_daily.index, avg_daily.values, marker="o")
plt.title("Average Daily Load Shape - Oskarshamn Port")
plt.xlabel("Hour of Day")
plt.ylabel("Average Load [kW]")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Weekly sample
sample_week = df.loc[f"{year}-01-06":f"{year}-01-12"]

plt.figure(figsize=(14, 5))
plt.plot(sample_week.index, sample_week["base_kW"], label="Base")
plt.plot(sample_week.index, sample_week["hvac_kW"], label="HVAC")
plt.plot(sample_week.index, sample_week["equipment_kW"], label="Equipment")
plt.plot(sample_week.index, sample_week["shore_power_kW"], label="Shore power")
plt.plot(sample_week.index, sample_week["total_load_kW"], label="Total", linewidth=2)
plt.title("Sample Week Load Breakdown - Oskarshamn Port")
plt.xlabel("Time")
plt.ylabel("Load [kW]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Monthly energy contribution
monthly = df.resample("M")[["base_kW", "hvac_kW", "equipment_kW", "shore_power_kW"]].sum()

plt.figure(figsize=(12, 5))
plt.plot(monthly.index, monthly["base_kW"], label="Base")
plt.plot(monthly.index, monthly["hvac_kW"], label="HVAC")
plt.plot(monthly.index, monthly["equipment_kW"], label="Equipment")
plt.plot(monthly.index, monthly["shore_power_kW"], label="Shore power")
plt.title("Monthly Energy Contribution by Component - Oskarshamn Port")
plt.xlabel("Month")
plt.ylabel("Energy [kWh/month]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()