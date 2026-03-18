from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_yearly_production(
    csv_path="outputs/pvsam_ac_outputs.csv",
    power_col="ac_gross_kw",
    timestep_hours=1.0,
):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if power_col not in df.columns:
        raise ValueError(f"Column '{power_col}' not found. Available columns: {list(df.columns)}")

    # Energy per timestep in kWh
    df["energy_kwh"] = df[power_col] * timestep_hours

    # Annual total
    annual_energy_kwh = df["energy_kwh"].sum()

    print(f"Annual production = {annual_energy_kwh:,.2f} kWh")
    print(f"Annual production = {annual_energy_kwh/1000:,.2f} MWh")

    # Plot power profile over the year
    plt.figure(figsize=(12, 5))
    plt.plot(df["timestep"], df[power_col])
    plt.xlabel("Timestep")
    plt.ylabel("AC Power (kW)")
    plt.title("Yearly PV Production Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: monthly energy plot if hourly data
    if len(df) in [8760, 8784]:
        start_year = 2025 if len(df) == 8760 else 2024  # leap-year fallback for 8784
        dt_index = pd.date_range(start=f"{start_year}-01-01 00:00:00", periods=len(df), freq="h")
        df["datetime"] = dt_index
        monthly_energy = df.resample("M", on="datetime")["energy_kwh"].sum()

        plt.figure(figsize=(10, 5))
        plt.plot(monthly_energy.index.strftime("%b"), monthly_energy.values, marker="o")
        plt.xlabel("Month")
        plt.ylabel("Energy (kWh)")
        plt.title("Monthly PV Energy Production")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df, annual_energy_kwh