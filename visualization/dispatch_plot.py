"""
visualization/dispatch_plot.py
===============================
Sample-week dispatch chart: PV / load / grid flows + battery power + SoC.

Usage:
    from visualization.dispatch_plot import plot_dispatch_week
    plot_dispatch_week(res, bat, out_dir="results")
"""

from pathlib import Path
import numpy  as np
import pandas as pd
import matplotlib.pyplot   as plt
import matplotlib.dates    as mdates
import matplotlib.ticker   as mticker

from battery.lfp_model import LFPBattery


# =============================================================================

def _pick_sample_week(res: pd.DataFrame) -> pd.DataFrame:
    """
    Return a 168-row (7-day) slice from July — the middle of the year,
    typically shows interesting charge/discharge behaviour.
    Falls back to the first 168 rows if July is not present.
    """
    jul = res[pd.to_datetime(res["datetime"]).dt.month == 7]
    if len(jul) >= 168:
        start = jul.index[0]
        return res.iloc[start: start + 168].copy()
    return res.iloc[:168].copy()


def plot_dispatch_week(
    res:     pd.DataFrame,
    bat:     LFPBattery,
    out_dir: str | Path = "results",
    show:    bool       = False,
) -> Path:
    """
    Three-panel dispatch chart for a sample week.

    Panel 1 — Power flows : PV generation, site load, grid import, grid export
    Panel 2 — Battery     : charge (positive) / discharge (negative) power
    Panel 3 — SoC         : state of charge [%] with min/max limits

    Parameters
    ----------
    res     : pd.DataFrame  — hourly dispatch from extractor
    bat     : LFPBattery
    out_dir : str | Path    — output directory
    show    : bool          — display interactively (True for notebooks)

    Returns
    -------
    Path  — saved figure path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    week  = _pick_sample_week(res)
    dates = pd.to_datetime(week["datetime"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # ── Panel 1: Power flows ──────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(dates, week["pv_kw"],
                    alpha=0.55, color="#EF9F27", label="PV generation")
    ax.fill_between(dates, week["load_kw"],
                    alpha=0.25, color="#378ADD", label="Site load")
    ax.plot(dates, week["P_imp_kw"],
            color="#E24B4A", lw=1.4, label="Grid import")
    ax.plot(dates, week["P_exp_kw"],
            color="#1D9E75", lw=1.0, ls="--", label="Grid export")
    ax.set_ylabel("Power [kW]", fontsize=10)
    ax.set_title("Sample week — BESS dispatch", fontsize=12, pad=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.18, linestyle="--")
    ax.set_ylim(bottom=0)

    # ── Panel 2: Battery charge / discharge ───────────────────────────────────
    ax = axes[1]
    ax.bar(dates, week["P_c_kw"],
           width=1/24, color="#534AB7", alpha=0.80, label="Charge")
    ax.bar(dates, -week["P_d_kw"],
           width=1/24, color="#D85A30", alpha=0.80, label="Discharge")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axhline( bat.P_charge_max,    color="#534AB7", lw=0.8, ls=":",
                alpha=0.6, label=f"Max charge {bat.P_charge_max:.0f} kW")
    ax.axhline(-bat.P_discharge_max, color="#D85A30", lw=0.8, ls=":",
                alpha=0.6, label=f"Max discharge {bat.P_discharge_max:.0f} kW")
    ax.set_ylabel("BESS Power [kW]", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.18, linestyle="--")

    # ── Panel 3: SoC ──────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(dates, week["soc_pct"],
            color="#0F6E56", lw=1.6, label="SoC")
    ax.fill_between(dates, week["soc_pct"],
                    alpha=0.15, color="#0F6E56")
    ax.axhline(bat.soc_min * 100, color="#E24B4A", ls=":", lw=1.2,
               label=f"SoC min {bat.soc_min*100:.0f}%")
    ax.axhline(bat.soc_max * 100, color="#EF9F27", ls=":", lw=1.2,
               label=f"SoC max {bat.soc_max*100:.0f}%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("State of Charge [%]", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.18, linestyle="--")

    # ── X-axis formatting ─────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))

    plt.tight_layout()
    path = out_dir / "dispatch_week.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[viz] Dispatch week chart → {path}")
    return path
