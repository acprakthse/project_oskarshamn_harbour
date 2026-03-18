"""
visualization/monthly_bar.py
=============================
Monthly energy summary bar chart and demand charge bar chart.

Charts produced:
  1. monthly_summary.png  — PV / load / import / export by month [MWh]
  2. monthly_demand.png   — peak grid import per month [kW] with demand cost

Usage:
    from visualization.monthly_bar import plot_monthly_summary, plot_monthly_demand
    plot_monthly_summary(res, out_dir="results")
    plot_monthly_demand(monthly_demand_series, out_dir="results")
"""

from pathlib import Path
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.ticker  as mticker

from config.tariff import Tariff

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


# =============================================================================

def plot_monthly_summary(
    res:     pd.DataFrame,
    out_dir: str | Path = "results",
    show:    bool       = False,
) -> Path:
    """
    Grouped bar chart: PV, load, import, export by month [MWh].

    Parameters
    ----------
    res     : pd.DataFrame  — hourly dispatch
    out_dir : str | Path
    show    : bool

    Returns
    -------
    Path  — saved figure path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = res.copy()
    res["month"] = pd.to_datetime(res["datetime"]).dt.month

    monthly = (
        res.groupby("month")
        .agg(
            pv_MWh   = ("pv_kw",    lambda x: x.sum() / 1000),
            load_MWh = ("load_kw",  lambda x: x.sum() / 1000),
            imp_MWh  = ("P_imp_kw", lambda x: x.sum() / 1000),
            exp_MWh  = ("P_exp_kw", lambda x: x.sum() / 1000),
            chg_MWh  = ("P_c_kw",   lambda x: x.sum() / 1000),
            dis_MWh  = ("P_d_kw",   lambda x: x.sum() / 1000),
        )
        .reindex(range(1, 13), fill_value=0)
    )

    x = np.arange(12)
    w = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(x - 1.5*w, monthly["pv_MWh"],   w, label="PV gen",      color="#EF9F27", alpha=0.88)
    ax.bar(x - 0.5*w, monthly["load_MWh"], w, label="Load",         color="#378ADD", alpha=0.88)
    ax.bar(x + 0.5*w, monthly["imp_MWh"],  w, label="Grid import",  color="#E24B4A", alpha=0.88)
    ax.bar(x + 1.5*w, monthly["exp_MWh"],  w, label="Grid export",  color="#1D9E75", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS, fontsize=10)
    ax.set_ylabel("Energy [MWh]", fontsize=10)
    ax.set_title("Monthly energy summary — PV + BESS optimised", fontsize=12, pad=8)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    path = out_dir / "monthly_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[viz] Monthly summary    → {path}")
    return path


def plot_monthly_demand(
    monthly_demand: pd.Series,
    out_dir:        str | Path = "results",
    show:           bool       = False,
) -> Path:
    """
    Bar chart of peak monthly grid import [kW] with demand cost annotation.

    Parameters
    ----------
    monthly_demand : pd.Series  index=month (1–12), values=peak import [kW]
    out_dir        : str | Path
    show           : bool

    Returns
    -------
    Path  — saved figure path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demand = monthly_demand.reindex(range(1, 13), fill_value=0)
    costs  = demand * Tariff.DEMAND_CHG

    x = np.arange(12)
    fig, ax1 = plt.subplots(figsize=(11, 4))

    bars = ax1.bar(x, demand.values, color="#534AB7", alpha=0.82,
                   label="Peak import [kW]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MONTH_LABELS, fontsize=10)
    ax1.set_ylabel("Peak grid import [kW]", fontsize=10, color="#534AB7")
    ax1.tick_params(axis="y", labelcolor="#534AB7")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.grid(True, axis="y", alpha=0.2, linestyle="--")

    # Annotate bars with demand cost
    for i, (bar, cost) in enumerate(zip(bars, costs.values)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + demand.max() * 0.01,
            f"{cost:,.0f}",
            ha="center", va="bottom", fontsize=7.5, color="#3C3489",
        )

    ax2 = ax1.twinx()
    ax2.plot(x, costs.values, color="#D85A30", marker="o",
             ms=5, lw=1.5, label="Demand cost")
    ax2.set_ylabel("Demand charge [currency]", fontsize=10, color="#D85A30")
    ax2.tick_params(axis="y", labelcolor="#D85A30")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax1.set_title("Monthly peak grid import & demand charge", fontsize=12, pad=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    plt.tight_layout()
    path = out_dir / "monthly_demand.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[viz] Monthly demand     → {path}")
    return path
