"""
visualization/heatmap.py
========================
Annual SoC heatmap — 365 days × 24 hours grid coloured by SoC [%].

Reveals seasonal patterns in battery utilisation at a glance:
  - Dark green = fully charged (solar surplus stored)
  - Red        = nearly depleted (heavy discharge or poor solar day)

Usage:
    from visualization.heatmap import plot_soc_heatmap
    plot_soc_heatmap(res, out_dir="results")
"""

from pathlib import Path
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.ticker  as mticker

from config.settings import N_HOURS


# =============================================================================

def plot_soc_heatmap(
    res:     pd.DataFrame,
    out_dir: str | Path = "results",
    show:    bool       = False,
) -> Path:
    """
    Annual SoC heatmap (365 × 24).

    Parameters
    ----------
    res     : pd.DataFrame  — hourly dispatch (must have soc_pct column)
    out_dir : str | Path
    show    : bool

    Returns
    -------
    Path  — saved figure path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reshape to (365, 24)  — use first 8760 rows
    soc = res["soc_pct"].values[:N_HOURS]
    if len(soc) < N_HOURS:
        soc = np.pad(soc, (0, N_HOURS - len(soc)), constant_values=np.nan)
    soc_2d = soc.reshape(365, 24)

    fig, ax = plt.subplots(figsize=(14, 4))

    im = ax.imshow(
        soc_2d.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.015)
    cbar.set_label("SoC [%]", fontsize=10)

    # X-axis: months
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_xlabel("Month", fontsize=10)

    # Y-axis: hours
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00","06:00","12:00","18:00","23:00"], fontsize=9)
    ax.set_ylabel("Hour of day", fontsize=10)

    ax.set_title("Annual LFP battery SoC heatmap  [%]", fontsize=12, pad=8)

    plt.tight_layout()
    path = out_dir / "soc_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[viz] SoC heatmap        → {path}")
    return path


def plot_import_heatmap(
    res:     pd.DataFrame,
    out_dir: str | Path = "results",
    show:    bool       = False,
) -> Path:
    """
    Annual grid import heatmap — same layout as SoC heatmap.
    Useful for identifying remaining peak demand hours.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imp = res["P_imp_kw"].values[:N_HOURS]
    if len(imp) < N_HOURS:
        imp = np.pad(imp, (0, N_HOURS - len(imp)), constant_values=0)
    imp_2d = imp.reshape(365, 24)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(
        imp_2d.T,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.015)
    cbar.set_label("Grid import [kW]", fontsize=10)

    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00","06:00","12:00","18:00","23:00"], fontsize=9)
    ax.set_ylabel("Hour of day", fontsize=10)
    ax.set_title("Annual grid import heatmap  [kW]", fontsize=12, pad=8)

    plt.tight_layout()
    path = out_dir / "import_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[viz] Import heatmap     → {path}")
    return path
