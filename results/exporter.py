"""
results/exporter.py
====================
Write optimisation results to disk.

Outputs
-------
  results/bess_dispatch_8760.csv  — full hourly dispatch table
  results/kpis.csv                — flat KPI summary row
  results/monthly_demand.csv      — monthly peak demand [kW]

Usage:
    from results.exporter import export_all
    export_all(res, kpis, monthly_demand, out_dir="results")
"""

import os
import pandas as pd
from pathlib import Path


def export_dispatch(res: pd.DataFrame, out_dir: Path) -> Path:
    """Write the 8760-row dispatch DataFrame to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bess_dispatch_8760.csv"
    res.to_csv(path, index=False, float_format="%.4f")
    print(f"[exporter] Dispatch CSV → {path}")
    return path


def export_kpis(kpis: dict, out_dir: Path) -> Path:
    """Write the KPI summary to a single-row CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "kpis.csv"
    pd.DataFrame([kpis]).to_csv(path, index=False, float_format="%.4f")
    print(f"[exporter] KPI CSV      → {path}")
    return path


def export_monthly_demand(monthly: pd.Series, out_dir: Path) -> Path:
    """Write monthly peak demand to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "monthly_demand.csv"
    monthly.reset_index().rename(
        columns={"index": "month"}
    ).to_csv(path, index=False, float_format="%.2f")
    print(f"[exporter] Monthly CSV  → {path}")
    return path


def export_all(
    res:            pd.DataFrame,
    kpis:           dict,
    monthly_demand: pd.Series,
    out_dir:        str | Path = "results",
) -> None:
    """
    Export all result files.

    Parameters
    ----------
    res            : pd.DataFrame  — hourly dispatch (from extractor)
    kpis           : dict          — KPI summary (from kpis.py)
    monthly_demand : pd.Series     — monthly peak demand (from extractor)
    out_dir        : str | Path    — output directory
    """
    out_dir = Path(out_dir)
    export_dispatch(res, out_dir)
    export_kpis(kpis, out_dir)
    export_monthly_demand(monthly_demand, out_dir)
    print(f"[exporter] All results saved to: {out_dir.resolve()}/")
