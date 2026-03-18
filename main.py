"""
main.py
=======
CLI entry point for the BESS MILP optimiser.

Orchestrates:
  1. Load & preprocess data
  2. Configure LFP battery
  3. Build MILP model
  4. Solve
  5. Extract results & KPIs
  6. Export CSVs
  7. Generate plots

Usage
-----
    # With real HelioScope + load CSVs:
    python main.py

    # With synthetic demo data (no CSVs needed):
    python main.py --demo

    # Custom paths & solver settings:
    python main.py \\
        --helioscope input_data/hs_export.csv \\
        --load       input_data/site_load.csv \\
        --outdir     results \\
        --mipgap     0.005 \\
        --timelimit  3600 \\
        --e-cap      1726 \\
        --p-max      863
"""

import argparse
import sys
from pathlib import Path

from battery.lfp_model   import LFPBattery
from config.settings     import HELIOSCOPE_CSV, LOAD_CSV, RESULTS_DIR
from data.loader         import load_data
from data.demo_data      import make_demo_data
from data.preprocessor   import preprocess
from optimization.model_builder import build_milp_model
from optimization.solver        import solve_model
from results.extractor          import extract_dispatch, extract_monthly_demand
from results.kpis               import compute_kpis, print_kpis
from results.exporter           import export_all
from visualization.dispatch_plot import plot_dispatch_week
from visualization.heatmap       import plot_soc_heatmap, plot_import_heatmap
from visualization.monthly_bar   import plot_monthly_summary, plot_monthly_demand


# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="BESS MILP Optimiser — Pyomo + Gurobi | LFP battery"
    )
    p.add_argument("--helioscope", type=Path, default=HELIOSCOPE_CSV,
                   help="Path to HelioScope 8760 CSV")
    p.add_argument("--load",       type=Path, default=LOAD_CSV,
                   help="Path to load profile CSV")
    p.add_argument("--outdir",     type=Path, default=RESULTS_DIR,
                   help="Output directory for results and plots")
    p.add_argument("--mipgap",     type=float, default=None,
                   help="Override MIPGap (e.g. 0.005 = 0.5%%)")
    p.add_argument("--timelimit",  type=int,   default=None,
                   help="Override solver time limit [seconds]")
    p.add_argument("--e-cap",      type=float, default=None,
                   help="Battery usable capacity [kWh]  (overrides default)")
    p.add_argument("--p-max",      type=float, default=None,
                   help="Battery max power [kW]          (overrides default)")
    p.add_argument("--demo",       action="store_true",
                   help="Run with synthetic data (no CSVs required)")
    p.add_argument("--no-plots",   action="store_true",
                   help="Skip plot generation")
    p.add_argument("--project-years", type=float, default=20.0,
                   help="Project life for lifetime savings calculation")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 58)
    print("  BESS MILP Optimiser  |  Pyomo + Gurobi  |  LFP")
    print("=" * 58)

    # ── 1. Battery configuration ──────────────────────────────────────────────
    bat = LFPBattery()
    if args.e_cap is not None:
        bat.E_cap_kwh = args.e_cap
    if args.p_max is not None:
        bat.P_max_kw  = args.p_max
    bat.summarise()

    # ── 2. Data ───────────────────────────────────────────────────────────────
    if args.demo or not (args.helioscope.exists() and args.load.exists()):
        if not args.demo:
            print(
                "\n[main] CSV files not found — running with synthetic demo data.\n"
                "       Pass --helioscope and --load for real data.\n"
            )
        df = make_demo_data(pv_peak_kw=bat.P_max_kw)
    else:
        df = load_data(args.helioscope, args.load)

    df = preprocess(df)

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("\nBuilding MILP model …")
    model, month_map = build_milp_model(df, bat)

    # ── 4. Solve ──────────────────────────────────────────────────────────────
    solve_model(
        model,
        mipgap    = args.mipgap,
        timelimit = args.timelimit,
    )

    # ── 5. Extract & KPIs ─────────────────────────────────────────────────────
    print("\nExtracting results …")
    res            = extract_dispatch(model, df, bat)
    monthly_demand = extract_monthly_demand(model)
    kpis           = compute_kpis(res, model, bat,
                                   project_years=args.project_years)
    print_kpis(kpis)

    # ── 6. Export ─────────────────────────────────────────────────────────────
    print("\nExporting results …")
    export_all(res, kpis, monthly_demand, out_dir=args.outdir)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nGenerating plots …")
        plot_dispatch_week(res, bat, out_dir=args.outdir)
        plot_soc_heatmap(res,        out_dir=args.outdir)
        plot_import_heatmap(res,     out_dir=args.outdir)
        plot_monthly_summary(res,    out_dir=args.outdir)
        plot_monthly_demand(monthly_demand, out_dir=args.outdir)

    print(f"\nDone.  All outputs in: {args.outdir.resolve()}/")
    print("\nNext step: run ga/ga_runner.py to optimise E_cap_kwh and P_max_kw.")


if __name__ == "__main__":
    main()
