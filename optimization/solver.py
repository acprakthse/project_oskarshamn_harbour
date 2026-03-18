"""
optimization/solver.py
=======================
Solver wrapper for the Pyomo MILP model.

Handles:
  - Gurobi (primary)
  - CBC    (open-source fallback)
  - GLPK   (secondary fallback)

Usage:
    from optimization.solver import solve_model
    result = solve_model(model)
"""

import sys
from pyomo.environ import SolverFactory, value
from pyomo.opt     import SolverStatus, TerminationCondition

from config.settings import SOLVER_NAME, SOLVER_FALLBACK, SOLVER_OPTIONS


# =============================================================================

def _get_solver():
    """
    Return an available solver instance.
    Tries SOLVER_NAME first, then SOLVER_FALLBACK, then GLPK.
    """
    for name in [SOLVER_NAME, SOLVER_FALLBACK, "glpk"]:
        s = SolverFactory(name)
        if s.available():
            if name != SOLVER_NAME:
                print(f"[solver] '{SOLVER_NAME}' not available — using '{name}'")
            return s, name
    raise RuntimeError(
        "No MILP solver found.\n"
        "Install options:\n"
        "  Gurobi: https://www.gurobi.com/downloads/\n"
        "  CBC:    conda install -c conda-forge coincbc\n"
        "  GLPK:   conda install -c conda-forge glpk"
    )


def solve_model(
    model,
    mipgap:    float = None,
    timelimit: int   = None,
    verbose:   bool  = True,
) -> object:
    """
    Solve the MILP model and return the solver result.

    Parameters
    ----------
    model     : pyomo.ConcreteModel  — built by model_builder
    mipgap    : float | None         — override MIPGap (0–1)
    timelimit : int   | None         — override TimeLimit [seconds]
    verbose   : bool                 — show solver log

    Returns
    -------
    result : SolverResults  (pyomo object)

    Raises
    ------
    RuntimeError  if solve status is infeasible or solver not found
    """
    solver, solver_name = _get_solver()

    # Build options dict for the chosen solver
    opts = dict(SOLVER_OPTIONS.get(solver_name, {}))
    if mipgap    is not None:
        _mipgap_key = {"gurobi": "MIPGap", "cbc": "ratioGap", "glpk": "mipgap"}
        opts[_mipgap_key.get(solver_name, "MIPGap")] = mipgap
    if timelimit is not None:
        _tl_key = {"gurobi": "TimeLimit", "cbc": "sec", "glpk": "tmlim"}
        opts[_tl_key.get(solver_name, "TimeLimit")] = timelimit

    print(
        f"\n[solver] Solving with {solver_name.upper()} | "
        f"MIPGap={opts.get('MIPGap', opts.get('ratioGap', '?'))} | "
        f"TimeLimit={opts.get('TimeLimit', opts.get('sec', '?'))} s"
    )

    result = solver.solve(model, options=opts, tee=verbose)

    # ── Check termination ─────────────────────────────────────────────────────
    status    = result.solver.status
    condition = result.solver.termination_condition

    if condition == TerminationCondition.infeasible:
        raise RuntimeError(
            "[solver] Model is INFEASIBLE.\n"
            "Check battery parameters (SoC bounds, C-rate) and data quality."
        )
    if condition == TerminationCondition.unbounded:
        raise RuntimeError("[solver] Model is UNBOUNDED — check objective sign.")

    if condition in (TerminationCondition.optimal,
                     TerminationCondition.feasible):
        obj_val = value(model.obj)
        print(f"[solver] Solution found | Objective = {obj_val:,.2f}")
    elif condition == TerminationCondition.maxTimeLimit:
        print(
            "[solver] WARNING: time limit reached — solution may not be optimal.\n"
            "Increase SOLVER_OPTIONS TimeLimit or relax MIPGap."
        )
    else:
        print(f"[solver] WARNING: termination={condition}, status={status}")

    return result


def solution_status(result) -> str:
    """Return a human-readable solution status string."""
    cond = result.solver.termination_condition
    return str(cond)
