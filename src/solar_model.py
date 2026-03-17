import json
import csv
from pathlib import Path
from typing import Dict, Any, Union, Optional

import PySAM.Pvsamv1 as pv


def run_pvsam_and_export_ac_outputs(
    json_path: Union[str, Path],
    csv_path: Union[str, Path] = "output/pySAM_PV/pvsam_ac_outputs.csv",
    print_summary: bool = True,
) -> Dict[str, Any]:
    json_path = Path(json_path)
    csv_path = Path(csv_path)

    # --- build model ---
    pv_model = pv.new()

    with json_path.open("r", encoding="utf-8") as f:
        pv_inputs = json.load(f)

    for k, v in pv_inputs.items():
        if k != "number_inputs":
            pv_model.value(k, v)

    # --- run ---
    pv_model.execute()

    # --- collect outputs (time-series) ---
    seq_outputs = {
        "ac_gross_kw": pv_model.Outputs.ac_gross,
        "ac_lifetime_loss_kw": pv_model.Outputs.ac_lifetime_loss,
        "ac_perf_adj_loss_kw": pv_model.Outputs.ac_perf_adj_loss,
        "ac_transmission_loss_kw": pv_model.Outputs.ac_transmission_loss,
        "ac_wiring_loss_kw": pv_model.Outputs.ac_wiring_loss,
    }

    n = len(seq_outputs["ac_gross_kw"])

    for name, seq in seq_outputs.items():
        if len(seq) != n:
            raise ValueError(f"Output length mismatch: {name} has {len(seq)} != {n}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep"] + list(seq_outputs.keys()))
        for i in range(n):
            writer.writerow([i] + [seq_outputs[key][i] for key in seq_outputs.keys()])

    if print_summary:
        print(f"Wrote {n} rows to: {csv_path}")
        try:
            print(
                f"Annual AC gross (kWh): {pv_model.Outputs.annual_ac_gross:,.0f}"
            )
        except Exception:
            pass

    return {
        "csv_path": str(csv_path),
        "seq_outputs": seq_outputs,
        "n_timesteps": n,
    }
