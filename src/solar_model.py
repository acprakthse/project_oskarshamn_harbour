import json
import csv
from pathlib import Path
from typing import Dict, Any, Union
import PySAM.Pvsamv1 as pv

BASE_DIR = Path(__file__).resolve().parents[1]

def run_pvsam_and_export_ac_outputs(
    json_path: Union[str, Path] = "data/Json_pySAM/untitled_pvsamv1.json",
    csv_path: Union[str, Path] = "outputs/pvsam_ac_outputs.csv",
    print_summary: bool = True,
) -> Dict[str, Any]:
    json_path = (BASE_DIR / json_path).resolve()
    csv_path = (BASE_DIR / csv_path).resolve()
    print(f"[INFO] Reading JSON from: {json_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    pv_model = pv.new()

    with json_path.open("r", encoding="utf-8") as f:
        pv_inputs = json.load(f)

    for k, v in pv_inputs.items():
        if k == "number_inputs":
            continue
        try:
            pv_model.value(k, v)
        except Exception as e:
            print(f"[WARNING] Skipping input '{k}': {e}")

    # --- Run simulation ---
    pv_model.execute()

    # --- Candidate outputs ---
    candidate_outputs = {
        "ac_gross_kw": "ac_gross",
        "ac_lifetime_loss_kw": "ac_lifetime_loss",
        "ac_perf_adj_loss_kw": "ac_perf_adj_loss",
        "ac_transmission_loss_kw": "ac_transmission_loss",
        "ac_wiring_loss_kw": "ac_wiring_loss",
    }

    # --- Collect outputs safely ---
    seq_outputs = {}
    for col_name, attr_name in candidate_outputs.items():
        value = getattr(pv_model.Outputs, attr_name, None)
        if value is not None:
            seq_outputs[col_name] = value
        else:
            print(f"[WARNING] Output '{attr_name}' not found in PySAM")

    if not seq_outputs:
        raise ValueError("No AC outputs found from PySAM model")

    # --- Validate lengths ---
    first_key = next(iter(seq_outputs))
    n = len(seq_outputs[first_key])

    for name, seq in seq_outputs.items():
        if len(seq) != n:
            raise ValueError(f"Length mismatch: {name} has {len(seq)} != {n}")

    # --- Write CSV ---
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep"] + list(seq_outputs.keys()))

        for i in range(n):
            writer.writerow([i] + [seq_outputs[key][i] for key in seq_outputs])

    # --- Summary ---
    if print_summary:
        print(f"[INFO] Wrote {n} timesteps to: {csv_path}")

        annual_ac = getattr(pv_model.Outputs, "annual_ac_gross", None)
        if annual_ac is not None:
            print(f"[INFO] Annual AC gross (kWh): {annual_ac:,.0f}")

    return {
        "csv_path": str(csv_path),
        "seq_outputs": seq_outputs,
        "n_timesteps": n,
    }

# --- Optional: direct run (for testing) ---
if __name__ == "__main__":
    res = run_pvsam_and_export_ac_outputs()
    print(res["csv_path"], res["n_timesteps"])