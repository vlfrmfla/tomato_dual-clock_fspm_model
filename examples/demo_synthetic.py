"""
Demo with a synthetic climate (no measured data needed).

Builds an hourly greenhouse climate from simple functions, runs the model for one autumn-winter crop with three
leaf-pruning rules and prints the state of the crop on the last day. The climate is invented, so the numbers only
show how the model is called and what it returns; they are not results of the paper.

Usage (from the repository root):  python examples/demo_synthetic.py
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fspm import Params, simulate, truss_table                      # noqa: E402
from fspm import management as M                                    # noqa: E402
from fspm.greenhouse import roof_glass_transmittance, solar_position  # noqa: E402

TRANSPLANT, START, END = "2023-09-18", "2023-10-05", "2024-01-11"


def synthetic_climate(first=TRANSPLANT, last=END, seed=1):
    """Hourly and daily climate tables with the columns the model reads."""
    t = pd.date_range(first, pd.Timestamp(last) + pd.Timedelta(hours=23), freq="h")
    days = len(t) // 24
    rng = np.random.default_rng(seed)
    elev, _, _ = solar_position(t + pd.Timedelta(minutes=30))
    sky = np.repeat(rng.uniform(0.25, 0.75, days), 24)                      # daily transmissivity of the atmosphere
    rg = 1361 * np.sin(np.radians(np.clip(elev, 0, None))) * sky            # outside global radiation, W m-2
    day = np.clip(np.sin((t.hour + 0.5 - 7) / 12 * np.pi), 0, None)
    temp = 17 + 5 * day + 0.004 * rg                                        # air temperature, degC
    rh = 80 - 0.025 * rg                                                    # relative humidity, %
    es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    hourly = pd.DataFrame({"T": temp, "RH": rh, "Rg_out": rg, "VPD": es * (1 - rh / 100),   # VPD in kPa
                           "CO2": 450.0,                                    # µmol mol-1
                           "curt1": 0.0, "curt2": 0.0, "curt3": 0.0}, index=t)   # screen closure, %
    # solar position, diffuse fraction and transmittance of the glass roof (Supplementary Section S5 of the paper)
    hourly[["sun_elev", "sun_az", "kd", "tau_glass"]] = roof_glass_transmittance(t, hourly["Rg_out"])
    daily = pd.DataFrame({"Tmean": hourly["T"].resample("D").mean(),
                          "Rg_out_MJ": hourly["Rg_out"].resample("D").sum() * 3600 / 1e6})
    return daily, hourly


def main():
    daily, hourly = synthetic_climate()
    p = Params(transplant=TRANSPLANT)
    cal = ROOT / "data_processed/params_m2.json"          # calibrated parameters of the Haman crop (Table 1)
    if cal.exists():
        pj = json.loads(cal.read_text())
        p = replace(p, **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    runs = {
        "no pruning": (M.NoPruning(), p),
        "R1: keep 18 leaves, weekly": (M.KeepN(18, trigger="2023-11-14"), p),
        "R3: truss-linked, batches of >= 6": (M.TrussLinked(tt_after=235.0), replace(p, prune_min_batch=6)),
    }
    rows = []
    for name, (rule, par) in runs.items():
        s, plant = simulate(daily, par, rule, START, END, hourly=hourly, leaf_dw0=10.0, stem_dw0=4.0)
        e = s.iloc[-1]
        tt = truss_table(plant)
        rows.append({"rule": name, "nodes": int(e.nodes), "leaves present": int(e.leaves_present), "LAI": e.LAI,
                     "pruning events": int((s.removed > 0).sum()), "leaves removed": int(s.removed.sum()),
                     "trusses set": int(tt["set"].notna().sum()), "trusses harvested": int(e.n_harvested),
                     "leaf DW (g)": e.leaf_dw, "fruit DW incl. harvested (g)": e.fruit_dw + e.harvested_dw,
                     "transpiration (L)": s.et_g.sum() / 1000, "night share (%)": 100 * s.et_night_g.sum() / s.et_g.sum()})
    out = pd.DataFrame(rows).set_index("rule").T
    out = out.apply(lambda col: col.map(lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"))
    pd.set_option("display.width", 200)
    print(f"Synthetic climate, {START} to {END}, per plant\n")
    print(out.to_string())
    print("\nDaily output columns:", ", ".join(s.columns))


if __name__ == "__main__":
    main()
