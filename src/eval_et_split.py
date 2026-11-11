"""
로드셀 증산 평가를 Rd25 선정에 쓴 기간과 홀드아웃 기간으로 분리 (리뷰 대응)

Rd25(0.65)는 τ_struct × Rd25 격자에서 생장 목적함수와 10–11월 증산 비율(sens_rd_grid.py, et_ratio_octnov)로 골랐다.
→ 10/5–11/30 = 파라미터 선정에 사용, 12/1–1/12 = 홀드아웃
출력: data_processed/et_split.csv, 표준출력
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import INIT, START, END, load, r0_rule      # noqa: E402
from fspm import Params, simulate                             # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"
PERIODS = {"all": (START, END), "used_for_Rd (Oct–Nov)": (START, "2023-11-30"), "hold-out (Dec–Jan)": ("2023-12-01", END)}


def metrics(j):
    e = j["sim"] - j["obs"]
    return dict(n=len(j), RMSE=float(np.sqrt((e ** 2).mean())), nRMSE=float(np.sqrt((e ** 2).mean()) / j["obs"].mean()),
                bias=float(e.mean()), bias_pct=float(100 * e.mean() / j["obs"].mean()),
                R2=float(np.corrcoef(j["sim"], j["obs"])[0, 1] ** 2), ratio=float(j["sim"].sum() / j["obs"].sum()))


def main():
    clim, hr, obs, fin, ev = load()
    pj = json.loads((DP / "params_m2.json").read_text())
    p = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    s, _ = simulate(clim, p, r0_rule(obs, fin), START, END, hourly=hr, **INIT)
    lc = pd.read_csv(DP / "loadcell_daily.csv", parse_dates=["date"])
    lm = lc[lc["use"] == 1].pivot_table(index="date", columns="device", values="ET_plant_mL").mean(axis=1)
    j = pd.concat([s["et_g"].rename("sim"), lm.rename("obs")], axis=1, join="inner").dropna()
    rows = []
    for k, (a, b) in PERIODS.items():
        m = metrics(j.loc[a:b])
        rows.append(dict(period=k, **m))
        print(f"{k:24s} n={m['n']:3d}  RMSE={m['RMSE']:.0f} g  nRMSE={m['nRMSE']:.2f}  bias={m['bias']:+.0f} g ({m['bias_pct']:+.0f}%)  "
              f"R2={m['R2']:.2f}  ratio={m['ratio']:.2f}")
    pd.DataFrame(rows).to_csv(DP / "et_split.csv", index=False)


if __name__ == "__main__":
    main()
