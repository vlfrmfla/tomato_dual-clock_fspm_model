"""
잎 암호흡 Rd25 민감도 — 호흡↔투과율 비식별 문제 정리

각 Rd25 수준마다: 생산 보정(파괴조사) → 보정된 τ_struct, 적합도
                  → 증산 편향 (로드셀, 보정 미사용) → C1 보정 적합 (A vs B)
출력: data_processed/sens_rd.csv
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import (INIT, START, END, PROD, aic, fit_clock, fit_production, load,  # noqa: E402
                          prod_error, r0_rule)
from fspm import Params, simulate                                                           # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"


def et_metrics(s, lc):
    j = pd.concat([s["et_g"].rename("s"), lc.rename("o")], axis=1, join="inner").dropna()
    e = j["s"] - j["o"]
    r = (j["s"] / j["o"])
    return dict(et_bias=e.mean(), et_rmse=np.sqrt((e ** 2).mean()), et_r2=np.corrcoef(j["s"], j["o"])[0, 1] ** 2,
                ratio_oct=r.loc["2023-10"].mean(), ratio_dec=r.loc["2023-12"].mean())


if __name__ == "__main__":
    clim, hr, obs, fin, ev = load()
    lc = pd.read_csv(DP / "loadcell_daily.csv", parse_dates=["date"])
    lc = lc[lc["use"] == 1].groupby("date")["ET_plant_mL"].mean()
    base = Params()
    rows = []
    for rd25 in (0.55, 0.83, 1.10, 1.38, 1.65):
        b = replace(base, rd_frac=rd25 / base.Vcmax25)
        p1, r = fit_production(clim, hr, obs, fin, b)
        s1, pl1 = simulate(clim, p1, r0_rule(obs, fin), START, END, hourly=hr, **INIT)
        tot, det = prod_error(s1, obs, True, pl1)
        res = fit_clock(clim, hr, obs, fin, ev, p1)
        _, _, dA, kA = res["A_temp"]
        pB, rmseB, dB, kB = res["B_temp_x_source"]
        row = dict(Rd25=rd25, tau_struct=p1.tau_struct, tau_eff_dec=p1.tau_struct * 0.812,
                   leaf_area_pot=p1.leaf_area_pot, fruit_W_pot=p1.fruit_W_pot, pi=p1.fruit_priority,
                   prod_obj=tot, **{f"nrmse_{k}": v[1] for k, v in det.items()},
                   **et_metrics(s1, lc),
                   rd_over_gross_dec=(s1["leaf_rd"] / (s1["gross"] + s1["leaf_rd"])).loc["2023-12"].mean(),
                   C1_rmse_A=np.sqrt((dA ** 2).mean()), C1_rmse_B=rmseB, dAIC=aic(dB, kB) - aic(dA, kA))
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DP / "sens_rd.csv", index=False)
    print(out.round(3).to_string(index=False))
