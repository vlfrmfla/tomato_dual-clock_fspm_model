"""
Rd 식별성 — (i) 다른 파라미터 고정 시 Rd만 바꾼 반응, (ii) τ_struct × Rd25 격자

(i)  params_m2.json 고정(τ_struct 포함) → Rd25 0.5–1.8: 총 DM, 기관 DM nRMSE, 생장 목적함수
(ii) τ_struct 0.60–1.00 × Rd25 0.5–1.8: 생장 목적함수(파괴조사) + 10–11월 증산 비율(로드셀 v2)
     → 생장은 능선(ridge)을 만들고 증산이 능선 위 위치를 정함
출력: data_processed/rd_oat.csv, rd_grid.csv, figures/rd_identifiability.png
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import INIT, START, END, load, prod_error, r0_rule     # noqa: E402
from fspm import Params, simulate                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False


def run(p, clim, hr, rule, obs, lc):
    s, pl = simulate(clim, p, rule, START, END, hourly=hr, **INIT)
    tot, det = prod_error(s, obs, True, pl)
    e = s.iloc[-1]
    j = pd.concat([s["et_g"].rename("s"), lc.rename("o")], axis=1, join="inner").dropna().loc["2023-10-05":"2023-11-30"]
    return dict(obj=tot, total_DM=e.leaf_dw + e.stem_dw + e.fruit_dw + e.harvested_dw + e.removed_leaf_dw,
                fruit_DM=e.fruit_dw + e.harvested_dw, **{f"nrmse_{k}": v[1] for k, v in det.items()},
                et_ratio_octnov=(j["s"].sum() / j["o"].sum()))


if __name__ == "__main__":
    clim, hr, obs, fin, ev = load()
    lc = pd.read_csv(DP / "loadcell_daily.csv", parse_dates=["date"])
    lc = lc[lc["use"] == 1].groupby("date")["ET_plant_mL"].mean()
    pj = json.loads((DP / "params_m2.json").read_text())
    p0 = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    rule = r0_rule(obs, fin)
    de = pd.read_csv(DP / "obs_des.csv", parse_dates=["date"])
    obs_total_last = (de[de.date == "2023-12-28"][["leaf_DW", "stem_DW", "truss_struct_DW", "fruit_DW"]].sum(axis=1)).mean()

    # (i) one-at-a-time
    rows = []
    for rd in np.round(np.arange(0.5, 1.81, 0.1), 2):
        r = run(replace(p0, rd_frac=rd / p0.Vcmax25), clim, hr, rule, obs, lc)
        rows.append(dict(Rd25=rd, **r))
    oat = pd.DataFrame(rows)
    oat.to_csv(DP / "rd_oat.csv", index=False)
    ref = oat.loc[(oat.Rd25 - 1.1).abs().idxmin()]
    oat["dTotal_%"] = (oat.total_DM / ref.total_DM - 1) * 100
    oat["dFruit_%"] = (oat.fruit_DM / ref.fruit_DM - 1) * 100
    print("(i) τ 등 고정, Rd25만 변화 (기준 1.1)")
    print(oat[["Rd25", "total_DM", "dTotal_%", "fruit_DM", "dFruit_%", "obj", "nrmse_truss_fruit_DW", "nrmse_leaf_DW",
               "nrmse_stem_all_DW", "et_ratio_octnov"]].round(2).to_string(index=False))
    sl = np.polyfit(oat.Rd25, oat["dTotal_%"], 1)[0]
    print(f"  총 DM 탄성: Rd25 +0.1 당 {sl / 10:+.1f}%  (τ 고정)")

    # (ii) grid
    taus = np.round(np.arange(0.60, 1.001, 0.05), 2)
    rds = np.round(np.arange(0.5, 1.81, 0.15), 2)
    g = []
    for t in taus:
        for rd in rds:
            r = run(replace(p0, tau_struct=t, rd_frac=rd / p0.Vcmax25), clim, hr, rule, obs, lc)
            g.append(dict(tau_struct=t, Rd25=rd, **r))
    grid = pd.DataFrame(g)
    grid.to_csv(DP / "rd_grid.csv", index=False)
    best = grid.loc[grid.obj.idxmin()]
    print(f"\n(ii) 격자 최소 생장목적함수 {best.obj:.3f} @ τ={best.tau_struct}, Rd25={best.Rd25}")
    for rd in rds:
        sub = grid[grid.Rd25 == rd]
        b = sub.loc[sub.obj.idxmin()]
        print(f"  Rd25 {rd:4.2f}: 최적 τ {b.tau_struct:.2f} obj {b.obj:.3f} | 그 τ에서 10–11월 증산비 {b.et_ratio_octnov:.2f}")

    # 그림
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    a = ax[0]
    a.plot(oat.Rd25, oat["dTotal_%"], "o-", color="#2a78d6", lw=2, label="총 DM 변화 (%)")
    a.plot(oat.Rd25, oat["dFruit_%"], "s-", color="#eb6834", lw=2, label="과실 DM 변화 (%)")
    a.axvline(1.1, color="#52514e", lw=0.8, ls=":")
    a.axhline(0, color="#52514e", lw=0.6)
    a.set_xlabel("Rd25 (µmol/m²/s)")
    a.set_ylabel("기준(Rd25 1.1) 대비 %")
    a.set_title("(a) τ 고정, Rd만 변화", loc="left")
    a.legend(frameon=False, fontsize=8)
    a = ax[1]
    Z = grid.pivot(index="tau_struct", columns="Rd25", values="obj")
    E = grid.pivot(index="tau_struct", columns="Rd25", values="et_ratio_octnov")
    cs = a.contourf(Z.columns, Z.index, Z.values, levels=12, cmap="Blues_r")
    fig.colorbar(cs, ax=a, label="생장 목적함수 (낮을수록 적합)")
    ce = a.contour(E.columns, E.index, E.values, levels=[0.9, 1.0, 1.1], colors=["#eb6834"], linestyles=["--", "-", "--"])
    a.clabel(ce, fmt={0.9: "증산 0.9", 1.0: "증산 1.0", 1.1: "증산 1.1"}, fontsize=8)
    a.set_xlabel("Rd25 (µmol/m²/s)")
    a.set_ylabel("τ_struct")
    a.set_title("(b) 생장 적합 능선과 증산(10–11월) 제약", loc="left")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/rd_identifiability.png", dpi=150)
    print("그림 저장")
