"""
M2 결과 + 독립 검증

보정(calibrate_m2.py) 결과 params_m2.json 사용
  (a) 기관 DW — 파괴조사(보정) + 최종조사 1/11 (검증, 별도 16주)
  (b) 엽면적
  (c) Q/D 와 생식 시계 배율 g
  (d) 화방별 개화→착과 기간 — 관측 vs A(온도만) vs B(온도×Q/D)
  (e) 일 증산 — 모의 vs 로드셀 4대 (검증, 보정 미사용)
  (f) 증산 산점도
출력: figures/m2_source.png, 표준출력 지표
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
from calibrate_m2 import INIT, START, END, load, r0_rule, set_dates_obs   # noqa: E402
from fspm import Params, simulate, truss_table                           # noqa: E402
from fspm.phenology import source_modifier                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
FIG = ROOT / "figures"
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
OBS = "#52514e"


def metrics(sim, obs):
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1, join="inner").dropna()
    e = j["s"] - j["o"]
    r2 = np.corrcoef(j["s"], j["o"])[0, 1] ** 2
    return dict(n=len(j), RMSE=float(np.sqrt((e ** 2).mean())), nRMSE=float(np.sqrt((e ** 2).mean()) / j["o"].mean()),
                bias=float(e.mean()), R2=float(r2))


def main():
    clim, hr, obs, fin, ev = load()
    pj = json.loads((DP / "params_m2.json").read_text())
    pB = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    pA = replace(pB, use_source_clock=False, TT_anth_set=pj["TT_anth_set_tempOnly"])
    rule = r0_rule(obs, fin)
    sB, plB = simulate(clim, pB, rule, START, END, hourly=hr, **INIT)
    sA, plA = simulate(clim, pA, rule, START, END, hourly=hr, **INIT)

    # ---- 검증 1: 최종조사 (1/11, 별도 16주) ----
    last = sB.loc["2024-01-11"]
    print("[검증] 최종조사 2024-01-11 (별도 16주, 보정 미사용)  관측 평균±sd / 모의")
    for o, m, sc in (("leaf_DW", "leaf_dw", 1), ("stem_all_DW", "stem_dw", 1),
                     ("leaf_area_cm2", "leaf_area_m2", 1e4)):
        print(f"  {o:14s} {fin[o].mean():8.1f} ± {fin[o].std():6.1f}   모의 {last[m] * sc:8.1f}")
    print(f"  {'fruit_DW':14s} {fin['fruit_DW'].mean():8.1f} ± {fin['fruit_DW'].std():6.1f}   "
          f"모의(달린+수확) {last['fruit_dw'] + last['harvested_dw']:8.1f}  ※ 최종조사 주는 수확 제외주")

    # ---- 검증 2: 로드셀 증산 ----
    lc = pd.read_csv(DP / "loadcell_daily.csv", parse_dates=["date"])
    lc = lc[lc["use"] == 1].pivot_table(index="date", columns="device", values="ET_plant_mL")
    lc_m = lc.mean(axis=1)
    et = sB["et_g"]
    m_all = metrics(et, lc_m)
    print("\n[검증] 일 증산 (g/주/d) 모의 vs 로드셀 4대 평균:",
          {k: round(v, 2) for k, v in m_all.items()})
    # 적엽 직후 반응: 적엽일 전후 3일 평균 비교
    ev_days = sB.index[sB["removed"] > 0]
    for d in ev_days:
        b = slice(d - pd.Timedelta(days=3), d - pd.Timedelta(days=1))
        a = slice(d + pd.Timedelta(days=1), d + pd.Timedelta(days=3))
        rb_ = clim.loc[b, "Rg_out_MJ"].mean(), clim.loc[a, "Rg_out_MJ"].mean()
        print(f"  적엽 {d.date()} ({int(sB.loc[d, 'removed'])}장): 모의 {et[b].mean():5.0f}→{et[a].mean():5.0f} g  "
              f"로드셀 {lc_m.reindex(pd.date_range(b.start, b.stop)).mean():5.0f}→{lc_m.reindex(pd.date_range(a.start, a.stop)).mean():5.0f} mL  "
              f"(일사 {rb_[0]:.1f}→{rb_[1]:.1f} MJ)")

    # ---- C1: 개화→착과 기간 ----
    obs_lag = set_dates_obs(ev)
    lag = {}
    for nm, pl in (("A", plA), ("B", plB)):
        tt = truss_table(pl).set_index("truss")
        lag[nm] = (tt["set"] - tt["anthesis"]).dt.days

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(3, 2, figsize=(12, 11))
    kw = dict(fmt="o", ms=4, color=OBS, ecolor=OBS, elinewidth=1, capsize=0)
    a = ax[0, 0]
    de = pd.read_csv(DP / "obs_des.csv", parse_dates=["date"])
    de["stem_all_DW"] = de["stem_DW"] + de["truss_struct_DW"].fillna(0)
    g = de.groupby("date")
    for i, (o, m, lab) in enumerate((("leaf_DW", "leaf_dw", "잎"), ("stem_all_DW", "stem_dw", "줄기+화방축"),
                                     ("fruit_DW", "fruit_dw", "과실(달린)"))):
        a.plot(sB.index, sB[m], color=C[i], lw=2, label=f"{lab} 모의")
        a.errorbar(g[o].mean().index, g[o].mean(), g[o].std(), fmt="o", ms=4, color=C[i], ecolor=C[i], elinewidth=1)
        a.plot([pd.Timestamp("2024-01-11")], [fin[o].mean()], "D", ms=7, mfc="white", mec=C[i], mew=2)
    a.plot(sB.index, sB["harvested_dw"], color=C[3], lw=2, ls="--", label="누적 수확 과실 모의")
    a.set_title("(a) 기관 건물중 — 원: 파괴조사(보정), ◇: 최종조사(검증)", loc="left")
    a.set_ylabel("g DW / 주")

    a = ax[0, 1]
    a.plot(sB.index, sB["leaf_area_m2"] * 1e4, color=C[0], lw=2, label="모의")
    a.errorbar(g["leaf_area_cm2"].mean().index, g["leaf_area_cm2"].mean(), g["leaf_area_cm2"].std(), label="관측", **kw)
    a.plot([pd.Timestamp("2024-01-11")], [fin["leaf_area_cm2"].mean()], "D", ms=7, mfc="white", mec=OBS, mew=2,
           label="최종조사(검증)")
    a.set_title("(b) 엽면적", loc="left")
    a.set_ylabel("cm² / 주")

    a = ax[1, 0]
    a.plot(sB.index, sB["qd"].clip(upper=4), color=C[0], lw=1, alpha=0.5, label="Q/D (일)")
    a.plot(sB.index, sB["qd_ema"], color=C[0], lw=2, label="Q/D 이동평균")
    a.plot(sB.index, [source_modifier(q, pB) for q in sB["qd_ema"]], color=C[1], lw=2, label="g(Q/D) 생식 시계 배율")
    a.axhline(1, color=OBS, lw=0.8, ls=":")
    a.set_title("(c) source/sink 비와 생식 시계 배율", loc="left")
    a.set_ylabel("무차원")

    a = ax[1, 1]
    x = obs_lag.index
    a.plot(x, obs_lag.values, "o", ms=7, color=OBS, label="관측")
    a.plot(lag["A"].reindex(x).index, lag["A"].reindex(x).values, "s-", color=C[1], lw=1.5, ms=5, label="A 온도만")
    a.plot(lag["B"].reindex(x).index, lag["B"].reindex(x).values, "^-", color=C[0], lw=1.5, ms=6, label="B 온도×Q/D")
    a.set_title("(d) 화방별 개화→착과 기간 (C1)", loc="left")
    a.set_xlabel("화방 번호")
    a.set_ylabel("일")

    a = ax[2, 0]
    a.fill_between(lc.index, lc.min(axis=1), lc.max(axis=1), color=OBS, alpha=0.2, lw=0, label="로드셀 4대 범위")
    a.plot(lc_m.index, lc_m, color=OBS, lw=1.2, label="로드셀 평균")
    a.plot(et.index, et, color=C[0], lw=1.5, label="모의 (잎 단위 PM 합)")
    for d in ev_days:
        a.axvline(d, color=C[1], lw=0.8, ls="--")
    a.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
    a.set_title("(e) 일 증산 — 독립 검증 (점선: 모의 적엽일)", loc="left")
    a.set_ylabel("g 물 / 주 / 일")

    a = ax[2, 1]
    j = pd.concat([et.rename("s"), lc_m.rename("o")], axis=1, join="inner").dropna()
    a.plot(j["o"], j["s"], "o", ms=4, color=C[0], alpha=0.7)
    lim = [0, max(j.max()) * 1.05]
    a.plot(lim, lim, color=OBS, lw=0.8, ls=":")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("로드셀 (mL / 주 / 일)")
    a.set_ylabel("모의 (g / 주 / 일)")
    a.set_title(f"(f) 증산 1:1  R²={m_all['R2']:.2f}, RMSE={m_all['RMSE']:.0f}, bias={m_all['bias']:+.0f}", loc="left")

    for a in ax.flat:
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        if a.get_legend_handles_labels()[0]:
            a.legend(frameon=False, fontsize=8)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0], ax[2, 0]):
        a.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(FIG / "m2_source.png", dpi=150)
    print("\n그림:", FIG / "m2_source.png")


if __name__ == "__main__":
    main()
