"""
구조 시계 + 적엽 규칙 점검 — 최종 모델 (params_m2.json, source 연결)
  (2026-11-06: 이전 M1 = source 미연결 Q/D 1 → 마디 출현이 생장 중 과실 수에 따르므로 착과가 Q/D에 반응하는 최종 모델로 평가)

비교 관측
  비파괴: 누적 마디 수, 화방 출현·개화·착과 수 (주별 평균)
  파괴  : 현재 엽수, 엽면적
출력: figures/m1_structure.png, 표준출력에 RMSE
"""
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json                                              # noqa: E402
from dataclasses import replace                          # noqa: E402
from fspm import Params, simulate, truss_table           # noqa: E402
from fspm import management as M                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

START, END = "2023-10-05", "2024-01-12"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # 범주 색 고정 순서
OBS = "#52514e"


def load_obs():
    nd = pd.read_csv(DP / "obs_nondes.csv", parse_dates=["date"])
    de = pd.read_csv(DP / "obs_des.csv", parse_dates=["date"])
    fin = pd.read_csv(DP / "obs_final.csv", parse_dates=["date"])   # 2024-01-11, 별도 16주 (수확 제외주)
    de = pd.concat([de, fin[["date", "plant", "leaves_present", "leaf_area_cm2"]]], ignore_index=True)
    ev = pd.read_csv(DP / "truss_events.csv", parse_dates=["app", "anth", "set"])
    # 2024-01-03 현재 엽수는 누적값 오기입 (평균 50) → 제외
    de.loc[de["leaves_present"] > 35, "leaves_present"] = np.nan
    dates = sorted(nd["date"].unique())
    n_plants = ev["plant"].nunique()
    tr = pd.DataFrame(
        {f"n_{k}": [(ev[k] <= d).sum() / n_plants for d in dates] for k in ("app", "anth", "set")},
        index=pd.DatetimeIndex(dates, name="date"))
    nodes = nd.groupby("date")["nodes_cum"].agg(["mean", "std"])
    des = de.groupby("date")[["leaves_present", "leaf_area_cm2"]].agg(["mean", "std"])
    return nodes, tr, des


INIT = dict(leaf_dw0=10.0, stem_dw0=4.0)                 # calibrate_m2.INIT 과 같음


def final_params(**kw) -> Params:
    pj = json.loads((DP / "params_m2.json").read_text())
    return replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__}, **kw)


def load_hourly():
    return pd.read_csv(DP / "climate_hourly.csv", parse_dates=["datetime"]).set_index("datetime")


def run(clim, hr, p, rule):
    return simulate(clim, p, rule, START, END, hourly=hr, **INIT)


def rmse(sim: pd.Series, obs: pd.Series) -> float:
    j = pd.concat([sim, obs], axis=1, join="inner").dropna()
    return float(np.sqrt(((j.iloc[:, 0] - j.iloc[:, 1]) ** 2).mean()))


def main():
    clim = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"]).set_index("date")
    hr = load_hourly()
    nodes_obs, tr_obs, des_obs = load_obs()
    p = final_params()
    batch = final_params(prune_min_batch=6)                 # 두 화방분 쌓이면 일괄 작업

    lp = des_obs[("leaves_present", "mean")].dropna()
    rules = {
        "R1 18장 유지, 매주": (M.KeepN(18, trigger="2023-11-10"), p),
        "R1 Jeong 2026 N(T)": (M.KeepN(M.KeepN.jeong2026, trigger="harvest"), p),
        "R3 화방연동 235°Cd, 매주": (M.TrussLinked(tt_after=235), p),
        "R3 화방연동 235°Cd, 묶음": (M.TrussLinked(tt_after=235), batch),
    }
    sims = {}
    for name, (rule, pp) in rules.items():
        sims[name], plant = run(clim, hr, pp, rule)
        if name.startswith("R3") and "묶음" in name:
            tt = truss_table(plant)

    s0 = next(iter(sims.values()))
    s_const, _ = run(clim, hr, final_params(phyllo_mode="constant"), M.NoPruning())
    print("구조 시계 RMSE (최종 모델)")
    print(f"  누적 마디 (단일 {p.phyllochron} °Cd)      : {rmse(s_const['nodes'], nodes_obs['mean']):.2f} 마디")
    print(f"  누적 마디 ({p.phyllo_mode} {p.phyllochron_early}/{p.phyllochron_late}): {rmse(s0['nodes'], nodes_obs['mean']):.2f} 마디")
    for k, col in (("n_app", "trusses"), ("n_anth", "n_anthesis"), ("n_set", "n_set")):
        print(f"  화방 {k[2:]:5s}: {rmse(s0[col], tr_obs[k]):.2f} 개")
    print("\n적엽 규칙별 RMSE (현재 엽수 / 엽면적 cm²)")
    for name, s in sims.items():
        r1 = rmse(s["leaves_present"], des_obs[("leaves_present", "mean")])
        r2 = rmse(s["leaf_area_m2"] * 1e4, des_obs[("leaf_area_cm2", "mean")])
        print(f"  {name:22s} {r1:5.2f} 장  {r2:7.0f} cm²  적엽 {int(s['removed'].sum())}장/{int((s['removed']>0).sum())}회")
    hs = sims["R1 Jeong 2026 N(T)"]
    fh = hs.index[hs["harvested_dw"] > 0]
    print(f"\n첫 수확 (Jeong 규칙 촉발): {fh[0].date() if len(fh) else '-'}")
    print("\n모의 화방 일정 (R3 묶음)")
    print(tt[["truss", "node", "visible", "anthesis", "set", "ripe"]].to_string(index=False))

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    kw_obs = dict(fmt="o", ms=4, color=OBS, ecolor=OBS, elinewidth=1, capsize=0, label="관측")

    a = ax[0, 0]
    a.errorbar(nodes_obs.index, nodes_obs["mean"], nodes_obs["std"], **kw_obs)
    a.plot(s_const.index, s_const["nodes"], color=SERIES[1], lw=2, label="모의: 단일 phyllochron")
    a.plot(s0.index, s0["nodes"], color=SERIES[0], lw=2, label="모의: 과실 부하 φ")
    a.set_title("(a) 누적 마디 수 — 구조 시계", loc="left")
    a.set_ylabel("마디 / 주")

    a = ax[0, 1]
    for i, (k, col, lab) in enumerate((("n_app", "trusses", "출현"), ("n_anth", "n_anthesis", "개화"),
                                       ("n_set", "n_set", "착과"))):
        a.plot(tr_obs.index, tr_obs[k], "o", ms=4, color=SERIES[i], mfc="white", mew=1.5)
        a.plot(s0.index, s0[col], color=SERIES[i], lw=2, label=lab)
    a.set_title("(b) 화방 수 — 선: 모의, 원: 관측", loc="left")
    a.set_ylabel("화방 / 주")

    for col, a, title, scale in (("leaves_present", ax[1, 0], "(c) 현재 엽수 — 적엽 규칙", 1),
                                 ("leaf_area_m2", ax[1, 1], "(d) 엽면적", 1e4)):
        key = "leaves_present" if col == "leaves_present" else "leaf_area_cm2"
        a.errorbar(des_obs.index, des_obs[(key, "mean")], des_obs[(key, "std")], **kw_obs)
        for i, (name, s) in enumerate(sims.items()):
            a.plot(s.index, s[col] * scale, color=SERIES[i], lw=2, label=name)
        a.set_title(title, loc="left")
    ax[1, 0].set_ylabel("장 / 주")
    ax[1, 1].set_ylabel("cm² / 주")

    for a in ax.flat:
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "m1_structure.png", dpi=150)
    print(f"\n그림: {FIG / 'm1_structure.png'}")


if __name__ == "__main__":
    main()
