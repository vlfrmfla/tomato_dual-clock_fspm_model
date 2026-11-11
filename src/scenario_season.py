"""
계절별 적정 엽수 시나리오 — 잎 호흡(FvCB Rd)·광·습도 부하로 겨울/여름 관행 설명

작기 (A7-6 6구획 실측 기후, 정식 = 시작 − 17일, 시작 상태 8마디):
  winter  시작 2023-10-05, 분석 2023-12-01 ~ 2024-01-31
  spring  시작 2024-01-20, 분석 2024-03-15 ~ 2024-05-15
  summer  횡계팜 2025 여름 정식 작기 (재배 온실 실측 기후, cv. Dafnis 보정값), 정식 2025-08-11, 시작 09-01(관측 상태), 분석 09-15 ~ 11-13
          (함안 2024 여름은 무작물 구획 기후라 제외)
적엽: 시작 40일 후부터 N장 유지(매주 월요일), N = 12 … 30
지표 (분석 구간): 과실 DM 증가, 총 DM 증가(적엽분 포함), 증산·야간 증산, 최하위 잎 7일 순탄소수지, 잎 Rd/총동화
가이드라인: N95 = 과실 DM 증가가 최대의 95% 이상인 최소 N
           N0  = 분석 구간 평균 최하위 잎 7일 순수지가 0 이상인 최대 N (탄소 중립 엽수)
변형: base(Rd25 0.65, Rd∝cap) / Rd25 0.50·0.90 / Rd 엽령감소 없음(const) / Rd·양자수율 엽령감소(cap_alpha)
출력: data_processed/scenario_season.csv, figures/scenario_season.png
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
from fspm import Params, simulate, truss_table        # noqa: E402
from fspm import management as M                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEASONS = {
    "winter": ("2023-10-05", "2023-12-01", "2024-01-31"),
    "spring": ("2024-01-20", "2024-03-15", "2024-05-15"),
    # 여름 정식(고랭지): 횡계팜 2025 재배 온실 실측 기후, cv. Dafnis 보정값 (calibrate_hoenggye.py)
    #   정식 2025-08-11, 시작 9/01 (첫 정식 후 조사 관측 상태, run_hoenggye.initial_plant), 엽수 유지 9/15부터, 분석 9/15–11/13 (실내 일사 측정 구간)
    #   ※ 함안 2024 여름(무작물 구획) 기후는 사용하지 않음 (사용자 결정 2026-10-05)
    "summer": ("2025-09-01", "2025-09-15", "2025-11-13"),
}
HG_SEASON = "summer"
LABEL = {"winter": "겨울", "spring": "봄", "summer": "여름 정식(횡계)"}
COLOR = {"winter": "#2a78d6", "spring": "#1baf7a", "summer": "#eb6834"}
NS = list(range(12, 31, 2))
# (이름, Rd25, rd_age_mode) — 기준 + Rd 민감도 + 엽령별 호흡 가정
VARIANTS = [("base", 0.65, "cap", {}), ("Rd0.50", 0.50, "cap", {}), ("Rd0.90", 0.90, "cap", {}),
            ("Rd_const", 0.65, "const", {}), ("Rd+alpha", 0.65, "cap_alpha", {}),
            ("noExposure", 0.65, "cap", {"fruit_abs": 0.0})]   # 과실 = 기온 (같은 숙기 기준) → 하엽 노출 효과만 분리


def window_metrics(s, a, b, hr, p):
    w = s.loc[a:b]
    e0, e1 = s.loc[:a].iloc[-1], w.iloc[-1]
    fruit = (e1.fruit_dw + e1.harvested_dw) - (e0.fruit_dw + e0.harvested_dw)
    total = sum(e1[k] for k in ("fruit_dw", "harvested_dw", "leaf_dw", "stem_dw", "removed_leaf_dw")) - \
        sum(e0[k] for k in ("fruit_dw", "harvested_dw", "leaf_dw", "stem_dw", "removed_leaf_dw"))
    days = len(w)
    h = hr.loc[a:b]
    return dict(fruit_gain=fruit, total_gain=total, fruit_per_wk=fruit / days * 7,
                et_L=w.et_g.sum() / 1000, et_night_L=w.et_night_g.sum() / 1000,
                night_share=w.et_night_g.sum() / w.et_g.sum(),
                lowest_net7=w.lowest_net7.mean(), LAI=w.LAI.mean(), leaves=w.leaves_present.mean(),
                rd_frac_gross=w.leaf_rd.sum() / (w.gross + w.leaf_rd).sum(), qd=w.qd.mean(),
                T=h["T"].mean(), T_night=h["T"].between_time("20:00", "05:00").mean(),
                RH_night=h["RH"].between_time("20:00", "05:00").mean(),
                Rg_MJ=h["Rg_ext" if "Rg_ext" in h else "Rg_out"].resample("D").sum().mean() * 3600 / 1e6)


def main():
    clim_ha = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"]).set_index("date")
    hr_ha = pd.read_csv(DP / "climate_hourly.csv", parse_dates=["datetime"]).set_index("datetime")
    pj = json.loads((DP / "params_m2.json").read_text())
    p_ha = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    # 횡계: 재배 조건 + 품종 보정값, 정식일 관측 초기 상태
    import run_hoenggye as RH
    clim_hg, hr_hg, obs_hg, _ = RH.load()
    ph = json.loads((DP / "hg/params_hg.json").read_text())
    p_hg = replace(RH.base_params(), **{k: v for k, v in ph.items() if k in Params.__dataclass_fields__})
    init_hg = dict(plant0=RH.initial_plant(clim_hg, p_hg, obs_hg))
    rows = []
    for season, (start, a, b) in SEASONS.items():
        st = pd.Timestamp(start)
        end = (pd.Timestamp(b) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        hg = season == HG_SEASON
        clim, hr, p0 = (clim_hg, hr_hg, p_hg) if hg else (clim_ha, hr_ha, p_ha)
        init = init_hg if hg else dict(leaf_dw0=10.0, stem_dw0=4.0)
        trigger = a if hg else (st + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        for vname, rd, mode, over in VARIANTS:
            p = replace(p0, rd_frac=rd / p_ha.Vcmax25,         # Rd25 = 계수 × Vcmax25 (계수는 함안 기준)
                         rd_age_mode=mode, **over,
                        transplant=RH.START_TP if hg else (st - pd.Timedelta(days=17)).strftime("%Y-%m-%d"))
            for n in NS:
                rule = M.KeepN(n, trigger=trigger)
                s, pl = simulate(clim, p, rule, start, end, hourly=hr, **init)
                tt = truss_table(pl)
                rp = tt[(tt.ripe >= a) & (tt.ripe <= b)]
                rows.append(dict(season=season, variant=vname, Rd25=rd, mode=mode, N=n, **window_metrics(s, a, b, hr, p),
                                 harvest_gain=s.loc[:b].harvested_dw.iloc[-1] - s.loc[:a].harvested_dw.iloc[-1],
                                 n_ripe=len(rp), set_to_ripe_d=(rp.ripe - rp.set).dt.days.mean(),
                                 fruit_dT=rp.fruit_dT_mean.mean(), hot_h_per_truss=rp.hot_hours.mean()))
    d = pd.DataFrame(rows)
    d.to_csv(DP / "scenario_season.csv", index=False)

    # 가이드라인
    g = []
    for (season, rd), x in d.groupby(["season", "variant"]):
        x = x.sort_values("N")
        fmax = x.fruit_gain.max()
        n95 = int(x[x.fruit_gain >= 0.95 * fmax].N.min())
        pos = x[x.lowest_net7 >= 0]
        n0 = int(pos.N.max()) if len(pos) else np.nan
        r95 = x[x.N == n95].iloc[0]
        rmax = x.loc[x.fruit_gain.idxmax()]
        g.append(dict(season=season, variant=rd, N95=n95, N0=n0, N_fruitmax=int(rmax.N),
                      fruit_at_N95=r95.fruit_gain, fruit_max=fmax,
                      et_saving_pct=(1 - r95.et_L / x[x.N == 30].et_L.iloc[0]) * 100,
                      night_et_saving_pct=(1 - r95.et_night_L / x[x.N == 30].et_night_L.iloc[0]) * 100,
                      T=r95["T"], T_night=r95.T_night, RH_night=r95.RH_night, Rg_MJ=r95.Rg_MJ,
                      rd_frac_gross=r95.rd_frac_gross, qd=r95.qd))
    G = pd.DataFrame(g)
    G.to_csv(DP / "scenario_guideline.csv", index=False)
    pd.set_option("display.width", 220)
    print(G.round(2).to_string(index=False))
    base = d[d.variant == "base"]
    print("\n[base: Rd25 0.65, Rd∝cap] 계절 × N")
    print(base.pivot_table(index="N", columns="season", values=["fruit_per_wk", "lowest_net7", "et_L", "et_night_L"]).round(2).to_string())
    print("\n[숙기] base vs noExposure (과실 온도 = 기온)")
    rip = d[d.variant.isin(["base", "noExposure"])]
    print(rip.pivot_table(index="N", columns=["season", "variant"], values=["set_to_ripe_d"]).round(1).to_string())
    print(rip.pivot_table(index="N", columns=["season", "variant"], values=["harvest_gain"]).round(1).to_string())
    print(d[d.variant == "base"].pivot_table(index="N", columns="season", values=["fruit_dT", "hot_h_per_truss"]).round(2).to_string())

    # 그림
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for season in SEASONS:
        x = base[base.season == season].sort_values("N")
        c = COLOR[season]
        rel = x.fruit_gain / x.fruit_gain.max() * 100
        ax[0, 0].plot(x.N, rel, "o-", color=c, lw=2, label=LABEL[season])
        ax[0, 1].plot(x.N, x.lowest_net7, "o-", color=c, lw=2, label=LABEL[season])
        ax[1, 0].plot(x.N, x.et_L / x.et_L.max() * 100, "o-", color=c, lw=2, label=LABEL[season])
        ax[1, 1].plot(x.N, x.et_night_L, "o-", color=c, lw=2, label=LABEL[season])
    ax[0, 0].axhline(95, color="#52514e", lw=0.8, ls=":")
    ax[0, 1].axhline(0, color="#52514e", lw=0.8, ls=":")
    ax[0, 0].set_title("(a) 과실 DM 증가 (계절 최대 대비 %)", loc="left")
    ax[0, 1].set_title("(b) 최하위 잎 7일 순탄소수지 (g CH2O)", loc="left")
    ax[1, 0].set_title("(c) 증산 (계절 최대 대비 %)", loc="left")
    ax[1, 1].set_title("(d) 야간 증산 합 (L/주, 분석 구간)", loc="left")
    for a in ax.flat:
        a.set_xlabel("남길 엽수 N")
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/scenario_season.png", dpi=150)
    print("그림 저장")


if __name__ == "__main__":
    main()
