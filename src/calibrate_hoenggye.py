"""
횡계팜 2025 (cv. Dafnis) — 품종 파라미터 보정 + 독립 증산 검증 + 재배 온실 여름 미기상

보정 (파괴조사 4회 9/22–12/09 × LAI·잎 DW·줄기 DW·과실 총 DW(12/09 제외), 정규화 RMSE 합):
  자유: leaf_area_pot, fruit_W_pot, fruit_priority  (품종·작형 차이)
  고정: 함안 A7-6 최종값 (구조·생식 시계, 호흡 Rd25 0.65, 증산), SLA = 횡계 실측(엽면적/잎 DW, 정식일 제외 5회 평균),
        Vcmax25·Jmax25 = 횡계 실측 (run_hoenggye.base_params), 시작 = 9/01 관측 상태 (run_hoenggye.initial_plant)
  적엽: 기록 없음 → 함안 관행 R3 (착과 후 235 °C d 화방까지, 6장 이상 묶음, 월요일); 규칙별 재보정·관측 달린 엽수로 비교
  DM_Har = 과실 총 DW (착과 + 조사일 이전 누적 수확, 메타데이터 확인). 누적 수확 기록(12회)은 보정에 쓰지 않는 검증
  착색기 TT_color (두 온실 공통): 첫 수확일(관측 10/16)에 맞춤 — fit_color. 품종 파라미터와 2회 교대.
        결과는 params_hg.json에 기록하고 params.py 기본값(공통)으로 옮겨 씀
증산 검증 없음: 로드셀(RMF1)은 같은 온실의 3월 정식 작기 M 소속 (메타데이터.md)
적엽 규칙 일관성: 보정 파라미터로 규칙별 LAI nRMSE
미기상: 8/11–9/15 실내 − 외기 기온차 vs 외부 일사, 외기 > 28 °C 시간
출력: data_processed/hg/params_hg.json, hg_calib_log.txt, figures/hoenggye_validation.png
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_hoenggye as R                           # noqa: E402
from fspm import simulate                          # noqa: E402
from fspm import management as M                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HG = ROOT / "data_processed/hg"
ET_WIN = ("2025-09-15", "2025-11-13")
FREE = ["leaf_area_pot", "fruit_W_pot", "fruit_priority"]
X0 = [0.10, 8.3, 1.0]
BOUNDS = [(0.04, 0.20), (4.0, 16.0), (0.5, 4.0)]
VARS = [("LAI", "LAI"), ("leaf_DW", "leaf_DW"), ("stem_DW", "stem_DW"), ("fruit_total_DW", "harvest_DW")]


def setup():
    clim, hr, obs, lc = R.load()
    o = obs[obs["initial"] >= 0]
    sla = float((o["leaf_area_m2"] / o["leaf_DW"]).mean())
    p0 = replace(R.base_params(), SLA=sla, prune_min_batch=6)
    return clim, hr, obs, lc, p0


def run(clim, hr, obs, p, rule=None):
    return R.sim(clim, hr, obs, p, rule or M.TrussLinked(235))


def prod_obj(s, obs, detail=False):
    o = obs[obs["initial"] == 0]
    sim = pd.DataFrame({"LAI": s["LAI"], "leaf_DW": s["leaf_dw"], "stem_DW": s["stem_dw"],
                        "fruit_total_DW": s["fruit_dw"] + s["harvested_dw"]})
    tot, det = 0.0, {}
    for v, ov in VARS:
        j = pd.concat([sim[v], o[ov]], axis=1, join="inner").dropna()
        rm = float(np.sqrt(((j.iloc[:, 0] - j.iloc[:, 1]) ** 2).mean()))
        det[v] = (rm, rm / j.iloc[:, 1].mean(), 100 * (j.iloc[:, 0] - j.iloc[:, 1]).mean() / j.iloc[:, 1].mean())
        tot += det[v][1]
    return (tot, det) if detail else tot


def et_stats(s, lc, a, b):
    e = pd.concat([s["et_g"], lc["ET_obs"]], axis=1, join="inner").loc[a:b].dropna()
    e.columns = ["sim", "obs"]
    rm = float(np.sqrt(((e.sim - e.obs) ** 2).mean()))
    return dict(n=len(e), RMSE=rm, nRMSE=rm / e.obs.mean(), bias=float((e.sim - e.obs).mean()),
                R2=float(np.corrcoef(e.sim, e.obs)[0, 1] ** 2), ratio=float(e.sim.sum() / e.obs.sum())), e


def microclimate():
    raw = pd.read_csv(R.ROOT / "data_analysis/대관령_횡계팜_KIST/20250811_20251209_H_farm_tom.csv",
                      parse_dates=["datetime"]).set_index("datetime")
    h = raw.resample("h").mean(numeric_only=True).loc["2025-08-11":"2025-09-15"]
    h["dT"] = h["In_Temp"] - h["Out_Temp"]
    day = h[h["Out_Rad"] > 50]
    tab = day.groupby(pd.cut(day["Out_Rad"], [50, 200, 400, 600, 800, 1100]), observed=True).agg(
        n=("dT", "size"), dT=("dT", "mean"), vent=("Vent_state", "mean"), shade=("Shade_SC_state", "mean"),
        T_out=("Out_Temp", "mean"), RH_in=("In_Hum", "mean"))
    hot = h[h["Out_Temp"] > 28]
    return h, tab, dict(n=len(hot), dT=float(hot["dT"].mean()), Tin_max=float(h["In_Temp"].max()),
                        Tout_max=float(h["Out_Temp"].max()))


def leaves_fit(s, obs):
    """달린 엽수: 모의 vs 관측 (9/22–12/09) → (RMSE, 평균 편차)"""
    j = pd.concat([s["leaves_present"], obs.loc[obs["initial"] == 0, "leaves"]], axis=1, join="inner").dropna()
    d = j.iloc[:, 0] - j.iloc[:, 1]
    return float(np.sqrt((d ** 2).mean())), float(d.mean())


def first_lag(clim, hr, obs, p, rule=None):
    s, _ = run(clim, hr, obs, p, rule)
    return R.harvest_stats(s, obs)["first_harvest_lag_d"]


def fit_color(clim, hr, obs, p, lo=0.0, hi=450.0, n=10):
    """TT_color: 모의 첫 수확일이 관측 첫 수확일(10/16)과 같은 구간(일 단위 계단)의 가운데 값.
    첫 수확 편차는 TT_color에 대해 단조 증가 → 편차 ≥ 0 이 되는 최소값과 편차 ≥ 1 이 되는 최소값을 이분 탐색"""
    def edge(target):
        a, b = lo, hi
        if first_lag(clim, hr, obs, replace(p, TT_color=a)) >= target:
            return a
        for _ in range(n):
            m = (a + b) / 2
            if first_lag(clim, hr, obs, replace(p, TT_color=m)) >= target:
                b = m
            else:
                a = m
        return b
    e0, e1 = edge(0), edge(1)
    return (e0 + e1) / 2, (e0, e1)


def calibrate(clim, hr, obs, p0, rule=None):
    """품종 파라미터 3개 Nelder–Mead 보정 (적엽 규칙 지정 가능) → (x, 반복수)"""
    def f(x):
        x = [min(max(v, lo), hi) for v, (lo, hi) in zip(x, BOUNDS)]
        s, _ = run(clim, hr, obs, replace(p0, **dict(zip(FREE, x))), rule)
        return prod_obj(s, obs)

    r = minimize(f, X0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3, maxiter=120,
                                                           initial_simplex=[X0, [0.13, 8.3, 1.0], [0.10, 11.0, 1.0], [0.10, 8.3, 2.0]]))
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(r.x, BOUNDS)], r.nit


def rule_recalibration(clim, hr, obs, p0, P):
    """규칙별로 품종 파라미터를 각각 다시 보정 → 공정한 규칙 비교 (R3 가정 보정의 편향 제거)"""
    rows = []
    for rn, (rule, kw) in R.rules().items():
        pr0 = replace(p0, prune_min_batch=kw.get("prune_min_batch", 1))
        x, nit = calibrate(clim, hr, obs, pr0, rule)
        sr, _ = run(clim, hr, obs, replace(pr0, **dict(zip(FREE, x))), rule)
        obj, dr = prod_obj(sr, obs, detail=True)
        lv = leaves_fit(sr, obs)
        hv = R.harvest_stats(sr, obs)
        rows.append(dict(rule=rn, obj=obj, LAI_nRMSE=dr["LAI"][1], LAI_bias=dr["LAI"][2], leaf_nRMSE=dr["leaf_DW"][1],
                         stem_nRMSE=dr["stem_DW"][1], fruit_nRMSE=dr["fruit_total_DW"][1],
                         leaves_RMSE=lv[0], leaves_bias=lv[1], harv_RMSE=hv["harv_RMSE"],
                         leaf_area_pot=x[0], fruit_W_pot=x[1], fruit_priority=x[2], iters=nit))
        P(f"  {rn:18s} obj {obj:.3f}  LAI nRMSE {dr['LAI'][1]:.2f} ({dr['LAI'][2]:+.0f}%)  엽수 RMSE {lv[0]:.1f} ({lv[1]:+.1f})  "
          f"A_pot {x[0]*1e4:.0f} W {x[1]:.2f} π {x[2]:.2f}")
    d = pd.DataFrame(rows)
    d.to_csv(HG / "rule_recalibration.csv", index=False)
    return d


def main():
    import sys
    clim, hr, obs, lc, p0 = setup()
    log = []
    P = lambda s: (print(s, flush=True), log.append(s))

    for it in range(2):                                    # 품종 파라미터 ↔ 착색기 TT_color 교대
        x, nit = calibrate(clim, hr, obs, p0)
        p = replace(p0, **dict(zip(FREE, x)))
        tc, (e0, e1) = fit_color(clim, hr, obs, p)
        P(f"[착색기 {it + 1}] TT_color {p0.TT_color:.1f} → {tc:.1f} °C d (첫 수확 일치 구간 {e0:.0f}–{e1:.0f})")
        p0, p = replace(p0, TT_color=tc), replace(p, TT_color=tc)
    s, pl = run(clim, hr, obs, p)
    obj, det = prod_obj(s, obs, detail=True)
    P(f"[보정] 반복 {nit}, 목적함수 {obj:.3f}, SLA(실측) = {p0.SLA:.4f} m² g⁻¹ (함안 0.020)")
    for k, v in zip(FREE, x):
        P(f"  {k:15s} = {v:.4g}")
    for v, (rm, nr, b) in det.items():
        P(f"  {v:15s} RMSE {rm:7.3f}  nRMSE {nr:.2f}  bias {b:+.0f}%")
    hv = R.harvest_stats(s, obs)
    lv = leaves_fit(s, obs)
    P(f"  누적 수확 (검증) RMSE {hv['harv_RMSE']:.1f} g, 12/09 모의 {hv['harv_final_sim']:.0f} vs 관측 {hv['harv_final_obs']:.0f} g, "
      f"첫 수확 차 {hv['first_harvest_lag_d']:+.0f} d (관측 10/16)")
    P(f"  달린 엽수 (검증) RMSE {lv[0]:.1f}, 평균 편차 {lv[1]:+.1f}장")
    P("[적엽 규칙 일관성] 보정 파라미터로 규칙별 LAI nRMSE / 달린 엽수 / 10월 평균 엽수")
    rule_rows = []
    for rn, (rule, kw) in R.rules().items():
        pr = replace(p, prune_min_batch=kw.get("prune_min_batch", 1))
        sr, _ = run(clim, hr, obs, pr, rule)
        _, dr = prod_obj(sr, obs, detail=True)
        rule_rows.append((rn, dr["LAI"][1], dr["LAI"][2], float(sr.loc["2025-10-01":"2025-10-31", "leaves_present"].mean()),
                          int((sr["removed"] > 0).sum())))
        lv = leaves_fit(sr, obs)
        P(f"  {rn:18s} LAI nRMSE {dr['LAI'][1]:.2f} bias {dr['LAI'][2]:+.0f}%  엽수 RMSE {lv[0]:.1f} ({lv[1]:+.1f})  "
          f"10월 엽수 {rule_rows[-1][3]:.1f}  작업 {rule_rows[-1][4]}회")

    if "--rules" in sys.argv:
        P("[규칙별 재보정] 규칙마다 품종 파라미터 3개를 각각 보정")
        rule_recalibration(clim, hr, obs, p0, P)

    h, tab, hot = microclimate()
    P("[미기상] 8/11–9/15 실내 − 외기 기온 (주간, 외부 일사 구간별)")
    P(tab.round(2).to_string())
    P(f"  외기 > 28 °C {hot['n']} h: 실내 − 외기 {hot['dT']:+.2f} K, 실내 최고 {hot['Tin_max']:.1f} °C (외기 최고 {hot['Tout_max']:.1f})")

    out = {k: float(v) for k, v in zip(FREE, x)}
    out.update(SLA=float(p0.SLA), plant_density=float(p.plant_density), row_spacing=float(p.row_spacing),
               Vcmax25=float(p.Vcmax25), Jmax25=float(p.Jmax25), rd_frac=float(p.rd_frac), TT_color=float(p.TT_color),
               transplant=R.START_TP, start=R.START, prune_rule="R3 batch (assumed)")
    (HG / "params_hg.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (HG / "hg_calib_log.txt").write_text("\n".join(log))

    # 그림 (2×2): LAI · 잎·줄기 · 과실 · 재배 온실 여름 기온차
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.5))
    o = obs[obs["initial"] == 0]
    a = ax[0, 0]
    a.plot(s.index, s["LAI"], color="#1baf7a", lw=1.8, label="모의 (R3 묶음)")
    a.errorbar(o.index, o["LAI"], o["LAI_sd"], fmt="o", color="#1d2420", capsize=3, label="관측 ± sd")
    a2 = a.twinx()
    a2.step(s.index, s["leaves_present"], color="#52514e", lw=0.8, where="post")
    a2.set_ylabel("모의 엽수", color="#52514e")
    a.set_title("(a) LAI", loc="left")
    a = ax[0, 1]
    a.plot(s.index, s["leaf_dw"], color="#1baf7a", lw=1.8, label="잎 모의")
    a.plot(s.index, s["stem_dw"], color="#8a6100", lw=1.8, label="줄기 모의")
    a.errorbar(o.index, o["leaf_DW"], o["leaf_DW_sd"], fmt="o", color="#1baf7a", mec="#1d2420", capsize=3, label="잎 관측")
    a.errorbar(o.index, o["stem_DW"], o["stem_DW_sd"], fmt="s", color="#8a6100", mec="#1d2420", capsize=3, label="줄기 관측")
    a.set_title("(b) 잎 · 줄기 DW (g/주)", loc="left")
    a = ax[1, 0]
    a.plot(s.index, s["fruit_dw"] + s["harvested_dw"], color="#eb6834", lw=1.8, label="모의 (달린 + 수확)")
    a.plot(s.index, s["harvested_dw"], color="#eb6834", lw=1, ls="--", label="모의 수확 누적")
    a.errorbar(o.index, o["harvest_DW"], o["harvest_DW_sd"].replace(0, np.nan), fmt="o", color="#1d2420", capsize=3, label="관측")
    a.set_title("(c) 과실 총 DW (g/주)", loc="left")
    a = ax[1, 1]
    d = h[h["Out_Rad"] > 50]
    a.scatter(d["Out_Rad"], d["dT"], s=6, color="#eb6834", alpha=0.35)
    a.plot([(iv.left + iv.right) / 2 for iv in tab.index], tab["dT"], "o-", color="#1d2420", lw=1.5)
    a.axhline(0, color="#52514e", lw=0.8, ls=":")
    a.set_xlabel("외부 일사 (W/m²)"); a.set_ylabel("실내 - 외기 (K)")
    a.set_title("(d) 재배 온실 여름 기온차 (8/11–9/15)", loc="left")
    for a in ax.flat:
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.legend(frameon=False, fontsize=7.5)
        for lab in a.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/hoenggye_validation.png", dpi=150)
    print("그림: figures/hoenggye_validation.png")


if __name__ == "__main__":
    main()
