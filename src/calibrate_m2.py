"""
M2 보정

1단계 (생산): 파괴조사 기관 DW·엽면적 (2023-10-13 ~ 2024-01-03, 8주 × 13회)
    적엽 = R0 관측 엽수 궤적 (관리 효과를 생산 보정에서 분리)
    자유 파라미터: cover_trans, leaf_area_pot, fruit_W_pot, fruit_TT_growth, fruit_priority (cap_min=0.4 고정: 비식별)
    과실은 총량 대신 화방별 DW 곡선(1–8화방, ~12/28)으로 맞춤 — 총량은 수확 시점 차이에 민감
    모의에서 먼저 수확된 화방은 수확 시 DW로 비교에 남김 (DM 생장은 착색기 전에 끝남)
    목적함수: 변수별 관측 평균으로 정규화한 RMSE 합
2단계 (생식 시계, C1 검정): 비파괴 화방별 착과일 (8주 × 10화방)
    A 온도만     : use_source_clock=False,  자유 TT_anth_set          (1개)
    B 온도×Q/D   : use_source_clock=True,   자유 TT_anth_set, g_a, g_b (3개; Q/D 척도가 Kang 실험과 달라 a도 자유)
    비교: 화방별 착과일 RMSE(일), AIC
착색기 TT_color: 두 온실 공통값 (Params 기본값, 횡계 첫 수확일로 보정 — calibrate_hoenggye.py).
    함안은 수확 기록이 없고 1·2화방이 12/28–1/03 사이 함께 사라져(농가 일괄 수확) 숙기 시점이 아닌 늦은 쪽 경계로만 확인
검증 (보정 미사용): 최종조사 2024-01-11 (별도 16주), 로드셀 증산

출력: data_processed/params_m2.json, 표준출력
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fspm import Params, simulate, truss_table           # noqa: E402
from fspm import management as M                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
START, END = "2023-10-05", "2024-01-11"
TRUSS_FRUIT_END = pd.Timestamp("2023-12-28")
N_ALT = 2            # 생산 ↔ 생식 시계 교대 보정 횟수
INIT = dict(leaf_dw0=10.0, stem_dw0=4.0)


def load():
    clim = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"]).set_index("date")
    hr = pd.read_csv(DP / "climate_hourly.csv", parse_dates=["datetime"]).set_index("datetime")
    de = pd.read_csv(DP / "obs_des.csv", parse_dates=["date"])
    de.loc[de["leaves_present"] > 35, "leaves_present"] = np.nan
    de["stem_all_DW"] = de["stem_DW"] + de["truss_struct_DW"].fillna(0)
    obs = de.groupby("date")[["leaf_DW", "stem_all_DW", "fruit_DW", "leaf_area_cm2", "leaves_present"]].mean()
    fin = pd.read_csv(DP / "obs_final.csv", parse_dates=["date"])
    fin["stem_all_DW"] = fin["stem_DW"] + fin["truss_struct_DW"].fillna(0)
    ev = pd.read_csv(DP / "truss_events.csv", parse_dates=["app", "anth", "set"])
    # 화방별 과실 DW (주 평균) — 수확 영향 없는 2023-12-28 까지, 1–8화방
    tcols = {f"T{j}_fruit_DW": j for j in range(1, 9)}
    tf = de[de["date"] <= TRUSS_FRUIT_END].groupby("date")[list(tcols)].mean().rename(columns=tcols)
    obs.attrs["truss_fruit"] = tf.stack().rename("obs")
    return clim, hr, obs, fin, ev


def r0_rule(obs, fin):
    lp = obs["leaves_present"].dropna()
    lp.loc[pd.Timestamp("2024-01-11")] = fin["leaves_present"].mean()
    return M.ObservedTrajectory(pd.Series(lp.index), lp)


VARS = {"leaf_DW": "leaf_dw", "stem_all_DW": "stem_dw", "leaf_area_cm2": "leaf_area_m2"}


def truss_fruit_sim(plant):
    tl = pd.DataFrame(plant.truss_log, columns=["date", "truss", "sim"])
    return tl.set_index(["date", "truss"])["sim"]


def prod_error(s, obs, detail=False, plant=None):
    """잎·줄기·엽면적 (일자 평균) + 화방별 과실 DW (화방×일자) 정규화 RMSE 합"""
    out, tot = {}, 0.0
    if plant is not None:
        j = pd.concat([truss_fruit_sim(plant), obs.attrs["truss_fruit"]], axis=1, join="inner").dropna()
        rm = float(np.sqrt(((j["sim"] - j["obs"]) ** 2).mean()))
        nr = rm / j["obs"].mean()
        out["truss_fruit_DW"] = (rm, nr)
        tot += nr
    for o, m in VARS.items():
        sim = s[m] * (1e4 if m == "leaf_area_m2" else 1)
        j = pd.concat([sim, obs[o]], axis=1, join="inner").dropna()
        rm = float(np.sqrt(((j.iloc[:, 0] - j.iloc[:, 1]) ** 2).mean()))
        nr = rm / j.iloc[:, 1].mean()
        out[o] = (rm, nr)
        tot += nr
    return (tot, out) if detail else tot


PROD = [("tau_struct", 0.5, 1.0), ("leaf_area_pot", 0.04, 0.10),
        ("fruit_W_pot", 5.0, 25.0), ("fruit_TT_growth", 200.0, 680.0)]   # π 고정 (비식별, sens_rd.py)


def fit_production(clim, hr, obs, fin, base: Params):
    rule = r0_rule(obs, fin)

    def f(x):
        kw = {k: float(np.clip(v, lo, hi)) for (k, lo, hi), v in zip(PROD, x)}
        s, pl = simulate(clim, replace(base, **kw), rule, START, END, hourly=hr, **INIT)
        return prod_error(s, obs, plant=pl)

    x0 = [getattr(base, k) for k, _, _ in PROD]
    r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=800, xatol=1e-3, fatol=1e-4))
    kw = {k: float(np.clip(v, lo, hi)) for (k, lo, hi), v in zip(PROD, r.x)}
    return replace(base, **kw), r


HARVEST_WIN = (pd.Timestamp("2023-12-28"), pd.Timestamp("2024-01-03"))   # 1·2화방 소실 구간 (중간 파괴조사)


def ripe_check(clim, hr, obs, fin, p: Params, trusses=(1, 2, 3)):
    """공통 TT_color로 함안 화방별 숙기일 → 관측 소실 창(12/28–1/03, 늦은 쪽 경계)과 비교"""
    _, pl = simulate(clim, p, r0_rule(obs, fin), START, "2024-01-25", hourly=hr, **INIT)
    tt = truss_table(pl).set_index("truss")["ripe"]
    return {k: (str(tt[k].date()) if k in tt.index and pd.notna(tt[k]) else None) for k in trusses}


def set_dates_obs(ev):
    """화방별 관측 개화→착과 기간 (일, 개체 평균) — C1이 직접 겨냥하는 양"""
    return ev.dropna(subset=["anth_set_d"]).groupby("truss")["anth_set_d"].mean()


def set_error(p, clim, hr, rule, obs_lag):
    s, plant = simulate(clim, p, rule, START, END, hourly=hr, **INIT)
    tt = truss_table(plant).set_index("truss")
    lag = (tt["set"] - tt["anthesis"]).dt.days
    d = []
    for k, o in obs_lag.items():
        if k in lag.index and pd.notna(lag[k]):
            d.append(lag[k] - o)
        elif k in tt.index and pd.notna(tt.loc[k, "anthesis"]):     # 개화했으나 기간 내 미착과 → 하한 벌점
            d.append(max((pd.Timestamp(END) - tt.loc[k, "anthesis"]).days - o, 0) + 7)
    d = np.array(d, float)
    return float(np.sqrt((d ** 2).mean())), d


def fit_clock(clim, hr, obs, fin, ev, p_prod: Params):
    rule = r0_rule(obs, fin)
    obs_set = set_dates_obs(ev)
    res = {}
    # A: 온도만
    pa = replace(p_prod, use_source_clock=False)
    ra = minimize_scalar(lambda v: set_error(replace(pa, TT_anth_set=v), clim, hr, rule, obs_set)[0],
                         bounds=(20, 200), method="bounded")
    pa = replace(pa, TT_anth_set=float(ra.x))
    res["A_temp"] = (pa, *set_error(pa, clim, hr, rule, obs_set), 1)
    # B: 온도 × g(Q/D)
    pb = replace(p_prod, use_source_clock=True)

    def mk(x):
        return replace(pb, TT_anth_set=float(np.clip(x[0], 20, 200)), g_a=float(np.clip(x[1], 0.2, 20)),
                       g_b=float(np.clip(x[2], -0.5, 0.95)))
    best = None
    for x0 in ([60.0, 2.39, 0.12], [60.0, 5.0, 0.3], [50.0, 1.0, 0.0]):      # 다중 시작점
        r = minimize(lambda x: set_error(mk(x), clim, hr, rule, obs_set)[0], x0,
                     method="Nelder-Mead", options=dict(maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    pb = mk(best.x)
    res["B_temp_x_source"] = (pb, *set_error(pb, clim, hr, rule, obs_set), 3)
    return res


def aic(d, k):
    n = len(d)
    return n * np.log((d ** 2).sum() / n) + 2 * k


if __name__ == "__main__":
    clim, hr, obs, fin, ev = load()
    base = Params()

    s0, pl0 = simulate(clim, base, r0_rule(obs, fin), START, END, hourly=hr, **INIT)
    print("보정 전:", {k: f"{v[1]:.2f}" for k, v in prod_error(s0, obs, True, pl0)[1].items()}, "(nRMSE)")

    p1, r = fit_production(clim, hr, obs, fin, base)
    s1, pl1 = simulate(clim, p1, r0_rule(obs, fin), START, END, hourly=hr, **INIT)
    tot, det = prod_error(s1, obs, True, pl1)
    print("\n[1단계 생산 보정]", "반복", r.nit)
    for k, _, _ in PROD:
        print(f"  {k:14s} = {getattr(p1, k):.4g}")
    for k, (rm, nr) in det.items():
        print(f"  {k:14s} RMSE={rm:8.2f}  nRMSE={nr:.2f}")

    print("\n관측 화방별 개화→착과 기간(일):", set_dates_obs(ev).round(1).to_dict())
    print("\n[2단계 생식 시계 — C1 검정] 화방별 개화→착과 기간")
    res = fit_clock(clim, hr, obs, fin, ev, p1)
    # 생산 ↔ 생식 시계 교대 보정 (착과 시점이 과실 DM에 영향 → 서로 일관되게). TT_color = 공통 기본값
    for it in range(1, N_ALT + 1):
        pB = res["B_temp_x_source"][0]
        p1, r = fit_production(clim, hr, obs, fin, pB)
        res = fit_clock(clim, hr, obs, fin, ev, p1)
        s1, pl1 = simulate(clim, res["B_temp_x_source"][0], r0_rule(obs, fin), START, END, hourly=hr, **INIT)
        tot, det = prod_error(s1, obs, True, pl1)
        print(f"  교대 {it}: 생산 목적함수 {tot:.3f} ({', '.join(f'{k} {v[1]:.2f}' for k, v in det.items())}) | "
              + ", ".join(f"{k}={getattr(p1, k):.4g}" for k, _, _ in PROD))
    for name, (p, rmse, d, k) in res.items():
        print(f"  {name:16s} RMSE={rmse:5.2f} d  AIC={aic(d, k):6.1f}  TT_anth_set={p.TT_anth_set:.1f}  g_a={p.g_a:.2f} g_b={p.g_b:.2f}  잔차={np.round(d).astype(int).tolist()}")

    best = res["B_temp_x_source"][0]
    print(f"  공통 TT_color={best.TT_color:.0f} °C d → 함안 숙기 {ripe_check(clim, hr, obs, fin, best)} "
          f"(관측 1·2화방 소실 {HARVEST_WIN[0].date()}–{HARVEST_WIN[1].date()} = 늦은 쪽 경계)")
    out = {k: getattr(best, k) for k in ("tau_struct", "leaf_area_pot", "fruit_W_pot", "fruit_TT_growth", "fruit_priority",
                                         "TT_anth_set", "g_a", "g_b")}             # TT_color는 공통 기본값 (params.py)
    out["TT_anth_set_tempOnly"] = res["A_temp"][0].TT_anth_set
    (DP / "params_m2.json").write_text(json.dumps(out, indent=2))
    print("\n저장:", DP / "params_m2.json")
