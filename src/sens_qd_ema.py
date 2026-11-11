"""
Q/D 이동평균의 일 값 상한(3)과 시작값(1) 민감도 (2026-11-07, 원고 검토 질문 "상한과 초기값을 고정해도 되는가")

(1) 파라미터 고정 — 상한 2, 2.5, 3, 4, 없음 (시작값 1) / 시작값 0.5, 1.5, 2, 첫날 값 (상한 3)
    함안: 화방 1–10 평균 개화→착과 기간 RMSE (보정 목적함수), 화방별 착과일 최대 변화, 생산 목적함수, 과실 DW,
          모델 C (cv_radiation_summary.csv 보정값)
    횡계: 과실 총 DW 편차, 첫 수확 차이 (R1 20장)
(2) 재보정 — 같은 설정에서 B의 TT_set, a, b를 다시 보정 (생산 파라미터 고정) → RMSE, 파라미터
(3) 시나리오 — base 변형, 세 작기 N95 (scenario_season과 같은 설정)
출력: data_processed/sens_qd_ema.csv, sens_qd_ema_refit.csv, sens_qd_ema_n95.csv
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import INIT, START, END, load, r0_rule, set_dates_obs, set_error, prod_error   # noqa: E402
from fspm import Params, simulate, truss_table                   # noqa: E402
from fspm import management as M                                 # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"
INF = 1e9
SETTINGS = [("base (cap 3, init 1)", 3.0, 1.0), ("cap 2", 2.0, 1.0), ("cap 2.5", 2.5, 1.0), ("cap 4", 4.0, 1.0),
            ("no cap", INF, 1.0), ("init 0.5", 3.0, 0.5), ("init 1.5", 3.0, 1.5), ("init 2", 3.0, 2.0),
            ("init = first day", 3.0, np.nan)]
REFIT = ["cap 2", "no cap", "init 0.5", "init 2", "init = first day"]


def p_haman():
    pj = json.loads((DP / "params_m2.json").read_text())
    return replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})


def set_dates(p, clim, hr, rule):
    _, pl = simulate(clim, p, rule, START, END, hourly=hr, **INIT)
    return truss_table(pl).set_index("truss")["set"]


def fixed():
    import run_hoenggye as RH
    clim, hr, obs, fin, ev = load()
    rule, obs_set = r0_rule(obs, fin), set_dates_obs(ev)
    pb = p_haman()
    cs = pd.read_csv(DP / "cv_radiation_summary.csv").set_index("model").loc["C"]
    pc = replace(pb, clock_signal="rad", TT_anth_set=float(cs.TT_set), g_a=float(cs.a), g_b=float(cs.b))
    c2, h2, o2, lc2 = RH.load()
    ph = json.loads((DP / "hg/params_hg.json").read_text())
    phg = replace(RH.base_params(), **{k: v for k, v in ph.items() if k in Params.__dataclass_fields__})
    rows, base_set = [], None
    for name, cap, init in SETTINGS:
        kw = dict(qd_ema_cap=cap, qd_ema_init=init)
        p = replace(pb, **kw)
        rmse, _ = set_error(p, clim, hr, rule, obs_set)
        s, pl = simulate(clim, p, rule, START, END, hourly=hr, **INIT)
        sd = truss_table(pl).set_index("truss")["set"]
        base_set = sd if base_set is None else base_set
        dd = (sd - base_set.reindex(sd.index)).dt.days.abs()
        rmse_c, _ = set_error(replace(pc, **kw), clim, hr, rule, obs_set)
        p2 = replace(phg, **kw)
        s2, _ = RH.sim(c2, h2, o2, p2, M.KeepN(20, trigger=RH.START))
        ev2, _ = RH.evaluate(s2, o2, lc2)
        rows.append(dict(setting=name, cap=cap if cap < INF else np.inf, init=init,
                         set_rmse_B=rmse, max_set_shift_d=float(dd.max()), trusses_shifted=int((dd > 0).sum()),
                         prod_obj=prod_error(s, obs, plant=pl), fruit_total_end=float((s.fruit_dw + s.harvested_dw).iloc[-1]),
                         set_rmse_C=rmse_c, hg_fruit_bias_pct=ev2["fruit_total_DW_bias%"],
                         hg_first_harvest_lag_d=ev2["first_harvest_lag_d"], hg_harv_final=ev2["harv_final_sim"]))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def refit_one(name):
    cap, init = next((c, i) for n, c, i in SETTINGS if n == name)
    clim, hr, obs, fin, ev = load()
    rule, obs_set = r0_rule(obs, fin), set_dates_obs(ev)
    pb = replace(p_haman(), qd_ema_cap=cap, qd_ema_init=init)

    def mk(x):
        return replace(pb, TT_anth_set=float(np.clip(x[0], 20, 200)), g_a=float(np.clip(x[1], 0.2, 20)),
                       g_b=float(np.clip(x[2], -0.5, 0.95)))
    best = None
    for x0 in ([60.0, 2.39, 0.12], [60.0, 5.0, 0.3], [50.0, 1.0, 0.0], [pb.TT_anth_set, pb.g_a, pb.g_b]):
        r = minimize(lambda x: set_error(mk(x), clim, hr, rule, obs_set)[0], x0, method="Nelder-Mead", options=dict(maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    p = mk(best.x)
    return dict(setting=name, set_rmse_refit=float(best.fun), TT_set=p.TT_anth_set, a=p.g_a, b=p.g_b)


def n95_one(name):
    import scenario_season as S
    import run_hoenggye as RH
    cap, init = next((c, i) for n, c, i in SETTINGS if n == name)
    clim_ha = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"]).set_index("date")
    hr_ha = pd.read_csv(DP / "climate_hourly.csv", parse_dates=["datetime"]).set_index("datetime")
    p_ha = p_haman()
    c2, h2, o2, _ = RH.load()
    ph = json.loads((DP / "hg/params_hg.json").read_text())
    p_hg = replace(RH.base_params(), **{k: v for k, v in ph.items() if k in Params.__dataclass_fields__})
    out = {}
    for season, (start, a, b) in S.SEASONS.items():
        st = pd.Timestamp(start)
        end = (pd.Timestamp(b) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        hg = season == S.HG_SEASON
        clim, hr, p0 = (c2, h2, p_hg) if hg else (clim_ha, hr_ha, p_ha)
        p = replace(p0, rd_frac=0.65 / p_ha.Vcmax25, rd_age_mode="cap", qd_ema_cap=cap, qd_ema_init=init,
                    transplant=RH.START_TP if hg else (st - pd.Timedelta(days=17)).strftime("%Y-%m-%d"))
        init_kw = dict(plant0=RH.initial_plant(c2, p, o2)) if hg else dict(leaf_dw0=10.0, stem_dw0=4.0)
        trigger = a if hg else (st + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        g = {}
        for n in S.NS:
            s, _ = simulate(clim, p, M.KeepN(n, trigger=trigger), start, end, hourly=hr, **init_kw)
            g[n] = S.window_metrics(s, a, b, hr, p)["fruit_gain"]
        fmax = max(g.values())
        out[season] = min(n for n, v in g.items() if v >= 0.95 * fmax)
        out[season + "_fruit30"] = g[30]
    return dict(setting=name, **out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    with ProcessPoolExecutor(max_workers=8) as ex:
        fut_r = {n: ex.submit(refit_one, n) for n in REFIT}
        fut_n = {n: ex.submit(n95_one, n) for n in ["base (cap 3, init 1)", "cap 2", "no cap", "init 0.5", "init 2", "init = first day"]}
        F = fixed()
        F.to_csv(DP / "sens_qd_ema.csv", index=False)
        print("\n[1] 파라미터 고정\n", F.round(3).to_string(index=False), flush=True)
        N = pd.DataFrame([f.result() for f in fut_n.values()])
        N.to_csv(DP / "sens_qd_ema_n95.csv", index=False)
        print("\n[3] 시나리오 N95\n", N.round(1).to_string(index=False), flush=True)
        R = pd.DataFrame([f.result() for f in fut_r.values()])
        R.to_csv(DP / "sens_qd_ema_refit.csv", index=False)
        print("\n[2] B 재보정\n", R.round(3).to_string(index=False), flush=True)
