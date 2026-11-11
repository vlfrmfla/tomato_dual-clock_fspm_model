"""
구조 시계 마디 출현 간격 검정: 누적 마디 수 (비파괴 8주 × 27회)

  constant  : φ                                        (1 파라미터)
  two_phase : φ1, φ2, 정식 후 전환 열시간 TT_sw         (3) — 이전 기본 (대조)
  qd        : φ = φ1 · min(Q/D_ema, 1)^(−β)             (2) — 일별 source 제한 시 엽 출현 지연
  load      : φ = φ1 + (φ2 − φ1) · min(1, F/F*)         (3) — 생장 중 과실 수 F에 따른 포화형 전환 (현재 기본)
비교: RMSE, AIC (일자 평균 n=27). load 적합값은 params.py 기본값으로 옮겨 씀 (자동 저장 없음)
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import INIT, START, END, load, r0_rule        # noqa: E402
from fspm import Params, simulate                              # noqa: E402
from fspm.phenology import fruits_growing                      # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"


def run(p, clim, hr, rule, nodes_obs):
    s, _ = simulate(clim, p, rule, START, END, hourly=hr, **INIT)
    j = pd.concat([s["nodes"], nodes_obs], axis=1, join="inner").dropna()
    return j.iloc[:, 0] - j.iloc[:, 1]


def aic(e, k):
    n = len(e)
    return n * np.log((e ** 2).sum() / n) + 2 * k


if __name__ == "__main__":
    clim, hr, obs, fin, ev = load()
    pj = json.loads((DP / "params_m2.json").read_text())
    p0 = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    rule = r0_rule(obs, fin)
    nd = pd.read_csv(DP / "obs_nondes.csv", parse_dates=["date"])
    nodes_obs = nd.groupby("date")["nodes_cum"].mean()

    models = {
        "constant": (lambda x: replace(p0, phyllo_mode="constant", phyllochron=x[0]), [26.0]),
        "two_phase": (lambda x: replace(p0, phyllo_mode="two_phase", phyllochron_early=x[0],
                                        phyllochron_late=x[1], phyllo_switch_TT=x[2]), [22.4, 30.0, 675.0]),
        "qd": (lambda x: replace(p0, phyllo_mode="qd", phyllochron_early=x[0], phyllo_qd_beta=x[1]), [22.0, 0.5]),
        "load": (lambda x: replace(p0, phyllo_mode="load", phyllochron_early=x[0], phyllochron_late=x[1],
                                   phyllo_load_F=max(x[2], 0.5)), [p0.phyllochron_early, p0.phyllochron_late, p0.phyllo_load_F]),
    }
    fits = {}
    starts = {"load": [[18.0, 28.0, 8.0], [18.0, 28.0, 14.0]]}           # 과실 수 계단 → 다중 시작점
    for name, (mk, x0) in models.items():
        r = None
        for xs in [x0] + starts.get(name, []):
            ri = minimize(lambda x: float((run(mk(x), clim, hr, rule, nodes_obs) ** 2).sum()), xs,
                          method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-2, fatol=1e-3))
            if r is None or ri.fun < r.fun:
                r = ri
        e = run(mk(r.x), clim, hr, rule, nodes_obs)
        fits[name] = mk(r.x)
        print(f"  {name:10s} k={len(x0)}  RMSE={np.sqrt((e**2).mean()):.2f} 마디  AIC={aic(e, len(x0)):6.1f}  "
              f"params={np.round(r.x, 3).tolist()}")
    # 현재 기본값(params.py)의 적합도와 load 전환 시점
    e = run(p0, clim, hr, rule, nodes_obs)
    print(f"  default({p0.phyllo_mode}) φ1 {p0.phyllochron_early} φ2 {p0.phyllochron_late} F* {p0.phyllo_load_F}: "
          f"RMSE={np.sqrt((e**2).mean()):.2f}  AIC={aic(e, 3):6.1f}")
    pl = fits["load"]
    trace = []
    from fspm import model as MD
    from fspm import phenology as PH
    orig = MD.step_structure

    def traced(plant, dtt, tt_tp, date, p, qd=1.0):
        trace.append((date, tt_tp, fruits_growing(plant, p)))
        return orig(plant, dtt, tt_tp, date, p, qd=qd)
    MD.step_structure = traced
    run(pl, clim, hr, rule, nodes_obs)
    MD.step_structure = orig
    tr = pd.DataFrame(trace, columns=["date", "tt", "F"])
    hit = tr[tr.F >= pl.phyllo_load_F]
    half = tr[tr.F >= pl.phyllo_load_F / 2]
    print(f"  load: F ≥ F*/2 첫날 {half.date.iloc[0] if len(half) else '-'} (정식 후 {half.tt.iloc[0] if len(half) else np.nan:.0f} °C d), "
          f"F ≥ F* 첫날 {hit.date.iloc[0] if len(hit) else '-'} (정식 후 {hit.tt.iloc[0] if len(hit) else np.nan:.0f} °C d), "
          f"F 최대 {tr.F.max():.1f}")
