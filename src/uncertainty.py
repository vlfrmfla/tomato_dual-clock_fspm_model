"""
파라미터 불확실성 — 보정값이 넓게 퍼져도 결론(C1 우세, N95)이 유지되는지

(A) 생식 시계 프로파일: TT_set 격자(40–120 °C d) 고정 → g_a, g_b 재적합. RMSE·AIC·실효 반응 g(0.6)/g(1.5)
(B) 생산 파라미터 다중 시작: PROD 4개, 시작점 6개 (Nelder–Mead), 생식 시계는 최종값 고정
(C) 전파: (A) ΔAIC ≤ 2 인 점 × (B) 목적함수 ≤ 최적 × 1.05 인 해 → 겨울·봄 N95 (기준 변형)
출력: data_processed/uncertainty_clock.csv, uncertainty_prod.csv, uncertainty_n95.csv, 표준출력
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_m2 as C                              # noqa: E402
from fspm import Params, simulate                     # noqa: E402
from fspm import management as M                      # noqa: E402

DP = C.DP
rng = np.random.default_rng(1)


def g(x, a, b):
    return 1 - np.exp(-a * (x - b))


def aic(rmse, n, k):
    return n * np.log(rmse ** 2) + 2 * k


def main():
    clim, hr, obs, fin, ev = C.load()
    pj = json.loads((DP / "params_m2.json").read_text())
    p0 = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    rule = C.r0_rule(obs, fin)
    obs_set = C.set_dates_obs(ev)
    n = len(obs_set)

    if "--n95-only" in sys.argv:          # (A)·(B)는 보정 기간만 쓰므로 저장값 재사용, (C)만 다시 계산
        A = pd.read_csv(DP / "uncertainty_clock.csv")
        B = pd.read_csv(DP / "uncertainty_prod.csv")
        return propagate(clim, hr, p0, A, B)

    # (A) 생식 시계 프로파일
    rows = []
    for tt in range(40, 121, 10):
        def f(x):
            p = replace(p0, use_source_clock=True, TT_anth_set=float(tt), g_a=float(np.clip(x[0], 0.2, 20)),
                        g_b=float(np.clip(x[1], -0.5, 0.95)))
            return C.set_error(p, clim, hr, rule, obs_set)[0]
        best = None
        for x0 in ([1.5, 0.1], [0.8, 0.0], [3.0, 0.3], [p0.g_a, p0.g_b]):      # 마지막 = 보정값 (계단형 목적함수의 국소 최소 방지)
            r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=200, xatol=1e-3, fatol=1e-3))
            best = r if best is None or r.fun < best.fun else best
        a, b = float(np.clip(best.x[0], 0.2, 20)), float(np.clip(best.x[1], -0.5, 0.95))
        rows.append(dict(TT_set=tt, g_a=a, g_b=b, rmse=best.fun, aic=aic(best.fun, n, 3),
                         rate_ratio=g(0.6, a, b) / g(1.5, a, b), dur_ratio=g(1.5, a, b) / g(0.6, a, b)))
        print(f"  TT_set {tt:4d}: a {a:.2f} b {b:+.2f}  RMSE {best.fun:.2f} d  rate(0.6)/rate(1.5) {rows[-1]['rate_ratio']:.2f}", flush=True)
    A = pd.DataFrame(rows)
    A["dAIC"] = A["aic"] - A["aic"].min()
    A.to_csv(DP / "uncertainty_clock.csv", index=False)
    rmse_A = C.set_error(replace(p0, use_source_clock=False, TT_anth_set=pj["TT_anth_set_tempOnly"]),
                         clim, hr, rule, obs_set)[0]
    print(f"[A] 온도만 RMSE {rmse_A:.2f} d (AIC {aic(rmse_A, n, 1):.1f}); ΔAIC ≤ 2 인 TT_set: "
          f"{A[A.dAIC <= 2].TT_set.tolist()}  (최소 AIC {A.aic.min():.1f})", flush=True)

    if "--reuse-prod" in sys.argv:        # (B)는 생식 시계와 무관 → 저장값 재사용
        return propagate(clim, hr, p0, A, pd.read_csv(DP / "uncertainty_prod.csv"))

    # (B) 생산 다중 시작
    rows = []
    starts = [[getattr(p0, k) for k, _, _ in C.PROD]]
    for _ in range(5):
        starts.append([rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)) if k != "tau_struct" else rng.uniform(0.75, 1.0)
                       for k, lo, hi in C.PROD])
    for i, x0 in enumerate(starts):
        def f(x):
            kw = {k: float(np.clip(v, lo, hi)) for (k, lo, hi), v in zip(C.PROD, x)}
            s, pl = simulate(clim, replace(p0, **kw), rule, C.START, C.END, hourly=hr, **C.INIT)
            return C.prod_error(s, obs, plant=pl)
        r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-3, fatol=1e-4))
        kw = {k: float(np.clip(v, lo, hi)) for (k, lo, hi), v in zip(C.PROD, r.x)}
        rows.append(dict(start=i, obj=r.fun, **kw))
        print(f"  start {i}: obj {r.fun:.3f}  " + "  ".join(f"{k} {v:.4g}" for k, v in kw.items()), flush=True)
    B = pd.DataFrame(rows)
    B.to_csv(DP / "uncertainty_prod.csv", index=False)
    return propagate(clim, hr, p0, A, B)


def propagate(clim, hr, p0, A, B):
    # (C) 전파: N95 (겨울·봄, 기준 변형)
    import scenario_season as S
    seasons = {k: v for k, v in S.SEASONS.items() if k != S.HG_SEASON}
    okA = A[A.dAIC <= 2]
    okB = B[B.obj <= B.obj.min() * 1.05]
    rows = []
    sets = [("final", {})]
    sets += [(f"clock TT_set={int(r.TT_set)}", dict(TT_anth_set=float(r.TT_set), g_a=r.g_a, g_b=r.g_b)) for r in okA.itertuples()]
    sets += [(f"prod start {int(r.start)}", {k: getattr(r, k) for k, _, _ in C.PROD}) for r in okB.itertuples()]
    for name, kw in sets:
        p = replace(p0, **kw)
        out = {"set": name}
        for season, (start, a, b) in seasons.items():
            st = pd.Timestamp(start)
            end = (pd.Timestamp(b) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            pp = replace(p, transplant=(st - pd.Timedelta(days=17)).strftime("%Y-%m-%d"))
            fr = {}
            for N in S.NS:
                s, _ = simulate(clim, pp, M.KeepN(N, trigger=(st + pd.Timedelta(days=40)).strftime("%Y-%m-%d")),
                                start, end, hourly=hr, leaf_dw0=10.0, stem_dw0=4.0)
                fr[N] = S.window_metrics(s, a, b, hr, pp)["fruit_gain"]
            fm = max(fr.values())
            out[f"N95_{season}"] = min(N for N, v in fr.items() if v >= 0.95 * fm)
        rows.append(out)
        print("  ", out, flush=True)
    Cdf = pd.DataFrame(rows)
    Cdf.to_csv(DP / "uncertainty_n95.csv", index=False)
    print("[C] N95 범위:", {c: (int(Cdf[c].min()), int(Cdf[c].max())) for c in Cdf.columns if c.startswith("N95")})


if __name__ == "__main__":
    main()
