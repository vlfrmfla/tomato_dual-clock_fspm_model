"""
C1 개체 단위 교차검증 (leave-one-plant-out, 8겹)

각 겹: 7주로 A(온도만, TT_anth_set) / B(온도×g(Q/D), TT_anth_set·g_a·g_b) 보정
       → 남긴 1주의 화방별 개화→착과 기간(일) 예측
지표: 겹 합산 예측 RMSE, MAE, 개체별 RMSE 짝비교(Wilcoxon)
생산 파라미터는 params_m2.json 고정 (파괴조사로 보정 — 비파괴 개체와 독립)
출력: data_processed/cv_c1.csv
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import INIT, START, END, load, r0_rule        # noqa: E402
from fspm import Params, simulate, truss_table                 # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"


def sim_lag(p, clim, hr, rule):
    _, pl = simulate(clim, p, rule, START, END, hourly=hr, **INIT)
    tt = truss_table(pl).set_index("truss")
    lag = (tt["set"] - tt["anthesis"]).dt.days.astype(float)
    # 기간 내 미착과 → 종료일까지 + 7 (하한 벌점)
    miss = tt["anthesis"].notna() & tt["set"].isna()
    lag[miss] = (pd.Timestamp(END) - tt.loc[miss, "anthesis"]).dt.days + 7
    return lag


def err(lag, obs):
    """obs: DataFrame(truss, anth_set_d) 개체 단위"""
    pred = lag.reindex(obs["truss"]).to_numpy()
    return obs["anth_set_d"].to_numpy() - pred


def fit_A(base, clim, hr, rule, tr):
    f = lambda v: np.sqrt(np.nanmean(err(sim_lag(replace(base, TT_anth_set=v), clim, hr, rule), tr) ** 2))
    r = minimize_scalar(f, bounds=(20, 200), method="bounded", options=dict(xatol=0.5))
    return replace(base, TT_anth_set=float(r.x))


def fit_B(base, clim, hr, rule, tr, x0s):
    def mk(x):
        return replace(base, TT_anth_set=float(np.clip(x[0], 20, 200)), g_a=float(np.clip(x[1], 0.2, 20)),
                       g_b=float(np.clip(x[2], -0.5, 0.95)))
    f = lambda x: np.sqrt(np.nanmean(err(sim_lag(mk(x), clim, hr, rule), tr) ** 2))
    best = None
    for x0 in x0s:
        r = minimize(f, x0, method="Nelder-Mead",
                     options=dict(maxiter=200, xatol=0.05, fatol=0.01, initial_simplex=np.array(x0) * np.array(
                         [[1, 1, 1], [1.3, 1, 1], [1, 1.5, 1], [1, 1, 1]]) + np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0.15]])))
        if best is None or r.fun < best.fun:
            best = r
    return mk(best.x)


if __name__ == "__main__":
    clim, hr, obs, fin, ev = load()
    pj = json.loads((DP / "params_m2.json").read_text())
    p0 = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    rule = r0_rule(obs, fin)
    data = ev.dropna(subset=["anth_set_d"])[["plant", "truss", "anth_set_d"]]

    rows = []
    for k in sorted(data["plant"].unique()):
        tr, te = data[data["plant"] != k], data[data["plant"] == k]
        pa = fit_A(replace(p0, use_source_clock=False), clim, hr, rule, tr)
        pb = fit_B(replace(p0, use_source_clock=True), clim, hr, rule, tr,
                   x0s=([60.0, 2.39, 0.12], [80.0, 1.0, 0.0], [50.0, 5.0, 0.3]))   # 전체자료 보정값 미사용 (누설 방지)
        ea = err(sim_lag(pa, clim, hr, rule), te)
        eb = err(sim_lag(pb, clim, hr, rule), te)
        for (t, a, b) in zip(te["truss"], ea, eb):
            rows.append(dict(plant=k, truss=t, err_A=a, err_B=b))
        print(f"  개체 {k}: n={len(te)}  RMSE A={np.sqrt(np.mean(ea**2)):.2f}  B={np.sqrt(np.mean(eb**2)):.2f}  "
              f"(B: TT={pb.TT_anth_set:.0f}, a={pb.g_a:.2f}, b={pb.g_b:.2f})", flush=True)

    cv = pd.DataFrame(rows)
    cv.to_csv(DP / "cv_c1.csv", index=False)
    ra, rb = np.sqrt((cv["err_A"] ** 2).mean()), np.sqrt((cv["err_B"] ** 2).mean())
    print(f"\n[LOPO-CV, n={len(cv)}] 예측 RMSE  A 온도만 {ra:.2f} d  |  B 온도×Q/D {rb:.2f} d")
    print(f"                       MAE   A {cv['err_A'].abs().mean():.2f} d  |  B {cv['err_B'].abs().mean():.2f} d")
    per = pd.DataFrame({"A": cv.groupby("plant")["err_A"].apply(lambda e: np.sqrt((e ** 2).mean())),
                        "B": cv.groupby("plant")["err_B"].apply(lambda e: np.sqrt((e ** 2).mean()))})
    w = wilcoxon(per["A"], per["B"], alternative="greater")
    print(f"  개체별 RMSE: B < A 인 개체 {int((per.B < per.A).sum())}/{len(per)},  Wilcoxon(A>B) p={w.pvalue:.3f}")
    w2 = wilcoxon(cv["err_A"].abs(), cv["err_B"].abs(), alternative="greater")
    print(f"  관측 단위 |오차| 짝비교 Wilcoxon(A>B) p={w2.pvalue:.4f}")
