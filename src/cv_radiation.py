"""
대조 모델 C: 온도 × g(일사) — source–sink 항의 효과를 일사 효과와 분리 (내부 검토 2026-11-03, 6번)

C는 B와 같은 식·같은 자유 파라미터(TT_set, a, b)·같은 5일 이동평균(상한 3, 초기값 1)을 쓰고,
신호만 Q/D 대신 외부 일사 / 10 MJ m⁻² d⁻¹ 로 바꾼다 (clock_signal="rad").
(1) 전체 자료 보정: 화방 1–10 평균 개화→착과 기간 RMSE, AIC, AICc (A·B와 같은 방식, calibrate_m2.fit_clock)
(2) LOPO-CV: cv_c1과 같은 겹·시작점 → 점 오차, 관측 구간 기준 오차(eval_fruitset과 같은 L, U)
출력: data_processed/cv_radiation.csv (개체·화방별 err_A/B/C), cv_radiation_summary.csv, 표준출력
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import load, r0_rule, set_dates_obs, set_error, aic    # noqa: E402
import cv_c1 as CV                                                        # noqa: E402
import eval_fruitset as EF                                                # noqa: E402
from fspm import Params                                                   # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"
X0 = ([60.0, 2.0, 0.3], [80.0, 1.0, 0.0], [50.0, 5.0, 0.5])


def aicc(d, k):
    n = len(d)
    return aic(d, k) + 2 * k * (k + 1) / (n - k - 1)


def main():
    clim, hr, obs, fin, ev = load()
    pj = json.loads((DP / "params_m2.json").read_text())
    p0 = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    rule = r0_rule(obs, fin)
    obs_set = set_dates_obs(ev)

    # (1) 전체 자료: A·B는 보정값 그대로, C는 B와 같은 절차로 보정
    pa = replace(p0, use_source_clock=False, TT_anth_set=pj["TT_anth_set_tempOnly"])
    pb = replace(p0, use_source_clock=True)
    pc0 = replace(p0, use_source_clock=True, clock_signal="rad")

    def mk(x):
        return replace(pc0, TT_anth_set=float(np.clip(x[0], 20, 200)), g_a=float(np.clip(x[1], 0.2, 20)),
                       g_b=float(np.clip(x[2], -0.5, 0.95)))
    best = None
    for x0 in X0:
        r = minimize(lambda x: set_error(mk(x), clim, hr, rule, obs_set)[0], x0, method="Nelder-Mead", options=dict(maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    pc = mk(best.x)
    full = []
    for name, p, k in (("A", pa, 1), ("B", pb, 3), ("C", pc, 3)):
        rmse, d = set_error(p, clim, hr, rule, obs_set)
        full.append(dict(model=name, k=k, rmse=rmse, AIC=aic(d, k), AICc=aicc(d, k),
                         TT_set=p.TT_anth_set, a=p.g_a if k == 3 else np.nan, b=p.g_b if k == 3 else np.nan))
        print(f"[전체] {name}: RMSE {rmse:.2f} d  AIC {aic(d, k):.1f}  AICc {aicc(d, k):.1f}  TT_set {p.TT_anth_set:.1f}"
              + (f"  a {p.g_a:.2f}  b {p.g_b:.2f}" if k == 3 else ""), flush=True)

    # (2) LOPO-CV (A·B는 cv_c1.csv 재사용)
    data = ev.dropna(subset=["anth_set_d"])[["plant", "truss", "anth_set_d"]]
    rows = []
    for k in sorted(data["plant"].unique()):
        tr, te = data[data["plant"] != k], data[data["plant"] == k]
        pck = CV.fit_B(pc0, clim, hr, rule, tr, x0s=X0)
        ec = CV.err(CV.sim_lag(pck, clim, hr, rule), te)
        rows += [dict(plant=k, truss=t, err_C=e) for t, e in zip(te["truss"], ec)]
        print(f"  개체 {k}: RMSE C = {np.sqrt(np.mean(ec ** 2)):.2f}  (TT {pck.TT_anth_set:.0f}, a {pck.g_a:.2f}, b {pck.g_b:.2f})", flush=True)
    cv = pd.read_csv(DP / "cv_c1.csv").merge(pd.DataFrame(rows), on=["plant", "truss"])
    cv.to_csv(DP / "cv_radiation.csv", index=False)

    # 관측 구간 (eval_fruitset과 같은 L, U)
    _, iv = EF.interval_eval()
    d = iv[["plant", "truss", "obs", "L", "U"]].merge(cv, on=["plant", "truss"])
    summ = {}
    for m in ("A", "B", "C"):
        pred = d["obs"] - d[f"err_{m}"]
        d[f"eint_{m}"] = np.where(pred < d["L"], d["L"] - pred, np.where(pred > d["U"], pred - d["U"], 0.0))
        summ[m] = dict(rmse=np.sqrt((d[f"err_{m}"] ** 2).mean()), mae=d[f"err_{m}"].abs().mean(),
                       within=(d[f"eint_{m}"] == 0).mean(), rmse_int=np.sqrt((d[f"eint_{m}"] ** 2).mean()))
        print(f"[LOPO-CV] {m}: RMSE {summ[m]['rmse']:.2f}  MAE {summ[m]['mae']:.2f}  구간 안 {summ[m]['within']:.0%}  "
              f"구간 RMSE {summ[m]['rmse_int']:.2f}")
    per = d.groupby("plant").apply(lambda g: pd.Series({m: np.sqrt((g[f"err_{m}"] ** 2).mean()) for m in "ABC"}))
    within = d.groupby("plant").apply(lambda g: pd.Series({m: (g[f"eint_{m}"] == 0).mean() for m in "ABC"}))
    tests = {}
    for x, y in (("A", "C"), ("C", "B")):
        tests[f"{x}>{y}_rmse"] = wilcoxon(per[x], per[y], alternative="greater").pvalue
        print(f"  개체별 RMSE {y} < {x}: {int((per[y] < per[x]).sum())}/8, Wilcoxon p = {tests[f'{x}>{y}_rmse']:.3f} | "
              f"구간 안 비율 {y} > {x}: {int((within[y] > within[x]).sum())}/8, 같음 {int((within[y] == within[x]).sum())}")
    out = pd.DataFrame(full).set_index("model")
    for m in "ABC":
        for k2, v in summ[m].items():
            out.loc[m, f"cv_{k2}"] = v
    out.to_csv(DP / "cv_radiation_summary.csv")
    print(out.round(3).to_string())
    print(tests)


if __name__ == "__main__":
    main()
