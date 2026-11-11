"""
착과 시점 평가 보완 (리뷰 대응)

(1) 구간 검열(interval-censored) 평가
    개화·착과일은 '처음 양성으로 관측된 조사일'이므로 실제 날짜는 (직전 조사일, 첫 양성 조사일] 구간에 있다.
    → 개화→착과 기간 D ∈ [L, U],  L = (직전 착과 조사일 + 1) − 첫 개화 관측일,  U = 첫 착과 관측일 − (직전 개화 조사일 + 1)
    → 예측이 구간 안이면 오차 0, 밖이면 가까운 경계까지 거리. LOPO-CV 예측(cv_c1.csv: err = 관측 − 예측)을 그대로 사용
    첫 조사일(10/5)에 이미 양성인 사건은 직전 조사일을 7 d 전(최대 조사 간격)으로 가정
(2) 반복 관측(식물당 여러 화방)을 고려한 일사 효과
    함안: anth_set_TT ~ Rg7 + truss + (1 | plant)        (선형 혼합모형, statsmodels MixedLM, ML)
    AI Hub: lag_TT ~ Rg7 + truss + greenhouse + (1 | plant)
    + 비교용 OLS (식물 군집 강건 표준오차)
출력: data_processed/fruitset_interval.csv, fruitset_eval_log.txt, 표준출력
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
FIRST_GAP = 7


def interval_eval():
    ev = pd.read_csv(DP / "truss_events.csv", parse_dates=["anth", "set"]).dropna(subset=["anth", "set"])
    nd = pd.read_csv(DP / "obs_nondes.csv", parse_dates=["date"])
    cv = pd.read_csv(DP / "cv_c1.csv")
    surveys = {p: np.array(sorted(g["date"].unique()), dtype="datetime64[D]") for p, g in nd.groupby("plant")}

    def prev(p, d):
        s = surveys[p]
        before = s[s < np.datetime64(d.date())]
        return pd.Timestamp(before[-1]) if len(before) else d - pd.Timedelta(days=FIRST_GAP)

    rows = []
    for r in ev.itertuples():
        pa, ps = prev(r.plant, r.anth), prev(r.plant, r.set)
        L = (ps + pd.Timedelta(days=1) - r.anth).days
        U = (r.set - (pa + pd.Timedelta(days=1))).days
        rows.append(dict(plant=r.plant, truss=r.truss, obs=(r.set - r.anth).days, L=max(L, 0), U=U))
    d = pd.DataFrame(rows).merge(cv, on=["plant", "truss"], how="inner")
    for m in ("A", "B"):
        pred = d["obs"] - d[f"err_{m}"]
        d[f"pred_{m}"] = pred
        d[f"eint_{m}"] = np.where(pred < d["L"], d["L"] - pred, np.where(pred > d["U"], pred - d["U"], 0.0))
        d[f"in_{m}"] = (d[f"eint_{m}"] == 0).astype(int)
    d.to_csv(DP / "fruitset_interval.csv", index=False)
    per = d.groupby("plant").apply(lambda g: pd.Series({m: np.sqrt((g[f"eint_{m}"] ** 2).mean()) for m in ("A", "B")}))
    w = wilcoxon(per["A"], per["B"], alternative="greater")
    res = dict(n=len(d), width_median=float((d["U"] - d["L"]).median()), width_mean=float((d["U"] - d["L"]).mean()),
               rmse_point_A=float(np.sqrt((d["err_A"] ** 2).mean())), rmse_point_B=float(np.sqrt((d["err_B"] ** 2).mean())),
               rmse_int_A=float(np.sqrt((d["eint_A"] ** 2).mean())), rmse_int_B=float(np.sqrt((d["eint_B"] ** 2).mean())),
               within_A=float(d["in_A"].mean()), within_B=float(d["in_B"].mean()),
               plants_B_better=int((per["B"] < per["A"]).sum()), plants=int(len(per)), p_wilcoxon=float(w.pvalue))
    return res, d


def mixed_models():
    out = {}
    ev = pd.read_csv(DP / "truss_events.csv").dropna(subset=["anth_set_TT", "Rg7"])
    ev["plant"] = ev["plant"].astype(str)
    m = smf.mixedlm("anth_set_TT ~ Rg7 + truss", ev, groups=ev["plant"]).fit(reml=False)
    o = smf.ols("anth_set_TT ~ Rg7 + truss", ev).fit(cov_type="cluster", cov_kwds={"groups": ev["plant"]})
    out["haman"] = dict(n=len(ev), plants=ev["plant"].nunique(), b_mixed=m.params["Rg7"], se_mixed=m.bse["Rg7"], p_mixed=m.pvalues["Rg7"],
                        b_truss_mixed=m.params["truss"], p_truss_mixed=m.pvalues["truss"],
                        b_ols_cl=o.params["Rg7"], p_ols_cl=o.pvalues["Rg7"])
    ex = pd.read_csv(DP / "external_aihub_events.csv").dropna(subset=["lag_TT", "Rg7"])
    ex["gid"] = ex["farm"] + "_" + ex["plant"].astype(str)
    m2 = smf.mixedlm("lag_TT ~ Rg7 + truss + C(farm)", ex, groups=ex["gid"]).fit(reml=False)
    o2 = smf.ols("lag_TT ~ Rg7 + truss + C(farm)", ex).fit(cov_type="cluster", cov_kwds={"groups": ex["gid"]})
    out["aihub"] = dict(n=len(ex), plants=ex["gid"].nunique(), b_mixed=m2.params["Rg7"], se_mixed=m2.bse["Rg7"], p_mixed=m2.pvalues["Rg7"],
                        b_ols_cl=o2.params["Rg7"], p_ols_cl=o2.pvalues["Rg7"])
    # 온실별 (혼합모형)
    for farm, g in ex.groupby("farm"):
        mf = smf.mixedlm("lag_TT ~ Rg7 + truss", g, groups=g["gid"]).fit(reml=False)
        out[f"aihub_{farm}"] = dict(n=len(g), plants=g["gid"].nunique(), b_mixed=mf.params["Rg7"], p_mixed=mf.pvalues["Rg7"],
                                    b_truss=mf.params["truss"], p_truss=mf.pvalues["truss"])
    return out


if __name__ == "__main__":
    log = []
    P = lambda s: (print(s), log.append(s))
    res, d = interval_eval()
    P("[1] 구간 검열 평가 (LOPO-CV 예측, n = %d 화방)" % res["n"])
    P(f"  관측 구간 폭 중앙값 {res['width_median']:.0f} d (평균 {res['width_mean']:.1f} d)")
    P(f"  점 RMSE        A {res['rmse_point_A']:.2f}  B {res['rmse_point_B']:.2f} d")
    P(f"  구간 RMSE      A {res['rmse_int_A']:.2f}  B {res['rmse_int_B']:.2f} d")
    P(f"  구간 안 비율   A {res['within_A']:.0%}  B {res['within_B']:.0%}")
    P(f"  개체별 구간 RMSE B < A: {res['plants_B_better']}/{res['plants']}, Wilcoxon(A>B) p = {res['p_wilcoxon']:.3f}")
    mm = mixed_models()
    P("[2] 일사 효과 (반복 관측 고려)")
    for k, v in mm.items():
        P(f"  {k:12s} " + "  ".join(f"{a}={b:.3g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    (DP / "fruitset_eval_log.txt").write_text("\n".join(log))
