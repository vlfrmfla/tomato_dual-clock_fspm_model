"""
비파괴조사 → 화방×개체 단위 페놀로지 이벤트 + 생식 시계 가설(C1) 검정

이벤트 (조사일 기준, 3–7일 간격이라 구간 중도절단):
  app  : 화방 출현 (화방개수 > 0 첫 조사일)
  anth : 개화     (전개화방 > 0 첫 조사일)
  set  : 착과     (착과수 > 0 첫 조사일)

출력: data_processed/truss_events.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
TBASE = 10.0


def load_climate():
    c = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"]).set_index("date")
    c["dTT"] = (c["Tmean"] - TBASE).clip(lower=0)
    c["TT"] = c["dTT"].cumsum()
    return c


def build_events(c):
    nd = pd.read_csv(DP / "obs_nondes.csv", parse_dates=["date"])
    rows = []
    for p, g in nd.groupby("plant"):
        g = g.sort_values("date")
        for t in range(1, 15):
            def first(col):
                if col not in g:
                    return pd.NaT
                s = g.loc[g[col] > 0, "date"]
                return s.min() if len(s) else pd.NaT
            rows.append(dict(truss=t, plant=p,
                             app=first(f"T{t}_flowers"),
                             anth=first(f"T{t}_open"),
                             set=first(f"T{t}_set")))
    d = pd.DataFrame(rows).dropna(subset=["app"])
    tt = c["TT"]
    for a, b in (("app", "anth"), ("anth", "set")):
        ok = d[a].notna() & d[b].notna() & (d[b] >= d[a])
        d.loc[ok, f"{a}_{b}_d"] = (d.loc[ok, b] - d.loc[ok, a]).dt.days
        d.loc[ok, f"{a}_{b}_TT"] = tt.reindex(d.loc[ok, b]).values - tt.reindex(d.loc[ok, a]).values
    # 개화 후 7일 환경 (착과 결정 구간)
    d["Rg7"] = d["anth"].apply(lambda x: c["Rg_out_MJ"][x:x + pd.Timedelta(days=6)].mean() if pd.notna(x) else np.nan)
    d["T7"] = d["anth"].apply(lambda x: c["Tmean"][x:x + pd.Timedelta(days=6)].mean() if pd.notna(x) else np.nan)
    d.to_csv(DP / "truss_events.csv", index=False)
    return d


def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + X)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    se = np.sqrt(np.diag(e @ e / (n - k) * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(abs(b / se), n - k)
    r2 = 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))
    for nm, bb, ss, pp in zip(["const"] + names, b, se, p):
        print(f"  {nm:6s} b={bb:8.2f}  se={ss:6.2f}  p={pp:.1e}")
    print(f"  R2={r2:.2f}  n={n}")


if __name__ == "__main__":
    c = load_climate()
    d = build_events(c)
    print(d.groupby("truss")[["app_anth_d", "app_anth_TT", "anth_set_d", "anth_set_TT", "Rg7", "T7"]]
          .mean().round(1).to_string())

    s = d.dropna(subset=["anth_set_TT", "Rg7"])
    print("\n[개화→착과, °Cd] 단변량 Spearman")
    for x in ("Rg7", "T7", "truss"):
        r, p = stats.spearmanr(s[x], s["anth_set_TT"])
        print(f"  vs {x:5s} rho={r:+.2f} p={p:.1e} n={len(s)}")
    print("\n[개화→착과, °Cd] ~ Rg7 + truss")
    ols(s["anth_set_TT"].values, [s["Rg7"].values, s["truss"].values], ["Rg7", "truss"])
    rx = s["Rg7"] - np.polyval(np.polyfit(s["truss"], s["Rg7"], 1), s["truss"])
    ry = s["anth_set_TT"] - np.polyval(np.polyfit(s["truss"], s["anth_set_TT"], 1), s["truss"])
    r, p = stats.spearmanr(rx, ry)
    print(f"  partial rho(lag, Rg7 | truss) = {r:+.2f}  p={p:.1e}")

    a = d.dropna(subset=["app_anth_TT", "Rg7"])
    r, p = stats.spearmanr(a["Rg7"], a["app_anth_TT"])
    print(f"\n[출현→개화, °Cd] vs Rg7  rho={r:+.2f} p={p:.2f} n={len(a)}  (대조: 구조 시계)")

    nd = pd.read_csv(DP / "obs_nondes.csv", parse_dates=["date"])
    gap = pd.Series(sorted(nd["date"].unique())).diff().dt.days
    print(f"\n조사 간격: 중앙값 {gap.median():.0f}일, 최대 {gap.max():.0f}일 (이벤트 시점 해상도)")
