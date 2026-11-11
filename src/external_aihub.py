"""
AI Hub 097 (2021–22, 농가 tom2·tom3) 외부 검증 — 이중 시계 가설

자료: data_external/aihub/labels.csv (b 개화군), labels_c.csv (c 착과군), TS_Timeseries.zip (Priva 환경)
  개화일 = 개체·화방별 b 라벨(floweringNode)에 처음 나타난 날
  착과일 = 같은 화방의 c 라벨 착과수 > 0 첫 날
  조사 간격 ~7일 (A7-6은 ~4일)
검정
  (1) 생식 시계: 개화→착과 °Cd ~ 개화 후 7일 일사(Rg7) [+ 화방순서, 농가] — A7-6과 같은 형식
  (2) 구조 시계: 개체별 개화 화방번호 ~ 누적 °Cd 기울기 → °Cd/화방, 전·후반 비교
출력: data_processed/external_aihub_events.csv, 표준출력
"""
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data_external/aihub"
DP = ROOT / "data_processed"
TBASE = 10.0
LAG_MAX = 40          # 일 — 이보다 긴 개화→착과는 라벨 오류로 간주 (tom2 10월 평균 46일 등)

TEMP_COLS = {"tom2": ["meas grh temp:Cmp 1 (°C)", "meas grh temp:Cmp 2 (°C)", "meas grh temp:Cmp 3 (°C)"],
             "tom3": ["meas grh temp:CMP (°C)"]}
RAD_COL = {"tom2": "meas radiation:WEATHER RAD SENS (W/m²)", "tom3": "radiation:WEATHER RAD SENS (W/m²)"}


def load_env(farm):
    zf = zipfile.ZipFile(EXT / "TS_Timeseries.zip")
    raw = [n for n in zf.namelist() if n.startswith(farm)
           and "환경데이터" in n.encode("cp437").decode("cp949", "replace") and n.endswith(".csv")][0]
    d = pd.read_csv(io.BytesIO(zf.read(raw)), parse_dates=["datetime"]).set_index("datetime")
    T = d[TEMP_COLS[farm]].mean(axis=1)
    out = pd.DataFrame({"Tmean": T.resample("D").mean(),
                        "Rg_MJ": d[RAD_COL[farm]].resample("D").mean() * 86400 / 1e6})
    out["dTT"] = (out["Tmean"] - TBASE).clip(lower=0)
    out["TT"] = out["dTT"].cumsum()
    return out


def events():
    lab = pd.read_csv(EXT / "labels.csv", dtype={"plant": str, "truss": str, "date": str})
    c = pd.read_csv(EXT / "labels_c.csv", dtype={"plant": str, "truss": str, "date": str})
    for d in (lab, c):
        d["date"] = pd.to_datetime(d["date"])
        d["truss"] = pd.to_numeric(d["truss"], errors="coerce")
    b = lab[lab["cat"].str.startswith("b")]
    anth = b.groupby(["farm", "plant", "truss"])["date"].min().rename("anth")
    setd = c[c["numberOfFruitPerPlant"] > 0].groupby(["farm", "plant", "truss"])["date"].min().rename("set")
    e = pd.concat([anth, setd], axis=1).dropna().reset_index()
    e["lag_d"] = (e["set"] - e["anth"]).dt.days
    return e, b


def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + X)
    bb = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ bb
    n, k = X.shape
    se = np.sqrt(np.diag(r @ r / (n - k) * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(abs(bb / se), n - k)
    return {nm: (b_, p_) for nm, b_, p_ in zip(["const"] + names, bb, p)}, 1 - r @ r / ((y - y.mean()) @ (y - y.mean()))


if __name__ == "__main__":
    e, b = events()
    rows = []
    for farm in ("tom2", "tom3"):
        env = load_env(farm)
        x = e[(e["farm"] == farm) & (e["lag_d"] >= 0) & (e["lag_d"] <= LAG_MAX)].copy()
        x = x[x["anth"] >= env.index.min()]
        x["lag_TT"] = env["TT"].reindex(x["set"]).to_numpy() - env["TT"].reindex(x["anth"]).to_numpy()
        x["Rg7"] = [env["Rg_MJ"][a:a + pd.Timedelta(days=6)].mean() for a in x["anth"]]
        x["T7"] = [env["Tmean"][a:a + pd.Timedelta(days=6)].mean() for a in x["anth"]]
        rows.append(x.dropna(subset=["lag_TT", "Rg7"]))

        # 구조 시계: 개체별 개화 화방번호 ~ TT
        bb = b[b["farm"] == farm].groupby(["plant", "date"])["truss"].max().reset_index()
        bb = bb[bb["date"] >= env.index.min()]
        bb["TT"] = env["TT"].reindex(bb["date"]).to_numpy()
        slopes, early, late = [], [], []
        mid = env.index.min() + (env.index.max() - env.index.min()) / 2
        for p, g in bb.dropna().groupby("plant"):
            if g["date"].nunique() >= 5 and g["truss"].nunique() >= 3:
                s = np.polyfit(g["TT"], g["truss"], 1)[0]
                if s > 0:
                    slopes.append(1 / s)
                for part, store in ((g[g["date"] < mid], early), (g[g["date"] >= mid], late)):
                    if part["truss"].nunique() >= 2 and len(part) >= 3:
                        sp = np.polyfit(part["TT"], part["truss"], 1)[0]
                        if sp > 0:
                            store.append(1 / sp)
        sl = np.array(slopes)
        print(f"[{farm}] 구조 시계: 화방 개화 간격 중앙값 {np.median(sl):.0f} °Cd/화방 (IQR {np.percentile(sl, 25):.0f}–{np.percentile(sl, 75):.0f}, n={len(sl)}주)"
              f" | 전반 {np.median(early):.0f} vs 후반 {np.median(late):.0f} °Cd  (Mann-Whitney p={stats.mannwhitneyu(early, late).pvalue:.3f})"
              f" | 평균기온 {env['Tmean'].mean():.1f} °C")

    d = pd.concat(rows, ignore_index=True)
    d.to_csv(DP / "external_aihub_events.csv", index=False)
    print(f"\n[생식 시계] 개화→착과 쌍 n={len(d)} (tom2 {int((d.farm == 'tom2').sum())}, tom3 {int((d.farm == 'tom3').sum())}), "
          f"중앙값 {d['lag_d'].median():.0f}일 / {d['lag_TT'].median():.0f} °Cd")
    for farm, g in [("전체", d)] + list(d.groupby("farm")):
        r, p = stats.spearmanr(g["Rg7"], g["lag_TT"])
        rt, pt = stats.spearmanr(g["truss"], g["lag_TT"])
        print(f"  {farm:4s} lag_TT vs Rg7 rho={r:+.2f} (p={p:.1e}) | vs 화방순서 rho={rt:+.2f} (p={pt:.1e}) | n={len(g)}")
    co, r2 = ols(d["lag_TT"].to_numpy(float),
                 [d["Rg7"].to_numpy(float), d["truss"].to_numpy(float), (d["farm"] == "tom3").to_numpy(float)],
                 ["Rg7", "truss", "farm_tom3"])
    print("  OLS lag_TT ~ Rg7 + truss + farm:", {k: f"b={v[0]:+.2f}, p={v[1]:.1e}" for k, v in co.items()}, f"R2={r2:.2f}")
    # 출현→개화 대조는 AI Hub에 화방 출현 조사가 없어 불가
