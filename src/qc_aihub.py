"""
AI Hub 토마토 자료(6농가) 정합성 점검 — 외부 검증에 tom2·tom3만 쓴 근거

점검
  (1) 환경 자료: 농가별 조사 기간 중 기온·일사 기록이 있는 시간 비율, 기록 간격, 범위 이상값
  (2) 조사 일정: 개체별 조사 간격, 조사일별 개체 수
  (3) 라벨 일관성: 중복, 개화 화방번호 역행, 화방별 착과수 감소, 화방번호·착과마디 불일치
  (4) 사건 도출: 개화→착과 쌍, 음수·40일 초과 간격, 농가별 간격 분포
출력: 표준출력 표, figures/qc/aihub_qc.png·pdf, data_processed/qc_aihub_summary.csv
"""
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data_external/aihub"
FARMS = ["tom1", "tom2", "tom3", "tom4", "tom5", "tom6"]
LAG_MAX = 40
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6})


def _names(zf):
    return {(n.encode("cp437").decode("cp949", "replace") if not n.isascii() else n): n for n in zf.namelist()}


def _csv(zf, name, **kw):
    raw = zf.read(name)
    for enc in ("utf-8", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, skipinitialspace=True, low_memory=False, **kw)
        except UnicodeDecodeError:
            continue


def load_climate():
    """농가별 시각·기온·일사 (원 기록 간격 그대로)"""
    zf = zipfile.ZipFile(EXT / "TS_Timeseries.zip")
    nm = _names(zf)
    out = {}
    for farm in FARMS:
        if farm in ("tom2", "tom3"):
            f = [k for k in nm if k.startswith(farm) and "환경데이터" in k][0]
            d = _csv(zf, nm[f])
            t = pd.to_datetime(d["datetime"])
            T = d[[c for c in d.columns if "grh temp" in c]].mean(axis=1)
            R = d[[c for c in d.columns if c.startswith(("meas radiation", "radiation:"))][0]]
            src = "Priva climate computer (5 min)"
        elif farm == "tom1":
            d = _csv(zf, nm["tom1/01_(함수율저울,열화상센서,작물환경센서)/MV.csv"])
            t = pd.to_datetime(d["time"])
            T, R = d["내부PT100온도센서1번건구"], d["내부광량"]
            src = "crop sensor unit (inside light)"
        else:
            fs = sorted(k for k in nm if k.startswith(farm) and k.endswith("Gdata.csv"))
            d = pd.concat([_csv(zf, nm[f]) for f in fs], ignore_index=True)
            d.columns = [c.strip() for c in d.columns]
            t = pd.to_datetime(d["날짜"].astype(str).str.strip() + " " + d["시간"].astype(str).str.strip(),
                               format="%Y-%m-%d %H:%M:%S", errors="coerce")
            T = pd.to_numeric(d["실내온도(도)"], errors="coerce")
            R = pd.to_numeric(d["현재일사(W)"], errors="coerce")
            src = "irrigation controller (Oct–Jan files)"
        x = pd.DataFrame({"T": pd.to_numeric(T, errors="coerce").to_numpy(),
                          "R": pd.to_numeric(R, errors="coerce").to_numpy()}, index=t).sort_index()
        out[farm] = (x[x.index.notna()], src)
    return out


def labels():
    b = pd.read_csv(EXT / "labels.csv", dtype={"plant": str, "truss": str, "date": str})
    c = pd.read_csv(EXT / "labels_c.csv", dtype={"plant": str, "truss": str, "date": str})
    for d in (b, c):
        d["date"] = pd.to_datetime(d["date"])
        d["truss_n"] = pd.to_numeric(d["truss"], errors="coerce")
    return b[b["cat"].str.startswith("b")].copy(), c


def main():
    clim = load_climate()
    b, c = labels()
    rows, cover = [], {}
    for farm in FARMS:
        x, src = clim[farm]
        bf, cf = b[b.farm == farm], c[c.farm == farm]
        d0, d1 = min(bf.date.min(), cf.date.min()), max(bf.date.max(), cf.date.max())
        hrs = pd.date_range(d0, d1 + pd.Timedelta(hours=23), freq="h")
        okT = (x["T"] > 0) & (x["T"] < 50)                 # tom4–6 실내온도는 전 기간 0.0 → 결측
        okT = okT.groupby(x.index.floor("h")).any().reindex(hrs, fill_value=False)
        okR = x["R"].between(0, 1500).groupby(x.index.floor("h")).any().reindex(hrs, fill_value=False)
        cover[farm] = okT.groupby(okT.index.floor("D")).mean().combine(
            okR.groupby(okR.index.floor("D")).mean(), min)
        step = x.index.to_series().diff().dt.total_seconds().median() / 60
        # 조사 일정
        gap = cf.sort_values("date").groupby("plant")["date"].apply(lambda s: s.drop_duplicates().diff().dt.days).dropna()
        per_date = cf.groupby("date")["plant"].nunique()
        # 라벨 일관성
        mx = bf.groupby(["plant", "date"])["truss_n"].max().reset_index().sort_values("date")
        back = int((mx.groupby("plant")["truss_n"].diff() < 0).sum())
        # 주의: numberOfFruitPerTruss는 화방번호와 같고, 화방별 착과수는 numberOfFruitPerPlant에 있음
        g = cf.groupby(["plant", "truss", "date"])["numberOfFruitPerPlant"]
        conflict = int((g.nunique() > 1).sum())
        cnt = g.max().reset_index().sort_values("date")
        drop = cnt.groupby(["plant", "truss"])["numberOfFruitPerPlant"].diff()
        n_drop, n_pairs_c = int((drop < 0).sum()), int(drop.notna().sum())
        field_is_truss = (pd.to_numeric(cf["numberOfFruitPerTruss"], errors="coerce") == cf["truss_n"]).mean()
        # 사건
        anth = bf.groupby(["plant", "truss_n"])["date"].min().rename("anth")
        setd = cf[cf["numberOfFruitPerPlant"] > 0].groupby(["plant", "truss_n"])["date"].min().rename("set")
        e = pd.concat([anth, setd], axis=1)
        both = e.dropna()
        lag = (both["set"] - both["anth"]).dt.days
        rows.append(dict(farm=farm, climate_source=src, record_min=round(step, 1),
                         days_complete=f"{(cover[farm] >= 0.9).mean():.0%}",
                         T_all_zero=bool((x["T"].fillna(0) == 0).all()),
                         survey_gap_d=f"{gap.median():.0f} ({gap.quantile(.1):.0f}–{gap.quantile(.9):.0f})",
                         plants_per_date=f"{per_date.median():.0f}", n_c_records=len(cf), images_per_key=round(len(cf) / g.ngroups, 1),
                         conflicting_keys=f"{conflict} / {g.ngroups}",
                         truss_backwards=back, fruit_count_drops=f"{n_drop} / {n_pairs_c}",
                         perTruss_field_is_truss=f"{field_is_truss:.0%}", trusses_anth=int(e["anth"].notna().sum()),
                         pairs=len(both), lag_neg=int((lag < 0).sum()), lag_gt40=int((lag > LAG_MAX).sum()),
                         lag_median_d=f"{lag[(lag >= 0) & (lag <= LAG_MAX)].median():.0f}"))
        cover[farm + "_lag"] = lag
    s = pd.DataFrame(rows).set_index("farm")
    pd.set_option("display.width", 250)
    print(s.T.to_string())
    s.to_csv(ROOT / "data_processed/qc_aihub_summary.csv")
    d2, d3 = (clim[f][0].resample("D").mean() for f in ("tom2", "tom3"))
    both = pd.concat([d2["R"], d3["R"]], axis=1).dropna()
    print("\n일사 교차 점검 tom2 vs tom3 일평균: r = %.3f, 평균비 %.2f" % (both.corr().iloc[0, 1], both.iloc[:, 0].mean() / both.iloc[:, 1].mean()))
    zf = zipfile.ZipFile(EXT / "TS_Timeseries.zip"); nm = _names(zf)
    t2 = _csv(zf, nm[[k for k in nm if k.startswith("tom2") and "환경데이터" in k][0]])
    tc = t2[[c for c in t2.columns if "grh temp" in c]]
    print("tom2 구획 3곳 기온 차 (최대−최소) 중앙값 %.2f K, 95%% %.2f K" % ((tc.max(axis=1) - tc.min(axis=1)).median(), (tc.max(axis=1) - tc.min(axis=1)).quantile(.95)))
    ev = pd.read_csv(ROOT / "data_processed/external_aihub_events.csv")
    print("\n외부 검증에 쓴 사건:", len(ev), "쌍,", ev.groupby("farm")["plant"].nunique().to_dict(), "개체",
          "| 개화→착과 중앙값(일):", ev.groupby("farm")["lag_d"].median().to_dict())

    # 그림: (a) 환경 기록 완결도, (b) 개화→착과 간격 분포
    fig, ax = plt.subplots(1, 2, figsize=(180 / 25.4, 62 / 25.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for i, farm in enumerate(FARMS):
        cv = cover[farm]
        ax[0].scatter(cv.index, np.full(len(cv), i), c=cv.values, cmap="Greys", vmin=0, vmax=1,
                      marker="s", s=9, linewidths=0)
    ax[0].set_yticks(range(6), FARMS)
    ax[0].set_ylim(5.6, -0.6)
    ax[0].xaxis.set_major_locator(mdates.MonthLocator())
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[0].set_xlim(pd.Timestamp("2021-09-01"), pd.Timestamp("2022-02-01"))
    ax[0].set_xlabel("Date (2021–2022)")
    ax[0].set_title("a  Share of hours with air temperature and radiation", loc="left", fontsize=8)
    data = [cover[f + "_lag"].clip(-20, 80).values for f in FARMS]
    ax[1].boxplot(data, vert=False, widths=0.6, flierprops=dict(markersize=2), medianprops=dict(color="k"))
    ax[1].axvspan(0, LAG_MAX, color="0.9", zorder=0)
    ax[1].set_yticks(range(1, 7), FARMS)
    ax[1].set_ylim(6.6, 0.4)
    ax[1].set_xlim(-20, 80)
    ax[1].set_xticks(range(-20, 81, 20))
    ax[1].set_xlabel("Anthesis to fruit set (d)")
    ax[1].set_title("b  Labeled anthesis-to-set intervals", loc="left", fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "figures/qc"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"aihub_qc.{ext}", dpi=300)


if __name__ == "__main__":
    main()
