"""
원자료(A7-6, 2023-24 RedKan) → 모델 보정/검증용 정리 테이블

출력 (data_processed/):
  climate_hourly.csv / climate_daily.csv  — Cmp 6 온실 기상 (loadcell 온도와 r=0.979로 실험 구획 확인)
  obs_nondes.csv     — 비파괴조사, 주·날짜별 (누적 발생 엽수, 화방별 화수/개화/착과)
  obs_des.csv        — 파괴조사, 주·날짜별 (현재 엽수, 엽면적, 기관별 FW/DW)
  obs_final.csv      — 최종 파괴조사 (16주)
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "A7-6_tomato_redcan"
OUT = ROOT / "data_processed"
OUT.mkdir(exist_ok=True)

N_TRUSS = 14
FINAL_DATE = pd.Timestamp("2024-01-11")   # 최종생육조사 (16주)


def sheet_date(name: str) -> pd.Timestamp:
    """'2023.10.5' / '10.23' / '01.02' → Timestamp (작기: 2023-09 ~ 2024-03)"""
    p = name.strip().split(".")
    if len(p) == 3:
        y, m, d = map(int, p)
    else:
        m, d = map(int, p)
        y = 2024 if m < 9 else 2023
    return pd.Timestamp(y, m, d)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------------------
# 1. 기상
# ---------------------------------------------------------------------------
CTRL = RAW / "env_control_priva"
ERA5_PATH = ROOT / "data_external/era5/haman_hourly.csv"
CO2_FILL = 420.0     # 10/25 이전은 필터 문제로 미기록(사용자 확인)·결측 → 외기 수준 가정


def _read_priva(path):
    d = pd.read_csv(path, sep=";", skiprows=[1, 2], na_values=[""])
    d = d.rename(columns={d.columns[0]: "datetime"})
    d["datetime"] = pd.to_datetime(d["datetime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    return d.dropna(subset=["datetime"]).set_index("datetime").sort_index()


def load_controller() -> pd.DataFrame:
    """
    Priva 내보내기 (Haman-RDA, CMP 6) → 시간 평균
      curt1 : CURTAIN 1 개도 %  — 차광 (고일사 시 한낮 부분 폐쇄: 외부 >800 W/m²에서 평균 73%)
      curt2 : CURTAIN 2 개도 %  — 보온 (야간 폐쇄)
      curt3 : CURTAIN 3 개도 %  — 보온 (야간 폐쇄)
      CO2   : ppm, 300–3000 범위 밖은 센서 이상으로 결측 처리
    """
    c = _read_priva(CTRL / "A6_2308-2508_2.csv")
    g = _read_priva(CTRL / "A6_2308-2508_3.csv")
    out = pd.DataFrame({
        "curt1": num(c["calc curtain 1"]),
        "curt2": num(c["calc curtain 2"]),
        "curt3": num(c["calc curtain 3"]),
        "CO2": num(g["meas CO2 conc"]),
    })
    out.loc[(out["CO2"] < 300) | (out["CO2"] > 3000), "CO2"] = np.nan
    return out.resample("h").mean()
def build_climate():
    # 열 이름이 중복되므로 위치로 지정 (0: 시각, 12: WEATHER RAD SENS, 13-15: Cmp 6 T/HD/RH)
    df = pd.read_csv(RAW / "A6_2309-2409.csv", skiprows=[1, 2])
    df.columns = ["datetime"] + [f"c{i}" for i in range(1, df.shape[1])]
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    clim = pd.DataFrame({
        "T": num(df["c13"]),        # Cmp 6 기온 (°C)
        "HD": num(df["c14"]),       # Cmp 6 포차 (g/m3)
        "RH": num(df["c15"]),       # Cmp 6 상대습도 (%)
        "Rg_out": num(df["c12"]),   # 외부 일사 (W/m2)
    })
    clim = clim.dropna(how="all")
    hourly = clim.resample("h").mean()
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"))
    hourly["gapfill"] = 0

    # --- 결측 보간 (2023-10-14 ~ 10-23 등) ---
    # T, RH: 로드셀 4대 평균 센서 → Cmp 6에 선형 회귀 보정
    lc = []
    for i in range(1, 5):
        l = pd.read_csv(ROOT / "data_analysis/data/loadcell" / f"dev_{i}.csv",
                        usecols=["datetime", "temp", "humidity"], dtype={"datetime": str})
        l["datetime"] = pd.to_datetime(l["datetime"], format="%Y%m%d%H%M")
        lc.append(l.set_index("datetime").resample("h").mean())
    lc = pd.concat(lc).groupby(level=0).mean().reindex(hourly.index)
    gap = hourly["T"].isna() & lc["temp"].notna()
    for col, src in (("T", "temp"), ("RH", "humidity")):
        ok = hourly[col].notna() & lc[src].notna()
        a, b = np.polyfit(lc.loc[ok, src], hourly.loc[ok, col], 1)
        hourly.loc[gap, col] = a * lc.loc[gap, src] + b
    hourly["RH"] = hourly["RH"].clip(upper=100)
    hourly.loc[gap, "gapfill"] = 1
    # 로드셀 센서도 없는 나머지 결측(2024-03-27 10시 ~ 03-28 19시, 34 h): 같은 시각 ±7일 평균
    #   (채우지 않으면 그날 광합성·Q/D가 NaN → Q/D 이동평균이 NaN으로 남아 이후 착과가 멈춤)
    rest_t = hourly["T"].isna()
    for col in ("T", "RH"):
        hod = hourly[col].groupby(hourly.index.hour).transform(lambda s: s.rolling(15, center=True, min_periods=3).mean())
        hourly.loc[hourly[col].isna(), col] = hod[hourly[col].isna()]
    hourly.loc[rest_t & hourly["T"].notna(), "gapfill"] = 3
    # Rg_out 결측: 같은 좌표 ERA5 단파 (Open-Meteo, 시각 = 구간 끝 → 1 h 앞당김)
    #   검증(결측 없는 9/20–1/31): 시간 r 0.980, 일 합 비율 0.998. 기상청 ASOS 자료가 생기면 ERA5_PATH 대신 사용
    #   ERA5 없는 시간만 같은 시각 ±7일 평균
    rg_gap = hourly["Rg_out"].isna()
    era = pd.read_csv(ERA5_PATH, parse_dates=["time"]).set_index("time")["shortwave_radiation"].shift(-1)
    hourly.loc[rg_gap, "Rg_out"] = era.reindex(hourly.index)[rg_gap]
    rest = hourly["Rg_out"].isna()
    clim_hod = hourly["Rg_out"].groupby(hourly.index.hour).transform(
        lambda s: s.rolling(15, center=True, min_periods=3).mean())
    hourly.loc[rest, "Rg_out"] = clim_hod[rest]
    hourly.loc[rg_gap, "gapfill"] = 2

    # --- Priva 제어기 (Cmp 6): 커튼 3종 개도, CO2 ---
    ctrl = load_controller().reindex(hourly.index)
    for col in ("curt1", "curt2", "curt3"):
        # 결측(10/13–10/26 등)은 복원하지 않음 → 스크린 열림(0) 가정 (사용자 결정 2026-10-05)
        ctrl[col] = ctrl[col].fillna(0.0)
    hourly[["curt1", "curt2", "curt3"]] = ctrl[["curt1", "curt2", "curt3"]]
    hourly["CO2"] = ctrl["CO2"]
    hourly["co2_filled"] = hourly["CO2"].isna().astype(int)
    hourly["CO2"] = hourly["CO2"].fillna(CO2_FILL)

    # --- 지붕 유리 투과율 (CAD: 남북동 벤로 27.9°, 유리 5 mm) ---
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fspm.greenhouse import roof_glass_transmittance
    geo = roof_glass_transmittance(hourly.index, hourly["Rg_out"].to_numpy())
    hourly[["sun_elev", "sun_az", "kd", "tau_glass"]] = geo[["sun_elev", "sun_az", "kd", "tau_glass"]]

    es = 0.6108 * np.exp(17.27 * hourly["T"] / (hourly["T"] + 237.3))
    hourly["VPD"] = es * (1 - hourly["RH"] / 100)  # kPa
    hourly.index.name = "datetime"
    hourly.to_csv(OUT / "climate_hourly.csv")

    g = hourly.resample("D")
    daily = pd.DataFrame({
        "Tmean": g["T"].mean(),
        "Tmax": g["T"].max(),
        "Tmin": g["T"].min(),
        "RH": g["RH"].mean(),
        "VPD": g["VPD"].mean(),
        "Rg_out_MJ": g["Rg_out"].sum() * 3600 / 1e6,   # 시간평균 W/m2 → MJ/m2/d
        "gapfill": g["gapfill"].max(),
    })
    daily.index.name = "date"
    daily.to_csv(OUT / "climate_daily.csv")
    return daily


# ---------------------------------------------------------------------------
# 2. 비파괴조사
# ---------------------------------------------------------------------------
def build_nondes():
    xl = pd.ExcelFile(RAW / "A7-6 토마토 비파괴조사.xlsx")
    rows = []
    for s in xl.sheet_names:
        date = sheet_date(s)
        df = xl.parse(s, header=None, index_col=0)
        df.index = df.index.map(lambda x: str(x).strip())
        plants = num(df.loc["처리번호"]).dropna()
        for col, pid in plants.items():
            g = lambda k: num(pd.Series([df.at[k, col]])).iloc[0] if k in df.index else np.nan
            r = {
                "date": date, "plant": int(pid),
                "height_cm": g("초장(cm)"),
                "growth_cm": g("생장량(cm)"),
                "nodes_cum": g("총엽수"),           # 누적 발생 엽수(=마디 수)
                "leaves_removed": g("적엽수"),
            }
            for i, k in enumerate(["엽장(1)(cm)", "엽장(2)", "엽장(3)"], 1):
                r[f"leaf_len_{i}"] = g(k)
            for i, k in enumerate(["엽폭(1)(cm)", "엽폭(2)", "엽폭(3)"], 1):
                r[f"leaf_wid_{i}"] = g(k)
            for i, k in enumerate(["경경(1)(mm)", "경경(2)", "경경(3)", "경경(4)", "경경(5)"], 1):
                r[f"stem_d_{i}"] = g(k)
            for t in range(1, N_TRUSS + 1):
                r[f"T{t}_flowers"] = g(f"화방개수({t})")   # 화방 내 화수
                r[f"T{t}_open"] = g(f"전개화방({t})")      # 개화(전개) 화수
                r[f"T{t}_set"] = g(f"착과수({t})")         # 착과수
            rows.append(r)
    out = pd.DataFrame(rows).sort_values(["date", "plant"])
    out = out.dropna(axis=1, how="all")
    out.to_csv(OUT / "obs_nondes.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 3. 파괴조사 (생체중 + 건물중)
# ---------------------------------------------------------------------------
FRUIT_RE = re.compile(r"과실\s*(\d+)-(\d+)")


def _parse_des(xl: pd.ExcelFile, sheet: str, kind: str) -> pd.DataFrame:
    df = xl.parse(sheet, header=None, index_col=0)
    df.index = df.index.map(lambda x: str(x).strip())
    plants = num(df.loc["조사항목"]).dropna()
    rows = []
    for col, pid in plants.items():
        r = {"plant": int(pid)}
        fruit = {}
        for k in df.index:
            v = num(pd.Series([df.at[k, col]])).iloc[0]
            m = FRUIT_RE.match(k)
            if m:
                if not np.isnan(v):
                    fruit.setdefault(int(m.group(1)), []).append(v)
                continue
            key = {
                "초장": "height_cm", "엽수": "leaves_present", "총엽면적": "leaf_area_cm2",
                "잎(생체중)": "leaf", "엽(생체중)": "leaf", "줄기(생체중)": "stem",
                "잎(건물중)": "leaf", "엽(건물중)": "leaf", "줄기(건물중)": "stem",
                "화방(건물중)": "truss_struct",
            }.get(k)
            if key in ("leaf", "stem", "truss_struct"):
                r[f"{key}_{kind}"] = v
            elif key:
                r[key] = v
            elif kind == "FW" and k.endswith("화방(생체중)"):
                r[f"T{int(k.split('화방')[0])}_struct_FW"] = v
        for t, vals in fruit.items():
            r[f"T{t}_fruit_{kind}"] = sum(vals)
            r[f"T{t}_nfruit"] = len(vals)
        r[f"fruit_{kind}"] = sum(sum(v) for v in fruit.values())
        r["nfruit"] = sum(len(v) for v in fruit.values())
        rows.append(r)
    return pd.DataFrame(rows)


def build_des():
    fw = pd.ExcelFile(RAW / "A7-6 토마토 파괴조사(생체중).xlsx")
    dw = pd.ExcelFile(RAW / "A7-6 토마토 파괴조사(건물중).xlsx")
    fw_map = {sheet_date(s): s for s in fw.sheet_names}
    dw_map = {sheet_date(s): s for s in dw.sheet_names}
    out = []
    for date in sorted(set(fw_map) | set(dw_map)):
        a = _parse_des(fw, fw_map[date], "FW") if date in fw_map else pd.DataFrame({"plant": []})
        b = _parse_des(dw, dw_map[date], "DW") if date in dw_map else pd.DataFrame({"plant": []})
        m = a.merge(b, on="plant", how="outer", suffixes=("", "_dw"))
        m = m.drop(columns=[c for c in m.columns if c.endswith("_dw")])
        m.insert(0, "date", date)
        out.append(m)
    out = pd.concat(out, ignore_index=True)
    # 빈 시트(2024-01-11: 최종조사로 대체) → 과실 합이 0으로 들어가므로 영양기관 값이 전부 없는 행 제거
    organ = [c for c in ("leaf_FW", "stem_FW", "leaf_DW", "stem_DW") if c in out]
    out = out.dropna(subset=organ, how="all")
    struct = [c for c in out.columns if c.endswith("_struct_FW")]
    out["truss_struct_FW"] = out[struct].sum(axis=1, min_count=1)
    out = out.drop(columns=struct)
    out["shoot_DW"] = out[["leaf_DW", "stem_DW", "truss_struct_DW", "fruit_DW"]].sum(axis=1, min_count=1)
    out = out.sort_values(["date", "plant"])
    out.to_csv(OUT / "obs_des.csv", index=False)
    return out


def build_final():
    xl = pd.ExcelFile(RAW / "토마토 최종생육조사.xlsx")
    a = _parse_des(xl, "생체중", "FW")
    b = _parse_des(xl, "건물중", "DW")
    m = a.merge(b, on="plant", how="outer", suffixes=("", "_dw"))
    m = m.drop(columns=[c for c in m.columns if c.endswith("_dw")])
    struct = [c for c in m.columns if c.endswith("_struct_FW")]
    m["truss_struct_FW"] = m[struct].sum(axis=1, min_count=1)
    m = m.drop(columns=struct)
    m["shoot_DW"] = m[["leaf_DW", "stem_DW", "truss_struct_DW", "fruit_DW"]].sum(axis=1, min_count=1)
    # 파괴조사 01.11 시트가 비어 있고 최종조사가 그 자리를 대신함 (사용자 확인: 2024-01-11)
    m.insert(0, "date", FINAL_DATE)
    m.to_csv(OUT / "obs_final.csv", index=False)
    return m


if __name__ == "__main__":
    d = build_climate()
    print("climate_daily", d.shape, d.index.min().date(), "~", d.index.max().date())
    for f in (build_nondes, build_des, build_final):
        x = f()
        print(f.__name__, x.shape)
