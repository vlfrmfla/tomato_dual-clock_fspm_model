"""
대관령 횡계팜 (KIST) 2025 작기 → 모델 입력·검증 자료 (data_processed/hg/)

원자료 (data_analysis/대관령_횡계팜_KIST/)
  기후·제어: 20250811_20251209_H_farm_tom.csv (1분, 2025-08-11 ~ 12-10; 메타데이터 §6: 2025 작기 환경 입력은 이 파일)
  CO₂ 센서 보정: 20250811_20260930_H_farm_env.csv 의 무작물 기간 (아래)
  생육: H_farm_tom_growth_2025_2026.csv (조사일 정리표, 측정 생육 단일 출처 — 값 = 주당 × 2.21주/m²)
  수확: H_farm_tom_harvest_2025.csv (12회, 10/16–12/09, 생체중 → 건물률 5.6 %)
재배 (사용자 2026-10-05 + 메타데이터.md 2026-10-08판): Venlo 연동(도면: 산광 F-CLEAN 필름), 작물 열 남북, 단줄기, 2025-08-11 정식(작기 A),
  Dafnis, 총 2,970주 / 1,344 m² = 2.21주/m²(통로 포함 바닥), 28라인 × 108주, 라인 길이 ~20.5 m → 라인 1 m당 5.27주,
  라인 간격 평균 2.0 m (56 m / 28; 경간 안 1.77, 기둥선 사이 2.68) → 재배면적 기준 2.63주/m² (함안 2.67과 같은 기준)
  같은 온실 50 %는 2025-03 정식 작기 M — 로드셀(RMF1)은 작기 M 소속 → 작기 A 증산 검증에 쓰지 않음
  실내 일사 센서는 차광 스크린 아래 → 캐노피 상단 일사로 직접 사용
※ 2026-11-03 수정: 이전 판은 레거시 파일 생육 열(옛 환산 = 주당 DW × 0.45 × 3.6, LAI = 주당 엽면적 × 3.6)을
  mg DW m⁻²·2.5주/m²로 보고 /2.5 → 주당 엽면적 1.44배 과대, DW 35 % 과소였음 (이전 결과 hg/_pre_conv_fix_1103/)
생육 (주당): 정리표 값 ÷ 2.21. 2025-12-09 착과는 미측정(엑셀 고정 +30 g/주) → 과실 총량 결측.
  initial: −1 = 정식일(8/11, 묘), 1 = 모의 시작 상태(9/01 첫 정식 후 조사), 0 = 평가
기후 → 모델 입력
  T, RH = 실내 측정. CO2 = 실내 측정 + 오프셋 (아래), 보정 후 200–3000 밖은 결측 → 420
    오프셋: 작물이 없던 2026-02–03 (슬래브 무게 < 3 kg) 10–15시·천창 ≥ 10 % 시간의 센서 중앙값이 외기(420)와 같아야 함
    (야간 호흡·주간 광합성 영향이 없는 조건). 교차 확인: 2025-08–09 천창 ≥ 60 % 밤 429, 한낮 404 ppm
  Rg_out 열 = 실내 일사 In_Rad (캐노피 상단 단파로 직접 사용: tau_glass = 1, 커튼 0, tau_struct = 1)
    In_Rad 는 2025-11-14부터 0 (센서 고장) → 외부 일사 × 직전 3주(10/24–11/13) 일 투과율 중앙값
  sun_elev/sun_az/kd = 횡계 좌표, 외부 일사(Out_Rad)로 Erbs 산란비
로드셀: build_loadcell.daily_et_from_minute (QC v2), 슬라브당 8주
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_loadcell import daily_et_from_minute            # noqa: E402
from fspm.greenhouse import solar_position, erbs_diffuse_fraction   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data_analysis/대관령_횡계팜_KIST"
RAW = DIR / "20250811_20251209_H_farm_tom.csv"
ENV = DIR / "20250811_20260930_H_farm_env.csv"
GROWTH = DIR / "H_farm_tom_growth_2025_2026.csv"
HARVEST = DIR / "H_farm_tom_harvest_2025.csv"
OUT = ROOT / "data_processed/hg"
LAT, LON = 37.677, 128.718          # 강원 평창군 대관령면 횡계리 (해발 ~770 m)
DENSITY_FLOOR = 2970 / 1344         # 정리표 환산 밀도 (통로 포함 바닥) — 주당 = 정리표 / 이 값
PLANTS_PER_M_ROW = 108 / 20.5       # 라인 1 m당 주수
ROW_SPACING = 56.0 / 28             # 라인 간격 평균 m
DENSITY = PLANTS_PER_M_ROW / ROW_SPACING   # 재배면적 기준 2.63주/m² (모델 plant_density, 관측 LAI 환산)
PLANTS_PER_SLAB = 8
HARVEST_DMC = 0.056                 # 완숙 과실 건물률 (Heuvelink 1995; 정리표 누적 수확과 같은 값)
CO2_OUTSIDE = 420.0
SENSOR_FAIL = "2025-11-14"
START_TP, START = "2025-08-11", "2025-09-01"


def vpd(T, RH):
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    return es * (1 - RH / 100)


def co2_offset():
    """무작물 기간(슬래브 무게 < 3 kg) 주간·천창 열림 시간의 센서 중앙값을 외기 420으로 맞추는 오프셋"""
    e = pd.read_csv(ENV, parse_dates=["datetime"], usecols=["datetime", "In_CO2", "Vent_State", "RMF1_Weight"])
    h = e.set_index("datetime").resample("h").agg({"In_CO2": "median", "Vent_State": "mean", "RMF1_Weight": "mean"})
    x = h.loc["2026-02-01":"2026-03-31 11:00"]
    ref = x[(x["RMF1_Weight"] < 3) & x.index.hour.isin(range(10, 16)) & (x["Vent_State"] >= 10)]
    return CO2_OUTSIDE - float(ref["In_CO2"].median()), len(ref), ref.index.normalize().nunique()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(RAW, parse_dates=["datetime"]).set_index("datetime").sort_index()
    m = m[~m.index.duplicated()]
    # 메타데이터 §6: 실내 센서·제어 상태 값 멈춤 → 결측 후 시간 보간
    frozen = [("2025-10-22 13:42", "2025-10-22 16:19"), ("2025-11-17 12:07", "2025-11-18 08:30")]
    in_cols = ["In_Temp", "In_Hum", "In_CO2", "In_Rad", "Shade_SC_state", "Thermal_SC_state", "Vent_state"]
    for fa, fb in frozen:
        m.loc[fa:fb, in_cols] = np.nan

    # ---------- 시간 기후
    h = m.resample("h").mean(numeric_only=True)
    h[in_cols] = h[in_cols].interpolate(limit=24, limit_area="inside")
    off, n_ref, d_ref = co2_offset()
    co2 = (h["In_CO2"] + off).where((h["In_CO2"] + off).between(200, 3000))
    day = pd.DataFrame({"out": h["Out_Rad"].resample("D").sum(), "inn": h["In_Rad"].resample("D").sum()})
    ref = day.loc["2025-10-24":"2025-11-13"]
    tau_fill = float((ref["inn"] / ref["out"]).median())
    rin = h["In_Rad"].copy()
    fail = rin.index >= SENSOR_FAIL
    rin[fail] = h.loc[fail, "Out_Rad"] * tau_fill
    beta, az, E0 = solar_position(h.index + pd.Timedelta(minutes=30), lat=LAT, lon=LON)
    kd = erbs_diffuse_fraction(h["Out_Rad"].to_numpy(), beta, E0)
    hr = pd.DataFrame({
        "T": h["In_Temp"], "RH": h["In_Hum"].clip(0, 100), "Rg_out": rin.clip(lower=0),
        "Rg_ext": h["Out_Rad"].clip(lower=0), "T_out": h["Out_Temp"], "RH_out": h["Out_Hum"],
        "CO2": co2.fillna(420.0), "co2_filled": co2.isna().astype(int), "CO2_raw": h["In_CO2"],
        "curt1": 0.0, "curt2": 0.0, "curt3": 0.0, "tau_glass": 1.0,
        "sun_elev": beta, "sun_az": az, "kd": kd,
        "shade_state": h["Shade_SC_state"], "thermal_state": h["Thermal_SC_state"], "vent_state": h["Vent_state"],
        "rad_filled": fail.astype(int),
    })
    hr["VPD"] = vpd(hr["T"], hr["RH"])
    hr.index.name = "datetime"
    hr.to_csv(OUT / "climate_hourly.csv")

    g = hr.resample("D")
    daily = pd.DataFrame({"Tmean": g["T"].mean(), "Tmax": g["T"].max(), "Tmin": g["T"].min(),
                          "RH": g["RH"].mean(), "VPD": g["VPD"].mean(),
                          "Rg_out_MJ": g["Rg_ext"].sum() * 3600 / 1e6, "Rg_in_MJ": g["Rg_out"].sum() * 3600 / 1e6,
                          "T_out": g["T_out"].mean()})
    daily.index.name = "date"
    daily.to_csv(OUT / "climate_daily.csv")

    # ---------- 파괴조사 (주당)
    gr = pd.read_csv(GROWTH, parse_dates=["date"])
    gr = gr[gr["season"] == 2025].set_index("date")
    k = 1000.0 * DENSITY_FLOOR                          # mg m⁻² → g 주⁻¹
    obs = pd.DataFrame({
        "leaf_area_m2": gr["LAI"] / DENSITY_FLOOR, "leaf_area_sd": gr["LAI_SD"] / DENSITY_FLOOR,
        "leaf_DW": gr["DM_Leaf"] / k, "leaf_DW_sd": gr["DM_Leaf_SD"] / k,
        "stem_DW": gr["DM_Stem"] / k, "stem_DW_sd": gr["DM_Stem_SD"] / k,
        "fruit_on_DW": gr["DM_Fruit"] / k, "harvest_cum_DW": gr["DM_HarCum"] / k,
        "harvest_DW": gr["DM_Har"] / k, "harvest_DW_sd": gr["DM_Har_SD"] / k,     # 과실 총 DW = 착과 + 조사일 이전 누적 수확
        "leaves": gr["Leaves"], "leaves_sd": gr["Leaves_SD"], "n": gr["n"],
    })
    obs.loc["2025-12-09", ["fruit_on_DW", "harvest_DW", "harvest_DW_sd"]] = np.nan   # 착과 미측정 (엑셀 고정값)
    obs["LAI"] = obs["leaf_area_m2"] * DENSITY           # 재배면적 기준 (모델과 같은 밀도)
    obs["LAI_sd"] = obs["leaf_area_sd"] * DENSITY
    obs["SLA_cm2_g"] = obs["leaf_area_m2"] * 1e4 / obs["leaf_DW"]
    obs["initial"] = np.select([obs.index == START_TP, obs.index == START], [-1, 1], 0)
    obs.index.name = "date"
    obs.to_csv(OUT / "obs_des.csv")

    hv = pd.read_csv(HARVEST, parse_dates=["date"]).set_index("date")
    harv = pd.DataFrame({"plants": hv["plants"], "harvest_FW_cum_g": hv["Har_FW_per_plant_cum_g"],
                         "harvest_cum_DW": hv["Har_FW_per_plant_cum_g"] * HARVEST_DMC})
    harv.index.name = "date"
    harv.to_csv(OUT / "obs_harvest.csv")

    # ---------- 로드셀
    l = m[["RMF1_Weight", "RMF1_InletCum", "RMF1_OutletCum"]].rename(
        columns={"RMF1_Weight": "weight", "RMF1_InletCum": "input", "RMF1_OutletCum": "output"})
    l = l.reindex(pd.date_range(l.index.min().floor("D"), l.index.max().ceil("D"), freq="min"))
    lc = daily_et_from_minute(l, PLANTS_PER_SLAB)
    lc.to_csv(OUT / "loadcell_daily.csv")
    return hr, daily, obs, harv, lc, tau_fill, (off, n_ref, d_ref)


if __name__ == "__main__":
    hr, daily, obs, harv, lc, tau_fill, (off, n_ref, d_ref) = build()
    print(f"CO2 오프셋 = {off:+.1f} ppm (무작물 2026-02–03, 10–15시·천창 ≥ 10 %: {n_ref} h, {d_ref}일)")
    x = hr.loc["2025-08-11":"2025-09-30"]
    nt = x[x.index.hour.isin(range(0, 5)) & (x["vent_state"] >= 60)]
    dy = x[x.index.hour.isin(range(11, 15)) & (x["vent_state"] >= 60)]
    print(f"  교차 확인 2025-08–09 천창 ≥ 60 %: 밤 {nt['CO2'].median():.0f} ppm (n {len(nt)}), 한낮 {dy['CO2'].median():.0f} ppm (n {len(dy)})")
    print("  보정 CO2 월 중앙값:", hr["CO2"].resample("MS").median().round(0).to_dict())
    print(f"In_Rad 대체 투과율 (10/24–11/13 중앙값) = {tau_fill:.2f}")
    print(f"재식밀도: 바닥 {DENSITY_FLOOR:.2f}, 재배면적 {DENSITY:.2f}주/m² (라인 {PLANTS_PER_M_ROW:.2f}주/m, 간격 {ROW_SPACING:.2f} m)")
    print("파괴조사 (주당):"); print(obs.round(3).to_string())
    print("수확 (주당 누적):"); print(harv.round(2).to_string())
