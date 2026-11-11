"""
로드셀 4대 (가운데 2열, 코이어 슬라브 1개 = 4주, 처리 아닌 반복) → 일 증산량

원자료(1분): weight [kg, 슬라브+배지+식물, 분해능 10 g], input/output [mL, 자정 리셋 누적 급액/배액]

두 방법으로 계산하고 서로 대조한다.
 (A) 보정 물수지  ET_A = 급액 − 배액 − (ΔW_자정→자정 − Σ계단변화)
     계단변화 = 급·배액이 없는 1분에 |ΔW| ≥ STEP_KG 인 변화 (적엽·수확·유인·접촉 — 증산 아님)
 (B) dW/dt       조용한 분(급·배액 이벤트 및 그 후 PAD_MIN분 제외, 계단변화 제외)의 무게 감소율을
                 시간별 평균 → 조용한 분이 MIN_QUIET분 미만인 시간은 앞뒤 시간으로 선형 보간 → 일 적분
                 급·배액 센서에 의존하지 않음
품질 플래그 (0 = 사용)
  1 1분 자료 < 95%      2 자정 무게 결측     3 ET_A 음수/비현실(>15 L)
  4 슬라브 제거(W<3 kg)  5 급액 펄스 이상(1분 > PULSE_MAX_ML)
  6 계단변화 합 > STEP_DAY_KG (작업일 — A는 보정되지만 B와 대조 권장)
  7 A·B 불일치 |A−B| / mean > AB_TOL
사용값: ET_plant_mL = ET_B_plant (flag 0·6, 보간 ≤ 6 h) — 작업일 계단의 부호가 양쪽(제거 vs 누락 급액)이라
        급·배액 센서에 의존하지 않는 B를 기본으로, A는 교차검증용 (flag 0에서 r=0.92, 평균 777 vs 779 mL)
  ※ 식물 생체중 증가(하루 수십 g)는 무시 — 증산 대비 1–3%

출력: data_processed/loadcell_daily.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data_analysis/data/loadcell"
OUT = ROOT / "data_processed"
PLANTS_PER_SLAB = 4

STEP_KG = 0.05        # 조용한 1분 |ΔW| 이 이상이면 비증산 계단변화 (분해능 10 g, 주간 증산 ~5–10 g/분)
PAD_MIN = 15          # 급·배액 이벤트 후 배수 지연 구간 제외
MIN_QUIET = 10        # 시간당 최소 조용한 분
PULSE_MAX_ML = 1000   # 1분 급액 증가가 이보다 크면 계측 이상 (정상 60–120 mL)
STEP_DAY_KG = 0.2
AB_TOL = 0.3


def load_minute(dev: int) -> pd.DataFrame:
    l = pd.read_csv(SRC / f"dev_{dev}.csv", dtype={"datetime": str},
                    usecols=["datetime", "weight", "input", "output"])
    l["datetime"] = pd.to_datetime(l["datetime"], format="%Y%m%d%H%M")
    l = l.set_index("datetime").sort_index()
    l = l[~l.index.duplicated()]
    return l.reindex(pd.date_range(l.index.min().floor("D"), l.index.max().ceil("D"), freq="min"))


def daily_et(dev: int) -> pd.DataFrame:
    d = daily_et_from_minute(load_minute(dev), PLANTS_PER_SLAB)
    d.insert(0, "device", dev)
    return d


def daily_et_from_minute(l: pd.DataFrame, plants_per_slab: int) -> pd.DataFrame:
    """1분 자료(weight kg, input/output mL 자정 리셋 누적) → 일 ET (A·B, 플래그). 다른 온실에도 사용"""
    same_day = l.index.to_series().diff().eq(pd.Timedelta("1min")) & (l.index.normalize() == l.index.normalize().to_series().shift().values)
    dw = l["weight"].diff().where(same_day.values)
    din = l["input"].diff().where(same_day.values).clip(lower=0)
    dout = l["output"].diff().where(same_day.values).clip(lower=0)
    event = (din.fillna(0) > 0) | (dout.fillna(0) > 0)
    after = event.astype(int).rolling(PAD_MIN, min_periods=1).max().astype(bool)
    quiet = ~after & dw.notna()
    step = quiet & (dw.abs() >= STEP_KG)
    good = quiet & ~step

    g = l.resample("D")
    d = pd.DataFrame({
        "irr": g["input"].max(),
        "drain": g["output"].max(),
        "n_obs": g["weight"].count(),
        "pulse_max": din.resample("D").max(),
        "step_kg": dw.where(step, 0).resample("D").sum(),            # 계단변화 순합 (음수 = 제거)
        "step_abs_kg": dw.where(step, 0).abs().resample("D").sum(),
    })
    w0 = l["weight"].between_time("00:00", "00:10").resample("D").median()
    d["W0"], d["W1"] = w0, w0.shift(-1)
    d["dW"] = (d["W1"] - d["W0"]) * 1000
    # (A) 보정 물수지: 계단변화는 증산이 아니므로 ΔW에서 제외
    d["ET_A_slab"] = d["irr"] - d["drain"] - (d["dW"] - d["step_kg"] * 1000)
    # (B) dW/dt: 조용한 분의 감소율 → 시간 평균 → 보간 → 적분
    rate = (-dw.where(good)).resample("h").mean()                     # kg/분
    nq = good.resample("h").sum()
    rate = rate.where(nq >= MIN_QUIET).interpolate(limit=6, limit_area="inside")
    d["ET_B_slab"] = (rate * 60 * 1000).resample("D").sum(min_count=20)
    d["hours_interp"] = (nq < MIN_QUIET).resample("D").sum()

    for k in ("A", "B"):
        d[f"ET_{k}_plant"] = d[f"ET_{k}_slab"] / plants_per_slab
    d["AB_rel"] = (d["ET_A_slab"] - d["ET_B_slab"]).abs() / d[["ET_A_slab", "ET_B_slab"]].mean(axis=1)

    d["flag"] = 0
    d.loc[d["AB_rel"] > AB_TOL, "flag"] = 7
    d.loc[d["step_abs_kg"] > STEP_DAY_KG, "flag"] = 6
    d.loc[d["pulse_max"] > PULSE_MAX_ML, "flag"] = 5
    d.loc[(d["ET_A_slab"] < 0) | (d["ET_A_slab"] > 15000), "flag"] = 3
    d.loc[d["n_obs"] < 0.95 * 1440, "flag"] = 1
    d.loc[d["W0"].isna() | d["W1"].isna(), "flag"] = 2
    d.loc[d["W0"] < 3, "flag"] = 4
    # 사용값: B(dW/dt, 센서 비의존) — flag 0 또는 6(작업일, B는 계단 제외로 견고), 보간 ≤ 6 h
    use = d["flag"].isin([0, 6]) & (d["hours_interp"] <= 6)
    d["ET_plant_mL"] = d["ET_B_plant"].where(use)
    d["use"] = use.astype(int)
    d.index.name = "date"
    return d


if __name__ == "__main__":
    out = pd.concat([daily_et(i) for i in range(1, 5)])
    out.to_csv(OUT / "loadcell_daily.csv")
    s = out[(out.index >= "2023-10-05") & (out.index <= "2024-01-12")]
    print("작기(10/5–1/12) 플래그 분포 (대×일):")
    print(s.groupby(["device", "flag"]).size().unstack(fill_value=0))
    ok = s[s["flag"] == 0]
    print("사용 일수/대 (use=1):", s[s["use"] == 1].groupby("device").size().to_dict())
    print("\nA vs B (flag 0): r =", round(np.corrcoef(ok["ET_A_plant"], ok["ET_B_plant"])[0, 1], 3),
          " 평균 A", round(ok["ET_A_plant"].mean()), " B", round(ok["ET_B_plant"].mean()), "mL/주/일")
    piv = s[s["use"] == 1].pivot_table(index="date", columns="device", values="ET_plant_mL")
    print("대 간 상관:"); print(piv.corr().round(2))
    print("월평균 (mL/주/일):", piv.mean(axis=1).resample("MS").mean().round(0).to_dict())
    f6 = s[s["flag"] == 6][["device", "step_kg", "ET_A_plant", "ET_B_plant"]]
    print("\n작업일(flag 6) 예:"); print(f6.round(2).head(15).to_string())
