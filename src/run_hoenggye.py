"""
횡계팜 2025 (Dafnis, F-CLEAN 필름 Venlo 연동) — 함안 A7-6 보정 모델의 적용 (재배 조건·초기 상태·실측 광합성)

입력: data_processed/hg/ (build_hoenggye.py)
설정: params_m2.json + 재배 조건 교체 (재배면적 기준 2.63주/m² = 라인 5.27주/m ÷ 평균 간격 2.0 m [메타데이터 2026-10-08판], 정식 2025-08-11)
      실내 일사 측정값을 캐노피 상단 일사로 사용 (τ_glass = τ_struct = 1, 커튼 0), CO₂ = 센서 + 오프셋 (build_hoenggye)
      광합성: Vcmax25·Jmax25 = 횡계 2025 실측 평균 (정식묘 제외 n 14, 상위 3번째 잎, Dynamic Assimilation Technique A/Ci, 25 °C),
              Rd25 = 함안과 같은 계수 rd_frac × Vcmax25 (= 0.0059 × 96 = 0.57; 실측 Rd는 부호 혼재로 미사용)
시작 상태 (함안처럼 첫 정식 후 조사일에서 시작): 정식(8/11, 7엽)부터 8/31까지 구조·생식 시계만(Q/D = 1) 진행해 마디·화방 단계를 정한 뒤,
      9/01 관측으로 맞춤 — 달린 잎 17장(하위 잎 제거), 잎·줄기 DW, 착과 과실 DW (initial_plant)
변형: (a) transfer — 함안 파라미터 그대로
      (b) SLA_obs   — SLA만 횡계 파괴조사 실측(엽면적/잎 DW 평균, 정식일 제외)
적엽: 기록 없음 → 규칙별 모의 (무적엽, R3 묶음/매주, R1 16·18·20장); 관측 달린 엽수(9/22–12/09)와 비교
검증: LAI·잎 DW·줄기 DW (주당, 4회 9/22–12/9), 과실 총 DW(달린 + 수확, 12/09 제외), 누적 수확(수확 기록 12회), 달린 엽수
      ※ 로드셀은 작기 M 소속 → 증산 비교는 참고용
출력: data_processed/hg/validation.csv, hg_runs.csv, 표준출력
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hoenggye import DENSITY, ROW_SPACING, START_TP, START    # noqa: E402
from fspm import Params, simulate                 # noqa: E402
from fspm import management as M                  # noqa: E402
from fspm.carbon import beta_cum                  # noqa: E402
from fspm.plant import TrussStage                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HG = ROOT / "data_processed/hg"
END = "2025-12-09"
_ph = pd.read_csv(ROOT / "data_analysis/대관령_횡계팜_KIST/H_farm_tom_photosynthesis_2025.csv")
_ph = _ph[_ph["use_for_model"]]
VCMAX25, JMAX25 = float(_ph["Vcmax25_umol_m2_s"].mean()), float(_ph["Jmax25_umol_m2_s"].mean())
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False


def load():
    clim = pd.read_csv(HG / "climate_daily.csv", parse_dates=["date"]).set_index("date")
    hr = pd.read_csv(HG / "climate_hourly.csv", parse_dates=["datetime"]).set_index("datetime")
    obs = pd.read_csv(HG / "obs_des.csv", parse_dates=["date"]).set_index("date")
    lc = pd.read_csv(HG / "loadcell_daily.csv", parse_dates=["date"]).set_index("date")
    # 물수지 A: 급액이 30분 간격으로 잦아 dW/dt(B)는 낮 시간 대부분이 보간 → 과소. 계측 이상(flag 1–5)만 제외
    ok = ~lc["flag"].isin([1, 2, 3, 4, 5]) & lc["ET_A_plant"].between(20, 3000)
    lc["ET_obs"] = lc["ET_A_plant"].where(ok)
    obs.attrs["harvest"] = pd.read_csv(HG / "obs_harvest.csv", parse_dates=["date"]).set_index("date")
    return clim, hr, obs, lc


def base_params():
    pj = json.loads((ROOT / "data_processed/params_m2.json").read_text())
    p = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    return replace(p, plant_density=DENSITY, row_spacing=ROW_SPACING, plants_per_slab=8, transplant=START_TP,
                   tau_struct=1.0, use_roof_geometry=True, use_curtains=True,
                   Vcmax25=VCMAX25, Jmax25=JMAX25)                 # rd_frac = 함안 계수 그대로 (Rd25 = rd_frac × Vcmax25)


def initial_plant(clim, p, obs):
    """9/01 시작 상태: 정식(7엽)부터 구조·생식 시계만(Q/D = 1) 진행한 식물을 9/01 관측(달린 엽수, 잎·줄기·과실 DW)으로 맞춤"""
    o = obs[obs["initial"] == 1].iloc[0]
    _, pl = simulate(clim, replace(p, use_source=False), M.NoPruning(), START_TP,
                     (pd.Timestamp(START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), n_nodes0=7)
    act = [n for n in pl.nodes if n.leaf.active]
    pl.remove_leaves([n.index for n in act[: max(len(act) - int(round(o["leaves"])), 0)]], START)
    act = [n.leaf for n in pl.nodes if n.leaf.active]
    a = np.array([lf.area for lf in act])
    for lf, w in zip(act, a / a.sum()):
        lf.dw = float(o["leaf_DW"]) * w
        lf.area = lf.dw * p.SLA
    pl.stem_dw = float(o["stem_DW"])
    st = [t for t in pl.trusses if t.stage == TrussStage.SET]
    if st and o["fruit_on_DW"] > 0:
        te = p.fruit_TT_growth
        w = np.array([t.n_fruit * beta_cum(t.tt_stage, te, p.fruit_tm_frac * te) for t in st])
        w = w / w.sum() if w.sum() > 0 else np.full(len(st), 1 / len(st))
        for t, wi in zip(st, w):
            t.fruit_dw = float(o["fruit_on_DW"]) * wi
    pl.removed_leaf_dw = pl.buffer = pl.harvested_dw = 0.0
    pl.truss_log = []
    return pl


def sim(clim, hr, obs, p, rule, end=END, start=START, **kw):
    """9/01 시작 상태(initial_plant)로 모의"""
    return simulate(clim, p, rule, start, end, hourly=hr, plant0=initial_plant(clim, p, obs), **kw)


def rules():
    return {
        "무적엽": (M.NoPruning(), {}),
        "R3 묶음 (함안 관행)": (M.TrussLinked(tt_after=235), dict(prune_min_batch=6)),
        "R3 매주": (M.TrussLinked(tt_after=235), {}),
        "R1 16장": (M.KeepN(16, trigger=START), {}),
        "R1 18장": (M.KeepN(18, trigger=START), {}),
        "R1 20장": (M.KeepN(20, trigger=START), {}),
    }


def evaluate(s, obs, lc):
    o = obs[obs["initial"] == 0]
    sim = pd.DataFrame({"LAI": s["LAI"], "leaf_DW": s["leaf_dw"], "stem_DW": s["stem_dw"],
                        "fruit_total_DW": s["fruit_dw"] + s["harvested_dw"]})
    out = {}
    for v, ov in (("LAI", "LAI"), ("leaf_DW", "leaf_DW"), ("stem_DW", "stem_DW"), ("fruit_total_DW", "harvest_DW")):
        j = pd.concat([sim[v], o[ov]], axis=1, join="inner").dropna()
        j.columns = ["sim", "obs"]
        rm = float(np.sqrt(((j.sim - j.obs) ** 2).mean()))
        out[f"{v}_nRMSE"] = rm / j.obs.mean()
        out[f"{v}_bias%"] = 100 * (j.sim - j.obs).mean() / j.obs.mean()
    e = pd.concat([s["et_g"], lc["ET_obs"]], axis=1, join="inner").dropna()
    e.columns = ["sim", "obs"]
    out["ET_n"] = len(e)
    out["ET_nRMSE"] = float(np.sqrt(((e.sim - e.obs) ** 2).mean())) / e.obs.mean()
    out["ET_R2"] = float(np.corrcoef(e.sim, e.obs)[0, 1] ** 2)
    out["ET_ratio"] = float(e.sim.sum() / e.obs.sum())
    j = pd.concat([s["leaves_present"], o["leaves"]], axis=1, join="inner").dropna()
    out["leaves_RMSE"] = float(np.sqrt(((j.iloc[:, 0] - j.iloc[:, 1]) ** 2).mean()))
    out["leaves_bias"] = float((j.iloc[:, 0] - j.iloc[:, 1]).mean())
    out.update(harvest_stats(s, obs))
    out["prune_events"] = int((s["removed"] > 0).sum())
    out["leaves_removed"] = int(s["removed"].sum())
    out["leaves_Oct"] = float(s.loc["2025-10-01":"2025-10-31", "leaves_present"].mean())
    return out, e


def harvest_stats(s, obs):
    """누적 수확 DW: 모의(그날 끝) vs 수확 기록 (그날 수확 포함 누적), 첫 수확일"""
    hv = obs.attrs["harvest"]
    j = pd.concat([s["harvested_dw"].rename("sim"), hv["harvest_cum_DW"].rename("obs")], axis=1, join="inner").dropna()
    hs = s.index[s["harvested_dw"] > 0]
    return dict(harv_RMSE=float(np.sqrt(((j.sim - j.obs) ** 2).mean())), harv_final_sim=float(j.sim.iloc[-1]),
                harv_final_obs=float(j.obs.iloc[-1]),
                first_harvest_lag_d=float((hs[0] - hv.index[0]).days) if len(hs) else np.nan)


def main():
    clim, hr, obs, lc = load()
    p0 = base_params()
    sla_obs = float((obs.loc[obs["initial"] >= 0, "leaf_area_m2"] / obs.loc[obs["initial"] >= 0, "leaf_DW"]).mean())
    variants = {"transfer": p0, "SLA_obs": replace(p0, SLA=sla_obs)}
    rows, runs = [], {}
    for vn, pv in variants.items():
        for rn, (rule, kw) in rules().items():
            p = replace(pv, **kw)
            s, pl = sim(clim, hr, obs, p, rule)
            ev, _ = evaluate(s, obs, lc)
            rows.append(dict(variant=vn, rule=rn, **ev))
            runs[(vn, rn)] = s
    res = pd.DataFrame(rows)
    res.to_csv(HG / "validation.csv", index=False)
    pd.concat({f"{k[0]}|{k[1]}": v for k, v in runs.items()}, names=["run", "date"]).to_csv(HG / "hg_runs.csv")
    pd.set_option("display.width", 220)
    print(f"SLA_obs = {sla_obs:.4f} m²/g (함안 {p0.SLA})")
    print(res.round(2).to_string(index=False))
    return res, runs, obs, lc, clim


if __name__ == "__main__":
    main()
