"""
탄소 버퍼 상한 민감도 (내부 검토 2026-11-03, 3번) — sens_source.py와 같은 틀

기본 모델은 버퍼(잎 DW × 0.3 g CH2O g⁻¹)를 넘는 동화산물을 버린다 (10월 23일, 총광합성의 8.7%).
상한을 1.0, 3.0(사실상 무제한 저장)으로 키워, 초기 잉여가 뒤에 과실로 가는 경로를 열었을 때의
함안 최종조사 건물중·증산 비와 계절별 N95 변화를 본다.
출력: data_processed/sens_buffer.csv, sens_buffer_n95.csv

(원 설명) 과실 DW 과소 추정(함안 최종조사 195 vs 331 g)이 엽수 간 상대 비교(N95)에 주는 영향 — source·sink 민감도 (리뷰 대응)

과소 추정의 두 후보 원인을 보정값에서 하나씩 키워 본다:
  source  τ_struct × 1.2, × 1.4  (모형에 없는 바닥·구조 반사광, 산란광 과소 → 캐노피 상단 광 +20·40%)
  sink    fruit_W_pot × 1.3       (과실 sink 세기 과소)
  both    τ_struct × 1.4 & fruit_W_pot × 1.3
지표: 함안 최종조사(2024-01-11) 과실 DW(달린 + 수확)·총 DM, 로드셀 증산 비(10–11월 / 12–1월),
      계절별 N95 와 N별 상대 과실 DM 증가(계절 최대 대비 %)
엽수 시나리오 설정(작기·분석 구간·KeepN 시작일·초기 상태)은 scenario_season.py 와 같음 (base 변형 = Rd25 0.65, Rd∝cap)
출력: data_processed/sens_source.csv (N별), sens_source_n95.csv, 표준출력
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import INIT, START, END, load, r0_rule      # noqa: E402
from fspm import Params, simulate                             # noqa: E402
from fspm import management as M                              # noqa: E402
from scenario_season import HG_SEASON, NS, SEASONS, window_metrics   # noqa: E402
import run_hoenggye as RH                                     # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"
VARIANTS = {"base": {}, "buffer1.0": {"buf": 1.0}, "buffer3.0": {"buf": 3.0}}


def apply(p, v):
    return replace(p, buf_frac=v.get("buf", p.buf_frac))


def main():
    clim, hr, obs, fin, ev = load()
    pj = json.loads((DP / "params_m2.json").read_text())
    p_ha = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    clim_hg, hr_hg, obs_hg, _ = RH.load()
    ph = json.loads((DP / "hg/params_hg.json").read_text())
    p_hg = replace(RH.base_params(), **{k: v for k, v in ph.items() if k in Params.__dataclass_fields__})
    init_hg = dict(plant0=RH.initial_plant(clim_hg, p_hg, obs_hg))

    # (1) 함안 최종조사·로드셀 증산: 관측 엽수 궤적(R0)으로 모의
    lc = pd.read_csv(DP / "loadcell_daily.csv", parse_dates=["date"])
    lm = lc[lc["use"] == 1].pivot_table(index="date", columns="device", values="ET_plant_mL").mean(axis=1)
    fin_rows = []
    for vn, v in VARIANTS.items():
        s, _ = simulate(clim, apply(p_ha, v), r0_rule(obs, fin), START, END, hourly=hr, **INIT)
        e = s.loc["2024-01-11"]
        j = pd.concat([s["et_g"].rename("sim"), lm.rename("obs")], axis=1, join="inner").dropna()
        ratio = lambda x: float(x["sim"].sum() / x["obs"].sum())
        fin_rows.append(dict(variant=vn, fruit_final=e.fruit_dw + e.harvested_dw, leaf_final=e.leaf_dw, stem_final=e.stem_dw,
                             total_final=e.fruit_dw + e.harvested_dw + e.leaf_dw + e.stem_dw,
                             et_ratio_octnov=ratio(j.loc[:"2023-11-30"]), et_ratio_decjan=ratio(j.loc["2023-12-01":])))
    F = pd.DataFrame(fin_rows)
    obs_tot = (fin["fruit_DW"] + fin["leaf_DW"] + fin["stem_all_DW"]).mean()
    print(f"[함안 최종조사 2024-01-11] 관측 과실 {fin['fruit_DW'].mean():.0f} ± {fin['fruit_DW'].std():.0f} g, 총 {obs_tot:.0f} g")
    print(F.round(1).to_string(index=False))

    # (2) 계절별 엽수 시나리오
    rows = []
    for season, (start, a, b) in SEASONS.items():
        st = pd.Timestamp(start)
        end = (pd.Timestamp(b) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        hg = season == HG_SEASON
        c, h, p0 = (clim_hg, hr_hg, p_hg) if hg else (clim, hr, p_ha)
        init = init_hg if hg else dict(leaf_dw0=10.0, stem_dw0=4.0)
        trigger = a if hg else (st + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        p0 = replace(p0, rd_frac=0.65 / p_ha.Vcmax25, rd_age_mode="cap",
                     transplant=RH.START_TP if hg else (st - pd.Timedelta(days=17)).strftime("%Y-%m-%d"))
        for vn, v in VARIANTS.items():
            for n in NS:
                s, _ = simulate(c, apply(p0, v), M.KeepN(n, trigger=trigger), start, end, hourly=h, **init)
                rows.append(dict(season=season, variant=vn, N=n, **window_metrics(s, a, b, h, p0)))
        print(f"  {season} 완료", flush=True)
    d = pd.DataFrame(rows)
    d["fruit_rel"] = d.groupby(["season", "variant"])["fruit_gain"].transform(lambda x: 100 * x / x.max())
    d.to_csv(DP / "sens_buffer.csv", index=False)

    g = []
    for (season, vn), x in d.groupby(["season", "variant"], sort=False):
        x = x.sort_values("N")
        n95 = int(x[x.fruit_rel >= 95].N.min())
        g.append(dict(season=season, variant=vn, N95=n95, fruit_max=x.fruit_gain.max(), qd=x[x.N == n95].qd.iloc[0],
                      rel16=x[x.N == 16].fruit_rel.iloc[0], rel20=x[x.N == 20].fruit_rel.iloc[0], rel24=x[x.N == 24].fruit_rel.iloc[0]))
    G = pd.DataFrame(g).merge(F, on="variant")
    G.to_csv(DP / "sens_buffer_n95.csv", index=False)
    pd.set_option("display.width", 200)
    print(G.round(2).to_string(index=False))
    # 상대 곡선 안정성: base 대비 N별 상대 과실 DM 증가 차 (%p)
    piv = d.pivot_table(index=["season", "N"], columns="variant", values="fruit_rel")
    dev = piv.sub(piv["base"], axis=0).drop(columns="base").abs()
    print("\nbase 대비 상대 과실 DM 증가 최대 차 (%p, N 12–30):")
    print(dev.groupby("season").max().round(1).to_string())


if __name__ == "__main__":
    main()
