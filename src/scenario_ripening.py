"""
과실 노출에 따른 숙기 단축 — 화방 짝 비교 (2026-11-11, 원고 검토 "숙기 단축은 해상도 한계")

scenario_season.py 의 set_to_ripe_d(분석 구간에 익은 화방의 착과→수확 일수 평균)를 두 변형(base, noExposure)에서 빼면
구간에 익은 화방 수가 다를 때 서로 다른 화방 집합을 비교하게 됨 (봄 12장: 노출 7화방 vs 비노출 6화방 → 2.4일로 과대, 짝 비교는 0.6일).
여기서는 같은 화방의 수확일 차이(비노출 − 노출, 일)를 구함. 모델은 수확일을 일 단위로 정하므로 화방별 차이는 0, 1, 2일.
  대상 화방 = 노출(base) 모의에서 분석 구간에 수확된 화방. 비노출 수확일이 구간 밖으로 밀려도 잡히도록 구간 뒤 15일까지 모의.
설정(작기·분석 구간·N장 유지 시작일·초기 상태)은 scenario_season.py 와 같음 (base 변형 = Rd25 0.65, Rd∝cap).
출력: data_processed/scenario_ripening.csv (season, N, n_truss, adv_mean_d, adv_max_d, n_shifted, shifts)
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fspm import Params, simulate, truss_table        # noqa: E402
from fspm import management as M                      # noqa: E402
from scenario_season import HG_SEASON, NS, SEASONS    # noqa: E402
import run_hoenggye as RH                             # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"
EXTRA_DAYS = 15


def main():
    clim_ha = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"]).set_index("date")
    hr_ha = pd.read_csv(DP / "climate_hourly.csv", parse_dates=["datetime"]).set_index("datetime")
    pj = json.loads((DP / "params_m2.json").read_text())
    p_ha = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    clim_hg, hr_hg, obs_hg, _ = RH.load()
    ph = json.loads((DP / "hg/params_hg.json").read_text())
    p_hg = replace(RH.base_params(), **{k: v for k, v in ph.items() if k in Params.__dataclass_fields__})
    init_hg = dict(plant0=RH.initial_plant(clim_hg, p_hg, obs_hg))
    rows = []
    for season, (start, a, b) in SEASONS.items():
        st = pd.Timestamp(start)
        hg = season == HG_SEASON
        clim, hr, p0 = (clim_hg, hr_hg, p_hg) if hg else (clim_ha, hr_ha, p_ha)
        end = min(clim.index.max(), pd.Timestamp(b) + pd.Timedelta(days=EXTRA_DAYS)).strftime("%Y-%m-%d")
        init = init_hg if hg else dict(leaf_dw0=10.0, stem_dw0=4.0)
        trigger = a if hg else (st + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        for n in NS:
            ripe = {}
            for name, over in (("exposed", {}), ("air", {"fruit_abs": 0.0})):      # air: 과실 온도 = 기온
                p = replace(p0, rd_frac=0.65 / p_ha.Vcmax25, rd_age_mode="cap", **over,
                            transplant=RH.START_TP if hg else (st - pd.Timedelta(days=17)).strftime("%Y-%m-%d"))
                _, pl = simulate(clim, p, M.KeepN(n, trigger=trigger), start, end, hourly=hr, **init)
                ripe[name] = truss_table(pl).set_index("truss")["ripe"]
            x = pd.concat(ripe, axis=1)
            x = x[(x.exposed >= a) & (x.exposed <= b)]
            d = (x.air - x.exposed).dt.days
            if d.isna().any():
                raise ValueError(f"{season} N={n}: 비노출 수확일이 모의 기간 밖 — EXTRA_DAYS 를 늘릴 것")
            rows.append(dict(season=season, N=n, n_truss=len(d), adv_mean_d=d.mean(), adv_max_d=int(d.max()),
                             n_shifted=int((d > 0).sum()), shifts=" ".join(str(int(v)) for v in d)))
    out = pd.DataFrame(rows)
    out.to_csv(DP / "scenario_ripening.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
