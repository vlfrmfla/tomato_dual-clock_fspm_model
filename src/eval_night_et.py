"""
야간 증산: 로드셀 실측 vs 모델 (함안, 보정 모델, R0 적엽)

야간 = 캐노피 상단 실내 단파 < p.sw_night (2 W m⁻²) 인 시간 — 모델과 실측에 같은 시간을 쓴다.
실측: build_loadcell 방법 B와 같은 시간별 무게 감소율(조용한 분, 보간 ≤ 6 h)을 야간 시간만 합산, 4대 평균.
      사용일(use == 1)이고 그날 야간 시간이 모두 값이 있는 날만.
모델: 일별 et_night_g. 비교용으로 이전 판정(단파 = 0인 시간만)도 계산.
출력: data_processed/night_et_eval.csv, 표준출력
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_loadcell as BL                                      # noqa: E402
from calibrate_m2 import INIT, START, END, load, r0_rule      # noqa: E402
from fspm import Params, simulate                             # noqa: E402
from fspm.source import inside_shortwave                      # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"
PERIODS = {"all": (START, END), "Oct–Nov": (START, "2023-11-30"), "Dec–Jan": ("2023-12-01", END)}


def hourly_et(dev):
    """방법 B의 시간별 증산 (g plant⁻¹ h⁻¹)"""
    l = BL.load_minute(dev)
    same_day = l.index.to_series().diff().eq(pd.Timedelta("1min")) & \
        (l.index.normalize() == l.index.normalize().to_series().shift().values)
    dw = l["weight"].diff().where(same_day.values)
    din = l["input"].diff().where(same_day.values).clip(lower=0)
    dout = l["output"].diff().where(same_day.values).clip(lower=0)
    event = (din.fillna(0) > 0) | (dout.fillna(0) > 0)
    after = event.astype(int).rolling(BL.PAD_MIN, min_periods=1).max().astype(bool)
    quiet = ~after & dw.notna()
    good = quiet & ~(quiet & (dw.abs() >= BL.STEP_KG))
    rate = (-dw.where(good)).resample("h").mean()
    nq = good.resample("h").sum()
    rate = rate.where(nq >= BL.MIN_QUIET).interpolate(limit=6, limit_area="inside")
    return rate * 60 * 1000 / BL.PLANTS_PER_SLAB


def main():
    clim, hr, obs, fin, ev = load()
    pj = json.loads((DP / "params_m2.json").read_text())
    p = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    h = hr.loc[START:END + " 23:00"]
    I = inside_shortwave(h["Rg_out"].to_numpy(), h[["curt1", "curt2", "curt3"]].to_numpy(), p, h["tau_glass"].to_numpy())
    night = pd.Series(I < p.sw_night, index=h.index)
    dark = h["sun_elev"] < 0
    print(f"야간 시간 / 해 아래 시간: {night[dark].mean():.0%} (이전 판정 단파 = 0: {pd.Series(I <= 0, index=h.index)[dark].mean():.0%})")

    lc = pd.read_csv(DP / "loadcell_daily.csv", parse_dates=["date"])
    use = lc[lc["use"] == 1].set_index(["device", "date"]).index
    per_dev = []
    for dev in sorted(lc["device"].unique()):
        r = hourly_et(dev).reindex(h.index)
        rn = r.where(night)
        ok = rn.notna().groupby(rn.index.date).sum() == night.groupby(night.index.date).sum()
        s = rn.groupby(rn.index.date).sum().where(ok)
        s.index = pd.to_datetime(s.index)
        keep = [(dev, d) in use for d in s.index]
        per_dev.append(s.where(keep).rename(dev))
    obs_n = pd.concat(per_dev, axis=1).mean(axis=1)

    s_new, _ = simulate(clim, p, r0_rule(obs, fin), START, END, hourly=hr, **INIT)
    s_old, _ = simulate(clim, replace(p, sw_night=1e-9), r0_rule(obs, fin), START, END, hourly=hr, **INIT)
    j = pd.concat([obs_n.rename("obs"), s_new["et_night_g"].rename("sim"), s_old["et_night_g"].rename("sim_old")],
                  axis=1, join="inner").dropna()
    rows = []
    for k, (a, b) in PERIODS.items():
        x = j.loc[a:b]
        e = x.sim - x.obs
        rows.append(dict(period=k, n=len(x), obs=x.obs.mean(), sim=x.sim.mean(), sim_old=x.sim_old.mean(),
                         ratio=x.sim.sum() / x.obs.sum(), r=np.corrcoef(x.sim, x.obs)[0, 1],
                         RMSE=np.sqrt((e ** 2).mean()), nRMSE=np.sqrt((e ** 2).mean()) / x.obs.mean(),
                         bias=e.mean(), bias_pct=100 * e.mean() / x.obs.mean()))
        print(f"{k:8s} n={len(x):3d}  실측 {x.obs.mean():5.0f}  모델 {x.sim.mean():5.0f}  (이전 판정 {x.sim_old.mean():4.0f}) "
              f"g plant⁻¹ night⁻¹ | 비 {x.sim.sum() / x.obs.sum():.2f}, r = {np.corrcoef(x.sim, x.obs)[0, 1]:.2f}")
    d = pd.DataFrame(rows); d.to_csv(DP / "night_et_eval.csv", index=False)
    print(d[["period", "n", "RMSE", "nRMSE", "bias", "bias_pct", "r"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
