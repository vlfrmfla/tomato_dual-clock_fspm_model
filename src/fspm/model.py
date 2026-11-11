"""
일 단위 시뮬레이션 루프 (광합성·증산은 그날 24시간 기상으로 시간 적분)

순서 (하루)
  1. 열시간 → 구조 시계 (마디 출현, 엽령)
  2. source: 활성 잎 위→아래 1D 광흡수 → 잎별 총동화·증산
  3. sink–source 분배 → Q/D, 기관 생장 (잎 면적 = DW × SLA)
  4. Q/D 지수이동평균 → 생식 시계 (개화→착과 속도 배율 g)
  5. 수확 (화방 성숙 시 과실 DW 이동)
  6. 적엽 규칙 집행
"""
import copy
from typing import Optional

import numpy as np
import pandas as pd

from .carbon import allocate
from .management import Context, PruningRule
from .params import Params
from .phenology import leaf_area_progress, step_structure, step_trusses, thermal_time
from .plant import Plant, TrussStage
from .fruit import fruit_day
from .source import canopy_day


def init_plant(n_nodes: int, p: Params, date: str, trusses_visible: int = 1,
               leaf_dw0: Optional[float] = None, stem_dw0: Optional[float] = None) -> Plant:
    """첫 조사일 관측 상태로 초기화. 잎 엽령은 위로 갈수록 phyllochron 씩 젊게."""
    plant = Plant()
    for _ in range(n_nodes):
        plant.add_node(date, p.truss_first_node, p.truss_interval)
    phy = p.phyllochron_early if p.phyllo_mode != "constant" else p.phyllochron
    for n in plant.nodes:
        n.leaf.tt_age = (n_nodes - n.index + 1) * phy
        prog = leaf_area_progress(n.leaf.tt_age, p)
        if p.use_source:
            n.leaf.area = p.leaf_area_pot * prog
            n.leaf.dw = n.leaf.area / p.SLA
        else:
            n.leaf.area = p.leaf_area_max * prog
    while len(plant.trusses) > trusses_visible:
        tr = plant.trusses.pop()
        plant.nodes[tr.node - 1].truss = None
        plant.next_truss_node -= p.truss_interval
    # 관측 초기값으로 맞춤 (있으면)
    if leaf_dw0 is not None and plant.leaf_dw() > 0:
        s = leaf_dw0 / plant.leaf_dw()
        for n in plant.nodes:
            n.leaf.dw *= s
            n.leaf.area = n.leaf.dw * p.SLA
    plant.stem_dw = stem_dw0 if stem_dw0 is not None else p.stem_dw_per_node * n_nodes * 0.5
    return plant


def simulate(climate: pd.DataFrame, p: Params, rule: PruningRule, start: str, end: str,
             n_nodes0: int = 8, hourly: Optional[pd.DataFrame] = None,
             leaf_dw0: Optional[float] = None, stem_dw0: Optional[float] = None,
             leaf_log: Optional[list] = None, plant0: Optional[Plant] = None):
    """
    climate : 일 기상 (index=date, Tmean, Rg_out_MJ ...)
    hourly  : 시간 기상 (index=datetime, T, Rg_out, VPD) — use_source=True 일 때 필요
    plant0  : 시작 상태 식물 (주면 복사해서 사용, n_nodes0·leaf_dw0·stem_dw0 무시)
    반환    : daily DataFrame, Plant
    """
    clim = climate.loc[pd.Timestamp(start) - pd.Timedelta(days=7): end].copy()
    clim["T_roll7"] = clim["Tmean"].rolling(7, min_periods=1).mean()
    if p.use_source:
        # 시간 기후 결측이 있으면 그날 광합성·Q/D가 NaN → Q/D 이동평균이 NaN으로 고정되어 이후 착과가 멈춤 → 미리 중단
        need = [c for c in ("T", "Rg_out", "VPD") if c in hourly.columns]
        bad = hourly.loc[start:end, need].isna().any(axis=1)
        if bad.any():
            days = sorted({str(d.date()) for d in bad.index[bad]})
            raise ValueError(f"hourly climate has missing {need} on {len(days)} day(s): {days[:5]} … — fill gaps first")
    pre = climate.loc[p.transplant: pd.Timestamp(start) - pd.Timedelta(days=1), "Tmean"]
    tt_tp = float(sum(thermal_time(T, p) for T in pre))
    plant = copy.deepcopy(plant0) if plant0 is not None else \
        init_plant(n_nodes0, p, start, leaf_dw0=leaf_dw0, stem_dw0=stem_dw0)
    harvest_started = False
    first_q = first_r = not np.isfinite(p.qd_ema_init)     # NaN → 첫날 값으로 시작 (민감도용)
    qd_ema = 1.0 if first_q else p.qd_ema_init
    rad_ema = qd_ema                                        # 대조 모델 C: 일사 이동평균 (같은 시간상수·상한·초기값)
    alpha = 1.0 / p.qd_ema_days
    par_hist = []           # 최근 7일 잎별 PAR (R2 규칙)
    rec = []

    for date, w in clim.loc[start:end].iterrows():
        ds = str(date.date())
        dtt = thermal_time(w["Tmean"], p)
        tt_tp += dtt

        # 1. 구조 시계
        new_nodes = step_structure(plant, dtt, tt_tp, ds, p, qd=qd_ema)

        # 2–3. source & 분배
        cday = dict(gross=np.nan, rm=np.nan, need=np.nan, avail=np.nan, qd=1.0,
                    g_leaf=np.nan, g_stem=np.nan, g_fruit=np.nan)
        et = par_top = et_night = np.nan
        leaf_par = None
        if p.use_source:
            h = hourly.loc[ds]
            act = [n for n in reversed(plant.nodes) if n.leaf.active]   # 위→아래
            # 착색기 화방의 과실 깊이 (캐노피 상단 기준): (최상위 마디 − 화방 마디 + 처짐) × 마디 간격
            coloring = [t for t in plant.trusses if t.stage == TrussStage.SET and t.tt_stage > p.fruit_TT_growth]
            top = plant.n_nodes
            fdep = np.array([(top - t.node + p.fruit_hang_nodes + 0.5) * p.internode_len for t in coloring])
            res = canopy_day(np.array([n.leaf.area for n in act]), np.array([n.leaf.tt_age for n in act]),
                             h["T"].to_numpy(), h["Rg_out"].to_numpy(), h["VPD"].to_numpy(), p,
                             curt_h=h[["curt1", "curt2", "curt3"]].to_numpy() if "curt1" in h else None,
                             co2_h=h["CO2"].to_numpy() if "CO2" in h else None,
                             tau_glass_h=h["tau_glass"].to_numpy() if "tau_glass" in h else None,
                             sun_h=(h["sun_elev"].to_numpy(), h["sun_az"].to_numpy(), h["kd"].to_numpy())
                             if "sun_az" in h else None,
                             fruit_depths=fdep if (p.fruit_temp_ripening and len(coloring)) else None)
            et, par_top, et_night = float(res["et_leaf"].sum()), res["par_top"], res["et_night"]
            par_hist.append({n.index: v for n, v in zip(act, res["par_leaf"])})
            if leaf_log is not None:                        # 엽위별 일 기록 (위→아래 순위 1, 2, …)
                leaf_log.append(dict(date=date, net=res["ch2o_leaf"] - res["rd_leaf"], gross=res["ch2o_leaf"],
                                     rd=res["rd_leaf"], et=res["et_leaf"], et_night=res["et_night_leaf"],
                                     par=res["par_leaf"], area=np.array([n.leaf.area for n in act])))
            net = res["ch2o_leaf"] - res["rd_leaf"]
            for n, v in zip(act, net):                      # 잎별 순탄소수지 (7일 이동합, 적엽 규칙·진단용)
                n.leaf.net_hist = (n.leaf.net_hist + [float(v)])[-7:]
            par_hist = par_hist[-7:]
            leaf_par = {i: sum(d.get(i, 0.0) for d in par_hist) for i in par_hist[-1]}
            leaf_gross = float(res["ch2o_leaf"].sum())
            leaf_rd = float(res["rd_leaf"].sum()) if p.leaf_resp_fvcb else 0.0
            cday = allocate(plant, leaf_gross - leaf_rd, w["Tmean"], dtt, p)
            cday["leaf_rd"] = leaf_rd
            cday["gross_leaf"] = leaf_gross
            if np.isfinite(cday["qd"]):
                v = min(cday["qd"], p.qd_ema_cap)
                qd_ema, first_q = (v if first_q else qd_ema + alpha * (v - qd_ema)), False

        # 4. 생식 시계 (착색기 화방은 과실 온도로)
        color_inc, fruit_dT = {}, np.nan
        if p.use_source and p.fruit_temp_ripening and len(coloring) and res.get("fruit_sw") is not None:
            inc, hot, dT = fruit_day(h["T"].to_numpy(), res["fruit_sw"], p)
            for t, ci, hh, d in zip(coloring, inc, hot, dT):
                color_inc[t.index] = float(ci)
                t.hot_hours += float(hh)
                t.dT_sum += float(d)
                t.dT_days += 1
            fruit_dT = float(np.mean(dT))
        if np.isfinite(w.get("Rg_out_MJ", np.nan)):
            v = min(w["Rg_out_MJ"] / p.rad_ref, p.qd_ema_cap)
            rad_ema, first_r = (v if first_r else rad_ema + alpha * (v - rad_ema)), False
        signal = rad_ema if p.clock_signal == "rad" else qd_ema
        ripe = step_trusses(plant, dtt, signal if p.use_source else 1.0, ds, p, color_inc=color_inc)

        # 5. 수확
        for t in ripe:
            plant.harvested_dw += t.fruit_dw
            t.stage = TrussStage.HARVESTED
            t.dates[TrussStage.HARVESTED.name] = ds
            harvest_started = True

        # 6. 적엽
        ctx = Context(date=date, T_roll7=w["T_roll7"], harvest_started=harvest_started,
                      leaf_weekly_par=leaf_par)
        removed = rule.execute(plant, ctx, p)

        for t in plant.trusses:
            if t.stage >= TrussStage.SET:
                plant.truss_log.append((date, t.index, t.fruit_dw))      # 수확 화방은 수확 시 DW 유지 (화방별 과실 비교에 남김)
        st = [t.stage for t in plant.trusses]
        rec.append(dict(
            date=date, Tmean=w["Tmean"], dTT=dtt, TT_tp=tt_tp,
            qd=cday["qd"], qd_ema=qd_ema, frac=cday.get("frac", np.nan), gross=cday["gross"], rm=cday["rm"], buffer=plant.buffer,
            g_leaf=cday["g_leaf"], g_stem=cday["g_stem"], g_fruit=cday["g_fruit"],
            fruit_dT=fruit_dT, n_coloring=len(color_inc),
            et_g=et, et_night_g=et_night, par_top=par_top, leaf_rd=cday.get("leaf_rd", np.nan),
            lowest_net7=next((sum(n.leaf.net_hist) for n in plant.nodes if n.leaf.active and len(n.leaf.net_hist) == 7), np.nan),
            n_neg_leaves=sum(1 for n in plant.nodes if n.leaf.active and len(n.leaf.net_hist) == 7
                             and sum(n.leaf.net_hist) < 0),
            nodes=plant.n_nodes + plant.node_clock, new_nodes=new_nodes,
            leaves_present=len(plant.active_leaves()),
            leaf_area_m2=plant.leaf_area(),
            LAI=plant.leaf_area() * p.plant_density,
            leaf_dw=plant.leaf_dw(), stem_dw=plant.stem_dw + sum(t.struct_dw for t in plant.trusses),
            fruit_dw=plant.fruit_dw(), harvested_dw=plant.harvested_dw,
            removed_leaf_dw=plant.removed_leaf_dw,
            lowest_leaf_node=plant.lowest_active_leaf_node(),
            removed=removed,
            trusses=len(st),
            n_anthesis=sum(s >= TrussStage.ANTHESIS for s in st),
            n_set=sum(s >= TrussStage.SET for s in st),
            n_harvested=sum(s == TrussStage.HARVESTED for s in st),
        ))
    out = pd.DataFrame(rec).set_index("date")
    out["TT"] = out["dTT"].cumsum()
    return out, plant


def truss_table(plant: Plant) -> pd.DataFrame:
    rows = []
    for t in plant.trusses:
        r = dict(truss=t.index, node=t.node, n_fruit=t.n_fruit, fruit_dw=t.fruit_dw, hot_hours=t.hot_hours,
                 fruit_dT_mean=t.dT_sum / t.dT_days if t.dT_days else np.nan)
        for s in TrussStage:
            r[s.name.lower()] = pd.Timestamp(t.dates[s.name]) if s.name in t.dates else pd.NaT
        rows.append(r)
    return pd.DataFrame(rows)
