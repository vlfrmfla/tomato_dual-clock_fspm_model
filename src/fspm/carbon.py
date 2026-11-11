"""
Sink–source 분배 (TOMSIM 계열, 기관 잠재 생장 = sink strength)

수요 (g DW/주/d, 오늘 dTT 기준)
  잎   : 잠재 최종 면적까지의 시그모이드 증가분 / SLA      (잎 크기가 source 의존이 됨)
  줄기 : 마디당 잠재 건물중 × 잎과 같은 진행도 증가분
  화방축: 화방당 잠재 건물중 × 개화 후 진행도
  과실 : 착과 후 열시간에 대한 beta 함수 (Yin et al. 2003) × 화방당 과실수
         DW 생장은 fruit_TT_growth 에서 끝나고 수확(TT_set_ripe)까지는 착색기
공급
  avail = 버퍼 + 총동화 − 유지호흡  (g CH2O)     → 실제 분배 비율 frac = min(avail/need, 1)
  Q/D   = (총동화 − 유지호흡) / Σ(수요_o / CVF_o)  → 발달 신호 (Kang et al. 2011, 버퍼 제외)
분배
  가용 ≥ 수요 → 모든 기관 잠재 생장, 남는 CH2O는 버퍼(상한 = buf_frac × 잎 DW)
  가용 < 수요 → 과실:영양기관 = π·수요_과실 : 수요_영양 비로 배분 (π = fruit_priority, 1이면 TOMSIM 비례)
"""
import numpy as np

from .params import Params
from .phenology import leaf_area_progress
from .plant import Plant, TrussStage


def beta_cum(t, te, tm):
    """Yin et al. (2003) beta 성장함수 누적형 (0→1), t∈[0,te]"""
    t = np.clip(t, 0, te)
    return (1 + (te - t) / (te - tm)) * (t / te) ** (te / (te - tm))


def fruit_potential_increment(tt0, dtt, p: Params):
    """과실 1개 잠재 DW 증가 (g): 착과 후 열시간 tt0 → tt0+dtt"""
    te, tm = p.fruit_TT_growth, p.fruit_tm_frac * p.fruit_TT_growth
    return p.fruit_W_pot * (beta_cum(tt0 + dtt, te, tm) - beta_cum(tt0, te, tm))


def maintenance(plant: Plant, T: float, p: Params) -> float:
    """유지호흡 g CH2O/주/d (Q10, 25 °C 기준). leaf_resp_fvcb=True 면 잎은 source.py의 FvCB Rd로 계산 → 여기선 제외"""
    f = p.Q10 ** ((T - 25.0) / 10.0)
    leaf = 0.0 if p.leaf_resp_fvcb else sum(n.leaf.dw for n in plant.nodes if n.leaf.active)
    fruit = sum(t.fruit_dw for t in plant.trusses if t.stage < TrussStage.HARVESTED)
    stem = plant.stem_dw + sum(t.struct_dw for t in plant.trusses)
    return f * (p.rm_leaf * leaf + p.rm_stem * stem + p.rm_fruit * fruit)


def demands(plant: Plant, dtt: float, p: Params) -> dict:
    """기관별 잠재 생장 (g DW). 잎은 마디별 dict"""
    leaf = {}
    stem = 0.0
    for n in plant.nodes:
        if not n.leaf.active:
            continue
        a0 = leaf_area_progress(n.leaf.tt_age - dtt, p)
        a1 = leaf_area_progress(n.leaf.tt_age, p)
        if a1 > a0:
            leaf[n.index] = p.leaf_area_pot * (a1 - a0) / p.SLA
            stem += p.stem_dw_per_node * (a1 - a0)
    struct, fruit = {}, {}
    for t in plant.trusses:
        if t.stage == TrussStage.HARVESTED:
            continue
        if TrussStage.ANTHESIS <= t.stage:
            struct[t.index] = p.truss_struct_pot * dtt / p.TT_struct
            if t.struct_dw >= p.truss_struct_pot:
                struct[t.index] = 0.0
        if t.stage == TrussStage.SET:
            fruit[t.index] = t.n_fruit * fruit_potential_increment(t.tt_stage - dtt, dtt, p)
    return dict(leaf=leaf, stem=stem, struct=struct, fruit=fruit)


def allocate(plant: Plant, gross: float, T: float, dtt: float, p: Params) -> dict:
    rm = maintenance(plant, T, p)
    d = demands(plant, dtt, p)
    need_f = sum(d["fruit"].values()) / p.cvf_fruit
    need_v = (sum(d["leaf"].values()) / p.cvf_leaf + d["stem"] / p.cvf_stem
              + sum(d["struct"].values()) / p.cvf_stem)
    need = need_f + need_v
    avail = max(plant.buffer + gross - rm, 0.0)
    if need <= 0:
        qd = 2.0
        frac = frac_f = frac_v = 1.0
    else:
        # 발달 신호 Q/D = 당일 순생산 / 수요 (Kang et al. 2011 정의: 버퍼 제외)
        qd = max(gross - rm, 0.0) / need
        # 실제 분배는 버퍼 포함 가용량으로
        frac = min(avail / need, 1.0)
        if frac >= 1.0:
            frac_f = frac_v = 1.0
        else:
            # 상대 우선권 π: 부족분을 과실:영양 = π·need_f : need_v 로 나눔 (π=1 → TOMSIM 비례 분배)
            share_f = p.fruit_priority * need_f / (p.fruit_priority * need_f + need_v)
            grant_f = min(need_f, avail * share_f)
            grant_v = min(need_v, avail - grant_f)
            grant_f = min(need_f, avail - grant_v)          # 영양기관이 다 못 쓴 몫은 과실로
            frac_f = grant_f / need_f if need_f > 0 else 1.0
            frac_v = grant_v / need_v if need_v > 0 else 1.0

    nodes = {n.index: n for n in plant.nodes}
    for i, g in d["leaf"].items():
        lf = nodes[i].leaf
        lf.dw += g * frac_v
        lf.area += g * frac_v * p.SLA
    plant.stem_dw += d["stem"] * frac_v
    tr = {t.index: t for t in plant.trusses}
    for j, g in d["struct"].items():
        tr[j].struct_dw += g * frac_v
    for j, g in d["fruit"].items():
        tr[j].fruit_dw += g * frac_f

    used = need_f * frac_f + need_v * frac_v
    leaf_dw = sum(n.leaf.dw for n in plant.nodes if n.leaf.active)
    plant.buffer = float(np.clip(avail - used, 0.0, p.buf_frac * leaf_dw))
    return dict(gross=gross, rm=rm, need=need, avail=avail, qd=qd, frac=frac,
                g_leaf=sum(d["leaf"].values()) * frac_v, g_stem=(d["stem"] + sum(d["struct"].values())) * frac_v,
                g_fruit=sum(d["fruit"].values()) * frac_f)
