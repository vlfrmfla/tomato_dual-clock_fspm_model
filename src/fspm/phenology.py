"""
이중 시계 페놀로지

구조 시계  : 마디 출현, 잎 전개, 화방 출현→개화  — 열시간 (마디 출현 간격은 생장 중 과실 수에 따라 φ1→φ2)
생식 시계  : 개화→착과 — 열시간 × g(Q/D),  착과→수확 — 열시간
            g(x) = 1 − exp[−a (x − b)]   (Kang et al. 2011 의 착과확률 형태를 발달 속도 배율로 사용)
"""
import math

from .params import Params
from .plant import Plant, TrussStage


def thermal_time(T: float, p: Params) -> float:
    """일 열시간 (°Cd): Tbase~Topt 선형 증가, Topt~Tmax 선형 감소"""
    if T <= p.Tbase or T >= p.Tmax:
        return 0.0
    if T <= p.Topt:
        return T - p.Tbase
    return (p.Tmax - T) / (p.Tmax - p.Topt) * (p.Topt - p.Tbase)


def source_modifier(qd: float, p: Params) -> float:
    """생식 시계 배율 g(Q/D) ∈ [0, 1)"""
    if not p.use_source_clock:
        return 1.0
    return max(0.0, 1.0 - math.exp(-p.g_a * (qd - p.g_b)))


def leaf_area_progress(tt_age: float, p: Params) -> float:
    """잎 전개 진행도 0→1 (smoothstep 시그모이드)"""
    x = min(max(tt_age / p.TT_leaf_expand, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fruits_growing(plant: Plant, p: Params) -> float:
    """생장 중 과실 수 F (개/주): 착과 후 DM 생장기(fruit_TT_growth)에 있는 화방의 과실 수 합"""
    return sum(t.n_fruit for t in plant.trusses if t.stage == TrussStage.SET and t.tt_stage <= p.fruit_TT_growth)


def phyllochron(tt_since_transplant: float, qd: float, p: Params, F: float = 0.0) -> float:
    """
    'constant'  : 단일 phyllochron
    'two_phase' : 정식 후 열시간 전환점 전후 (대조 모형)
    'qd'        : φ = φ_ref · min(Q/D, 1)^(−β)  — source 제한 시 엽 출현 지연 (Kang 2011 관찰)
    'load'      : φ = φ1 + (φ2 − φ1) · min(1, F/F*)  — 생장 중 과실 수 F에 따른 포화형 전환 (기본)
    """
    if p.phyllo_mode == "load":
        return p.phyllochron_early + (p.phyllochron_late - p.phyllochron_early) * min(1.0, F / p.phyllo_load_F)
    if p.phyllo_mode == "two_phase":
        return p.phyllochron_late if tt_since_transplant >= p.phyllo_switch_TT else p.phyllochron_early
    if p.phyllo_mode == "qd":
        return p.phyllochron_early * max(min(qd, 1.0), 0.05) ** (-p.phyllo_qd_beta)
    return p.phyllochron


def step_structure(plant: Plant, dtt: float, tt_since_transplant: float, date: str, p: Params,
                   qd: float = 1.0) -> int:
    """구조 시계 1일 진행. 반환: 새 마디 수. source 모드에서 잎 면적은 carbon.allocate가 갱신"""
    for n in plant.nodes:
        if n.leaf.active:
            n.leaf.tt_age += dtt
            if not p.use_source:
                n.leaf.area = p.leaf_area_max * leaf_area_progress(n.leaf.tt_age, p)
    F = fruits_growing(plant, p) if p.phyllo_mode == "load" else 0.0
    plant.node_clock += dtt / phyllochron(tt_since_transplant, qd, p, F)
    new = 0
    while plant.node_clock >= 1.0:
        plant.node_clock -= 1.0
        plant.add_node(date, p.truss_first_node, p.truss_interval)
        new += 1
    return new


def step_trusses(plant: Plant, dtt: float, qd: float, date: str, p: Params, color_inc: dict = None) -> list:
    """
    화방 단계 진행. 반환: 오늘 수확 가능해진 화방 목록
    color_inc : {화방 번호: 과실 온도 기준 착색 증분 °Cd} — fruit_temp_ripening 이면
                착과 후 DM 생장기(fruit_TT_growth, 기온 °Cd) 이후 착색기를 과실 온도로 진행
    """
    ripe_today = []
    g = source_modifier(qd, p)
    for t in plant.trusses:
        if t.stage == TrussStage.VISIBLE:
            t.advance(dtt / p.TT_app_anth, dtt, date, TrussStage.ANTHESIS)
        elif t.stage == TrussStage.ANTHESIS:
            t.advance(dtt / p.TT_anth_set * g, dtt, date, TrussStage.SET)
            if t.stage == TrussStage.SET:
                fpt = p.fruits_per_truss
                t.n_fruit = fpt[t.index - 1] if t.index <= len(fpt) else p.fruits_default
        elif t.stage == TrussStage.SET and p.fruit_temp_ripening and p.use_source:
            t.tt_stage += dtt
            if t.tt_stage > p.fruit_TT_growth:
                t.color_tt += (color_inc or {}).get(t.index, 0.0)
            if t.color_tt >= p.TT_color:
                t.stage = TrussStage.RIPE
                t.dates[TrussStage.RIPE.name] = date
                ripe_today.append(t)
        elif t.stage == TrussStage.SET:
            t.advance(dtt / p.TT_set_ripe, dtt, date, TrussStage.RIPE)
            if t.stage == TrussStage.RIPE:
                ripe_today.append(t)
    return ripe_today
