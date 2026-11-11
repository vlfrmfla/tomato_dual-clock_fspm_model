"""
적엽 규칙 모듈

모든 규칙은 "남길 최하위 마디"를 정하는 결정(decide)과, 그것을 작업 패턴
(작업 요일·주기, 1회 최대 제거 수)으로 집행하는 부분(execute)으로 나뉜다.
결정 = 무엇을 원하는가 (규칙마다 다름) / 집행 = 사람이 실제로 언제 얼마나 하는가 (공통)

R0 ObservedTrajectory : 관측된 현재 엽수 궤적 재현 (보정·검증용 기준선)
R1 KeepN              : 목표 엽수 유지 — 상수 또는 온도 함수 (Jeong & Choi 2026)
R2 LightThreshold     : 최하위 잎 주간 광적산 < 광보상점 (Kim & Kubota 2025)
R3 TrussLinked        : 단계 ≥ s* 인 최상위 화방 아래 잎 제거 (관행; 되먹임 고리 형성)
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .params import Params
from .plant import Plant, TrussStage


@dataclass
class Context:
    date: pd.Timestamp
    T_roll7: float                     # 7일 이동평균 기온
    harvest_started: bool
    leaf_weekly_par: Optional[dict] = None   # node → mol m⁻² week⁻¹ (광 모듈 연결 후)


class PruningRule:
    name = "base"

    def decide(self, plant: Plant, ctx: Context) -> List[int]:
        """제거를 원하는 잎의 마디 번호 (아래부터)"""
        raise NotImplementedError

    def execute(self, plant: Plant, ctx: Context, p: Params) -> int:
        day = ctx.date.weekday()
        if day != p.prune_weekday:
            return 0
        want = self.decide(plant, ctx)
        if len(want) < p.prune_min_batch:
            return 0
        want = sorted(want)[: p.prune_max_per_event]
        return plant.remove_leaves(want, str(ctx.date.date()))


def _lowest_active(plant: Plant, k: int) -> List[int]:
    return [n.index for n in plant.nodes if n.leaf.active][: max(k, 0)]


class NoPruning(PruningRule):
    name = "none"

    def decide(self, plant, ctx):
        return []


class ObservedTrajectory(PruningRule):
    """파괴조사 현재 엽수(일자별 평균)를 선형 보간한 목표를 따라감"""
    name = "R0_observed"

    def __init__(self, dates: pd.Series, leaves: pd.Series):
        s = pd.Series(leaves.values, index=pd.to_datetime(dates.values)).sort_index()
        self.target = s

    def decide(self, plant, ctx):
        s = self.target
        if ctx.date < s.index[0]:
            return []
        x = np.interp(ctx.date.value, s.index.view("int64"), s.values)
        n_active = len(plant.active_leaves())
        return _lowest_active(plant, n_active - int(round(x)))


class KeepN(PruningRule):
    """목표 엽수 유지. n_keep: 상수 또는 f(T_roll7). trigger: 'harvest' | 날짜 문자열"""
    name = "R1_keepN"

    def __init__(self, n_keep, trigger="harvest"):
        self.n_keep: Callable[[float], int] = n_keep if callable(n_keep) else (lambda T, k=n_keep: k)
        self.trigger = trigger

    @staticmethod
    def jeong2026(T: float) -> int:
        """Jeong & Choi (2026) Eq. 8"""
        if T >= 20:
            return 18
        if T > 18:
            return 16
        return 14

    def _armed(self, ctx):
        if self.trigger == "harvest":
            return ctx.harvest_started
        return ctx.date >= pd.Timestamp(self.trigger)

    def decide(self, plant, ctx):
        if not self._armed(ctx):
            return []
        n_active = len(plant.active_leaves())
        return _lowest_active(plant, n_active - self.n_keep(ctx.T_roll7))


class LightThreshold(PruningRule):
    """최하위 잎부터 주간 광적산이 광보상점 기준 미만이면 제거 (광 모듈 필요)"""
    name = "R2_light"

    def __init__(self, wli_min: float = 11.2):   # mol m⁻² week⁻¹ PAR, Kim & Kubota 2025
        self.wli_min = wli_min

    def decide(self, plant, ctx):
        if ctx.leaf_weekly_par is None:
            raise RuntimeError("R2 규칙은 광흡수 모듈(leaf_weekly_par)이 필요")
        out = []
        for n in plant.nodes:
            if not n.leaf.active:
                continue
            if ctx.leaf_weekly_par.get(n.index, np.inf) < self.wli_min:
                out.append(n.index)
            else:
                break        # 아래부터 연속된 잎만 제거
        return out


class TrussLinked(PruningRule):
    """
    착과 후 tt_after °Cd 이상 지난 화방 중 가장 위 화방의 마디까지(포함) 잎을 모두 제거.
    관측(A7-6): 적엽 시점에 기준 화방은 착과 후 231·240 °Cd → tt_after ≈ 235
    """
    name = "R3_truss"

    def __init__(self, tt_after: float = 235.0, offset: int = 0):
        self.tt_after = tt_after
        self.offset = offset

    def decide(self, plant, ctx):
        ok = [t for t in plant.trusses
              if t.stage > TrussStage.SET
              or (t.stage == TrussStage.SET and t.tt_stage >= self.tt_after)]
        if not ok:
            return []
        limit = max(t.node for t in ok) - self.offset
        return [n.index for n in plant.nodes if n.leaf.active and n.index <= limit]
