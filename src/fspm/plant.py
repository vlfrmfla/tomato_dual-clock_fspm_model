"""
식물 구조: 마디(Node) 단위로 잎·화방을 추적.
마디 번호는 아래(1)→위(N)로 고유하며 적엽 후에도 유지된다.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


class TrussStage(IntEnum):
    VISIBLE = 0      # 화방 출현
    ANTHESIS = 1     # 개화
    SET = 2          # 착과
    RIPE = 3         # 수확 가능
    HARVESTED = 4


@dataclass
class Leaf:
    node: int
    tt_age: float = 0.0            # 출현 후 °Cd
    area: float = 0.0              # m²
    dw: float = 0.0                # g
    removed_on: Optional[str] = None
    net_hist: list = field(default_factory=list)   # 최근 7일 일 순탄소수지 g CH2O (총동화 − FvCB Rd)

    @property
    def active(self) -> bool:
        return self.removed_on is None


@dataclass
class Truss:
    index: int                     # 화방 번호 (1 = 최하위)
    node: int
    stage: TrussStage = TrussStage.VISIBLE
    clock: float = 0.0             # 현 단계 진행도 (0→1)
    tt_stage: float = 0.0          # 현 단계 진입 후 누적 °Cd (적엽 규칙 R3가 참조)
    n_fruit: float = 0.0           # 착과 수 (적과 후, 관리 입력)
    fruit_dw: float = 0.0          # g
    struct_dw: float = 0.0         # 화방축 g
    color_tt: float = 0.0          # 착색기 누적 (과실 온도 기준 °Cd)
    hot_hours: float = 0.0         # 착색기 과실 온도 > color_hot 누적 시간
    dT_sum: float = 0.0            # 착색기 일별 주간 평균 (과실 − 기온) 합
    dT_days: int = 0
    dates: dict = field(default_factory=dict)   # 단계 → 도달일

    def advance(self, rate: float, dtt: float, date: str, stage_next: TrussStage):
        self.clock += rate
        self.tt_stage += dtt
        if self.clock >= 1.0:
            self.clock -= 1.0
            self.tt_stage = 0.0
            self.stage = stage_next
            self.dates[stage_next.name] = date


@dataclass
class Node:
    index: int
    leaf: Leaf
    truss: Optional[Truss] = None


@dataclass
class Plant:
    nodes: List[Node] = field(default_factory=list)
    trusses: List[Truss] = field(default_factory=list)
    node_clock: float = 0.0        # 다음 마디까지 진행도 (°Cd/phyllochron)
    next_truss_node: float = 0.0   # 다음 화방이 달릴 마디 (실수, 누적)
    stem_dw: float = 0.0           # g
    buffer: float = 0.0            # 비구조 탄수화물 g CH2O
    removed_leaf_dw: float = 0.0   # 누적 적엽 DW g
    harvested_dw: float = 0.0      # 누적 수확 과실 DW g
    truss_log: list = field(default_factory=list)   # (date, truss, fruit_dw) 일별 기록

    # --- 조회 ---
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def active_leaves(self) -> List[Leaf]:
        return [n.leaf for n in self.nodes if n.leaf.active]

    def leaf_area(self) -> float:
        return sum(l.area for l in self.active_leaves())

    def lowest_active_leaf_node(self) -> Optional[int]:
        for n in self.nodes:
            if n.leaf.active:
                return n.index
        return None

    def truss_at_stage(self, stage: TrussStage) -> List[Truss]:
        return [t for t in self.trusses if t.stage == stage]

    # --- 구조 생성 ---
    def add_node(self, date: str, truss_first: float, truss_interval: float) -> Node:
        idx = self.n_nodes + 1
        node = Node(index=idx, leaf=Leaf(node=idx))
        if self.next_truss_node == 0.0:
            self.next_truss_node = truss_first
        if idx >= round(self.next_truss_node):
            tr = Truss(index=len(self.trusses) + 1, node=idx)
            tr.dates[TrussStage.VISIBLE.name] = date
            node.truss = tr
            self.trusses.append(tr)
            self.next_truss_node += truss_interval
        self.nodes.append(node)
        return node

    # --- 관리 ---
    def remove_leaves(self, node_ids: List[int], date: str) -> int:
        k = 0
        targets = set(node_ids)
        for n in self.nodes:
            if n.index in targets and n.leaf.active:
                n.leaf.removed_on = date
                self.removed_leaf_dw += n.leaf.dw
                k += 1
        return k

    def leaf_dw(self) -> float:
        return sum(l.dw for l in self.active_leaves())

    def fruit_dw(self) -> float:
        return sum(t.fruit_dw for t in self.trusses if t.stage < TrussStage.HARVESTED)
