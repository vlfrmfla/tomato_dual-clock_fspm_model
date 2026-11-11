"""
논문 그림 (영문, COMPAG 폭 190 mm) — figures/paper/Fig1–7.png/pdf, FigS1

Fig 1  (a) 일반 개념: 적엽 한 번이 바꾸는 세 경로  (b) 모델 흐름
Fig 2  구조 시계와 관찰된 적엽 관행: 누적 마디, 화방, 현재 엽수(규칙 비교)
Fig 3  엽위별 잎의 가치: 수광(Beer vs 열), 순탄소수지, 낮·밤 증산, 누적 기여 (계절별, 30장)
Fig 4  생장·증산 검증 (함안)
Fig 5  생식 시계: 화방별 개화→착과, LOPO 개체 오차, 외부 온실 화방 진행, 개화 후 1주 일사 구간별 기간
Fig 6  두 번째 온실 (횡계, 여름 정식)
Fig 7  계절 시나리오: 과실 생장–증산 절감 트레이드오프, 야간 증산, 과실 30 °C 초과, 숙기 단축
Fig S1 Rd 식별성
사용: python paper_figs.py [1 2 3 ...]  (인자 없으면 전부)
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fspm import Params, simulate, truss_table          # noqa: E402
from fspm import management as M                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
OUT = ROOT / "figures/paper"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "font.family": ["Arial", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
    "legend.frameon": True, "legend.framealpha": 0.85, "legend.edgecolor": "none", "legend.fancybox": False,
    "legend.borderpad": 0.3, "legend.handlelength": 1.6, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.unicode_minus": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
}


def style():
    """다른 모듈 import가 rcParams(한글 폰트)를 바꾸므로 그림마다 다시 적용"""
    matplotlib.rcParams.update(STYLE)


style()
W2 = 7.48                                    # 190 mm
INK, GREY = "#1d2420", "#7a837e"
SEA = {"winter": ("#2a78d6", "Winter (Haman)"), "spring": ("#1baf7a", "Spring (Haman)"),
       "summer": ("#eb6834", "Summer-planted (Hoenggye)")}
CA, CB = "#eb6834", "#2a78d6"                # model A / model B


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def tag(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")


def _unclip(ax):
    """한계 위(눈금 끝)에 놓인 점·선이 반쯤 잘리지 않게 — 한계 안에 전부 들어오는 요소만 clip 해제"""
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    tx, ty = 1e-9 * (x1 - x0), 1e-9 * (y1 - y0)

    def inside(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        return ((xy[:, 0] >= x0 - tx) & (xy[:, 0] <= x1 + tx) & (xy[:, 1] >= y0 - ty) & (xy[:, 1] <= y1 + ty)).all()

    for ln in ax.lines:
        if ln.get_transform() == ax.transData and inside(ln.get_xydata()):
            ln.set_clip_on(False)
    for c in ax.collections:
        if isinstance(c, matplotlib.collections.LineCollection):
            pts = [s for s in c.get_segments() if len(s)]
            pts = np.vstack(pts) if pts else np.zeros((0, 2))
        elif isinstance(c, matplotlib.collections.PathCollection):
            pts = c.get_offsets()
        else:
            continue
        if inside(pts):
            c.set_clip_on(False)


def snap(ax, x=True, y=True, xlim=None, ylim=None):
    """축 양 끝 = 눈금 (사용자 요청 2026-10-17)
    데이터 범위(+ xlim/ylim 로 지정한 꼭 포함할 값, 예: 범례 여백)를 감싸는 가장 가까운 눈금 배수로 한계를 정하고
    같은 간격의 눈금으로 고정. 0 바로 아래의 작은 음수(오차 막대)는 0으로 봄"""
    for which, do, need in (("x", x, xlim), ("y", y, ylim)):
        if not do:
            continue
        axis = ax.xaxis if which == "x" else ax.yaxis
        lo, hi = (ax.dataLim.intervalx if which == "x" else ax.dataLim.intervaly)
        data_lo = lo
        if need is not None:
            lo = lo if need[0] is None else min(lo, need[0])
            hi = hi if need[1] is None else max(hi, need[1])
        if lo < 0 <= hi and -lo < 0.03 * (hi - lo):
            lo = 0.0
        ticks = np.asarray(axis.get_major_locator().tick_values(lo, hi), float)
        if len(ticks) < 2:
            continue
        step = float(np.median(np.diff(ticks)))
        new_lo = np.floor(lo / step + 1e-9) * step
        new_hi = np.ceil(hi / step - 1e-9) * step
        if which == "y" and data_lo - new_lo < 0.03 * (new_hi - new_lo):
            new_lo = data_lo - 0.04 * (new_hi - new_lo)      # 자료가 x축 선에 붙으면 아래만 여백 (눈금 맞춤 예외, 사용자 요청)
        (ax.set_xlim if which == "x" else ax.set_ylim)(new_lo, new_hi)
        axis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
    _unclip(ax)


def datefmt(ax):
    """날짜 축: 매월 1일 주 눈금(라벨) + 15일 보조 눈금, 양 끝 = 데이터를 감싸는 1일 또는 15일"""
    lo, hi = ax.dataLim.intervalx
    t0 = pd.Timestamp(matplotlib.dates.num2date(lo)).tz_localize(None)
    t1 = pd.Timestamp(matplotlib.dates.num2date(hi)).tz_localize(None)
    s0 = t0.normalize().replace(day=1 if t0.day < 15 else 15)
    t1 = t1.normalize() + (pd.Timedelta(days=1) if t1 > t1.normalize() else pd.Timedelta(0))
    s1 = t1 if t1.day in (1, 15) else (t1.replace(day=15) if t1.day < 15 else t1 + pd.offsets.MonthBegin(1))
    ax.set_xlim(s0, s1)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator(bymonthday=[15]))
    _unclip(ax)


def haman():
    import calibrate_m2 as C
    style()
    clim, hr, obs, fin, ev = C.load()
    pj = json.loads((DP / "params_m2.json").read_text())
    p = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    return C, clim, hr, obs, fin, ev, pj, p


# ------------------------------------------------------------------ Fig 1
def fig1():
    fig = plt.figure(figsize=(W2, 4.3))
    ax = fig.add_axes([0.0, 0.0, 0.40, 0.96])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.grid(False)
    ax.text(0.1, 9.85, "a", fontsize=10, fontweight="bold", va="top")
    x0, z0, dz = 3.0, 2.0, 0.335
    nodes = 22
    ax.plot([x0, x0], [z0, z0 + nodes * dz + 0.2], color=INK, lw=1.6)
    cut = 7                                      # nodes ≤ cut removed
    for i in range(1, nodes + 1):
        z = z0 + i * dz
        side = 1 if i % 2 else -1
        size = min(1.0, 0.35 + 0.65 * min(1, (nodes - i) / 9))      # young leaves at the top are small
        removed = i <= cut
        e = Ellipse((x0 + side * (0.55 * size + 0.05), z), 1.1 * size, 0.28 * size,
                    fc="none" if removed else "#8fc9a8", ec=GREY if removed else "#2f6b4f",
                    ls=":" if removed else "-", lw=0.8)
        ax.add_patch(e)
        if (i - 3) % 3 == 0 and i >= 3:
            col = "#c0392b" if i <= 9 else ("#eb6834" if i <= 15 else "#9bc46b")
            for k in range(3):
                ax.add_patch(Ellipse((x0 + 0.32 + 0.22 * k, z - 0.38), 0.2, 0.2, fc=col, ec="none"))
    zc = z0 + (cut + 0.5) * dz
    ax.plot([x0 - 1.6, x0 + 1.6], [zc, zc], color="#c0392b", lw=1.2, ls="--")
    ax.text(x0 - 1.65, zc + 0.08, "pruning\nline", color="#c0392b", fontsize=7, ha="right", va="bottom")
    # decision rules box
    ax.add_patch(FancyBboxPatch((0.15, 0.1), 4.6, 1.45, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc="#f6e6e0", ec="#c0392b", lw=0.8))
    ax.text(0.3, 1.45, "Decision rule (weekly)", fontsize=7.2, fontweight="bold", color="#c0392b", va="top")
    ax.text(0.3, 1.05, "R1 keep N leaves\nR2 light below threshold\nR3 truss set + 235 °C d", fontsize=6.6, va="top")
    # three pathways
    boxes = [
        (8.55, "Carbon", "#2f6b4f", "lower leaves: little light,\nrespiration continues\n→ source–sink ratio\n→ fruit-set rate", (x0 + 1.0, z0 + 10 * dz)),
        (5.75, "Water", "#2a78d6", "fewer leaves: less day\ntranspiration and night\nhumidity load\n→ irrigation volume", (x0 + 1.05, z0 + 15 * dz)),
        (2.95, "Truss microclimate", "#eb6834", "exposed trusses: higher\nfruit temperature\n→ ripening rate,\n> 30 °C quality risk", (x0 + 0.85, z0 + 6 * dz - 0.38)),
    ]
    for yb, title, col, body, target in boxes:
        ax.add_patch(FancyBboxPatch((5.35, yb - 1.25), 4.45, 1.95, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc="white", ec=col, lw=1.0))
        ax.text(5.5, yb + 0.55, title, fontsize=7.4, fontweight="bold", color=col, va="top")
        ax.text(5.5, yb + 0.12, body, fontsize=6.5, va="top", linespacing=1.25)
        ax.add_patch(FancyArrowPatch((5.35, yb - 0.3), target, arrowstyle="-|>", mutation_scale=7, color=col, lw=0.8,
                                     connectionstyle="arc3,rad=0.15"))
    ax.annotate("", xy=(x0 - 0.6, zc - 0.02), xytext=(1.6, 1.6), arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=0.8))

    # (b) flow
    bx = fig.add_axes([0.42, 0.0, 0.58, 0.96])
    bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis("off"); bx.grid(False)
    bx.text(0.0, 9.85, "b", fontsize=10, fontweight="bold", va="top")

    def box(x, y, w, h, title, body, col):
        bx.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=col, alpha=0.12, ec=col, lw=0.9))
        bx.text(x + 0.1, y + h - 0.1, title, fontsize=7.2, fontweight="bold", color=col, va="top")
        bx.text(x + 0.1, y + h - 0.5, body, fontsize=6.2, va="top", linespacing=1.3)

    def arr(p, q, col=GREY, rad=0.0):
        bx.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=col, lw=0.8, connectionstyle=f"arc3,rad={rad}"))

    box(0.3, 8.25, 9.4, 1.25, "Hourly climate-computer records",
        "air temperature, humidity, outside radiation, screen positions, CO$_2$", "#52514e")
    box(0.3, 5.7, 3.0, 2.15, "Light (row canopy)",
        "Beer's law along\ndirectional paths through\nhedgerows (direct + 18\nsky directions)\n→ irradiance per leaf\n   and per coloring truss", "#b08400")
    box(3.55, 5.7, 3.0, 2.15, "Leaf processes",
        "FvCB photosynthesis\ndark respiration\n(leaf age, temperature)\nPenman–Monteith\ntranspiration\n7-d net carbon balance", "#1baf7a")
    box(6.8, 5.7, 2.9, 2.15, "Carbon partitioning",
        "buffer, maintenance\nrespiration, fruit priority\nsource–sink ratio Q/D\n(5-d moving average)", "#2a78d6")
    box(0.3, 3.05, 3.0, 2.2, "Structural clock",
        "thermal time;\ntwo-phase phyllochron\n(switch at onset of\nfruit growth)\nnodes, leaves, trusses", "#52514e")
    box(3.55, 3.05, 3.0, 2.2, "Reproductive clock",
        "anthesis → set:\n  rate = ΔTT/TT$_{set}$ · g(Q/D)\nset → end of growth:\n  beta function\ncoloring → harvest:\n  fruit thermal time", "#eb6834")
    box(6.8, 3.05, 2.9, 2.2, "Pruning rules",
        "R1 keep N leaves\nR2 light threshold\nR3 truss-linked\n(work schedule,\nbatches)", "#c0392b")
    box(0.3, 0.2, 9.4, 1.65, "Decision-relevant outputs",
        "leaves present, leaf area and organ dry weights · fruit-set and harvest dates\n"
        "day and night transpiration · fruit temperature, hours > 30 °C\n→ pruning and irrigation guidance", "#52514e")
    for x in (1.8, 5.05, 8.25):
        arr((x, 8.25), (x, 7.85))
    arr((3.3, 6.8), (3.55, 6.8)); arr((6.55, 6.8), (6.8, 6.8))
    arr((8.25, 5.7), (5.6, 5.25), "#2a78d6"); bx.text(7.0, 5.43, "Q/D", color="#2a78d6", fontsize=6.8)
    arr((1.8, 5.25), (1.8, 5.7), "#b08400"); bx.text(1.9, 5.36, "canopy structure", fontsize=6, color=GREY)
    arr((3.3, 5.95), (4.6, 5.25), "#b08400")
    arr((3.3, 4.1), (3.55, 4.1)); arr((6.55, 4.1), (6.8, 4.1), "#eb6834")
    arr((8.25, 3.05), (1.8, 3.05), "#c0392b", rad=-0.18)
    bx.text(5.05, 2.18, "leaf removal → fewer leaf layers, exposed trusses", color="#c0392b", fontsize=6.3, ha="center",
            bbox=dict(fc="white", ec="none", pad=0.3))
    save(fig, "Fig1")


def figS2():
    """모델 과정도 (형태 · 생리 · 이중 시계 페놀로지, 화살표에 식 번호) — 보충 그림 (2026-11-06, 이전 일 단위 흐름도 대체)
    값은 최종 파라미터(params_m2.json + Params 기본값)에서 읽음. 엽출현 간격은 생장 중 과실 수 F에 따라 φ1→φ2"""
    from matplotlib.patches import Polygon
    style()
    pj = json.loads((DP / "params_m2.json").read_text())
    pp = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
    OR, BLUE, GOLD, HEAD = "#eb6834", CB, "#b08400", "#e6e9e7"
    fig = plt.figure(figsize=(W2, 6.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 92); ax.axis("off"); ax.grid(False)

    def frame(x0, y0, x1, y1, head, hh=3.6):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="square,pad=0", fc="none", ec=GREY, lw=0.8))
        ax.add_patch(FancyBboxPatch((x0, y1 - hh), x1 - x0, hh, boxstyle="square,pad=0", fc=HEAD, ec=GREY, lw=0.8))
        ax.text((x0 + x1) / 2, y1 - hh / 2, head, ha="center", va="center", fontsize=7, weight="bold", color=INK)

    def box(xc, yc, w, h, title, sub=None, fc="white", ec=INK, lw=0.8):
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        if sub:
            ax.text(xc, yc + h * 0.17, title, ha="center", va="center", fontsize=6.4, weight="bold", color=INK, zorder=4)
            ax.text(xc, yc - h * 0.22, sub, ha="center", va="center", fontsize=5.5, color=INK, zorder=4)
        else:
            ax.text(xc, yc, title, ha="center", va="center", fontsize=6.4, weight="bold", color=INK, zorder=4)

    def arrow(pts, color=INK, lw=0.7, label=None, lpos=None, lkw=None):
        """pts: 꺾은선 꼭짓점, 마지막 구간에 화살촉"""
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="butt", zorder=2)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=6.5, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0, zorder=2))
        if label:
            kw = dict(fontsize=5.5, color=color, ha="center", va="center", zorder=5, bbox=dict(fc="white", ec="none", pad=0.4))
            kw.update(lkw or {})
            ax.text(*lpos, label, **kw)

    # 입력 · 틀
    ax.add_patch(FancyBboxPatch((1, 84), 85, 6, boxstyle="round,pad=0,rounding_size=0.7", fc="#f4f5f4", ec=GREY, lw=0.8))
    ax.text(43.5, 88.3, "Climate computer records (hourly)", ha="center", va="center", fontsize=7, weight="bold", color=INK)
    ax.text(43.5, 85.8, "air temperature · humidity (VPD) · CO$_2$ · outside radiation · screen positions",
            ha="center", va="center", fontsize=6.2, color=INK)
    frame(7, 52, 35.5, 80.5, "Morphology: canopy structure")
    frame(7, 1, 35.5, 49.5, "Physiology (hourly, per leaf)")
    frame(40, 1, 99, 80.5, "Phenology: dual clock (daily)")

    # 이중 시계
    box(64, 70.5, 43, 5.0, "Thermal time ΔTT (Eq. 1)", f"T$_b$ {pp.Tbase:.0f}, T$_{{opt}}$ {pp.Topt:.0f}, T$_{{max}}$ {pp.Tmax:.0f} °C",
        fc="#fbfbfb")
    LX, TX, BW, BH = 52.5, 76, 19, 5.6
    h2 = BH / 2
    ax.text(43.2, 65.3, "Structural clock", ha="left", va="center", fontsize=6.4, style="italic", color=INK)
    ax.text(66.2, 65.3, "Truss clock\n(each truss)", ha="left", va="center", fontsize=6.4, style="italic", color=INK, linespacing=1.0)
    L = {"app": 58.5, "exp": 48.5, "age": 38.5, "rem": 28.5}
    T = {"app": 58.5, "anth": 48.5, "set": 38.5, "grow": 28.5, "col": 18.5, "harv": 8.5}
    box(LX, L["app"], BW, BH, "Node and leaf appearance",
        f"φ$_1$ {pp.phyllochron_early:.1f} → φ$_2$ {pp.phyllochron_late:.1f} °C d node$^{{-1}}$")
    box(LX, L["exp"], BW, BH, "Leaf expansion", f"to A$_{{pot}}$ over {pp.TT_leaf_expand:.0f} °C d; area = DW × SLA")
    box(LX, L["age"], BW, BH, "Leaf aging",
        f"capacity and R$_d$ × f$_{{age}}$ ({pp.cap_decline_start:.0f}–{pp.cap_decline_end:.0f} °C d)")
    box(LX, L["rem"], BW, BH, "Leaf removal", "pruning rule R1–R3 on the work day", ec=OR, lw=1.1)
    box(TX, T["app"], 20, BH, "Truss appearance",
        f"on node {pp.truss_first_node:.1f}, {pp.truss_first_node + pp.truss_interval:.1f}, … (structural clock)")
    box(TX, T["anth"], 20, BH, "Anthesis", f"{pp.TT_app_anth:.0f} °C d after appearance (T only)")
    box(TX, T["set"], 20, BH, "Fruit set", "dr/dt = ΔTT / TT$_{set}$ · g($\\overline{Q/D}$) (Eq. 2)", ec=BLUE, lw=1.1)
    box(TX, T["grow"], 20, BH, "Fruit dry-matter growth", "duration TT$_{fg}$ by T; rate by assimilates")
    box(TX, T["col"], 20, BH, "Coloring", "fruit thermal time > 5.7 °C to TT$_{col}$")
    box(TX, T["harv"], 20, BH, "Harvest")
    arrow([(81, 84), (81, 73)], label="T", lpos=(82.3, 82), lkw=dict(fontsize=6))
    arrow([(LX + 6, 68), (LX + 6, L["app"] + h2)])
    arrow([(TX + 10.6, 68), (87.0, 68), (87.0, T["anth"]), (TX + 10, T["anth"])], label="ΔTT", lpos=(87.0, 56),
          lkw=dict(fontsize=5.6, rotation=90))
    for a_, b_ in zip(list(L.values())[:-1], list(L.values())[1:]):
        arrow([(LX, a_ - h2), (LX, b_ + h2)])
    for a_, b_ in zip(list(T.values())[:-1], list(T.values())[1:]):
        arrow([(TX, a_ - h2), (TX, b_ + h2)])
    arrow([(LX + BW / 2, L["app"] + 1.2), (TX - 10, T["app"] + 1.2)])            # 마디 → 화방 출현
    ax.text(64, 62.9, f"every\n{pp.truss_interval:.2f} nodes", ha="center", va="center", fontsize=5.0, color=INK, linespacing=0.95)
    # 과실 부하 → 마디 출현 간격 (생장 중 과실 수 F)
    arrow([(TX - 10, T["set"] + 2.1), (63.7, T["set"] + 2.1), (63.7, L["app"] - 1.9), (LX + BW / 2, L["app"] - 1.9)], color=GOLD,
          lw=1.1, label="growing fruits F", lpos=(63.7, 48.5), lkw=dict(fontsize=5.3, rotation=90))
    # R3: 착과 후 235 °C d 지난 화방까지 잎 제거
    arrow([(TX - 10, T["set"] - 1.6), (63.0, T["set"] - 1.6), (63.0, L["rem"] + 0.8), (LX + BW / 2, L["rem"] + 0.8)], color=OR, lw=1.1,
          label="R3", lpos=(63.0, 33.2), lkw=dict(fontsize=5.6))

    # 생육 단계 화살표 (오른쪽)
    ax.add_patch(Polygon([(90, 72), (97, 72), (97, 9), (99, 9), (93.5, 3), (88, 9), (90, 9)], closed=True, fc=HEAD, ec=GREY, lw=0.8))
    ax.text(93.5, 74.5, "Transplanting", ha="center", va="center", fontsize=6, color=INK)
    for y, s in ((61, "Vegetative (φ$_1$)"), (37, f"Fruit load (φ → φ$_2$ as F → {pp.phyllo_load_F:.0f} fruits)"),
                 (14, "Harvest period")):
        ax.text(93.5, y, s, ha="center", va="center", fontsize=6.3, rotation=270, color=INK)

    # 형태 (왼쪽 위)
    MX, MW, MH = 21.25, 25, 5.0
    Mo = {"n": 71.5, "a": 63.5, "lad": 55.5}
    box(MX, Mo["n"], MW, MH, "Leaves present", "identified nodes, top to bottom")
    box(MX, Mo["a"], MW, MH, "Area of each leaf, A$_i$")
    box(MX, Mo["lad"], MW, MH, "Hedgerow canopy, LAD$_i$", "row spacing, hedgerow width, Δz (Eq. 3)")
    arrow([(MX, Mo["n"] - MH / 2), (MX, Mo["a"] + MH / 2)])
    arrow([(MX, Mo["a"] - MH / 2), (MX, Mo["lad"] + MH / 2)])
    RE = MX + MW / 2
    arrow([(LX - BW / 2, L["app"] + 1.2), (38.9, L["app"] + 1.2), (38.9, Mo["n"] + 1.1), (RE, Mo["n"] + 1.1)])
    arrow([(LX - BW / 2, L["exp"]), (37.7, L["exp"]), (37.7, Mo["a"]), (RE, Mo["a"])])
    arrow([(LX - BW / 2, L["rem"]), (36.5, L["rem"]), (36.5, Mo["n"] - 1.1), (RE, Mo["n"] - 1.1)], color=OR, lw=1.1,
          label="removes", lpos=(36.5, 42), lkw=dict(fontsize=5.3, rotation=90))

    # 생리 (왼쪽 아래)
    PX, PW, PH = 21.25, 25, 4.4
    Ph = {"light": 41.0, "pg": 35.0, "resp": 29.0, "part": 23.0, "qd": 17.0, "et": 11.0, "tf": 5.0}
    box(PX, Ph["light"], PW, PH, "Light in the row canopy (Eq. 4)", "directional Beer's law, floor reflection")
    box(PX, Ph["pg"], PW, PH, "Gross photosynthesis", "FvCB; PAR, T, CO$_2$; leaf age")
    box(PX, Ph["resp"], PW, PH, "Respiration", "leaf R$_d$ (Eq. 5); stem, fruit maintenance")
    box(PX, Ph["part"], PW, PH, "Assimilate buffer and partitioning", "potential growth; fruit priority π")
    box(PX, Ph["qd"], PW, PH, "Source–sink ratio Q/D (Eq. 6)", "5-d moving average", ec=BLUE, lw=1.1)
    box(PX, Ph["et"], PW, PH, "Transpiration, day and night (Eq. 7)", "Penman–Monteith per leaf")
    box(PX, Ph["tf"], PW, PH, "Fruit temperature (Eq. 8)", "irradiance at the truss")
    arrow([(MX + 11, Mo["lad"] - MH / 2), (MX + 11, Ph["light"] + PH / 2)])
    for a_, b_ in (("light", "pg"), ("pg", "resp"), ("resp", "part"), ("part", "qd")):
        arrow([(PX, Ph[a_] - PH / 2), (PX, Ph[b_] + PH / 2)])
    LE = PX - PW / 2
    arrow([(LE, Ph["light"]), (8.0, Ph["light"]), (8.0, Ph["et"]), (LE, Ph["et"])], color=GREY)
    arrow([(8.0, Ph["et"]), (8.0, Ph["tf"]), (LE, Ph["tf"])], color=GREY)
    arrow([(3.5, 84), (3.5, 30), (7, 30)], label="radiation, T, VPD, CO$_2$", lpos=(3.5, 57), lkw=dict(rotation=90, fontsize=5.6))
    arrow([(RE, Ph["part"] + 0.9), (41.5, Ph["part"] + 0.9), (41.5, L["exp"] - 1.3), (LX - BW / 2, L["exp"] - 1.3)],
          label="leaf DW", lpos=(41.5, 41.5), lkw=dict(fontsize=5.3, rotation=90))
    arrow([(RE, Ph["part"] - 0.9), (62.3, Ph["part"] - 0.9), (62.3, T["grow"] - 1.0), (TX - 10, T["grow"] - 1.0)],
          label="fruit DW", lpos=(52, Ph["part"] - 0.9), lkw=dict(fontsize=5.3))
    arrow([(RE, Ph["qd"]), (64.6, Ph["qd"]), (64.6, T["set"] + 0.2), (TX - 10, T["set"] + 0.2)], color=BLUE, lw=1.1,
          label="g(Q/D), Eq. 2", lpos=(52, Ph["qd"]), lkw=dict(fontsize=5.6))
    arrow([(RE, Ph["tf"]), (87.2, Ph["tf"]), (87.2, T["col"]), (TX + 10, T["col"])],
          label="Eq. 8", lpos=(60, Ph["tf"]), lkw=dict(fontsize=5.6))

    # 범례
    for y, c, lw, s in ((90.0, INK, 0.7, "flow"), (88.2, BLUE, 1.1, "source–sink signal"),
                        (86.4, GOLD, 1.1, "fruit-load signal"), (84.6, OR, 1.1, "leaf pruning")):
        ax.plot([87.5, 90.5], [y, y], color=c, lw=lw); ax.text(91.1, y, s, fontsize=5.6, va="center", color=INK)
    save(fig, "FigS2")



# ------------------------------------------------------------------ Fig 2
def fig2():
    import run_m1 as R1
    clim = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"]).set_index("date")
    nodes_obs, tr_obs, des_obs = R1.load_obs()
    hr = R1.load_hourly()
    p = R1.final_params()                                   # 최종 모델 (source 연결; 마디 출현 = 과실 부하 φ)
    batch = R1.final_params(prune_min_batch=6)
    rules = [("Keep 18 leaves (R1)", M.KeepN(18, trigger="2023-11-10"), p, "#7a837e", ":"),
             ("Temperature-based N(T) (Jeong and Choi, 2026)", M.KeepN(M.KeepN.jeong2026, trigger="harvest"), p, "#eda100", "-."),
             ("Truss-linked, weekly (R3)", M.TrussLinked(235), p, "#2a78d6", "--"),
             ("Truss-linked, batches (R3)", M.TrussLinked(235), batch, "#c0392b", "-")]
    style()
    sims = {nm: R1.run(clim, hr, pp, r)[0] for nm, r, pp, _, _ in rules}
    s2 = sims["Truss-linked, batches (R3)"]
    sc, _ = R1.run(clim, hr, R1.final_params(phyllo_mode="constant"), M.NoPruning())
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6))
    a = ax[0]
    a.errorbar(nodes_obs.index, nodes_obs["mean"], nodes_obs["std"], fmt="o", ms=2.5, color=INK, elinewidth=0.6, label="Observed (n = 8)")
    a.plot(sc.index, sc["nodes"], color=GREY, lw=1.2, ls="--", label="Constant phyllochron")
    a.plot(s2.index, s2["nodes"], color=CB, lw=1.4, label="Fruit-load phyllochron")
    a.set_ylabel("Nodes per plant"); a.legend(loc="upper left"); tag(a, "a")
    a = ax[1]
    for k, col, lab, c in (("n_app", "trusses", "visible", "#7a837e"), ("n_anth", "n_anthesis", "at anthesis", "#eda100"),
                           ("n_set", "n_set", "set", "#c0392b")):
        a.plot(tr_obs.index, tr_obs[k], "o", ms=2.5, color=c)
        a.plot(s2.index, s2[col], color=c, lw=1.2, label=f"Trusses {lab}")
    a.set_ylabel("Trusses per plant"); a.legend(loc="upper left")            # 선 = 모델, 점 = 관측 → 캡션
    tag(a, "b")
    a = ax[2]
    lp = des_obs[("leaves_present", "mean")]
    a.errorbar(lp.index, lp, des_obs[("leaves_present", "std")], fmt="o", ms=3, color=INK, elinewidth=0.6, label="Observed", zorder=5)
    for nm, _, _, col, ls in rules:
        a.plot(sims[nm].index, sims[nm]["leaves_present"], color=col, ls=ls, lw=1.1, label=nm)
    for d, y in (("2023-11-16", 16.8), ("2023-12-07", 18.2)):     # 주요 적엽 2회 (화방 2·4까지, 착과 후 231·240 °C d → 캡션)
        a.annotate("", xy=(pd.Timestamp(d), y - 0.6), xytext=(pd.Timestamp(d), y - 7),
                   arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=0.8))
    a.set_ylabel("Leaves present per plant"); a.legend(loc="upper left", fontsize=5.8)
    tag(a, "c")
    for a in ax:
        datefmt(a)
    snap(ax[0], x=False); snap(ax[1], x=False); snap(ax[2], x=False, ylim=(0, 60))
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig2")


# ------------------------------------------------------------------ Fig 3
def leaf_rank_runs(over=None):
    """엽위별 일 평균 (30엽 유지, 분석 구간). over: 파라미터 덮어쓰기 (민감도용)"""
    import scenario_season as S
    import run_hoenggye as RH
    C, clim, hr, obs, fin, ev, pj, p = haman()
    out = {}
    for season, (start, a, b) in S.SEASONS.items():
        st = pd.Timestamp(start)
        end = (pd.Timestamp(b) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        L = []
        if season == S.HG_SEASON:
            c2, h2, o2, _ = RH.load()
            ph = json.loads((DP / "hg/params_hg.json").read_text())
            pp = replace(RH.base_params(), **{k: v for k, v in ph.items() if k in Params.__dataclass_fields__}, **(over or {}))
            simulate(c2, pp, M.KeepN(30, trigger=a), start, end, hourly=h2, leaf_log=L,
                     plant0=RH.initial_plant(c2, pp, o2))
        else:
            pp = replace(p, transplant=(st - pd.Timedelta(days=17)).strftime("%Y-%m-%d"), **(over or {}))
            simulate(clim, pp, M.KeepN(30, trigger=(st + pd.Timedelta(days=40)).strftime("%Y-%m-%d")), start, end,
                     hourly=hr, leaf_log=L, leaf_dw0=10.0, stem_dw0=4.0)
        L = [x for x in L if pd.Timestamp(a) <= x["date"] <= pd.Timestamp(b)]
        n = 30
        def mean_rank(key):
            m = np.full((len(L), n), np.nan)
            for i, x in enumerate(L):
                v = x[key][:n]
                m[i, :len(v)] = v
            return np.nanmean(m, axis=0)
        out[season] = {k: mean_rank(k) for k in ("net", "gross", "rd", "et", "et_night", "par", "area")}
    return out


def fig3():
    from fspm.light_row import row_light
    C, clim, hr, obs, fin, ev, pj, p = haman()
    s, pl = simulate(clim, p, C.r0_rule(obs, fin), C.START, "2023-12-15", hourly=hr, **C.INIT)
    act = [n for n in reversed(pl.nodes) if n.leaf.active]
    la = np.array([n.leaf.area for n in act]); la = la * 0.95 / la.sum()
    h = hr.loc["2023-12-15"]; I = h["Rg_out"].to_numpy()
    pos = np.arange(1, len(la) + 1)
    dL = la * (312 / 224); beer = np.exp(-0.7 * (np.cumsum(dL) - dL / 2))
    inc, inc0 = (row_light(la, replace(p, canopy_width=0.9, floor_refl=rho), h["sun_elev"].to_numpy(), h["sun_az"].to_numpy(),
                           (1 - h["kd"].to_numpy()) * I, h["kd"].to_numpy() * I).sum(axis=0) for rho in (p.floor_refl, 0.0))
    R = leaf_rank_runs()
    style()
    rk = np.arange(1, 31)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.85))
    a = ax[0]
    a.plot(pos, beer * 100, "o-", ms=2.5, lw=1.1, color=GREY, label="Homogeneous")
    a.plot(pos, inc / inc[0] * 100, "o-", ms=2.5, lw=1.3, color="#b08400", label="Row")
    a.plot(pos, inc0 / inc0[0] * 100, "--", lw=1.0, color="#b08400", label="Row, no reflection")
    a.set_xlabel("Leaf position from the top"); a.set_ylabel("Irradiance relative to top leaf (%)")
    a.legend(loc="lower left", fontsize=5.8, handlelength=1.6)        # 왼쪽 아래 = 곡선 없는 영역
    snap(a, ylim=(0, None)); tag(a, "a")
    a = ax[1]
    for k, (c, lab) in SEA.items():
        a.plot(rk, R[k]["net"], color=c, lw=1.3, label=lab)
    a.axhline(0, color=INK, lw=0.6)
    a.set_xlabel("Leaf position from the top"); a.set_ylabel("Net carbon gain (g CH₂O leaf⁻¹ d⁻¹)")
    snap(a); tag(a, "b")
    a = ax[2]
    for k, (c, lab) in SEA.items():
        a.plot(rk, R[k]["et"] - R[k]["et_night"], color=c, lw=1.3)
        a.plot(rk, R[k]["et_night"], color=c, lw=1.1, ls="--")
    a.plot([], [], color=INK, lw=1.2, label="day"); a.plot([], [], color=INK, lw=1.0, ls="--", label="night")
    a.set_xlabel("Leaf position from the top"); a.set_ylabel("Transpiration (g leaf⁻¹ d⁻¹)")
    a.legend(); snap(a); tag(a, "c")
    a = ax[3]
    w = R["winter"]
    for key, lab, ls in (("net", "net carbon gain", "-"), ("et", "transpiration", "--"), ("et_night", "night transpiration", ":")):
        v = np.clip(w[key], 0, None)
        a.plot(rk, np.cumsum(v) / v.sum() * 100, color=SEA["winter"][0], ls=ls, lw=1.3, label=lab)
    a.axvspan(20.5, 30, color=GREY, alpha=0.12, lw=0)          # 음영 = 21–30번째 잎, 선 종류 → 캡션
    a.set_xlabel("Leaf position from the top"); a.set_ylabel("Cumulative share from the top (%)")
    snap(a); tag(a, "d")
    hs = [matplotlib.lines.Line2D([], [], color=c, lw=1.5) for c, _ in SEA.values()]
    fig.legend(hs, [lab for _, lab in SEA.values()], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.6, 1.0))
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.93))
    save(fig, "Fig3")
    rows = []
    for k in R:
        v = {key: np.clip(R[k][key], 0, None) for key in ("net", "et", "et_night", "gross", "rd")}
        rows.append(dict(season=k, share21_30_net=v["net"][20:].sum() / v["net"].sum(),
                         share21_30_et=v["et"][20:].sum() / v["et"].sum(),
                         share21_30_night=v["et_night"][20:].sum() / max(v["et_night"].sum(), 1e-9),
                         first_neg_rank=int(np.argmax(R[k]["net"] < 0)) + 1 if (R[k]["net"] < 0).any() else None,
                         net_rank30=R[k]["net"][29], rd_share_21_30=R[k]["rd"][20:].sum() / R[k]["gross"][20:].sum()))
    d = pd.DataFrame(rows); d.to_csv(DP / "leaf_rank_summary.csv", index=False); print(d.round(3).to_string(index=False))


# ------------------------------------------------------------------ Fig 4
def fig4():
    C, clim, hr, obs, fin, ev, pj, p = haman()
    s, pl = simulate(clim, p, C.r0_rule(obs, fin), C.START, C.END, hourly=hr, **C.INIT)
    de = pd.read_csv(DP / "obs_des.csv", parse_dates=["date"])
    de["stem_all_DW"] = de["stem_DW"] + de["truss_struct_DW"].fillna(0)
    g = de.groupby("date")
    lc = pd.read_csv(DP / "loadcell_daily.csv", parse_dates=["date"])
    lc = lc[lc["use"] == 1].pivot_table(index="date", columns="device", values="ET_plant_mL")
    lm = lc.mean(axis=1)
    fig = plt.figure(figsize=(W2, 4.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], height_ratios=[1, 1])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[0, 2])]
    a = ax[0]
    for o, m, lab, c in (("leaf_DW", "leaf_dw", "Leaf", "#1baf7a"), ("stem_all_DW", "stem_dw", "Stem + truss axes", "#8a6100"),
                         ("fruit_DW", "fruit_dw", "Fruit on plant", "#eb6834")):
        a.plot(s.index, s[m], color=c, lw=1.3, label=lab)
        a.errorbar(g[o].mean().index, g[o].mean(), g[o].std(), fmt="o", ms=2.5, color=c, elinewidth=0.6)
        a.plot([pd.Timestamp("2024-01-11")], [fin[o].mean()], "D", ms=4.5, mfc="white", mec=c, mew=1.2)
    # 최종조사 주는 수확하지 않은 주 → 비교 대상은 모의 '달린 + 수확' 과실 (점선)
    a.plot(s.index, s["fruit_dw"] + s["harvested_dw"], color="#eb6834", lw=1, ls="--", label="Fruit on plant + harvested")
    a.set_ylabel("Dry weight (g plant⁻¹)"); a.legend(loc="upper left"); tag(a, "a")
    a = ax[1]
    a.plot(s.index, s["leaf_area_m2"], color="#1baf7a", lw=1.3, label="Model")
    a.errorbar(g["leaf_area_cm2"].mean().index, g["leaf_area_cm2"].mean() / 1e4, g["leaf_area_cm2"].std() / 1e4,
               fmt="o", ms=2.5, color=INK, elinewidth=0.6, label="Destructive survey")
    a.plot([pd.Timestamp("2024-01-11")], [fin["leaf_area_cm2"].mean() / 1e4], "D", ms=4.5, mfc="white", mec=INK, mew=1.2,
           label="Final survey (evaluation)")
    a.set_ylabel("Leaf area (m² plant⁻¹)"); a.legend(loc="lower right", fontsize=6.4); tag(a, "b")
    a = ax[2]
    HOLD = pd.Timestamp("2023-12-01")            # 10–11월 = Rd25 선정에 사용, 12–1월 = 홀드아웃 (eval_et_split.py)
    a.axvspan(HOLD - pd.Timedelta(hours=12), pd.Timestamp(C.END), color="#e8eef7", lw=0, zorder=0)
    lcw, lmw = lc.loc[C.START:C.END], lm.loc[C.START:C.END]
    a.fill_between(lcw.index, lcw.min(axis=1), lcw.max(axis=1), color=GREY, alpha=0.25, lw=0, label="Load cells (range of 4)")
    a.plot(lmw.index, lmw, color=INK, lw=0.9, label="Load cells (mean)")
    a.plot(s.index, s["et_g"], color=CB, lw=1.1, label="Model")
    for d in s.index[s["removed"] > 0]:
        a.axvline(d, color="#c0392b", lw=0.5, ls=":")
    a.set_ylabel("Transpiration (g plant⁻¹ d⁻¹)"); a.legend(loc="lower left", ncol=3)   # 음영·점선 설명 → 캡션
    tag(a, "c", x=-0.05)
    a = ax[3]
    j = pd.concat([s["et_g"].rename("s"), lm.rename("o")], axis=1, join="inner").dropna()
    m = 1600; a.plot([0, m], [0, m], color=GREY, lw=0.7, ls=":")
    lines = []
    for (lo, hi), lab, c, mk in (((C.START, "2023-11-30"), "Oct–Nov", GREY, "o"), (("2023-12-01", C.END), "Dec–Jan (hold-out)", CB, "o")):
        k = j.loc[lo:hi]
        a.scatter(k["o"], k["s"], s=6, color=c, alpha=0.8, marker=mk)
        e = k["s"] - k["o"]
        bias = f"{100 * e.mean() / k['o'].mean():+.0f}%".replace("-", "−")
        lines.append((f"{lab}\nnRMSE = {np.sqrt((e**2).mean())/k['o'].mean():.2f}\nR² = {np.corrcoef(k['s'], k['o'])[0, 1]**2:.2f}\n"
                      f"bias = {bias}", c))
    for i, (txt, c) in enumerate(lines):
        a.text(0.04, 0.97 - 0.25 * i, txt, transform=a.transAxes, va="top", fontsize=6.2, linespacing=1.15,
               color=c if c == CB else INK)
    a.set_xlabel("Measured (g plant⁻¹ d⁻¹)"); a.set_ylabel("Simulated (g plant⁻¹ d⁻¹)"); tag(a, "d", x=-0.2)
    for a in ax[:3]:
        datefmt(a)
        snap(a, x=False)
    for ax_ in (ax[3].xaxis, ax[3].yaxis):
        ax_.set_major_locator(matplotlib.ticker.MultipleLocator(400))
    ax[3].set_xlim(0, m); ax[3].set_ylim(0, m); _unclip(ax[3])
    fig.tight_layout(w_pad=1.2, h_pad=1.2)
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig 5
def fig5():
    import external_aihub as EA
    C, clim, hr, obs, fin, ev, pj, p = haman()
    rule = C.r0_rule(obs, fin)
    pA = replace(p, use_source_clock=False, TT_anth_set=pj["TT_anth_set_tempOnly"])
    sB, plB = simulate(clim, p, rule, C.START, C.END, hourly=hr, **C.INIT)
    _, plA = simulate(clim, pA, rule, C.START, C.END, hourly=hr, **C.INIT)
    lag = {}
    for nm, pl in (("A", plA), ("B", plB)):
        tt = truss_table(pl).set_index("truss")
        lag[nm] = (tt["set"] - tt["anthesis"]).dt.days
    style()
    ttB = truss_table(plB).set_index("truss")
    o = ev.dropna(subset=["anth_set_d"]).groupby("truss")["anth_set_d"].agg(["mean", "std"])
    fig = plt.figure(figsize=(W2, 5.1))
    gs = fig.add_gridspec(2, 7)
    ax = [fig.add_subplot(gs[0, 0:5]), fig.add_subplot(gs[0, 5:7]), fig.add_subplot(gs[1, 0:4]), fig.add_subplot(gs[1, 4:7])]
    a = ax[0]
    a.errorbar(o.index, o["mean"], o["std"], fmt="o", ms=3.5, color=INK, elinewidth=0.6, label="Observed (mean ± s.d.)")
    a.plot(o.index, lag["A"].reindex(o.index), "s-", ms=3, lw=1, color=CA, label="A: temperature only")
    a.plot(o.index, lag["B"].reindex(o.index), "^-", ms=3.5, lw=1.2, color=CB, label="B: temperature × Q/D")
    a.set_xlabel("Truss number"); a.set_ylabel("Anthesis to fruit set (d)")
    a2 = a.twinx(); a2.grid(False); a2.spines["right"].set_visible(True)
    qd = [sB["qd_ema"].asof(d) if pd.notna(d) else np.nan for d in ttB["anthesis"].reindex(o.index)]
    a2.plot(o.index, qd, color="#b08400", lw=0.8, ls=":", marker=".", ms=3)
    a2.set_ylabel("Q/D at anthesis", color="#b08400"); a2.tick_params(axis="y", colors="#b08400")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=6.5, borderaxespad=0)
    a.set_xticks(range(1, 11)); snap(a); snap(a2, x=False)
    tag(a, "a", x=-0.08, y=1.1)
    a = ax[1]
    cv = pd.read_csv(DP / "cv_c1.csv")
    r = cv.groupby("plant").apply(lambda g: pd.Series({"A": np.sqrt((g.err_A ** 2).mean()), "B": np.sqrt((g.err_B ** 2).mean())}))
    for _, x in r.iterrows():
        a.plot([0, 1], [x.A, x.B], color=GREY, lw=0.7, marker="o", ms=3, mfc="white")
    a.plot([0, 1], [np.sqrt((cv.err_A ** 2).mean()), np.sqrt((cv.err_B ** 2).mean())], color=INK, lw=1.6, marker="o", ms=4)
    a.set_xticks([0, 1], ["A", "B"]); a.set_xlim(-0.45, 1.45)
    a.set_ylabel("Prediction RMSE, withheld plant (d)"); tag(a, "b", x=-0.3, y=1.1)   # 회색 = 개체, 검정 = 전체 → 캡션
    snap(a, x=False)
    # c: truss progression aligned at first observation
    a = ax[2]
    e_, b = EA.events()
    cols = {"tom2": "#7a837e", "tom3": "#b08400"}
    GR = {"tom2": "GR1", "tom3": "GR2"}                     # 원고 이름 (데이터셋 코드 대응은 보충자료 S3)
    for farm in ("tom2", "tom3"):
        env = EA.load_env(farm)
        bb = b[b["farm"] == farm].groupby(["plant", "date"])["truss"].max().reset_index()
        bb = bb[bb["date"] >= env.index.min()]
        bb["TT"] = env["TT"].reindex(bb["date"]).to_numpy()
        bb = bb.dropna()
        pts = []
        for pl_, g in bb.groupby("plant"):
            if g["truss"].nunique() >= 3:
                g = g.sort_values("TT")
                pts.append(pd.DataFrame({"dTT": g["TT"] - g["TT"].iloc[0], "dtr": g["truss"] - g["truss"].iloc[0]}))
        P = pd.concat(pts)
        bins = np.arange(0, P.dTT.max() + 60, 60)
        q = P.groupby(pd.cut(P.dTT, bins), observed=True)["dtr"].agg(["median", lambda v: v.quantile(.25), lambda v: v.quantile(.75), "size"])
        q = q[q["size"] >= 10]
        xm = [iv.mid for iv in q.index]
        a.fill_between(xm, q.iloc[:, 1], q.iloc[:, 2], color=cols[farm], alpha=0.18, lw=0)
        a.plot(xm, q["median"], color=cols[farm], lw=1.3, label=f"{GR[farm]} (median, IQR)")
    # A7-6: 개체별 개화 화방 vs TT
    tt_h = clim["Tmean"].apply(lambda T: max(T - 10, 0)).cumsum()
    pts = []
    for pl_, g in ev.dropna(subset=["anth"]).groupby("plant"):
        g = g.sort_values("anth")
        T_ = tt_h.reindex(g["anth"]).to_numpy()
        pts.append(pd.DataFrame({"dTT": T_ - T_[0], "dtr": g["truss"].to_numpy() - g["truss"].iloc[0]}))
    P = pd.concat(pts)
    a.plot(P.dTT, P.dtr, "o", ms=2, color=CB, alpha=0.5, label="Haman (plants)")
    xx = np.linspace(0, 700, 10); a.plot(xx, xx / 82, color=CB, lw=1.2, ls="--", label="82 °C d truss⁻¹ (Haman)")
    a.set_xlim(0, 700); a.set_ylim(-0.4, 10)               # 0 에서 시작하는 선·점이 x축에 붙지 않게
    a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100)); a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    _unclip(a)
    a.set_xlabel("Thermal time since first observation (°C d)"); a.set_ylabel("Additional trusses at anthesis")
    a.legend(loc="upper left", fontsize=6.5); tag(a, "c", x=-0.1)
    # d: 상대 개화→착과 기간 (온실별 중앙값 대비) vs 개화 후 1주 일사
    a = ax[3]
    ex = pd.read_csv(DP / "external_aihub_events.csv")
    ha = ev.dropna(subset=["anth_set_d", "Rg7"])[["anth_set_d", "Rg7"]].rename(columns={"anth_set_d": "lag_d"})
    ha["site"] = "Haman"
    ex = ex.rename(columns={"farm": "site"})[["site", "lag_d", "Rg7"]]
    D = pd.concat([ha, ex])
    D = D[D["lag_d"] > 0]
    D["rel"] = D["lag_d"] / D.groupby("site")["lag_d"].transform("median")
    bins = np.arange(4, 20, 2)
    for st, c, lab in (("Haman", CB, "Haman"), ("tom2", "#7a837e", "GR1"), ("tom3", "#b08400", "GR2")):
        x = D[D.site == st]
        a.scatter(x["Rg7"], x["rel"], s=4, color=c, alpha=0.25, lw=0)
        q = x.groupby(pd.cut(x["Rg7"], bins), observed=True)["rel"].agg(["mean", "sem", "size"])
        q = q[q["size"] >= 5]
        a.errorbar([iv.mid for iv in q.index], q["mean"], q["sem"], fmt="o-", ms=3.5, lw=1.3, color=c, elinewidth=0.7,
                   label=f"{lab} (median {x['lag_d'].median():.0f} d)")
    a.axhline(1, color=GREY, lw=0.6, ls=":")
    a.set_ylim(0, 3)                       # 상대 기간 > 3 인 소수 점은 범위 밖 (눈금 0.5 → 끝 = 눈금)
    snap(a, y=False)
    a.set_xlabel("Light in the week after anthesis (MJ m⁻² d⁻¹)")
    a.set_ylabel("Anthesis to set, relative to site median")
    a.legend(fontsize=6.3, loc="upper right"); tag(a, "d", x=-0.12)
    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    save(fig, "Fig5")


# ------------------------------------------------------------------ Fig 6
def fig6():
    import calibrate_hoenggye as CH
    import run_hoenggye as RH
    clim, hr, obs, lc, p0 = CH.setup()
    ph = json.loads((DP / "hg/params_hg.json").read_text())
    p = replace(p0, **{k: v for k, v in ph.items() if k in Params.__dataclass_fields__})
    s, _ = CH.run(clim, hr, obs, p)
    h, tab, hot = CH.microclimate()
    # 규칙별 재보정 (calibrate_hoenggye.py --rules): 규칙마다 품종 파라미터 3개를 따로 보정한 LAI 궤적
    rc = pd.read_csv(DP / "hg/rule_recalibration.csv").set_index("rule")
    RULE_EN = {"무적엽": ("No pruning", GREY, "--"), "R1 16장": ("R1, 16 leaves", "#f2b179", "-"),
               "R1 18장": ("R1, 18 leaves", "#e07b39", "-"), "R1 20장": ("R1, 20 leaves", "#a8461b", "-"),
               "R3 매주": ("R3, weekly", CB, "-"), "R3 묶음 (함안 관행)": ("R3, batch", "#1baf7a", "-")}
    recal = {}
    for rn, (rule, kw) in RH.rules().items():
        x = rc.loc[rn]
        pr = replace(p0, prune_min_batch=kw.get("prune_min_batch", 1), leaf_area_pot=float(x.leaf_area_pot),
                     fruit_W_pot=float(x.fruit_W_pot), fruit_priority=float(x.fruit_priority))
        recal[rn] = CH.run(clim, hr, obs, pr, rule)[0]
    style()
    o = obs[obs["initial"] == 0]
    hv = obs.attrs["harvest"]
    fig = plt.figure(figsize=(W2, 5.0))
    gs = fig.add_gridspec(2, 3)
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    a = ax[0]
    a.plot(s.index, s["LAI"], color="#1baf7a", lw=1.3, label="Model (R3, batch)")
    a.errorbar(o.index, o["LAI"], o["LAI_sd"], fmt="o", ms=3, color=INK, elinewidth=0.6)
    a2 = a.twinx(); a2.grid(False); a2.spines["right"].set_visible(True)
    a2.step(s.index, s["leaves_present"], color=GREY, lw=0.7, where="post")
    a2.errorbar(o.index, o["leaves"], o["leaves_sd"], fmt="s", ms=3, mfc="white", color=GREY, elinewidth=0.6)
    a2.set_ylabel("Leaves present", color=GREY)
    a2.tick_params(axis="y", colors=GREY)
    a.set_ylabel("Leaf area index"); tag(a, "a", x=-0.2)
    a = ax[1]
    for o_, m_, lab, c in (("leaf_DW", "leaf_dw", "Leaf", "#1baf7a"), ("stem_DW", "stem_dw", "Stem", "#8a6100")):
        a.plot(s.index, s[m_], color=c, lw=1.3, label=lab)
        a.errorbar(o.index, o[o_], o[f"{o_}_sd"], fmt="o", ms=3, color=c, mec=INK, elinewidth=0.6)   # ● = 파괴조사 (색 = 기관)
    a.set_ylabel("Dry weight (g plant⁻¹)"); a.legend(loc="upper left", fontsize=6.2); tag(a, "b", x=-0.2)
    a = ax[2]
    a.plot(s.index, s["fruit_dw"] + s["harvested_dw"], color="#eb6834", lw=1.3, label="Total")
    a.plot(s.index, s["harvested_dw"], color="#eb6834", lw=0.9, ls="--", label="Harvested")
    a.errorbar(o.index, o["harvest_DW"], o["harvest_DW_sd"].replace(0, np.nan), fmt="o", ms=3, color=INK, elinewidth=0.6)
    a.plot(hv.index, hv["harvest_cum_DW"], "^", ms=3, mfc="white", color=INK, mew=0.7)
    a.set_ylabel("Fruit dry weight (g plant⁻¹)"); a.legend(loc="upper left", fontsize=6.0); tag(a, "c", x=-0.2)
    for k, col, ylab in ((3, "LAI", "Leaf area index"), (4, "leaves_present", "Leaves present")):
        a = ax[k]
        for rn, (lab, c, ls) in RULE_EN.items():
            sr = recal[rn]
            kw = dict(color=c, lw=1.5 if rn.startswith("R3 묶음") else 1.0, ls=ls, label=lab)
            (a.step(sr.index, sr[col], where="post", **kw) if col == "leaves_present" else a.plot(sr.index, sr[col], **kw))
        if col == "LAI":
            a.errorbar(o.index, o["LAI"], o["LAI_sd"], fmt="o", ms=3.5, color=INK, elinewidth=0.6, zorder=5)
        else:
            a.errorbar(o.index, o["leaves"], o["leaves_sd"], fmt="s", ms=3.5, mfc="white", color=GREY, elinewidth=0.6, zorder=5)
        a.set_ylabel(ylab)
        tag(a, "de"[k - 3], x=-0.2)
    ax[3].legend(loc="upper right", fontsize=5.8, ncol=2, columnspacing=0.8, handlelength=1.6)
    a = ax[5]
    d = h[h["Out_Rad"] > 50].dropna(subset=["dT", "Out_Rad"])
    a.scatter(d["Out_Rad"], d["dT"], s=4, color="#eb6834", alpha=0.35, lw=0, label="Hourly observations")
    x, y = d["Out_Rad"].to_numpy(), d["dT"].to_numpy()
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    res = y - (b0 + b1 * x); n = len(x); se = np.sqrt((res ** 2).sum() / (n - 2))
    ci = 1.96 * se * np.sqrt(1 / n + (xx - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    a.fill_between(xx, b0 + b1 * xx - ci, b0 + b1 * xx + ci, color=INK, alpha=0.18, lw=0)
    a.plot(xx, b0 + b1 * xx, color=INK, lw=1.4, label="Linear regression (95% CI)")
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    a.text(0.03, 0.04, f"ΔT = {b0:.2f} − {abs(b1):.4f} $\\mathregular{{I_{{out}}}}$\nR² = {r2:.2f}, n = {n} h", transform=a.transAxes, fontsize=6.3)
    a.axhline(0, color=GREY, lw=0.6, ls=":")
    a.legend(loc="upper right", fontsize=6.3)
    print(f"Fig6f regression: dT = {b0:.3f} + {b1:.5f} x, R2 {r2:.2f}, n {n}, dT at 800 W = {b0 + b1*800:.2f}")
    a.set_xlabel("Outside radiation (W m⁻²)"); a.set_ylabel("Inside − outside air temperature (K)")   # 기간 → 캡션
    snap(a); tag(a, "f", x=-0.2)
    for a in ax[:5]:
        datefmt(a)
    snap(ax[0], x=False); snap(a2, x=False); snap(ax[1], x=False); snap(ax[2], x=False)
    snap(ax[3], x=False, ylim=(0, 3.6)); snap(ax[4], x=False)     # d: 범례 여백
    # 관측 기호 공통 범례 (패널 안 겹침 없이 모든 패널에 같은 기호)
    hs = [matplotlib.lines.Line2D([], [], ls="none", marker="o", ms=4, color=INK),
          matplotlib.lines.Line2D([], [], ls="none", marker="s", ms=4, mfc="white", color=GREY),
          matplotlib.lines.Line2D([], [], ls="none", marker="^", ms=4, mfc="white", color=INK, mew=0.7)]
    fig.legend(hs, ["Destructive survey (mean ± s.d.)", "Leaves present, observed (mean ± s.d.)", "Harvest records"],
               loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.2, h_pad=1.3, rect=(0, 0, 1, 0.95))
    save(fig, "Fig6")


# ------------------------------------------------------------------ Fig 7
def fig7():
    d = pd.read_csv(DP / "scenario_season.csv")
    G = pd.read_csv(DP / "scenario_guideline.csv")
    R = pd.read_csv(DP / "scenario_ripening.csv")       # 숙기 단축 = 같은 화방의 수확일 차이 (src/scenario_ripening.py)
    base = d[d.variant == "base"]                        # 12–30장 전체 (표 6·본문이 12장 값을 인용)
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1])
    ax = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1])]
    ax_e = fig.add_subplot(gs[1, 2])
    a = ax[0]
    for k, (c, lab) in SEA.items():
        x = d[(d.variant == "base") & (d.season == k)].set_index("N")
        fr = x.fruit_gain / x.fruit_gain.max() * 100
        et = (1 - x.et_L / x.loc[30, "et_L"]) * 100
        a.plot(et, fr, "-", color=c, lw=1.3, label=lab)
        a.scatter(et, fr, s=9, color=c, zorder=3)
        n95 = int(G[(G.season == k) & (G.variant == "base")].N95.iloc[0])
        a.scatter([et[n95]], [fr[n95]], s=70, marker="*", color=c, ec=INK, lw=0.5, zorder=4)
        off = {"winter": (4, 3), "spring": (-4, -10), "summer": (-6, -8)}[k]
        star = {"winter": (5, 3, "baseline"), "spring": (4, 4, "baseline"), "summer": (6, 0, "center")}[k]   # 별표 번호는 오른쪽 (왼쪽 위는 세 곡선이 겹침)
        for N in sorted(({12, 16, 20} - {n95 - 2, n95 + 2}) | {n95}):     # N95 별표에 바로 붙는 이웃 번호는 생략, 22장 이상은 곡선이 겹쳐 번호 없음
            o = star[:2] if N == n95 else off
            a.annotate(str(N), (et[N], fr[N]), xytext=o, textcoords="offset points", fontsize=6.3, color=c,
                       ha="left" if o[0] > 0 else "right", va=star[2] if N == n95 else "baseline")
    a.axhline(95, color=GREY, lw=0.6, ls=":")
    a.set_xlabel("Transpiration saving vs 30 leaves (%)"); a.set_ylabel("Fruit DM gain in analysis period (% of maximum)")
    a.text(0.98, 95.4, "95%", transform=a.get_yaxis_transform(), ha="right", fontsize=6, color=GREY, va="bottom")
    snap(a); tag(a, "a", x=-0.1, y=1.02)   # 숫자·별표 설명 → 캡션
    a = ax[1]
    for k, (c, lab) in SEA.items():
        x = base[base.season == k]
        a.plot(x.N, x.et_night_L, "o-", ms=2.5, lw=1.2, color=c)
    a.set_xlabel("Leaves retained"); a.set_ylabel("Night transpiration\n(L plant⁻¹ per 2 months)"); tag(a, "b")
    a = ax[2]
    for k, (c, lab) in SEA.items():
        x = base[base.season == k]
        a.plot(x.N, x.hot_h_per_truss, "o-", ms=2.5, lw=1.2, color=c)
    a.set_xlabel("Leaves retained"); a.set_ylabel("Hours > 30 °C\nper coloring truss"); tag(a, "c")
    a = ax[3]
    for k, (c, lab) in SEA.items():
        x = R[R.season == k]
        a.plot(x.N, x.adv_mean_d, "o-", ms=2.5, lw=1.2, color=c)
    a.set_xlabel("Leaves retained"); a.set_ylabel("Ripening advance by exposure (d)"); tag(a, "d")
    a = ax_e
    for k, (c, lab) in SEA.items():
        x = base[base.season == k]
        a.plot(x.N, x.LAI, "o-", ms=2.5, lw=1.2, color=c)
    a.set_xlabel("Leaves retained"); a.set_ylabel("Mean leaf area index"); tag(a, "e")
    for a in ax[1:] + [ax_e]:
        a.set_xticks(range(12, 31, 2))
        snap(a)
    hs = [matplotlib.lines.Line2D([], [], color=c, lw=1.5, marker="o", ms=3) for c, _ in SEA.values()]
    fig.legend(hs, [lab for _, lab in SEA.values()], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.0, h_pad=1.2, rect=(0, 0, 1, 0.95))
    save(fig, "Fig7")


# ------------------------------------------------------------------ Fig S1
def figS1():
    oat = pd.read_csv(DP / "rd_oat.csv")
    grid = pd.read_csv(DP / "rd_grid.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.6))
    a = ax[0]
    ref = oat[np.isclose(oat.Rd25, 1.1)].iloc[0]
    a.plot(oat.Rd25, (oat.total_DM / ref.total_DM - 1) * 100, "o-", ms=3, color=INK, lw=1.2, label="Total dry matter")
    a.plot(oat.Rd25, (oat.fruit_DM / ref.fruit_DM - 1) * 100, "s-", ms=3, color="#eb6834", lw=1.2, label="Fruit dry matter")
    a.axvline(0.65, color=CB, lw=0.8, ls="--")
    a.set_xlabel("$R_{d,25}$ (µmol m$^{-2}$ s$^{-1}$)"); a.set_ylabel("Change relative to $R_{d,25}$ = 1.1 (%)")
    a.legend(); snap(a); tag(a, "a")
    a = ax[1]
    pv = grid.pivot_table(index="tau_struct", columns="Rd25", values="obj")
    im = a.pcolormesh(pv.columns, pv.index, pv.values, cmap="viridis_r", shading="nearest")
    cs = a.contour(pv.columns, pv.index, grid.pivot_table(index="tau_struct", columns="Rd25", values="et_ratio_octnov").values,
                   levels=[0.8, 0.85, 0.9, 0.95], colors="white", linewidths=0.6)
    a.clabel(cs, fontsize=5.5, fmt="%.2f")
    fig.colorbar(im, ax=a, label="Growth objective (sum of nRMSE)")
    a.set_xlabel("$R_{d,25}$ (µmol m$^{-2}$ s$^{-1}$)"); a.set_ylabel("$\\mathregular{τ_{struct}}$"); a.grid(False); tag(a, "b")
    fig.tight_layout()
    save(fig, "FigS1")


if __name__ == "__main__":
    which = sys.argv[1:] or ["2", "3", "4", "5", "6", "7", "S1", "S2"]   # Fig 1 = 사용자 PDF (figures/paper/Fig1.pdf)
    for w in which:
        globals()[f"fig{w}"]()
