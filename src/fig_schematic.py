"""Fig. 1 모식도: (a) 열 캐노피·마디 구조·적엽·과실 위치, (b) 모델 구성과 일 단위 흐름"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 8
C_GRAY, C_BLUE, C_ORANGE, C_GREEN, C_RED = "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#c0392b"

fig = plt.figure(figsize=(7.48, 4.6))
axa = fig.add_axes([0.0, 0.02, 0.32, 0.94])
axb = fig.add_axes([0.33, 0.02, 0.665, 0.94])

# ---------------------------------------------------------------- (a) 열 캐노피
ax = axa
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-0.5, 4.7)
ax.set_aspect("equal")
ax.axis("off")
ax.text(-1.55, 4.68, "(a)", fontsize=10, fontweight="bold", va="top")

# 지붕 (벤로 스팬 4 m, 경사 27.9°) — 도식
ax.add_patch(Polygon([[-1.6, 3.9], [0.0, 4.35], [1.6, 3.9]], closed=False, fill=False, lw=1.0, ec=C_GRAY))
ax.text(0.0, 4.42, "glass roof, slope 27.9°, N–S ridge", ha="center", fontsize=6.5, color=C_GRAY)

# 이웃 열 (산란·그림자) 반투명 직사각형
for xc in (-1.5, 1.5):
    ax.add_patch(Rectangle((xc - 0.45, 1.25), 0.9, 2.35, fc=C_GREEN, alpha=0.12, ec="none"))
# 중앙 열: 활성 잎 층
nz = 20
z0 = 1.25
dz = 2.35 / nz
for i in range(nz):
    a = 0.55 - 0.4 * i / nz                       # 위쪽 진하게
    ax.add_patch(Rectangle((-0.45, z0 + i * dz), 0.9, dz * 0.92, fc=C_GREEN, alpha=a, ec="none"))
# 줄기
ax.plot([0, 0], [0.0, 3.75], color=C_GRAY, lw=1.6)
# 제거된 잎 (점선)
for i in range(5):
    z = 0.25 + i * 0.2
    ax.plot([0, 0.32], [z, z + 0.06], ls=":", color=C_GRAY, lw=0.9)
    ax.plot([0, -0.32], [z + 0.1, z + 0.16], ls=":", color=C_GRAY, lw=0.9)
ax.annotate("removed leaves\n(pruning rule)", xy=(0.3, 0.55), xytext=(0.62, 0.25), fontsize=6.5,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C_GRAY), color=C_GRAY)
# 화방 + 과실 (화방 마디 3마디 아래로 처짐)
for zt, col in ((1.55, C_RED), (2.25, C_ORANGE), (2.95, "#7fb069")):
    ax.plot([0, 0.18], [zt, zt - 0.22], color=C_GRAY, lw=0.8)
    for dx in (0.12, 0.22, 0.30):
        ax.add_patch(Ellipse((dx + 0.03, zt - 0.27), 0.11, 0.11, fc=col, ec="none"))
ax.annotate("colouring truss:\nfruit temperature\n$T_f = T_a + a_f I_f / h_f$", xy=(0.36, 1.27), xytext=(0.62, 1.05),
            fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.6, color=C_GRAY))
ax.annotate("set → anthesis\ntrusses", xy=(0.33, 2.7), xytext=(0.62, 2.55), fontsize=6.5,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C_GRAY))
ax.annotate("leaf layers\n(one leaf per node,\n$\\Delta z$ = 0.081 m)", xy=(-0.4, 1.6), xytext=(-1.6, 0.55), fontsize=6.5,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C_GRAY))
# 직달·산란 광선
for x0 in (-1.2, -0.7):
    ax.add_patch(FancyArrowPatch((x0, 3.85), (x0 + 0.75, 2.4), arrowstyle="-|>", mutation_scale=7, color="#e0a800", lw=1.0))
ax.text(-1.25, 3.0, "direct\nbeam", fontsize=6.5, color="#b08400", ha="right")
for ang in (-0.5, 0.0, 0.5):
    ax.add_patch(FancyArrowPatch((1.05 + ang * 0.3, 3.8), (1.05 + ang * 0.3 - ang * 0.6, 3.2), arrowstyle="-|>",
                                 mutation_scale=6, color=C_BLUE, lw=0.7, alpha=0.8))
ax.text(1.05, 3.12, "diffuse\n(18 dirs)", ha="center", va="top", fontsize=6.5, color=C_BLUE)
# 치수
ax.annotate("", xy=(-0.45, 3.68), xytext=(0.45, 3.68), arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(0.0, 3.74, "$w_c$ = 0.9 m", ha="center", fontsize=6.5, va="bottom")
ax.annotate("", xy=(0.0, -0.2), xytext=(1.5, -0.2), arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(0.75, -0.27, "$w_r$ = 1.5 m (2.67 plants m$^{-2}$)", ha="center", fontsize=6.5, va="top")
# 슬라브 + 로드셀
ax.add_patch(Rectangle((-0.35, -0.08), 0.7, 0.08, fc="#c8b49a", ec=C_GRAY, lw=0.5))
ax.text(-0.4, -0.04, "load cell", ha="right", va="center", fontsize=6.5, color=C_GRAY)

# ---------------------------------------------------------------- (b) 흐름도
ax = axb
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(0.05, 9.85, "(b)", fontsize=10, fontweight="bold", va="top")


def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=col, alpha=0.13, ec=col, lw=1.0))
    ax.text(x + 0.12, y + h - 0.12, title, fontsize=6.8, fontweight="bold", va="top", color=col)
    ax.text(x + 0.12, y + h - 0.5, body, fontsize=5.9, va="top", linespacing=1.3)
    return (x, y, w, h)


def arrow(p, q, col=C_GRAY, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, color=col, lw=0.9, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


b_in = box(0.3, 8.0, 9.4, 1.45, "Hourly drivers (climate computer log)",
           "air T, RH, outside global radiation, screen positions (shade 0.5, energy 0.9), CO$_2$\n"
           "+ solar position and diffuse fraction (roof transmittance)", C_GRAY)
b_li = box(0.3, 5.55, 3.0, 2.05, "Light (2-D row canopy)",
           "Beer's law along directional\npaths through hedgerows\n"
           "direct + 18 sky directions\n→ irradiance of each leaf\n   and colouring truss", "#b08400")
b_lf = box(3.55, 5.55, 3.0, 2.05, "Leaf processes (per leaf)",
           "FvCB gross photosynthesis\n$R_d$ (leaf age, temperature)\nPenman–Monteith\n  transpiration ($r_s$: I, CO$_2$)\n7-d net carbon balance", C_GREEN)
b_cb = box(6.8, 5.55, 2.9, 2.05, "Carbon partitioning",
           "buffer, maintenance resp.\nfruit priority $\\pi$\nsource–sink ratio\n"
           "$Q/D$ = (P$_g$ − R) / demand\n(5-d moving average)", C_BLUE)
b_sc = box(0.3, 3.0, 3.0, 2.15, "Structural clock",
           "thermal time\nphyllochron 19.3 → 28.1 °C d\n(switch: onset of fruit growth)\n"
           "node, leaf and truss\nappearance; truss → anthesis", C_GRAY)
b_rc = box(3.55, 3.0, 3.0, 2.15, "Reproductive clock",
           "anthesis → set (per truss):\n  rate = $\\Delta TT/TT_{set}\\cdot g(\\overline{Q/D})$\n"
           "set → end of growth: beta fn.\ncolouring: fruit-temperature\n  thermal time → harvest", C_ORANGE)
b_pr = box(6.8, 3.0, 2.9, 2.15, "Pruning rules",
           "R0 observed\nR1 keep N leaves, N(T)\nR2 light-based (WLI)\n"
           "R3 truss-linked: leaves\nbelow truss set +235 °C d,\nweekly, batched", C_RED)
b_out = box(0.3, 0.2, 9.4, 1.75, "Decision-relevant outputs (validation data)",
            "leaf number & area [surveys]  ·  organ dry weights [destructive]  ·  anthesis & set dates [surveys, AI Hub]\n"
            "daily & night transpiration [load cells]  ·  fruit temperature, hours > 30 °C, harvest date\n"
            "→ seasonal leaf-number guideline", C_GRAY)

# 화살표
arrow((1.8, 8.0), (1.8, 7.6))
arrow((5.05, 8.0), (5.05, 7.6))
arrow((8.25, 8.0), (8.25, 7.6))
arrow((3.3, 6.6), (3.55, 6.6))                               # 광 → 잎
arrow((6.55, 6.6), (6.8, 6.6))                               # 잎 → 탄소
arrow((8.25, 5.55), (5.6, 5.15), col=C_BLUE)                 # Q/D → 생식 시계
ax.text(7.2, 5.32, "$\\overline{Q/D}$", fontsize=7, color=C_BLUE)
arrow((1.8, 5.55), (1.8, 5.15), col="#b08400", style="<|-")   # 구조 → 광 (잎 층)
ax.text(1.88, 5.27, "canopy structure", fontsize=6, color=C_GRAY)
arrow((3.3, 5.75), (4.6, 5.15), col="#b08400")               # 광 → 과실 온도
ax.text(3.45, 5.22, "$I_f$", fontsize=7, color="#b08400")
arrow((3.3, 4.0), (3.55, 4.0))                               # 구조 → 생식 (truss 출현)
arrow((6.55, 4.0), (6.8, 4.0), col=C_ORANGE)                 # truss 단계 → 적엽 규칙
ax.text(6.67, 3.7, "set", fontsize=6, color=C_ORANGE, ha="center")
arrow((8.25, 3.0), (1.8, 3.0), col=C_RED, rad=-0.18)        # 적엽 → 캐노피
ax.text(5.05, 2.2, "leaf removal → fewer leaf layers, exposed trusses", fontsize=6.3, color=C_RED, ha="center",
        bbox=dict(fc="white", ec="none", pad=0.4))

out = ROOT / "figures/fig1_schematic.png"
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print(out)
