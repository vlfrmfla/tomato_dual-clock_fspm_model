"""외부 검증 그림: A7-6 vs AI Hub tom2·tom3 — 구조 시계(화방 개화 간격), 생식 시계(개화→착과 °Cd vs 일사)"""
from pathlib import Path
import sys
import matplotlib, matplotlib.pyplot as plt, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_aihub import events, load_env   # noqa: E402
ROOT = Path(__file__).resolve().parents[1]; DP = ROOT / "data_processed"
matplotlib.rcParams["font.family"] = "AppleGothic"; matplotlib.rcParams["axes.unicode_minus"] = False
C = {"A7-6": "#2a78d6", "tom2": "#eb6834", "tom3": "#1baf7a"}; INK = "#52514e"

c = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"]).set_index("date"); c["TT"] = (c.Tmean - 10).clip(lower=0).cumsum()
ev = pd.read_csv(DP / "truss_events.csv", parse_dates=["anth"]).dropna(subset=["anth"])
iv = {"A7-6": [1 / np.polyfit(c.TT.reindex(g.anth).to_numpy(), g.truss, 1)[0] for _, g in ev.groupby("plant")]}
e, b = events()
for farm in ("tom2", "tom3"):
    env = load_env(farm); bb = b[b.farm == farm].groupby(["plant", "date"]).truss.max().reset_index()
    bb = bb[bb.date >= env.index.min()]; bb["TT"] = env.TT.reindex(bb.date).to_numpy(); s = []
    for _, g in bb.dropna().groupby("plant"):
        if g.date.nunique() >= 5 and g.truss.nunique() >= 3:
            k = np.polyfit(g.TT, g.truss, 1)[0]
            if k > 0: s.append(1 / k)
    iv[farm] = s
ex = pd.read_csv(DP / "external_aihub_events.csv")
a7 = pd.read_csv(DP / "truss_events.csv").dropna(subset=["anth_set_TT", "Rg7"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
a = ax[0]
bp = a.boxplot([iv[k] for k in iv], widths=0.5, patch_artist=True, showfliers=False)
a.set_xticks(range(1, len(iv) + 1)); a.set_xticklabels([f"{k}\n(n={len(iv[k])})" for k in iv])
for patch, k in zip(bp["boxes"], iv): patch.set_facecolor(C[k]); patch.set_alpha(0.35); patch.set_edgecolor(C[k])
for med in bp["medians"]: med.set_color(INK)
a.set_ylabel("°Cd / 화방 (기준 10 °C)"); a.set_title("(a) 구조 시계: 화방 개화 간격", loc="left")
a = ax[1]
a.scatter(a7.Rg7, a7.anth_set_TT, s=22, color=C["A7-6"], alpha=0.8, label=f"A7-6 (n={len(a7)}, 조사 4일)")
for farm in ("tom2", "tom3"):
    g = ex[ex.farm == farm]; a.scatter(g.Rg7, g.lag_TT, s=12, color=C[farm], alpha=0.45, label=f"{farm} (n={len(g)}, 조사 7일)")
a.set_xlabel("개화 후 7일 외부 일사 (MJ/m²/일)"); a.set_ylabel("개화→착과 (°Cd)"); a.set_title("(b) 생식 시계: 개화→착과 vs 일사", loc="left")
for a in ax: a.grid(alpha=0.25, lw=0.6); a.spines[["top", "right"]].set_visible(False)
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "figures/external_aihub.png", dpi=150); print("ok")
