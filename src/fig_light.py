"""하엽 광 분포: Beer(바닥 평균 LAI) vs 열 단면 광선추적 — 2023-12-15, 관측 엽면적 0.95 m²/주로 정규화"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_m2 import INIT, START, load, r0_rule      # noqa: E402
from fspm import Params, simulate                         # noqa: E402
from fspm.light_row import row_light                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
C = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a"]

clim, hr, obs, fin, ev = load()
pj = json.loads((ROOT / "data_processed/params_m2.json").read_text())
p = replace(Params(), **{k: v for k, v in pj.items() if k in Params.__dataclass_fields__})
s, pl = simulate(clim, p, r0_rule(obs, fin), START, "2023-12-15", hourly=hr, **INIT)
act = [n for n in reversed(pl.nodes) if n.leaf.active]
la = np.array([n.leaf.area for n in act])
la = la * 0.95 / la.sum()
h = hr.loc["2023-12-15"]
I = h["Rg_out"].to_numpy()
pos = np.arange(1, len(la) + 1)
rows = []
fig, ax = plt.subplots(figsize=(6.2, 4.6))
# Beer, 구 밀도(1.39)·바닥 평균
dL = la * (312 / 224)
Lm = np.cumsum(dL) - dL / 2
beer = np.exp(-0.7 * Lm)
ax.plot(pos, beer * 100, "o-", color=C[0], lw=2, ms=4, label="Beer–Lambert, 1.39 plants m⁻² (floor)")
rows.append(("Beer 1.39", beer[-1], beer[-5:].mean()))
for c, wc in zip(C[1:], (0.6, 0.9, 1.2)):
    q = replace(p, canopy_width=wc)
    inc = row_light(la, q, h["sun_elev"].to_numpy(), h["sun_az"].to_numpy(),
                    (1 - h["kd"].to_numpy()) * I, h["kd"].to_numpy() * I)
    d = inc.sum(axis=0)
    rel = d / d[0]
    ax.plot(pos, rel * 100, "o-", color=c, lw=2, ms=4, label=f"Row model, w_c = {wc} m")
    rows.append((f"row {wc}", rel[-1], rel[-5:].mean()))
ax.set_xlabel("Leaf position from the top")
ax.set_ylabel("Daily irradiance relative to the top leaf (%)")
ax.set_ylim(0, 105)
ax.grid(alpha=0.25, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/light_profile.png", dpi=150)
for r in rows:
    print(f"{r[0]:10s} 최하위 {r[1]*100:.0f}%  하위5장 {r[2]*100:.0f}%")
