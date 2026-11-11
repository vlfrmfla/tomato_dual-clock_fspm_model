"""
하위 10엽(21–30번)의 야간 증산 비중은 엽면적 비중과 얼마나 다른가 (내부 검토 2026-11-03, 4번)

야간에는 Rn = 0이고 기공저항(일사 0에서의 값)·경계층저항·기온·VPD가 모든 잎에 같으므로,
기본 모델의 잎별 야간 증산은 엽면적에 비례한다 → 하위 잎 비중 ≈ 엽면적 비중.
민감도: gs_age_coupling=True (기공저항을 엽령 능력으로 나눔 → 오래된 하위 잎의 야간 기공이 더 닫힘)
출력: data_processed/night_share.csv, 표준출력
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figs as P      # noqa: E402

DP = Path(__file__).resolve().parents[1] / "data_processed"


def shares(R, tag):
    rows = []
    for k, v in R.items():
        net, et, night, area = (np.clip(v[x], 0, None) for x in ("net", "et", "et_night", "area"))
        rows.append(dict(variant=tag, season=k, area=area[20:].sum() / area.sum(), net=net[20:].sum() / net.sum(),
                         et=et[20:].sum() / et.sum(), night=night[20:].sum() / night.sum(),
                         night_total=night.sum(), night_per_area_21_30_vs_1_20=(night[20:].sum() / area[20:].sum())
                         / (night[:20].sum() / area[:20].sum())))
    return rows


if __name__ == "__main__":
    d = pd.DataFrame(shares(P.leaf_rank_runs(), "default") + shares(P.leaf_rank_runs({"gs_age_coupling": True}), "rs_age"))
    d.to_csv(DP / "night_share.csv", index=False)
    print(d.round(3).to_string(index=False))
