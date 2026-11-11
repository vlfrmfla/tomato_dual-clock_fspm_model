"""
열(hedgerow) 작물 광 분포 — 마디 높이 기반 2D 광선 추적 (turbid medium)

기하
  열 중심 x = k·w_r, 캐노피 띠 |x − k·w_r| ≤ w_c/2, 높이 z ∈ [z_bot, z_top]
  활성 잎 i 는 위에서부터 마디 간격 dz 로 층을 이룸 (층 i: z_top − i·dz ~ z_top − (i+1)·dz)
  엽면적밀도 LAD_i = 잎 면적 × 열 길이당 주수 / (w_c · dz)   [m² m⁻³]
투과
  잎 i 층 중앙의 지점(열 단면 여러 x 위치 평균)에서 방향 (β, φ_rel)으로 위쪽 광선
  수평 이동 = cotβ·|sin φ_rel| per 높이 (열에 수직인 성분)
  경로상 엽면적 = Σ LAD·(경로가 캐노피 띠 안에 있는 길이),  T = exp(−G · 엽면적경로)
  G = 0.5 (구형 엽각 분포)
흡수 (잎 단위면적, 평균)
  직달: I_b,h / sinβ · G · T_b       산란(균일 흐린 하늘): I_d · Σ_k ω_k T_k  (ω = 입체각 비율)
  × 흡수율 a (산란 손실 근사)
바닥 반사 (p.floor_refl > 0, 바닥 백색 필름)
  바닥 입사 = 캐노피 상단 입사 − 캐노피 차단 (Σ 잎 입사 × 잎 면적 × 재식밀도), 상향 산란 I_up = ρ · 바닥 입사
  잎·지점 입사에 I_up · T_up 추가 (T_up = 캐노피를 위아래로 뒤집어 같은 산란 투과로 계산, 1회 반사만)
"""
import numpy as np

G = 0.5
# 하늘 방향 (고도 15/45/75° × 방위 6) — 균일 흐린 하늘의 입체각 가중
_EL = np.radians([15.0, 45.0, 75.0])
_EL_W = np.array([np.cos(np.radians(15)) * 30, np.cos(np.radians(45)) * 30, np.cos(np.radians(75)) * 30])
_EL_W = _EL_W / _EL_W.sum()
_AZ = np.radians(np.arange(0, 180, 30) + 15)          # 열에 대한 상대 방위 (대칭이므로 0–180)
SKY = [(b, a, w / len(_AZ)) for b, w in zip(_EL, _EL_W) for a in _AZ]


def _transmit_dirs(z_pts, lad_layers, dz, z_top, w_r, w_c, beta, phi_rel, x0s, nsub=6):
    """z_pts: (지점,) 높이, lad_layers: (n,) 위→아래 층 LAD, beta·phi_rel: (k,) 방향 → (k, 지점) 평균 투과율"""
    n = len(lad_layers)
    beta, phi_rel = np.atleast_1d(beta), np.atleast_1d(phi_rel)
    tb = (1.0 / np.tan(beta) * np.abs(np.sin(phi_rel)))[:, None, None]
    s_per_z = (1.0 / np.sin(beta))[:, None]
    k = (np.arange(n * nsub) + 0.5) / nsub
    zm = z_top - k * dz
    lad_sub = np.repeat(lad_layers, nsub)[None, :] * (dz / nsub) * s_per_z   # (방향, n·nsub)
    above = zm[None, :] > z_pts[:, None]                            # (지점, n·nsub)
    dzm = zm[None, :] - z_pts[:, None]
    out = np.zeros((len(beta), len(z_pts)))
    for x0 in x0s:
        xr = np.mod(x0 + tb * dzm[None] + w_r / 2, w_r) - w_r / 2
        inside = np.abs(xr) <= w_c / 2
        acc = ((above[None] & inside) * lad_sub[:, None, :]).sum(axis=2)
        out += np.exp(-G * acc)
    return out / len(x0s)


_SKY_B = np.array([s[0] for s in SKY]); _SKY_A = np.array([s[1] for s in SKY]); _SKY_W = np.array([s[2] for s in SKY])


def _sky_transmit(z_pts, lad_layers, dz, z_top, w_r, w_c, x0s):
    """균일 흐린 하늘 가중 투과 Σ ω_k T_k"""
    return _SKY_W @ _transmit_dirs(z_pts, lad_layers, dz, z_top, w_r, w_c, _SKY_B, _SKY_A, x0s)


def row_light(leaf_area, p, sun_beta, sun_az, I_b_h, I_d_h, extra_depths=None):
    """
    leaf_area : (n,) 활성 잎 면적 (위→아래), m²/주
    sun_beta, sun_az : (24,) 태양 고도·방위 (°)
    I_b_h, I_d_h : (24,) 캐노피 상단 직달·산란 (수평면, 같은 단위)
    extra_depths : (m,) 캐노피 상단에서의 깊이 m (과실 위치 등) — 주어지면 (잎 입사, 지점 입사) 튜플 반환
    반환 : (24, n) 잎 단위면적당 평균 입사(흡수 전) 복사, 같은 단위
    """
    n = len(leaf_area)
    w_r, w_c, dz = p.row_spacing, p.canopy_width, p.internode_len
    plants_per_m = p.plant_density * w_r
    lad = leaf_area * plants_per_m / (w_c * dz)
    z_top = n * dz
    z_pts = z_top - (np.arange(n) + 0.5) * dz
    m = 0 if extra_depths is None else len(extra_depths)
    if m:
        z_pts = np.concatenate([z_pts, z_top - np.asarray(extra_depths, float)])
    x0s = np.linspace(-w_c / 2 * 0.8, w_c / 2 * 0.8, 3)
    # 산란: 날마다 캐노피가 같으므로 1회
    Td = _sky_transmit(z_pts, lad, dz, z_top, w_r, w_c, x0s)
    out = I_d_h[:, None] * Td[None, :]
    row_az = np.radians(p.row_azimuth)
    hs = np.flatnonzero((np.asarray(sun_beta) > 2) & (np.asarray(I_b_h) > 0))
    if len(hs):
        b = np.radians(np.asarray(sun_beta)[hs])
        Tb = _transmit_dirs(z_pts, lad, dz, z_top, w_r, w_c, b, np.radians(np.asarray(sun_az)[hs]) - row_az, x0s)
        out[hs] += (np.asarray(I_b_h)[hs] / np.sin(b) * G)[:, None] * Tb
    rho = getattr(p, "floor_refl", 0.0)
    if rho > 0:
        floor = np.clip(np.asarray(I_b_h) + np.asarray(I_d_h) - (out[:, :n] * leaf_area[None, :]).sum(axis=1) * p.plant_density, 0, None)
        Tu = _sky_transmit(z_top - z_pts, lad[::-1], dz, z_top, w_r, w_c, x0s)   # 뒤집은 캐노피: 아래 층이 위, 지점 높이 z → z_top − z
        out = out + (rho * floor)[:, None] * Tu[None, :]
    if m:
        return out[:, :n], out[:, n:]
    return out
