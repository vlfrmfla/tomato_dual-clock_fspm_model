"""
Source: 1D 마디 층위 광흡수 → 잎별 FvCB 광합성 + 잎별 Penman–Monteith 증산 (시간 단위)

층위 = 활성 잎을 위(최상위 마디) → 아래로 정렬. 잎 i 위의 누적 엽면적지수 L_above(i)
  잎 i 단위면적당 흡수광 = I_top · k · exp(−k · (L_above + ΔL_i/2))
  ΔL_i = 잎 면적(m²/주) × 재식밀도(주/m²)
적엽 = 아래층 제거 → 남은 잎의 흡수광은 변하지 않고(위에서 가려지므로), 캐노피 총흡수와 증산 표면만 감소

FvCB: week1/3_pgm_tomato.py 파라미터 (Vanthoor et al. 2011 기반), 잎 엽령에 따라 Vcmax·Jmax 감소
증산 : 잎 단위 Penman–Monteith (Stanghellini 1987 형태), 기공저항 = Boulard et al. (1991) 토마토식을
       잎이 받는 단파복사로 적용 rs = rs_min·[1 + 1/exp(0.05(I_leaf − 50))] × f(CO2) (Stanghellini 1987 Eq. 3.52),
       경계층저항 ra = 200 s/m
"""
import numpy as np

from .params import Params

R_GAS = 8.314


def _arr(k25, Ha, T):
    TK = T + 273.15
    return k25 * np.exp(np.clip(Ha / R_GAS * (1 / 298.15 - 1 / TK), -50, 50))


def _peaked(k25, Ha, Hd, DS, T):
    TK = T + 273.15
    num = _arr(k25, Ha, T)
    return num * (1 + np.exp((DS * 298.15 - Hd) / (R_GAS * 298.15))) / (1 + np.exp((DS * TK - Hd) / (R_GAS * TK)))


def fvcb_gross(par_abs, T, Ca, cap, p: Params):
    """총광합성 (µmol CO2 m⁻² leaf s⁻¹). 배열 브로드캐스트. cap = 엽령 광합성 능력 배율"""
    Vc = _peaked(p.Vcmax25, 65330.0, 149250.0, 486.0, T) * cap
    Jm = _peaked(p.Jmax25, 37000.0, 220000.0, 710.0, T) * cap
    Kc = _arr(404.9, 79430.0, T)
    Ko = _arr(278400.0, 36380.0, T)
    Gs = _arr(42.75, 37830.0, T)
    Ci = p.Ci_ratio * Ca
    Ac = Vc * (Ci - Gs) / (Ci + Kc * (1 + 210000.0 / Ko))
    alpha = p.alpha_J * cap if p.rd_age_mode == "cap_alpha" else p.alpha_J     # 노화엽 양자수율 저하 (선택)
    a, b, c = p.theta_J, -(alpha * par_abs + Jm), alpha * par_abs * Jm
    J = (-b - np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a)
    Aj = J * (Ci - Gs) / (4 * Ci + 8 * Gs)
    return np.maximum(np.minimum(Ac, Aj), 0.0)


def leaf_capacity(tt_age, p: Params):
    """엽령에 따른 광합성 능력 (1 → cap_min), 하위 노화엽"""
    x = (tt_age - p.cap_decline_start) / (p.cap_decline_end - p.cap_decline_start)
    return 1.0 - (1.0 - p.cap_min) * np.clip(x, 0, 1)


def _pm_leaf(Rn, T, VPD_kPa, rs, ra):
    """잎 단위면적당 잠열 (W m⁻²): Stanghellini형 PM, 양면 현열교환"""
    rho_cp = 1200.0                                   # J m⁻³ K⁻¹
    gamma = 0.066                                     # kPa K⁻¹
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    delta = 4098 * es / (T + 237.3) ** 2              # kPa K⁻¹
    return np.maximum((delta * Rn + 2 * rho_cp * VPD_kPa / ra) / (delta + gamma * (1 + rs / ra)), 0.0)


def inside_shortwave(Rg_out_h, curt_h, p: Params, tau_glass_h=None):
    """
    실내 캐노피 상단 단파 W m⁻²
      use_roof_geometry: 외부 × τ_struct × τ_glass(시간, CAD 지붕 기하) × Π(1 − 차광률_k × 개도_k)
      아니면            : 외부 × cover_trans(상수) × 커튼
    """
    if p.use_roof_geometry and tau_glass_h is not None:
        I = np.nan_to_num(Rg_out_h) * p.tau_struct * np.nan_to_num(np.asarray(tau_glass_h, float), nan=0.8)
    else:
        I = np.nan_to_num(Rg_out_h) * p.cover_trans
    if p.use_curtains and curt_h is not None:
        c = np.nan_to_num(np.asarray(curt_h, float)) / 100.0          # (24, 3)
        f = (1 - p.curt_shade_frac * c[:, 0]) * (1 - p.curt_energy_frac * c[:, 1]) * (1 - p.curt_energy_frac * c[:, 2])
        I = I * f
    return I


def canopy_day(leaf_area, leaf_age, T_h, Rg_out_h, VPD_h, p: Params, curt_h=None, co2_h=None, tau_glass_h=None,
               sun_h=None, fruit_depths=None):
    """
    leaf_area : (n,) 활성 잎 면적 m²/주, **위→아래 순서**
    leaf_age  : (n,) 엽령 °Cd
    T_h, Rg_out_h, VPD_h : (24,) 시간 기온 °C, 외부 일사 W m⁻², VPD kPa
    curt_h    : (24, 3) 커튼 1(차광)·2·3(보온) 개도 %, co2_h : (24,) ppm — 없으면 p.Ca
    반환 dict:
      ch2o_leaf  (n,) 잎별 일 총동화 g CH2O/주
      rd_leaf    (n,) 잎별 일 암호흡 g CH2O/주 (FvCB Rd) → 순수지 = ch2o − rd
      et_leaf    (n,) 잎별 일 증산 g H2O/주
      par_leaf   (n,) 잎별 일 흡수 PAR mol m⁻² leaf d⁻¹  (R2 광 규칙용)
      par_top    일 캐노피 상단 PAR mol m⁻² d⁻¹
    """
    n = len(leaf_area)
    if n == 0:
        z = np.zeros(0)
        return dict(ch2o_leaf=z, rd_leaf=z, et_leaf=z, et_night=0.0, par_leaf=z, par_top=0.0)
    dL = leaf_area * p.plant_density
    L_mid = np.cumsum(dL) - dL / 2                          # (n,)
    att = p.k_ext * np.exp(-p.k_ext * L_mid)                # (n,) 잎면적당 흡수 비율

    I_sw = inside_shortwave(Rg_out_h, curt_h, p, tau_glass_h)   # (24,) 실내 단파 W m⁻²
    par_top = I_sw * p.ppfd_per_W                           # µmol m⁻² s⁻¹ (일사 → PPFD 2.3 µmol J⁻¹)
    if p.light_model == "row" and sun_h is not None:
        from .light_row import row_light
        beta, az, kd = (np.nan_to_num(np.asarray(v, float)) for v in sun_h)
        if fruit_depths is not None and len(fruit_depths):
            inc, inc_f = row_light(leaf_area, p, beta, az, (1 - kd) * I_sw, kd * I_sw, extra_depths=fruit_depths)
            fruit_sw = 0.5 * inc_f                              # 구형 과실 표면적당 입사 W m⁻²
        else:
            inc = row_light(leaf_area, p, beta, az, (1 - kd) * I_sw, kd * I_sw)   # (24, n) 잎 입사 W m⁻²
            fruit_sw = None
        sw_inc = inc
        par_abs = inc * p.ppfd_per_W * p.a_par
        sw_abs = inc * p.a_sw
    else:
        fruit_sw = None
        par_abs = par_top[:, None] * att[None, :]           # (24, n)
        sw_abs = I_sw[:, None] * att[None, :]               # (24, n) W m⁻² leaf
        sw_inc = sw_abs / p.k_ext

    cap = leaf_capacity(leaf_age, p)[None, :]
    T = T_h[:, None]
    Ca = p.Ca if (co2_h is None or not p.use_measured_co2) else np.asarray(co2_h, float)[:, None]
    A = fvcb_gross(par_abs, T, Ca, cap, p)                  # µmol m⁻² s⁻¹
    ch2o = (A * 3600 * 30e-6).sum(axis=0) * leaf_area       # g CH2O/주/d
    # 잎 암호흡 Rd (FvCB): Rd25 = rd_frac·Vcmax25 (Collatz et al. 1991), Arrhenius Ha=46390, 엽령 cap 비례, 주야 24 h
    rd_scale = np.ones_like(cap) if p.rd_age_mode == "const" else cap
    Rd = _arr(p.rd_frac * p.Vcmax25, 46390.0, T) * rd_scale * np.ones_like(par_abs)
    if p.rd_light_inhib > 0:
        Rd = np.where(par_abs > 10, Rd * (1 - p.rd_light_inhib), Rd)   # 주간 호흡 광억제 (선택)
    rd = (Rd * 3600 * 30e-6).sum(axis=0) * leaf_area        # g CH2O/주/d

    rs = p.rs_min * (1 + 1 / np.exp(0.05 * (sw_inc - 50)))  # 잎 입사광 기준
    if p.rs_co2 and co2_h is not None:
        # Stanghellini (1987) Eq. 3.52: f(CO2) = 1 + 6.1e-7 (CO2 − 200)², CO2 ≥ 1100 vpm 에서 1.5
        c = np.asarray(co2_h, float)[:, None]
        rs = rs * np.where(c < 1100, 1 + 6.1e-7 * (c - 200) ** 2, 1.5)
    if p.gs_age_coupling:
        rs = rs / cap                                       # 기공 전도도 ∝ 광합성 능력 (엽령), 추가 파라미터 없음
    Rn = p.Rn_frac * sw_abs if p.light_model != "row" else 0.9 * sw_abs   # row: a_sw 반영된 흡수 단파의 90%
    LE = _pm_leaf(Rn, T, VPD_h[:, None], rs, p.ra)          # W m⁻² leaf
    lam = 2.45e6
    et_h = LE / lam * 3600 * 1000 * leaf_area[None, :]      # (24, n) g H2O/주/h
    et = et_h.sum(axis=0)                                   # g H2O/주/d
    night = I_sw < p.sw_night                               # 야간: 일사계 영점 오차를 0으로 판별
    et_night = float(et_h[night].sum())                     # 야간 증산 — 습도 부하
    et_night_leaf = et_h[night].sum(axis=0)                 # 잎별 야간 증산 (엽위 분석용)

    return dict(ch2o_leaf=ch2o, rd_leaf=rd, et_leaf=et, et_night=et_night, et_night_leaf=et_night_leaf, fruit_sw=fruit_sw,
                par_leaf=(par_abs * 3600e-6).sum(axis=0),
                par_top=float((par_top * 3600e-6).sum()))
