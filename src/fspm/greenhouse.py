"""
온실 지붕 기하 → 시간별 유리 투과율 (A6 동, CAD 도면 기준)

도면 (지붕평면도·정면도·배면도)
  벤로형 연동, 스팬 4.0 m, 스팬당 용마루 1개, 용마루 남북 방향 → 지붕면은 동향·서향
  용마루 GL+6.36 m, 처마 GL+5.30 m → 경사 atan(1.06/2.0) = 27.9°
  피복: 투명유리 5 mm, AL 서까래 @1.0 m

방법
  1. 태양 위치 (Spencer 1971 적위·균시차), KST(135°E 표준자오선)
  2. 직달/산란 분리: Erbs et al. (1982) 시간 청명지수 상관식
  3. 유리판 투과율: Fresnel 반사 + 흡수, 다중반사 포함 (Duffie & Beckman, s·p 편광 평균)
     직달 = 동·서 지붕면 입사각별 투과율을 각 면으로 들어오는 직달 비율로 가중
     산란 = 경사면 유효입사각 θe = 59.7 − 0.1388 s + 0.001497 s² (Brandemuehl & Beckman 1980)
  4. 구조재·서까래·거터 그림자, 오염 등은 τ_struct (보정 상수) — model 쪽에서 곱함
"""
import numpy as np
import pandas as pd

LAT, LON = 35.2299, 128.4247         # 국립원예특작과학원 시설원예연구소, 경남 함안군 함안면 봉성리 334 (OSM 봉성리 중심)
ROOF_SLOPE = 27.9                   # °
N_GLASS, KL_GLASS = 1.526, 0.02     # 굴절률, 소광계수×두께 (저철분 유리 K≈4 m⁻¹ × 5 mm)


def solar_position(times: pd.DatetimeIndex, lat=LAT, lon=LON):
    """태양 고도·방위 (°, 방위는 북=0 시계방향). times = KST 시각"""
    doy = times.dayofyear.to_numpy()
    hr = times.hour.to_numpy() + times.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))          # 분
    solar_time = hr + (4 * (lon - 135.0) + eot) / 60
    ha = np.radians(15 * (solar_time - 12))
    phi = np.radians(lat)
    sin_b = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    beta = np.arcsin(np.clip(sin_b, -1, 1))
    cos_az = (np.sin(dec) - np.sin(beta) * np.sin(phi)) / (np.cos(beta) * np.cos(phi) + 1e-12)
    az = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    az = np.where(ha > 0, 360 - az, az)
    E0 = 1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g) + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g)
    return np.degrees(beta), az, E0


def erbs_diffuse_fraction(Rg, beta_deg, E0):
    """Erbs et al. (1982) 시간 산란비 kd(kt)"""
    I0h = 1367.0 * E0 * np.maximum(np.sin(np.radians(beta_deg)), 1e-3)
    kt = np.clip(np.nan_to_num(Rg) / I0h, 0, 1.2)
    kd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.80, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                           0.165))
    return np.clip(kd, 0, 1)


def slab_transmittance(theta_deg, n=N_GLASS, KL=KL_GLASS):
    """유리판 투과율 (Fresnel 양면 + 흡수, 다중반사, 편광 평균)"""
    th = np.radians(np.clip(theta_deg, 0, 89.9))
    thr = np.arcsin(np.sin(th) / n)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = (np.sin(thr - th) / np.sin(thr + th)) ** 2
        rp = (np.tan(thr - th) / np.tan(thr + th)) ** 2
    r0 = ((n - 1) / (n + 1)) ** 2
    rs = np.where(th < 1e-6, r0, rs)
    rp = np.where(th < 1e-6, r0, rp)
    ta = np.exp(-KL / np.cos(thr))
    T = 0.0
    for r in (rs, rp):
        T = T + 0.5 * ta * (1 - r) ** 2 / (1 - (r * ta) ** 2)
    return T


def roof_glass_transmittance(times: pd.DatetimeIndex, Rg_out, slope=ROOF_SLOPE):
    """
    시간별 지붕 유리 투과율 (구조재 제외). 반환 DataFrame: beta, kd, tau_glass
    times 는 시간 평균 구간의 시작 시각 → 중앙(+30분)에서 태양 위치 계산
    """
    t_mid = times + pd.Timedelta(minutes=30)
    beta, az, E0 = solar_position(t_mid)
    kd = erbs_diffuse_fraction(Rg_out, beta, E0)
    s = np.radians(slope)
    b = np.radians(beta)
    cos_th = {}
    for name, gp in (("E", 90.0), ("W", 270.0)):
        cos_th[name] = np.cos(s) * np.sin(b) + np.sin(s) * np.cos(b) * np.cos(np.radians(az - gp))
    wE, wW = np.maximum(cos_th["E"], 0), np.maximum(cos_th["W"], 0)
    tot = wE + wW
    T_E = slab_transmittance(np.degrees(np.arccos(np.clip(cos_th["E"], 0, 1))))
    T_W = slab_transmittance(np.degrees(np.arccos(np.clip(cos_th["W"], 0, 1))))
    T_beam = np.where(tot > 0, (wE * T_E + wW * T_W) / np.where(tot > 0, tot, 1), 0.0)
    theta_e = 59.68 - 0.1388 * slope + 0.001497 * slope ** 2       # Brandemuehl & Beckman (1980) Eq. 7
    T_diff = float(slab_transmittance(theta_e))
    tau = np.where(beta > 0, (1 - kd) * T_beam + kd * T_diff, T_diff)
    return pd.DataFrame({"sun_elev": beta, "sun_az": az, "kd": kd, "tau_glass": tau}, index=times)
