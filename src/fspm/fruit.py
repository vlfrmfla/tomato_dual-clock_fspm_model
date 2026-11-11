"""
과실 미기상 → 착색기 숙기 (적엽이 화방 노출을 바꿔 생식 시계를 직접 당기는 경로)

과실 위치 : 화방 마디보다 fruit_hang_nodes 마디 아래 (그 사이 잎이 남아 있으면 과실을 가림)
과실 입사 : 열 광선추적(light_row)의 해당 깊이 입사 × 0.5
            (구형 과실의 표면적당 입사 = 직달 I_n/4, 산란 I_d/2 ; 구형 엽각 잎(G=0.5)의 절반)
과실 온도 : 정상상태 에너지수지  T_f = T_air + a_f · I_f / (h_c + h_r)
            a_f 단파 흡수율, h_c+h_r 대류+복사 열전달계수 (사과 FST 에너지수지: Li et al. 2014 형태를 단순화)
숙기 진행 : 과실 온도의 적산온도, 기준 5.7 °C 위 선형 (Adams et al. 2001: 14–26 °C에서 선형, 최적 없음,
            812 °Cd > 5.7 °C; 후기 성숙 단계가 온도에 더 민감) — 연구 범위 밖 color_Tplateau 이상은 평탄
품질 위험 : 착색기 과실 온도 > 30 °C 시간 (라이코펜 억제: Brandt et al. 2006; Tomes 1963) — 숙기 속도와 분리
근거 관측 : 숙기 화방 위 두 화방까지 적엽 시 과실 낮 고온·밤 저온, 1.2일 조기 숙기 (Adams et al. 2002)
"""
import numpy as np

from .params import Params


def fruit_temperature(T_air_h, I_fruit_h, p: Params):
    """(24,) 기온, (24, m) 과실 표면적당 입사 단파 W m⁻² → (24, m) 과실 온도 °C"""
    return np.asarray(T_air_h, float)[:, None] + p.fruit_abs * np.nan_to_num(I_fruit_h) / p.fruit_h


def color_rate(T, p: Params):
    """숙기 진행 °Cd/일 단위 (시간값 배열에 적용 후 /24 하여 합산): 기준온도 위 선형, 평탄 상한"""
    T = np.asarray(T, float)
    return np.clip(T - p.color_Tbase, 0, p.color_Tplateau - p.color_Tbase)


def fruit_day(T_air_h, I_fruit_h, p: Params):
    """반환: 과실별 일 착색 증분(°Cd), 30 °C 초과 시간, 주간 평균 과실-기온 차"""
    Tf = fruit_temperature(T_air_h, I_fruit_h, p)
    inc = color_rate(Tf, p).sum(axis=0) / 24.0
    hot = (Tf > p.color_hot).sum(axis=0)
    day = np.nan_to_num(I_fruit_h).sum(axis=1) > 0
    dT = (Tf - np.asarray(T_air_h, float)[:, None])[day].mean(axis=0) if day.any() else np.zeros(Tf.shape[1])
    return inc, hot, dT
