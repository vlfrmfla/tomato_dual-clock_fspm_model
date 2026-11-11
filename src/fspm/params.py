"""
모델 파라미터 (A7-6, RedKan, 2023-24 기준 초기값)

출처 표기: [obs] 우리 데이터 추정, [lit] 문헌, [cal] 보정 대상
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Params:
    # --- 재배 ---
    plant_density: float = 4 / 1.5        # 주/m² = 2.67 — 1 m 슬라브 4주(열당 4주/m) ÷ 열 간격 1.5 m [사용자 확인: 8 m 폭 중 식물 공간 6 m에 4열]
                                          # (구 312/224 = 1.39는 열이 없는 구획 길이까지 포함한 값)
    plants_per_slab: int = 4              # 로드셀 슬라브당 주수 [obs]

    transplant: str = "2023-09-18"

    # --- 구조 시계 ---
    Tbase: float = 10.0                   # °C [lit]
    Topt: float = 25.0                    # 이상에서 선형 감소 시작 [lit]
    Tmax: float = 35.0                    # 발달 정지 [lit]
    phyllo_mode: str = "load"             # 'constant' | 'two_phase' | 'qd' | 'load' (2026-11-06: two_phase → load)
    phyllochron: float = 23.4             # °Cd/마디, 단일 (대조) — 모델 내 적합 (src/test_phyllo.py), 누적 마디 RMSE 1.49
    phyllochron_early: float = 17.63      # φ1: 과실 부하 없을 때 — 모델 내 적합 (src/test_phyllo.py)
    phyllochron_late: float = 27.51       # φ2: 생장 중 과실 수 ≥ F* 일 때
    phyllo_load_F: float = 11.0           # F*: φ가 φ2에 이르는 생장 중 과실 수 (개/주) — φ = φ1 + (φ2−φ1)·min(1, F/F*)
    phyllo_switch_TT: float = 490.0       # 'two_phase' 대조 모형: 정식 후 °Cd 전환 (이전 기본값 19.3 → 28.1 °Cd)
    phyllo_qd_beta: float = 0.0           # 'qd' 모드 지수 [cal]; 0이면 온도만
    truss_first_node: float = 8.2         # 첫 화방 마디 [obs]
    truss_interval: float = 3.19          # 화방 간 마디 수 [obs]
    TT_app_anth: float = 76.0             # 화방 출현→개화 °Cd [obs] 일사 무관

    # --- 잎 ---
    leaf_area_max: float = 520e-4         # 개별 잎 최종 면적 m² [obs] 파괴조사 엽면적/엽수
    TT_leaf_expand: float = 250.0         # 출현→최종 면적 °Cd [cal]
    SLA: float = 0.020                    # m²/g [obs] ~200 cm²/g

    # --- 생식 시계 (화방 단위, 온도 × source) ---
    TT_anth_set: float = 60.0             # Q/D 비제한 시 개화→착과 °Cd [cal] (관측 최소 ~60)
    TT_set_ripe: float = 630.0            # 착과→수확 °Cd [obs] 1·2화방 12/28–1/03 수확. 과실온도 착색(fruit.py) 사용 시 630 → T1 12/28, T2 1/03
                                          # (기온만 사용 시 616–741 °Cd → 680)
    # g(Q/D) = 1 − exp[−a (Q/D − b)]  (Kang et al. 2011 형태, 확률 대신 발달 속도 배율로 사용)
    g_a: float = 2.39                     # [lit] Kang 2011 → [cal]
    g_b: float = 0.12                     # [lit] Kang 2011 → [cal]
    use_source_clock: bool = True         # False면 온도만 (대조 모델)
    clock_signal: str = "qd"              # 생식 시계 신호: "qd" = Q/D 이동평균, "rad" = 외부 일사/rad_ref 이동평균 (대조 모델 C)
    rad_ref: float = 10.0                 # MJ m⁻² d⁻¹ — "rad" 신호의 기준 (값 1)

    # --- Source (M2) ---
    use_source: bool = True               # False면 M1 (Q/D=1, 잎 면적 = 잠재)
    cover_trans: float = 0.65             # 상수 투과율 (use_roof_geometry=False 일 때) [cal]
    use_roof_geometry: bool = True        # CAD 지붕 기하로 시간별 유리 투과율 (greenhouse.py)
    tau_struct: float = 0.75              # 구조재·서까래·거터 그림자·오염 [cal]
    ppfd_per_W: float = 2.3               # µmol J⁻¹ — 일사(전천 단파) W m⁻² → PPFD 환산 [사용자 지정 2026-11-03; 이전 0.48 × 4.57 = 2.19]
    k_ext: float = 0.7                    # 소광계수 (light_model='beer' 일 때) [lit] 토마토 0.7–0.8
    light_model: str = "row"              # 'row' = 열 단면 광선추적 (light_row.py), 'beer' = 균일 층 Beer–Lambert
    row_spacing: float = 1.5              # 열 중심 간격 m [사용자 확인: 6 m / 4열]
    canopy_width: float = 0.9             # 열 캐노피 폭 m [재배자 입력, 사용자 확인 2026-11-07]
    row_azimuth: float = 0.0              # 열 방향 (0 = 남북, 용마루와 평행 가정)
    floor_refl: float = 0.65              # 바닥 단파 반사율 — 함안·횡계 모두 백색 필름, 0.65 [재배자 입력, 함안도 같은 값 — 사용자 확인 2026-11-07] (light_row 상향 산란)
    internode_len: float = 0.081          # 마디 간격 m [obs] (초장 367−64 cm)/(마디 45.5−8.2)
    a_par: float = 0.88                   # 잎 PAR 흡수율 [lit] Heuvelink 1996 Ch. 3.3: 토마토 잎 산란계수 0.12 → 흡수 0.88 (2026-11-06: 0.85 → 0.88)
    a_sw: float = 0.55                    # 잎 단파 전체 흡수율 (PAR+NIR) [lit]
    Ca: float = 450.0                     # µmol mol⁻¹ — use_measured_co2=False 일 때만
    use_measured_co2: bool = True         # Priva CMP 6 CO2 (10/25 이전·결측은 420 ppm)
    use_curtains: bool = True             # Priva CMP 6 커튼 개도로 실내 일사 감쇠
    curt_shade_frac: float = 0.5          # 커튼1(차광) 완전 폐쇄 시 차광률 [사용자 확인: 50%]
    curt_energy_frac: float = 0.9         # 커튼2·3(보온) 차광률 [사용자 지정: 90%]
    Vcmax25: float = 110.0                # [lit] week1 / Vanthoor
    Jmax25: float = 210.0                 # [lit] Vanthoor 2011
    alpha_J: float = 0.385
    theta_J: float = 0.7                  # 전자전달 곡률 [lit] Vanthoor 2011 (Jmax·α와 함께 추정된 세트). 0.9는 민감도만
                                          # (Graefe 2022의 0.962는 오이 잎 엽록체 수준 값 — 토마토 잎 전체 근거 아님, 2026-11-06 확인)
    Ci_ratio: float = 0.7
    leaf_resp_fvcb: bool = True           # 잎 호흡 = FvCB Rd (잎 면적·온도·엽령 기반). False면 rm_leaf × 잎 DW
    rd_frac: float = 0.65 / 110           # Rd25 0.65 µmol m⁻² s⁻¹ — 열 광모델에서 τ×Rd 격자의 생장 최적(τ≈0.95) & 증산비 0.87 (src/sens_rd_grid.py)
    rd_age_mode: str = "cap"              # 엽령에 따른 Rd: 'cap' Rd∝광합성능력 / 'const' 감소 없음 / 'cap_alpha' Rd∝cap & 양자수율∝cap
    rd_light_inhib: float = 0.0           # 주간 Rd 광억제 비율 (Kok 효과) — 기본 0 (표준 FvCB)
    cap_decline_start: float = 400.0      # 엽령 °Cd 이후 광합성 능력 감소 [cal]
    cap_decline_end: float = 1200.0
    cap_min: float = 0.4
    # 증산
    rs_min: float = 200.0                 # s m⁻¹ [lit] Boulard et al. 1991 토마토
    ra: float = 200.0                     # s m⁻¹ [lit] Stanghellini 1987 온실 대표값
    sw_night: float = 2.0                 # W m⁻² 실내 단파가 이보다 작으면 야간 (일사계 야간 영점 0.1–0.6 W m⁻² 제외)
    Rn_frac: float = 0.8                  # 순복사/흡수단파 [가정]
    rs_co2: bool = True                   # 기공저항 CO2 항 [lit] Stanghellini 1987 Eq. 3.52
    gs_age_coupling: bool = False         # True: rs/cap(엽령) — 사전 규칙, 로드셀로 조정하지 않음
    # 분배 (g DW 기준)
    leaf_area_pot: float = 650e-4         # 잠재 개별 잎 면적 m² [cal] (관측 최대 ~670 cm²)
    stem_dw_per_node: float = 1.3         # g/마디 [obs] 12/28 줄기 48.6 g / 41마디 [cal]
    truss_struct_pot: float = 0.8         # g/화방 [obs] 화방축 DW ~6 g / 8화방
    TT_struct: float = 300.0
    fruit_W_pot: float = 11.0             # 과실 1개 잠재 최종 DW g [cal] (관측 성숙과 ~9–11 g)
    fruit_TT_growth: float = 400.0        # 착과→DW 생장 종료 °Cd [cal] (관측: 착과 ~350 °Cd 후 DW 정체, 이후 착색)
    fruit_tm_frac: float = 0.45           # beta 함수 최대생장 시점 / 생장 종료
    fruits_per_truss: tuple = (4.8, 3.8, 3.0, 3.4, 3.0, 2.5, 2.5, 3.0, 2.6, 2.1, 1.9, 2.0)  # [obs] 최종조사 (적과 = 관리 입력)
    fruits_default: float = 2.5
    cvf_leaf: float = 0.72                # g DW/g CH2O [lit] Heuvelink 1996
    cvf_stem: float = 0.69
    cvf_fruit: float = 0.73
    rm_leaf: float = 0.03                 # g CH2O g⁻¹ DW d⁻¹ @25 °C [lit] TOMSIM
    rm_stem: float = 0.015
    rm_fruit: float = 0.01
    Q10: float = 2.0
    fruit_priority: float = 2.0           # 부족 시 과실 상대 우선권 π (1 = TOMSIM 비례). 목적함수 평평 → Rd 수준별 보정값 ~2.0으로 고정
    buf_frac: float = 0.3                 # 버퍼 상한 = 잎 DW × [cal]
    qd_ema_days: float = 5.0              # 생식 시계용 Q/D 지수이동평균 [lit] Kang: 꽃눈 후 2–7 phyllochron
    qd_ema_cap: float = 3.0               # 이동평균에 넣는 일 Q/D 상한 (모델 C 일사 신호도 같음) [가정] — 함안 99일 중 3일만 초과 (src/sens_qd_ema.py)
    qd_ema_init: float = 1.0              # 이동평균 시작값 (모델 C 같음) [가정]; NaN이면 첫날 값으로 시작 (민감도용)

    # --- 과실 미기상 → 착색기 숙기 (fruit.py) ---
    fruit_temp_ripening: bool = True      # True: 착색기(DM 생장 종료 후) 숙기를 과실 온도로 진행
    fruit_hang_nodes: float = 3.0         # 과실이 화방 마디보다 아래로 처지는 마디 수 (~24 cm) [가정]
    fruit_abs: float = 0.6                # 과실 단파 흡수율 [가정]
    fruit_h: float = 13.0                 # 과실 대류+복사 열전달계수 W m⁻² K⁻¹ (h_c 5–10 + h_r ~6) [가정]
    color_Tbase: float = 5.7              # 숙기 진행 기준온도 [lit] Adams et al. 2001 (14–26 °C 선형)
    color_Tplateau: float = 30.0          # 이 이상 평탄 [가정: 연구 범위 밖]
    TT_color: float = 186.5               # 착색기(DM 생장 종료→수확) 요구 °Cd (>5.7 °C, 과실 온도) — 두 온실 공통 [cal: 횡계 첫 수확 10/16,
                                          #   calibrate_hoenggye.fit_color, 첫 수확 일치 구간 180–193 → 가운데]; 함안 1·2화방 소실 12/28–1/03은 늦은 쪽 경계로만 확인
    color_hot: float = 30.0               # 품질 위험 임계 (라이코펜 억제) [lit] Brandt et al. 2006

    # --- 적엽 작업 패턴 (사람) ---
    prune_weekday: int = 0                # 작업 요일 0=월 [cal] (목요일 파괴조사 11/09 시점엔 미적엽)
    prune_max_per_event: int = 12         # 1회 최대 제거 수 [obs] 11/09→11/16 약 11장
    prune_min_batch: int = 1              # 이 수 이상 쌓여야 작업 (묶음 작업) [obs] ~6 = 2화방분

    def as_dict(self):
        return asdict(self)
