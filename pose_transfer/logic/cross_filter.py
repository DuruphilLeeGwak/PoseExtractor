"""
========================================
Cross-Filtering (교차 필터링) 로직
========================================

개요:
    Body 모델(COCO 17)과 DWPose(COCO-WholeBody 133)를 결합하여
    할루시네이션은 제거하고 디테일은 보존하는 하이브리드 필터링 시스템

핵심 아이디어:
    1. Body 모델 → "존재 여부" 검증 (감시자 역할)
       - 17개 주요 관절의 신뢰도로 "여기에 팔이 있는가?" 판단
       - 오검출이 적고 안정적이지만 디테일 부족 (손가락, 얼굴 없음)
    
    2. DWPose → "정밀 좌표" 제공 (아티스트 역할)
       - 133개 키포인트로 손가락, 얼굴 랜드마크까지 상세 묘사
       - 디테일이 풍부하지만 할루시네이션 (환각) 발생 가능
    
    3. 교차 검증 → "Body가 보증하면, DWPose를 쓴다"
       - Body가 "손목 있음" 확인 → DWPose의 21개 손가락 전부 승인
       - Body가 "손목 없음" → DWPose 손가락 좌표 무시 (할루시네이션 제거)

작동 원리:
    [단계 0] 사전 우회 시스템 (2-tier bypass)
        0-0. Clean Mode: 의심 키포인트가 0개 (디버깅 정보만, 임계값 유지)
        0-1. Full-Body Bypass: 17개 body가 모두 >6.0이면 전체 승인
        0-2. Individual Bypass: 단일 키포인트 >8.0이면 무조건 승인
    
    [단계 1] Body 17개 검증
        - Body 신뢰도 > threshold → DWPose 좌표 사용
    
    [단계 2] 부모-자식 종속 필터링
        - 손: 손목 승인 → 21개 손가락 승인
        - 발: 발목 승인 → 3개 발가락 승인
        - 얼굴: 코 or 어깨 승인 → 68개 얼굴 랜드마크 승인

할루시네이션 방지:
    - 손목 없는데 손가락만 suspicious 범위(0.05~2.0)에 있으면 할루시네이션으로 판단
    - Body 모델에는 손목까지만 있고 손가락(91-133)은 없어서 검증 불가
    - Clean Mode는 상태 표시용 (임계값 완화 없음)
"""
import numpy as np
from typing import Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class CrossFilterConfig:
    """
    교차 필터링 설정 클래스
    
    이 설정으로 필터링의 엄격함/관대함을 조절할 수 있습니다.
    """
    # ========== 기본 설정 ==========
    enabled: bool = True
    """필터링 활성화 여부 (False면 DWPose 원본 그대로 사용)"""
    
    # ========== Body 모델 신뢰도 임계값 (YOLO: 0~1 범위) ==========
    body_confidence_threshold: float = 0.5
    """
    Body 모델(YOLO)의 키포인트 신뢰도 임계값
    
    ⚠️ **Confidence 범위**: Body 모델은 **0~1 범위의 확률값** (Sigmoid 출력)
       - 0.5 이상: 높은 확신
       - 0.3~0.5: 중간 확신
       - 0.3 이하: 낮은 확신
    
    - 이 값보다 높은 신뢰도를 가진 Body 키포인트만 "진짜 존재"로 인정
    - 권장 범위: 0.3 ~ 0.8
    - 너무 낮으면: Body 오검출이 DWPose까지 영향 (할루시네이션 증가)
    - 너무 높으면: 실제 있는 부위도 제거됨 (과도한 필터링)
    """
    
    # ========== 부모-자식 종속 규칙 ==========
    enable_hand_dependency: bool = True
    """손목이 없으면 손가락도 없다 (Body는 손목까지만 검출)"""
    
    enable_foot_dependency: bool = True
    """발목이 없으면 발가락도 없다"""
    
    enable_face_dependency: bool = True
    """코 또는 목(어깨 중점)이 없으면 얼굴 랜드마크도 없다"""
    
    # ========== DWPose 자체 신뢰도 체크 ==========
    dw_min_confidence: float = 0.05
    """
    DWPose 최소 신뢰도 안전 장치
    - Body가 승인해도 DWPose 자체 신뢰도가 이보다 낮으면 제외
    - 권장: 0.05 ~ 0.1
    """
    
    # ========== DWPose 고신뢰도 보호 (개별 우회, rtmlib: 2.8~8.0+ 범위) ==========
    dw_high_confidence_threshold: float = 8.0
    """
    개별 키포인트 고신뢰도 보호 임계값
    
    ⚠️ **Confidence 범위**: DWPose(rtmlib)는 **2.8~8.0+ 범위의 로그 확률** (SimCC 출력)
       실제 측정값 (rtmlib Wholebody):
       - 평균: 5.5
       - 중앙값: 5.78 (50th percentile)
       - 75th percentile: 6.28
       - 95th percentile: 6.86
       - 99th percentile: 7.34
       - 최대: 8.0+
       
       ※ 0~1 확률값이 **아닙니다**! 이것은 정규화되지 않은 로그 확률/로짓입니다.
    
    - DWPose 신뢰도가 이 값을 초과하면 Body 검증 없이 무조건 승인
    - Body가 오검출하거나 보지 못한 경우에도 DWPose가 확신하면 보존
    - 권장 범위: 7.0 ~ 9.0 (상위 5% 이상)
    - 비활성화: 999.0 (실질적으로 도달 불가능한 값)
    """
    
    # ========== DWPose 전신 확신 모드 (전체 우회) ==========
    dw_full_body_confidence_threshold: float = 6.0
    """
    전신 확신 모드 임계값
    
    ⚠️ **Confidence 범위**: DWPose는 2.8~8.0+ 범위 사용!
       6.0 = "평균 이상" (50th: 5.78, 75th: 6.28)
    
    - DWPose의 Body 17개 키포인트가 모두 이 값 이상이면 Cross-Filter 전체 우회
    - 전신이 명확한 이미지에서 과도한 필터링 방지
    - 권장 범위: 5.0 ~ 7.0 (평균~상위 25%)
    - 비활성화: 999.0
    """
    
    # ========== Clean Mode 설정 ==========
    dw_suspicious_threshold: float = 2.0
    """
    의심 키포인트 판단 기준
    
    ⚠️ **Confidence 범위**: DWPose는 2.8~8.0+ 범위 사용!
       2.0 = "매우 낮은 확신" (1st percentile: 2.85 근처)
    
    - dw_min_confidence < score <= dw_suspicious_threshold 범위가 "의심 구간"
    - 이 구간의 키포인트가 0개면 Clean Mode 활성화
    - 할루시네이션은 보통 이 구간에 나타남 (낮은 신뢰도지만 0은 아님)
    """
    
    clean_mode_body_threshold: float = 0.2
    """
    Clean Mode일 때 Body 임계값 (사용 안 함 - 레거시)
    
    ⚠️ **현재 비활성화됨**: Clean Mode에서 임계값 완화 기능 제거됨
       이 설정은 하위 호환성을 위해 남아있지만 실제로는 사용되지 않음
    
    - 이전: Clean Mode에서 0.2로 임계값 완화
    - 현재: Clean Mode는 상태 표시만 하고 임계값은 body_confidence_threshold 유지
    """
    
    # ========== Hand 할루시네이션 방지 ==========
    hand_hallucination_check: bool = True
    """
    손목 없는데 손가락만 있는 할루시네이션 제거
    - Body 모델은 손목(wrist)까지만 있고 손가락(91-133)은 없음
    - 손목이 승인 안 되었는데 손가락이 suspicious 범위에 있으면 제거
    """
    
    # ========== Foot 할루시네이션 방지 ==========
    foot_hallucination_check: bool = True
    """
    발목 Body confidence가 낮을 때 발가락 할루시네이션 제거
    - 발목은 있지만 Body confidence가 낮으면 발가락 신뢰도도 의심
    - 발목 Body가 임계값 미달이면 발가락도 제거
    """
    
    foot_body_confidence_threshold: float = 0.25
    """
    발 할루시네이션 판정용 발목 Body 임계값
    - 일반 body_confidence_threshold와 동일하게 설정
    - 발목 Body가 이 값 미만이면 발가락 제거
    """
    
    foot_dw_min_confidence: float = 4.0
    """
    발가락 DWPose 최소 신뢰도
    - 발가락(17-22)에만 적용되는 더 높은 임계값
    - 일반 dw_suspicious_threshold(2.0)보다 높게 설정
    - 발가락은 할루시네이션이 많아 더 엄격하게 필터링
    """


class CrossFilter:
    """
    ========================================
    교차 필터링 (Cross-Filtering) 엔진
    ========================================
    
    부모-자식 종속 필터링 (Parent-Child Dependency Filtering)을 통해
    DWPose 할루시네이션을 제거하면서 디테일은 보존합니다.
    
    핵심 전략:
        "몸통이 보증하면, 디테일은 DWPose 것을 쓴다"
    
    역할 분리:
        - Body 모델 (감시자): "여기에 팔이 있는가?" (존재 여부만 판단)
          → COCO 17 keypoints (주요 관절만)
          → 안정적이지만 디테일 없음
          
        - DWPose (아티스트): "손가락 마디는 어디에?" (정밀 묘사 담당)
          → COCO-WholeBody 133 keypoints (손가락, 얼굴 포함)
          → 디테일 풍부하지만 할루시네이션 위험
    
    작동 예시:
        [성공 케이스]
        1. Body 모델: "왼손목 있음" (신뢰도 0.8)
           → DWPose의 왼손 21개 손가락 전부 승인
           → 손 디테일 보존
        
        2. Body 모델: "오른발목 있음" (신뢰도 0.6)
           → DWPose의 오른발 3개 발가락 승인
           → 발 디테일 보존
        
        [할루시네이션 제거]
        1. Body 모델: "무릎 없음" (신뢰도 0.2)
           → DWPose 무릎 좌표 무시
           → 환각 제거
        
        2. Body 모델: "왼손목 없음"
           + DWPose: 왼손가락들이 suspicious 범위(0.05~2.0)
           → Hand 할루시네이션 제거
           → 손목 없는데 손가락만 그려지는 현상 방지
    
    COCO-WholeBody 133 구조:
        - Body:       0-16   (17개) - 주요 관절
        - Feet:       17-22  (6개)  - 발가락 좌우 각 3개
        - Face:       23-90  (68개) - 얼굴 랜드마크
        - Left Hand:  91-111 (21개) - 왼손 손가락
        - Right Hand: 112-132(21개) - 오른손 손가락
    """
    
    def __init__(self, config: CrossFilterConfig = None):
        """
        초기화
        
        Args:
            config: 필터링 설정 (None이면 기본값 사용)
        """
        self.config = config or CrossFilterConfig()
        
        # ========== COCO 17 → COCO-WholeBody 133 매핑 ==========
        # Body 모델의 17개 인덱스를 Wholebody 133개 인덱스로 변환
        # Body와 WholeBody의 body 부분은 인덱스가 동일 (0-16)
        self.body_to_wholebody = {
            0: 0,   # Nose (코)
            1: 1,   # Left Eye (왼쪽 눈)
            2: 2,   # Right Eye (오른쪽 눈)
            3: 3,   # Left Ear (왼쪽 귀)
            4: 4,   # Right Ear (오른쪽 귀)
            5: 5,   # Left Shoulder (왼쪽 어깨)
            6: 6,   # Right Shoulder (오른쪽 어깨)
            7: 7,   # Left Elbow (왼쪽 팔꿈치)
            8: 8,   # Right Elbow (오른쪽 팔꿈치)
            9: 9,   # Left Wrist (왼쪽 손목) ← 손 종속의 부모
            10: 10, # Right Wrist (오른쪽 손목) ← 손 종속의 부모
            11: 11, # Left Hip (왼쪽 엉덩이)
            12: 12, # Right Hip (오른쪽 엉덩이)
            13: 13, # Left Knee (왼쪽 무릎)
            14: 14, # Right Knee (오른쪽 무릎)
            15: 15, # Left Ankle (왼쪽 발목) ← 발 종속의 부모
            16: 16  # Right Ankle (오른쪽 발목) ← 발 종속의 부모
        }
        
        # ========== 부모-자식 종속 규칙 정의 ==========
        
        # 손 종속: 손목(부모) → 손가락(자식)
        # Body 모델은 손목까지만 있고 손가락은 없음!
        self.hand_dependencies = {
            'left': (9, range(91, 112)),    # 왼손목(9) → 왼손가락 21개(91-111)
            'right': (10, range(112, 133))  # 오른손목(10) → 오른손가락 21개(112-132)
        }
        
        # 발 종속: 발목(부모) → 발가락(자식)
        self.foot_dependencies = {
            'left': (15, range(17, 20)),   # 왼발목(15) → 왼발가락 3개(17-19)
            'right': (16, range(20, 23))   # 오른발목(16) → 오른발가락 3개(20-22)
        }
        
        # 얼굴 종속: 코(0) 또는 목(가상, 어깨 중점) → 얼굴 랜드마크
        # 목은 가상 키포인트 (왼어깨 + 오른어깨) / 2로 계산
        self.face_indices = range(23, 91)  # 얼굴 68개 랜드마크 (23-90)
    
    def filter(
        self,
        body_keypoints: np.ndarray,
        body_scores: np.ndarray,
        dw_keypoints: np.ndarray,
        dw_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
        """
        ========================================
        교차 필터링 실행 (메인 함수)
        ========================================
        
        Body 모델과 DWPose 결과를 교차 검증하여 할루시네이션을 제거하고
        신뢰할 수 있는 키포인트만 승인합니다.
        
        처리 순서:
            [0-0] Clean Mode 체크
            [0-1] 전신 확신 체크 (Full-Body Bypass)
            [0-2] 개별 고신뢰도 보호 (Individual Bypass)
            [1]   Body 17개 검증
            [2]   손 디테일 (손목 종속)
            [3]   발 디테일 (발목 종속)
            [4]   얼굴 디테일 (코/목 종속)
        
        Args:
            body_keypoints: Body 모델 키포인트 좌표 (17, 2)
                - COCO 17 format [x, y]
                - 주요 관절 17개 (코, 눈, 귀, 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목)
            
            body_scores: Body 모델 신뢰도 (17,)
                - 각 키포인트의 confidence score
                - 0.0 ~ 10.0 범위 (실제로는 보통 0~5)
            
            dw_keypoints: DWPose 키포인트 좌표 (133, 2)
                - COCO-WholeBody format [x, y]
                - Body(17) + Feet(6) + Face(68) + Hands(42)
            
            dw_scores: DWPose 신뢰도 (133,)
                - 각 키포인트의 confidence score
                - 0.0 ~ 10.0 범위
        
        Returns:
            Tuple[np.ndarray, np.ndarray, Set[int]]:
                - filtered_keypoints: (133, 2) 필터링된 좌표
                  → 승인 안 된 키포인트는 (0, 0)
                
                - filtered_scores: (133,) 필터링된 신뢰도
                  → 승인 안 된 키포인트는 0.0
                
                - approved_indices: 승인된 인덱스 집합
                  → {0, 1, 5, 6, 9, 10, ...} 형태
                  → 렌더링 시 이 인덱스만 그림
        
        필터링 로직 상세:
            1. Clean Mode:
               - 의심 키포인트(0.05~2.0) 개수가 0개면 활성화
               - 상태 표시만 하고 임계값은 유지 (완화 기능 제거됨)
            
            2. Full-Body Bypass:
               - DWPose의 17개 body가 모두 >6.0이면
               - Body 검증 생략하고 133개 전부 승인
               - 명확한 전신 이미지에서 과도한 필터링 방지
            
            3. Individual Bypass:
               - DWPose 신뢰도 >8.0이면 무조건 승인
               - Body가 오검출해도 DWPose가 확신하면 보존
            
            4. Body 검증:
               - Body 신뢰도 > threshold면 DWPose 좌표 사용
               - 큰 관절만 검증 (디테일은 종속 규칙으로)
            
            5. 종속 규칙:
               - 손목 O → 손가락 21개 승인
               - 손목 X + 손가락 suspicious → 할루시네이션 제거
               - 발목 O → 발가락 3개 승인
               - 코 or 어깨 O → 얼굴 68개 승인
        """
        # 필터링이 비활성화되어 있으면 원본 그대로 반환
        if not self.config.enabled:
            return dw_keypoints.copy(), dw_scores.copy(), set(range(133))
        
        # ========== 결과 저장용 변수 초기화 ==========
        filtered_keypoints = np.zeros_like(dw_keypoints)  # (133, 2) 전부 0으로 초기화
        filtered_scores = np.zeros_like(dw_scores)        # (133,) 전부 0으로 초기화
        approved_indices: Set[int] = set()                 # 승인된 인덱스 집합
        
        # ========== 설정값 로드 (짧은 변수명으로) ==========
        threshold = self.config.body_confidence_threshold       # Body 임계값 (기본 0.5)
        dw_min = self.config.dw_min_confidence                  # DWPose 최소값 (0.05)
        dw_high = self.config.dw_high_confidence_threshold      # 개별 우회 (8.0)
        dw_full_body = self.config.dw_full_body_confidence_threshold  # 전신 우회 (6.0)
        dw_suspicious = self.config.dw_suspicious_threshold     # 의심 판단 (2.0)
        
        # ============================================
        # [0-0단계] Clean Mode 체크 (의심 키포인트 여부)
        # ============================================
        # 목적: 의심 키포인트가 없는 "깨끗한" 이미지는 임계값을 완화해서
        #       전신이 온전히 있는 경우 더 잘 통과시킴
        #
        # 의심 키포인트란?
        # - dw_min(0.05) < score <= dw_suspicious(2.0) 범위
        # - 너무 낮지도(0.05 이하), 높지도(2.0 초과) 않은 애매한 신뢰도
        # - 할루시네이션은 주로 이 범위에 나타남 (있는 듯 없는 듯)
        #
        # Clean Mode 발동 조건:
        # - 의심 키포인트 개수 == 0
        # - 즉, 모든 키포인트가 "매우 낮음(<0.05)" 또는 "충분히 높음(>2.0)"
        # - 이런 이미지는 할루시네이션 위험이 적음
        
        suspicious_count = sum(1 for score in dw_scores 
                              if dw_min < score <= dw_suspicious)
        
        is_clean_mode = (suspicious_count == 0)
        
        # Clean Mode 상태 표시만 하고, 임계값은 유지 (관대한 승인 비활성화)
        if is_clean_mode:
            print(f"   🟢 [Clean Mode] 의심 키포인트 0개 (Body 임계값 {threshold} 유지)")
        else:
            print(f"   🟡 [Normal Mode] 의심 키포인트 {suspicious_count}개 (Body 임계값 {threshold} 유지)")
        
        # ============================================
        # [0-1단계] DWPose 전신 확신 체크 (완전 우회)
        # ============================================
        # 목적: DWPose가 전신을 모두 높은 신뢰도로 검출했다면
        #       Body 검증 없이 전체 승인 (과도한 필터링 방지)
        #
        # 전신 확신 조건:
        # - DWPose의 Body 17개 키포인트가 모두 > dw_full_body(6.0)
        # - 코, 눈, 귀, 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목 전부 확실
        #
        # 왜 필요한가?
        # - Body 모델이 일부를 낮게 보더라도 DWPose가 전신 확신하면 믿어줌
        # - 예: Body가 손목을 0.4로 봐도, DWPose가 전신 6.0 이상이면 OK
        #
        # 결과:
        # - 조건 만족 시 → 133개 키포인트 전부 승인하고 즉시 return
        # - 조건 불만족 → 다음 단계로 진행
        
        body_17_indices = list(self.body_to_wholebody.values())  # [0, 1, 2, ..., 16]
        body_17_scores = [dw_scores[idx] for idx in body_17_indices]
        
        if all(score > dw_full_body for score in body_17_scores):
            # 전신 모두 확신 → Body 검증 우회, 133개 전부 승인 (단, suspicious 범위는 제외)
            for idx in range(133):
                score = dw_scores[idx]
                # suspicious 범위 제거: 0.05 < score <= 2.0인 키포인트는 제외
                if score > dw_suspicious:  # 2.0 초과만 승인
                    filtered_keypoints[idx] = dw_keypoints[idx]
                    filtered_scores[idx] = dw_scores[idx]
                    approved_indices.add(idx)
            return filtered_keypoints, filtered_scores, approved_indices
        
        # ============================================
        # [0-2단계] DWPose 고신뢰도 보호 (개별 우회)
        # ============================================
        # 목적: DWPose가 특정 키포인트에 대해 매우 확신한다면
        #       Body 검증 없이 무조건 승인 (디테일 보존)
        #
        # 2-tier 개별 우회:
        # 1) DWPose > dw_high(8.0): 초고신뢰도, 무조건 승인
        # 2) DWPose > dw_full_body(6.0): 중고신뢰도, suspicious 제외하고 승인
        #
        # 왜 필요한가?
        # - Body가 오검출하거나 보지 못한 경우에도 DWPose가 확실하면 보존
        # - 예: Body가 발목을 0.252로 봐도, DWPose가 6.244이면 승인
        #
        # 주의:
        # - 전체 우회(0-1)와 달리 개별 키포인트마다 판단
        # - 손가락 일부가 8.0 넘으면 그 손가락만 승인
        
        for idx in range(133):
            score = dw_scores[idx]
            # 초고신뢰도 승인 (8.0 초과)
            if score > dw_high:
                filtered_keypoints[idx] = dw_keypoints[idx]
                filtered_scores[idx] = dw_scores[idx]
                approved_indices.add(idx)
            # 중고신뢰도 승인 (6.0 초과, suspicious 범위 제외)
            elif score > dw_full_body and score > dw_suspicious:
                filtered_keypoints[idx] = dw_keypoints[idx]
                filtered_scores[idx] = dw_scores[idx]
                approved_indices.add(idx)
        
        # ============================================
        # [1단계] 큰 뼈대 검증 (Body 17개)
        # ============================================
        # 목적: Body 모델로 주요 관절의 존재 여부 검증
        #
        # 처리 과정:
        # 1. Body 17개를 순회
        # 2. Body 신뢰도 > threshold면 "진짜 있다"고 판단
        # 3. DWPose의 해당 키포인트를 승인
        #
        # 왜 DWPose 좌표를 쓰는가?
        # - Body는 "있다/없다"만 판단 (감시자)
        # - DWPose는 "정확히 어디"를 알려줌 (아티스트)
        # - Body가 보증하면 DWPose의 정밀 좌표 사용
        #
        # Safety check:
        # - Body가 승인해도 DWPose 자체 신뢰도가 너무 낮으면(< 0.05) 제외
        
        for body_idx, wb_idx in self.body_to_wholebody.items():
            body_conf = body_scores[body_idx]  # Body 모델 신뢰도
            
            # 이미 0-2단계에서 고신뢰도로 승인되었으면 스킵
            if wb_idx in approved_indices:
                continue
            
            # Body 모델이 "있다"고 확신하면
            if body_conf > threshold:
                # DWPose의 정밀한 좌표를 사용 (디테일 보존)
                dw_conf = dw_scores[wb_idx]
                
                # Safety check: suspicious 범위(0.05~2.0) 제거
                if dw_conf > dw_suspicious:
                    filtered_keypoints[wb_idx] = dw_keypoints[wb_idx]
                    filtered_scores[wb_idx] = dw_scores[wb_idx]
                    approved_indices.add(wb_idx)
        
        # ============================================
        # [2단계] 손 디테일 - 손목 종속성
        # ============================================
        # 목적: Body 모델은 손목까지만 있고 손가락은 없음
        #       손목이 진짜면 DWPose 손가락도 진짜로 간주
        #
        # 손 할루시네이션 문제:
        # - Body 모델에는 손목(wrist)까지만 있고 손가락(91-133)은 없음
        # - 손목이 없는데 DWPose가 손가락을 그리면 할루시네이션
        # - 예: 반신 사진에서 보이지 않는 손의 손가락이 화면 밖에 그려짐
        #
        # 처리 로직:
        # 1. 손목 승인 O → 손가락 21개 전부 승인
        #    - 손목이 진짜면 손가락도 진짜
        #    - DWPose 손가락 디테일 보존
        #
        # 2. 손목 승인 X + 손가락 suspicious → 할루시네이션 제거
        #    - 손목 없는데 손가락만 의심 구간(0.05~2.0)에 있으면
        #    - 할루시네이션으로 판단하고 제거
        #    - score를 명시적으로 0으로 설정
        #
        # 주의:
        # - 손목 없어도 손가락이 고신뢰도(>8.0)면 0-2단계에서 이미 승인됨
        # - 따라서 여기서는 suspicious 범위만 체크하면 됨
        
        if self.config.enable_hand_dependency:
            for side, (wrist_idx, finger_range) in self.hand_dependencies.items():
                # ===== 케이스 1: 손목 승인 O → 손가락 승인 =====
                if wrist_idx in approved_indices:
                    # 손목이 진짜라면, DWPose 손가락 승인 (단, suspicious 범위는 제외)
                    for finger_idx in finger_range:
                        # 이미 0-2단계에서 고신뢰도로 승인되었으면 스킵
                        if finger_idx in approved_indices:
                            continue
                        score = dw_scores[finger_idx]
                        # suspicious 범위 제거: 2.0 초과만 승인
                        if score > dw_suspicious:
                            filtered_keypoints[finger_idx] = dw_keypoints[finger_idx]
                            filtered_scores[finger_idx] = dw_scores[finger_idx]
                            approved_indices.add(finger_idx)
                
                # ===== 케이스 2: 손목 승인 X + 손가락 suspicious → 할루시네이션 제거 =====
                elif self.config.hand_hallucination_check:
                    # 손목이 없는데 손가락들이 의심 범위(suspicious)에 있으면 할루시네이션
                    for finger_idx in finger_range:
                        if dw_min < dw_scores[finger_idx] <= dw_suspicious:
                            # 이미 고신뢰도로 승인되지 않았으면 제거 (approved_indices에 추가 안 함)
                            if finger_idx not in approved_indices:
                                filtered_scores[finger_idx] = 0  # 명시적으로 0으로 설정
                                # filtered_keypoints는 이미 0으로 초기화되어 있음
        
        # ============================================
        # [3단계] 발 디테일 - 발목 종속성
        # ============================================
        # 목적: 발목이 진짜면 DWPose 발가락도 진짜로 간주
        #
        # 처리 로직:
        # 1. 발목 승인 O + 발목 Body 높음 → 발가락 승인
        # 2. 발목 승인 O + 발목 Body 낮음 → 발가락 제거 (할루시네이션 의심)
        # 3. 발목 승인 X → 발가락 제거
        #
        # 발 할루시네이션 방지:
        # - 발목 DWPose는 높지만 Body가 낮으면 (< foot_body_threshold)
        # - 발가락도 할루시네이션일 가능성 높음
        # - 예: 6.jpg의 발목 Body=0.303/0.437, 발가락들이 2.3~4.3 범위
        
        if self.config.enable_foot_dependency:
            for side, (ankle_idx, toe_range) in self.foot_dependencies.items():
                # 발목이 승인되었는가?
                if ankle_idx in approved_indices:
                    # 발 할루시네이션 체크: 발목 Body confidence 확인
                    ankle_body_idx = 15 if side == "left" else 16  # Body 인덱스
                    ankle_body_conf = body_scores[ankle_body_idx]
                    
                    # 발목 Body가 낮으면 발가락 제거 (할루시네이션 의심)
                    if self.config.foot_hallucination_check and ankle_body_conf < self.config.foot_body_confidence_threshold:
                        # 발목 Body가 낮으면 발가락을 승인하지 않음
                        continue
                    
                    # 발목이 진짜라면, DWPose 발가락 승인 (발가락은 더 높은 임계값)
                    for toe_idx in toe_range:
                        # 이미 고신뢰도로 승인되었으면 스킵
                        if toe_idx in approved_indices:
                            continue
                        score = dw_scores[toe_idx]
                        # 발가락 전용 임계값 사용 (4.0 초과만 승인)
                        if score > self.config.foot_dw_min_confidence:
                            filtered_keypoints[toe_idx] = dw_keypoints[toe_idx]
                            filtered_scores[toe_idx] = dw_scores[toe_idx]
                            approved_indices.add(toe_idx)
        
        # ============================================
        # [4단계] 얼굴 디테일 - 코/목 종속성
        # ============================================
        # 목적: 코 또는 어깨가 진짜면 DWPose 얼굴 랜드마크도 진짜로 간주
        #
        # 얼굴 종속 부모:
        # 1. 코(0) - 직접 검출
        # 2. 목(가상) - 왼어깨(5) + 오른어깨(6) 중점
        #    - COCO-WholeBody에는 목 키포인트가 없음
        #    - 양쪽 어깨가 있으면 목도 있다고 간주
        #
        # 처리 로직:
        # - (코 O) OR (양쪽 어깨 O) → 얼굴 68개 승인
        # - 둘 다 X → 얼굴 제거
        #
        # 왜 목을 가상으로?
        # - 측면 사진에서는 코가 안 보일 수 있음
        # - 하지만 어깨는 보이면 얼굴도 있다고 볼 수 있음
        # - 백업 조건으로 어깨 체크
        
        if self.config.enable_face_dependency:
            # 코(0)가 있는가?
            nose_approved = 0 in approved_indices
            
            # 목(가상) = 어깨 중점이 유효한가?
            # 왼어깨(5), 오른어깨(6)가 둘 다 있으면 목도 있다고 간주
            neck_approved = (5 in approved_indices and 6 in approved_indices)
            
            # 코 또는 목 중 하나라도 있으면 얼굴 전체 승인 (suspicious 범위는 제외)
            if nose_approved or neck_approved:
                for face_idx in self.face_indices:  # 23-90 (68개)
                    # 이미 고신뢰도로 승인되었으면 스킵
                    if face_idx in approved_indices:
                        continue
                    score = dw_scores[face_idx]
                    # suspicious 범위 제거: 2.0 초과만 승인
                    if score > dw_suspicious:
                        filtered_keypoints[face_idx] = dw_keypoints[face_idx]
                        filtered_scores[face_idx] = dw_scores[face_idx]
                        approved_indices.add(face_idx)
        
        # ========== 필터링 완료, 결과 반환 ==========
        return filtered_keypoints, filtered_scores, approved_indices
    
    def get_statistics(self, approved_indices: Set[int]) -> Dict[str, int]:
        """
        필터링 통계 계산
        
        승인된 키포인트를 영역별로 집계하여 반환합니다.
        디버깅 및 로깅에 유용합니다.
        
        Args:
            approved_indices: 승인된 키포인트 인덱스 집합
                예: {0, 1, 5, 6, 9, 10, 23, 24, ...}
        
        Returns:
            Dict[str, int]: 영역별 승인 개수
                {
                    'body': 17개 중 승인된 개수 (0-16),
                    'face': 68개 중 승인된 개수 (23-90),
                    'left_hand': 21개 중 승인된 개수 (91-111),
                    'right_hand': 21개 중 승인된 개수 (112-132),
                    'left_foot': 3개 중 승인된 개수 (17-19),
                    'right_foot': 3개 중 승인된 개수 (20-22),
                    'total': 133개 중 전체 승인된 개수
                }
        
        사용 예시:
            stats = cross_filter.get_statistics(approved_indices)
            print(f"Body: {stats['body']}/17")
            print(f"Face: {stats['face']}/68")
            print(f"Total: {stats['total']}/133")
        """
        # 각 영역별로 승인된 키포인트 개수 계산
        body_count = sum(1 for i in range(17) if i in approved_indices)
        face_count = sum(1 for i in range(23, 91) if i in approved_indices)
        lhand_count = sum(1 for i in range(91, 112) if i in approved_indices)
        rhand_count = sum(1 for i in range(112, 133) if i in approved_indices)
        lfoot_count = sum(1 for i in range(17, 20) if i in approved_indices)
        rfoot_count = sum(1 for i in range(20, 23) if i in approved_indices)
        
        return {
            'body': body_count,           # Body 17개 중 승인 (0-16)
            'face': face_count,           # Face 68개 중 승인 (23-90)
            'left_hand': lhand_count,     # Left Hand 21개 중 승인 (91-111)
            'right_hand': rhand_count,    # Right Hand 21개 중 승인 (112-132)
            'left_foot': lfoot_count,     # Left Foot 3개 중 승인 (17-19)
            'right_foot': rfoot_count,    # Right Foot 3개 중 승인 (20-22)
            'total': len(approved_indices) # 전체 승인 (최대 133)
        }
