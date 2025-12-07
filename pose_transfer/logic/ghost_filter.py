"""
Ghost Keypoint Filter v1.2
범용 Ghost 키포인트 필터링 - 손, 발, 하반신 등 모든 부위 처리

v1.2 변경사항:
- _is_hand_scattered 조건 완화: 손목이 유효하면 무조건 손 유지
- 손 키포인트 분산 체크 제거 (DWPose 손 추출이 불안정하므로)
- 핵심 원칙: 팔(어깨-팔꿈치-손목)이 보이면 손도 유지
"""
import numpy as np
from typing import Tuple, Set, List, Optional
from dataclasses import dataclass

# =============================================================================
# 키포인트 인덱스 정의
# =============================================================================

BODY_INDICES = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}

LOWER_BODY_INDICES = [11, 12, 13, 14, 15, 16]
FEET_INDICES = [17, 18, 19, 20, 21, 22]

LEFT_HAND_START = 91
LEFT_HAND_END = 112
RIGHT_HAND_START = 112
RIGHT_HAND_END = 133

FACE_START = 23
FACE_END = 91

# =============================================================================
# Ghost Filter Config
# =============================================================================

@dataclass
class GhostFilterConfig:
    """Ghost 필터링 설정"""
    body_score_threshold: float = 2.0
    hand_score_threshold: float = 1.0
    face_score_threshold: float = 3.0
    
    # 손 필터링 설정 (v1.2: 더 단순화)
    wrist_score_threshold: float = 2.0
    elbow_score_threshold: float = 2.0
    
    # 이미지 범위 검증
    check_image_bounds: bool = True
    bounds_margin: float = 0.05
    
    # 하반신 Ghost Leg 검증
    ghost_score_threshold: float = 2.0
    check_anatomy_order: bool = True


# =============================================================================
# Ghost Filter 클래스
# =============================================================================

class GhostFilter:
    """범용 Ghost 키포인트 필터 v1.2"""
    
    def __init__(self, config: Optional[GhostFilterConfig] = None):
        self.config = config or GhostFilterConfig()
    
    def filter(
        self, 
        kpts: np.ndarray, 
        scores: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Ghost 키포인트 필터링
        """
        filtered_scores = scores.copy()
        
        # 1. 하반신 Ghost Leg 필터링
        if self._is_ghost_lower_body(kpts, scores):
            filtered_scores = self._zero_indices(filtered_scores, LOWER_BODY_INDICES + FEET_INDICES)
            print("   🦵 [Ghost Filter] Lower body filtered (Ghost Leg detected)")
        
        # 2. 왼손 Ghost 필터링 (v1.2: 단순화)
        if self._is_ghost_hand_v3(kpts, scores, 'left', image_size):
            hand_indices = list(range(LEFT_HAND_START, LEFT_HAND_END))
            filtered_scores = self._zero_indices(filtered_scores, hand_indices)
            print("   🖐️ [Ghost Filter] Left hand filtered")
        
        # 3. 오른손 Ghost 필터링 (v1.2: 단순화)
        if self._is_ghost_hand_v3(kpts, scores, 'right', image_size):
            hand_indices = list(range(RIGHT_HAND_START, RIGHT_HAND_END))
            filtered_scores = self._zero_indices(filtered_scores, hand_indices)
            print("   🖐️ [Ghost Filter] Right hand filtered")
        
        # 4. 개별 키포인트 범위 체크
        if image_size and self.config.check_image_bounds:
            filtered_scores = self._filter_out_of_bounds(kpts, filtered_scores, image_size)
        
        return filtered_scores
    
    def _is_ghost_lower_body(self, kpts: np.ndarray, scores: np.ndarray) -> bool:
        """Ghost Leg 판별"""
        ghost_threshold = self.config.ghost_score_threshold
        
        def get_y(idx):
            return kpts[idx][1] if scores[idx] > 0.1 else None
        
        l_hip_y = get_y(11)
        r_hip_y = get_y(12)
        l_knee_y = get_y(13)
        r_knee_y = get_y(14)
        l_ankle_y = get_y(15)
        r_ankle_y = get_y(16)
        
        def check_leg_order(hip_y, knee_y, ankle_y):
            if hip_y is None or knee_y is None:
                return False
            hip_knee_ok = hip_y < knee_y
            if ankle_y is not None:
                return hip_knee_ok and (knee_y < ankle_y)
            return hip_knee_ok
        
        left_order_ok = check_leg_order(l_hip_y, l_knee_y, l_ankle_y)
        right_order_ok = check_leg_order(r_hip_y, r_knee_y, r_ankle_y)
        
        if not (left_order_ok or right_order_ok):
            return True
        
        max_knee = max(scores[13], scores[14])
        max_ankle = max(scores[15], scores[16])
        
        if max_knee < ghost_threshold and max_ankle < ghost_threshold:
            return True
        
        return False
    
    def _is_ghost_hand_v3(
        self, 
        kpts: np.ndarray, 
        scores: np.ndarray, 
        side: str,
        image_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Ghost Hand 판별 v1.2 (매우 단순화)
        
        핵심 원칙: 
        - 손목 점수가 유효하면 → 손 유지! (무조건)
        - 손목 점수가 낮아도 팔꿈치가 유효하면 → 손 유지!
        - 둘 다 낮으면 → 팔이 안 보이므로 손 필터링
        
        _is_hand_scattered 체크 제거! (DWPose 손 추출이 불안정)
        """
        if side == 'left':
            wrist_idx = 9
            elbow_idx = 7
            shoulder_idx = 5
        else:
            wrist_idx = 10
            elbow_idx = 8
            shoulder_idx = 6
        
        wrist_score = scores[wrist_idx]
        elbow_score = scores[elbow_idx]
        shoulder_score = scores[shoulder_idx]
        wrist_pos = kpts[wrist_idx]
        
        # ===================================================================
        # [조건 1] 손목이 완전히 이미지 밖이고 팔꿈치도 없으면 Ghost
        # ===================================================================
        if image_size:
            h, w = image_size
            if wrist_pos[0] < 0 or wrist_pos[0] > w or wrist_pos[1] < 0 or wrist_pos[1] > h:
                if elbow_score < self.config.elbow_score_threshold:
                    return True  # 손목 범위 밖 + 팔꿈치 없음 → Ghost
        
        # ===================================================================
        # [조건 2] 손목 유효하면 손 유지! (v1.2 핵심 - 분산 체크 제거)
        # ===================================================================
        if wrist_score >= self.config.wrist_score_threshold:
            return False  # 손목 유효 → 무조건 손 유지!
        
        # ===================================================================
        # [조건 3] 팔꿈치 유효하면 손 유지!
        # ===================================================================
        if elbow_score >= self.config.elbow_score_threshold:
            return False  # 팔꿈치 유효 → 손 유지!
        
        # ===================================================================
        # [조건 4] 어깨라도 유효하면 손 유지! (팔이 접혀있을 수 있음)
        # ===================================================================
        if shoulder_score >= self.config.body_score_threshold:
            return False  # 어깨 유효 → 손 유지!
        
        # 어깨, 팔꿈치, 손목 모두 점수 낮음 → 팔 전체가 안 보임 → Ghost
        return True
    
    def _filter_out_of_bounds(
        self, 
        kpts: np.ndarray, 
        scores: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """이미지 범위 밖 키포인트 필터링 (Body만)"""
        h, w = image_size
        margin = self.config.bounds_margin
        
        x_min, x_max = -w * margin, w * (1 + margin)
        y_min, y_max = -h * margin, h * (1 + margin)
        
        filtered_scores = scores.copy()
        
        for i in range(23):
            if scores[i] > 0:
                x, y = kpts[i]
                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    filtered_scores[i] = 0.0
        
        return filtered_scores
    
    def _zero_indices(self, scores: np.ndarray, indices: List[int]) -> np.ndarray:
        """특정 인덱스들의 점수를 0으로 설정"""
        for idx in indices:
            if idx < len(scores):
                scores[idx] = 0.0
        return scores


# =============================================================================
# 편의 함수
# =============================================================================

def filter_ghost_keypoints(
    kpts: np.ndarray, 
    scores: np.ndarray,
    image_size: Optional[Tuple[int, int]] = None,
    config: Optional[GhostFilterConfig] = None
) -> np.ndarray:
    """Ghost 키포인트 필터링 (편의 함수)"""
    ghost_filter = GhostFilter(config)
    return ghost_filter.filter(kpts, scores, image_size)