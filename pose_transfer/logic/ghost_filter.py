"""
Ghost Keypoint Filter v2.1 (Syntax Fix)
- GhostFilterConfig의 문법 오류(TypeError)를 수정했습니다.
"""
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

# 인덱스 정의
LOWER_BODY_INDICES = [11, 12, 13, 14, 15, 16] # Hips, Knees, Ankles
FEET_INDICES = [17, 18, 19, 20, 21, 22]
LEFT_HAND_INDICES = list(range(91, 112))
RIGHT_HAND_INDICES = list(range(112, 133))

@dataclass
class GhostFilterConfig:
    enabled: bool = True
    ghost_score_threshold: float = 0.5
    check_anatomy_order: bool = False
    check_image_bounds: bool = True
    bounds_margin: float = 0.05
    wrist_score_threshold: float = 0.3
    elbow_score_threshold: float = 0.3  # [수정] 문법 오류 수정 (: 0.3 -> : float = 0.3)

class GhostFilter:
    def __init__(self, config: Optional[GhostFilterConfig] = None):
        self.config = config or GhostFilterConfig()

    def filter(self, kpts: np.ndarray, scores: np.ndarray, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        필터링 메인 함수
        """
        # 필터가 꺼져있으면 원본 점수 그대로 반환
        if not self.config.enabled:
            return scores

        filtered_scores = scores.copy()

        # [1] 하반신 필터링 (다리 꼬임/Ghost Leg)
        if self._should_remove_lower_body(kpts, scores):
            filtered_scores = self._zero_indices(filtered_scores, LOWER_BODY_INDICES + FEET_INDICES)
            print("   🦵 [GhostFilter] Lower body removed based on config.")

        # [2] 손 필터링 (손 꼬임)
        # 왼손
        if self._should_remove_hand(scores, side='left'):
            filtered_scores = self._zero_indices(filtered_scores, LEFT_HAND_INDICES)
        # 오른손
        if self._should_remove_hand(scores, side='right'):
            filtered_scores = self._zero_indices(filtered_scores, RIGHT_HAND_INDICES)

        # [3] 화면 밖 키포인트 제거
        if image_size and self.config.check_image_bounds:
            filtered_scores = self._filter_out_of_bounds(kpts, filtered_scores, image_size)

        return filtered_scores

    def _should_remove_lower_body(self, kpts, scores) -> bool:
        """하반신 제거 여부 판단"""
        # 1. 점수 기반 체크 (사용자 설정 threshold)
        # 무릎이나 발목 중 하나라도 설정값보다 높은 게 있으면 -> 유효하다고 판단 (지우지 않음)
        max_leg_score = max(
            scores[13], scores[14], # Knees
            scores[15], scores[16]  # Ankles
        )
        
        # 모든 다리 관절 점수가 설정값 미만이면 -> 노이즈로 보고 제거
        if max_leg_score < self.config.ghost_score_threshold:
            return True

        # 2. 해부학적 순서 체크 (사용자가 켰을 때만 작동)
        if self.config.check_anatomy_order:
            l_hip_y, r_hip_y = kpts[11][1], kpts[12][1]
            l_knee_y, r_knee_y = kpts[13][1], kpts[14][1]
            
            # 무릎이 골반보다 위에 있으면(Y값이 작으면) 제거
            # (앉은 자세에서는 끄세요)
            if (scores[13] > 0.1 and l_knee_y < l_hip_y) or \
               (scores[14] > 0.1 and r_knee_y < r_hip_y):
                return True

        return False

    def _should_remove_hand(self, scores, side='left') -> bool:
        """손 제거 여부 판단"""
        if side == 'left':
            wrist_idx, elbow_idx = 9, 7
        else:
            wrist_idx, elbow_idx = 10, 8
            
        wrist_score = scores[wrist_idx]
        elbow_score = scores[elbow_idx]
        
        # 손목과 팔꿈치 점수가 모두 설정값 미만이면 손 제거
        # (손만 둥둥 떠있는 꼬임 방지)
        if wrist_score < self.config.wrist_score_threshold and \
           elbow_score < self.config.elbow_score_threshold:
            return True
            
        return False

    def _filter_out_of_bounds(self, kpts, scores, image_size):
        h, w = image_size
        margin = self.config.bounds_margin
        x_min, x_max = -w * margin, w * (1 + margin)
        y_min, y_max = -h * margin, h * (1 + margin)
        
        new_scores = scores.copy()
        for i in range(len(scores)):
            if scores[i] > 0:
                x, y = kpts[i]
                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    new_scores[i] = 0.0
        return new_scores

    def _zero_indices(self, scores, indices):
        for idx in indices:
            if idx < len(scores):
                scores[idx] = 0.0
        return scores

# 편의 함수
def filter_ghost_keypoints(kpts, scores, image_size=None, config=None):
    if config is None:
        # Config가 없으면 기본값 사용 (모두 허용하는 방향)
        config = GhostFilterConfig(enabled=False)
    
    filter_ = GhostFilter(config)
    return filter_.filter(kpts, scores, image_size)