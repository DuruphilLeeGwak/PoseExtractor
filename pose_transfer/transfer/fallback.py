"""
누락 키포인트 폴백 시스템

오클루전, 프레임 이탈 등으로 누락된 키포인트를 추정합니다.
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from ..extractors.keypoint_constants import (
    BODY_KEYPOINTS,
    FEET_KEYPOINTS,
    SYMMETRIC_BODY_PAIRS,
    SYMMETRIC_FEET_PAIRS,
    BODY_HIERARCHY,
    LEFT_HAND_START_IDX,
    RIGHT_HAND_START_IDX,
    get_keypoint_index,
    get_symmetric_pair
)
from ..utils.geometry import mirror_point_horizontal, interpolate_point


@dataclass
class FallbackResult:
    """폴백 적용 결과"""
    keypoints: np.ndarray
    scores: np.ndarray
    fallback_applied: Dict[int, str]  # 키포인트 인덱스: 적용된 폴백 방법


class FallbackStrategy:
    """
    누락 키포인트 복구 전략
    
    우선순위:
    1. 대칭 미러링 (반대쪽이 유효한 경우)
    2. 계층적 추정 (부모 키포인트 기반)
    3. 평균 비율 추정
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        enable_symmetric_mirror: bool = True,
        enable_hierarchical: bool = True,
        fallback_confidence_decay: float = 0.5
    ):
        """
        Args:
            confidence_threshold: 유효 키포인트 판단 임계값
            enable_symmetric_mirror: 대칭 미러링 활성화
            enable_hierarchical: 계층적 추정 활성화
            fallback_confidence_decay: 폴백 적용 시 신뢰도 감소율
        """
        self.confidence_threshold = confidence_threshold
        self.enable_symmetric_mirror = enable_symmetric_mirror
        self.enable_hierarchical = enable_hierarchical
        self.fallback_confidence_decay = fallback_confidence_decay
        
        # 대칭 쌍 인덱스 매핑
        self._init_symmetric_pairs()
        
        # 평균 본 비율 (통계적 데이터)
        self._init_average_ratios()
    
    def _init_symmetric_pairs(self):
        """대칭 쌍 인덱스 매핑 초기화"""
        self.symmetric_pairs = {}
        
        for left_name, right_name in SYMMETRIC_BODY_PAIRS:
            left_idx = get_keypoint_index(left_name)
            right_idx = get_keypoint_index(right_name)
            self.symmetric_pairs[left_idx] = right_idx
            self.symmetric_pairs[right_idx] = left_idx
        
        for left_name, right_name in SYMMETRIC_FEET_PAIRS:
            left_idx = get_keypoint_index(left_name)
            right_idx = get_keypoint_index(right_name)
            self.symmetric_pairs[left_idx] = right_idx
            self.symmetric_pairs[right_idx] = left_idx
        
        # 손 대칭 쌍
        for i in range(21):
            left_idx = LEFT_HAND_START_IDX + i
            right_idx = RIGHT_HAND_START_IDX + i
            self.symmetric_pairs[left_idx] = right_idx
            self.symmetric_pairs[right_idx] = left_idx
    
    def _init_average_ratios(self):
        """평균 본 비율 초기화 (인체 비율 통계)"""
        # 어깨 너비 = 1.0 기준
        self.avg_ratios = {
            # 상체
            'shoulder_to_elbow': 0.75,  # 상완 길이
            'elbow_to_wrist': 0.70,     # 하완 길이
            'shoulder_to_hip': 1.2,     # 몸통 길이
            
            # 하체
            'hip_to_knee': 1.1,         # 대퇴 길이
            'knee_to_ankle': 1.0,       # 하퇴 길이
            
            # 머리
            'shoulder_to_ear': 0.4,
            'ear_to_eye': 0.15,
            'eye_to_nose': 0.1,
        }
    
    def apply_fallback(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        image_size: Tuple[int, int]
    ) -> FallbackResult:
        """
        누락 키포인트에 폴백 적용
        
        Args:
            keypoints: (K, 2) 키포인트 좌표
            scores: (K,) 키포인트 신뢰도
            image_size: (height, width) 이미지 크기
        
        Returns:
            FallbackResult: 폴백 적용 결과
        """
        result_kpts = keypoints.copy()
        result_scores = scores.copy()
        fallback_applied = {}
        
        # 이미지 중심선 (대칭 미러링용)
        img_h, img_w = image_size
        center_x = img_w / 2
        
        # 몸 중심선 계산 (어깨 중심)
        left_shoulder_idx = BODY_KEYPOINTS['left_shoulder']
        right_shoulder_idx = BODY_KEYPOINTS['right_shoulder']
        
        if (scores[left_shoulder_idx] > self.confidence_threshold and
            scores[right_shoulder_idx] > self.confidence_threshold):
            body_center_x = (
                keypoints[left_shoulder_idx][0] + 
                keypoints[right_shoulder_idx][0]
            ) / 2
        else:
            body_center_x = center_x
        
        # 1. Body 키포인트 폴백
        for name, idx in BODY_KEYPOINTS.items():
            if scores[idx] > self.confidence_threshold:
                continue  # 유효한 키포인트는 스킵
            
            # 대칭 미러링 시도
            if self.enable_symmetric_mirror and idx in self.symmetric_pairs:
                mirror_idx = self.symmetric_pairs[idx]
                if scores[mirror_idx] > self.confidence_threshold:
                    result_kpts[idx] = mirror_point_horizontal(
                        keypoints[mirror_idx], body_center_x
                    )
                    result_scores[idx] = (
                        scores[mirror_idx] * self.fallback_confidence_decay
                    )
                    fallback_applied[idx] = 'symmetric_mirror'
                    continue
            
            # 계층적 추정 시도
            if self.enable_hierarchical:
                estimated, method = self._hierarchical_estimate(
                    name, result_kpts, result_scores
                )
                if estimated is not None:
                    result_kpts[idx] = estimated
                    result_scores[idx] = 0.3 * self.fallback_confidence_decay
                    fallback_applied[idx] = method
                    continue
        
        # 2. Feet 키포인트 폴백
        for name, idx in FEET_KEYPOINTS.items():
            if scores[idx] > self.confidence_threshold:
                continue
            
            if self.enable_symmetric_mirror and idx in self.symmetric_pairs:
                mirror_idx = self.symmetric_pairs[idx]
                if scores[mirror_idx] > self.confidence_threshold:
                    result_kpts[idx] = mirror_point_horizontal(
                        keypoints[mirror_idx], body_center_x
                    )
                    result_scores[idx] = (
                        scores[mirror_idx] * self.fallback_confidence_decay
                    )
                    fallback_applied[idx] = 'symmetric_mirror'
        
        # 3. 손 키포인트 폴백
        self._apply_hand_fallback(
            result_kpts, result_scores, fallback_applied,
            body_center_x, is_left=True
        )
        self._apply_hand_fallback(
            result_kpts, result_scores, fallback_applied,
            body_center_x, is_left=False
        )
        
        return FallbackResult(
            keypoints=result_kpts,
            scores=result_scores,
            fallback_applied=fallback_applied
        )
    
    def _hierarchical_estimate(
        self,
        keypoint_name: str,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str]:
        """계층적 추정"""
        # 부모-자식 관계에서 추정
        for parent_name, children in BODY_HIERARCHY.items():
            if keypoint_name in children:
                if parent_name == 'root':
                    # 루트 기반 추정
                    return self._estimate_from_root(
                        keypoint_name, keypoints, scores
                    )
                else:
                    parent_idx = get_keypoint_index(parent_name)
                    if scores[parent_idx] > self.confidence_threshold:
                        return self._estimate_from_parent(
                            parent_name, keypoint_name,
                            keypoints, scores
                        )
        
        return None, ''
    
    def _estimate_from_root(
        self,
        keypoint_name: str,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str]:
        """루트에서 추정"""
        left_hip_idx = BODY_KEYPOINTS['left_hip']
        right_hip_idx = BODY_KEYPOINTS['right_hip']
        
        if (scores[left_hip_idx] > self.confidence_threshold and
            scores[right_hip_idx] > self.confidence_threshold):
            
            root = (keypoints[left_hip_idx] + keypoints[right_hip_idx]) / 2
            hip_width = np.linalg.norm(
                keypoints[left_hip_idx] - keypoints[right_hip_idx]
            )
            
            if keypoint_name == 'left_hip':
                return keypoints[left_hip_idx], 'existing'
            elif keypoint_name == 'right_hip':
                return keypoints[right_hip_idx], 'existing'
        
        return None, ''
    
    def _estimate_from_parent(
        self,
        parent_name: str,
        child_name: str,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str]:
        """부모 키포인트에서 추정"""
        parent_idx = get_keypoint_index(parent_name)
        parent_pos = keypoints[parent_idx]
        
        # 어깨 너비를 기준으로 스케일 계산
        left_shoulder_idx = BODY_KEYPOINTS['left_shoulder']
        right_shoulder_idx = BODY_KEYPOINTS['right_shoulder']
        
        if (scores[left_shoulder_idx] > self.confidence_threshold and
            scores[right_shoulder_idx] > self.confidence_threshold):
            shoulder_width = np.linalg.norm(
                keypoints[left_shoulder_idx] - keypoints[right_shoulder_idx]
            )
        else:
            # 기본값
            shoulder_width = 100
        
        # 관계에 따른 오프셋 계산
        offset = self._get_estimated_offset(
            parent_name, child_name, shoulder_width
        )
        
        if offset is not None:
            return parent_pos + offset, 'hierarchical'
        
        return None, ''
    
    def _get_estimated_offset(
        self,
        parent_name: str,
        child_name: str,
        shoulder_width: float
    ) -> Optional[np.ndarray]:
        """추정 오프셋 계산"""
        # 부모-자식 관계별 기본 오프셋 (아래 방향 기준)
        offset_map = {
            ('left_shoulder', 'left_elbow'): np.array([0, 0.75]),
            ('left_elbow', 'left_wrist'): np.array([0, 0.70]),
            ('right_shoulder', 'right_elbow'): np.array([0, 0.75]),
            ('right_elbow', 'right_wrist'): np.array([0, 0.70]),
            ('left_hip', 'left_knee'): np.array([0, 1.1]),
            ('left_knee', 'left_ankle'): np.array([0, 1.0]),
            ('right_hip', 'right_knee'): np.array([0, 1.1]),
            ('right_knee', 'right_ankle'): np.array([0, 1.0]),
            ('left_shoulder', 'left_ear'): np.array([0, -0.4]),
            ('right_shoulder', 'right_ear'): np.array([0, -0.4]),
            ('left_ear', 'left_eye'): np.array([0.1, -0.1]),
            ('right_ear', 'right_eye'): np.array([-0.1, -0.1]),
            ('left_eye', 'nose'): np.array([0.05, 0.1]),
        }
        
        key = (parent_name, child_name)
        if key in offset_map:
            return offset_map[key] * shoulder_width
        
        return None
    
    def _apply_hand_fallback(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        fallback_applied: Dict[int, str],
        body_center_x: float,
        is_left: bool
    ):
        """손 키포인트 폴백"""
        hand_start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
        other_hand_start = RIGHT_HAND_START_IDX if is_left else LEFT_HAND_START_IDX
        
        for i in range(21):
            idx = hand_start + i
            
            if scores[idx] > self.confidence_threshold:
                continue
            
            # 반대 손에서 미러링
            if self.enable_symmetric_mirror:
                other_idx = other_hand_start + i
                if scores[other_idx] > self.confidence_threshold:
                    keypoints[idx] = mirror_point_horizontal(
                        keypoints[other_idx], body_center_x
                    )
                    scores[idx] = scores[other_idx] * self.fallback_confidence_decay
                    fallback_applied[idx] = 'hand_symmetric_mirror'


def apply_fallback(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_size: Tuple[int, int],
    confidence_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    편의 함수: 폴백 적용
    
    Returns:
        keypoints: 폴백 적용된 키포인트
        scores: 폴백 적용된 신뢰도
        fallback_applied: 적용된 폴백 방법
    """
    strategy = FallbackStrategy(confidence_threshold=confidence_threshold)
    result = strategy.apply_fallback(keypoints, scores, image_size)
    
    return result.keypoints, result.scores, result.fallback_applied
