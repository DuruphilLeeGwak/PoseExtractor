"""
본(Bone) 길이 계산 모듈

원본 이미지에서 신체 비율(각 본의 길이)을 추출합니다.
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from ..extractors.keypoint_constants import (
    BODY_KEYPOINTS,
    FEET_KEYPOINTS,
    BODY_BONES,
    FEET_BONES,
    HAND_BONES,
    LEFT_HAND_START_IDX,
    RIGHT_HAND_START_IDX,
    get_keypoint_index
)
from ..utils.geometry import calculate_distance


@dataclass
class BoneInfo:
    """본 정보"""
    name: str
    start_idx: int
    end_idx: int
    length: float
    confidence: float  # 두 키포인트 신뢰도의 최소값
    is_valid: bool


@dataclass
class BodyProportions:
    """신체 비율 정보"""
    bone_lengths: Dict[str, BoneInfo] = field(default_factory=dict)
    
    # 정규화 기준값
    shoulder_width: float = 0.0
    torso_length: float = 0.0  # 어깨 중심 ~ 엉덩이 중심
    
    # 정규화된 비율 (기준값 = 1.0)
    normalized_lengths: Dict[str, float] = field(default_factory=dict)
    
    def get_bone_length(self, bone_name: str) -> float:
        """본 길이 반환"""
        if bone_name in self.bone_lengths:
            return self.bone_lengths[bone_name].length
        return 0.0
    
    def get_normalized_length(self, bone_name: str) -> float:
        """정규화된 본 길이 반환"""
        return self.normalized_lengths.get(bone_name, 0.0)


class BoneCalculator:
    """
    본 길이 계산기
    
    키포인트에서 각 본의 길이를 계산하고 정규화합니다.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        normalization_base: str = 'shoulder_width'  # 'shoulder_width' or 'torso_length'
    ):
        """
        Args:
            confidence_threshold: 유효 키포인트 판단 임계값
            normalization_base: 정규화 기준 ('shoulder_width' or 'torso_length')
        """
        self.confidence_threshold = confidence_threshold
        self.normalization_base = normalization_base
        
        # 본 정의
        self._init_bone_definitions()
    
    def _init_bone_definitions(self):
        """본 정의 초기화"""
        self.body_bone_defs = []
        for start_name, end_name in BODY_BONES:
            start_idx = get_keypoint_index(start_name)
            end_idx = get_keypoint_index(end_name)
            bone_name = f"{start_name}_{end_name}"
            self.body_bone_defs.append((bone_name, start_idx, end_idx))
        
        self.feet_bone_defs = []
        for start_name, end_name in FEET_BONES:
            start_idx = get_keypoint_index(start_name)
            end_idx = get_keypoint_index(end_name)
            bone_name = f"{start_name}_{end_name}"
            self.feet_bone_defs.append((bone_name, start_idx, end_idx))
        
        # 손 본 정의 (왼손/오른손 구분)
        self.left_hand_bone_defs = []
        self.right_hand_bone_defs = []
        
        hand_joint_names = [
            'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]
        
        for start_local, end_local in HAND_BONES:
            start_name = hand_joint_names[start_local]
            end_name = hand_joint_names[end_local]
            bone_name = f"{start_name}_{end_name}"
            
            self.left_hand_bone_defs.append((
                f"left_{bone_name}",
                LEFT_HAND_START_IDX + start_local,
                LEFT_HAND_START_IDX + end_local
            ))
            self.right_hand_bone_defs.append((
                f"right_{bone_name}",
                RIGHT_HAND_START_IDX + start_local,
                RIGHT_HAND_START_IDX + end_local
            ))
    
    def calculate(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> BodyProportions:
        """
        키포인트에서 본 길이 계산
        
        Args:
            keypoints: (K, 2) 키포인트 좌표
            scores: (K,) 키포인트 신뢰도
        
        Returns:
            BodyProportions: 신체 비율 정보
        """
        proportions = BodyProportions()
        
        # Body 본 계산
        for bone_name, start_idx, end_idx in self.body_bone_defs:
            bone_info = self._calculate_bone(
                keypoints, scores, bone_name, start_idx, end_idx
            )
            proportions.bone_lengths[bone_name] = bone_info
        
        # Feet 본 계산
        for bone_name, start_idx, end_idx in self.feet_bone_defs:
            bone_info = self._calculate_bone(
                keypoints, scores, bone_name, start_idx, end_idx
            )
            proportions.bone_lengths[bone_name] = bone_info
        
        # Hand 본 계산
        for bone_name, start_idx, end_idx in self.left_hand_bone_defs:
            bone_info = self._calculate_bone(
                keypoints, scores, bone_name, start_idx, end_idx
            )
            proportions.bone_lengths[bone_name] = bone_info
        
        for bone_name, start_idx, end_idx in self.right_hand_bone_defs:
            bone_info = self._calculate_bone(
                keypoints, scores, bone_name, start_idx, end_idx
            )
            proportions.bone_lengths[bone_name] = bone_info
        
        # 정규화 기준값 계산
        self._calculate_normalization_base(proportions, keypoints, scores)
        
        # 정규화된 길이 계산
        self._normalize_lengths(proportions)
        
        return proportions
    
    def _calculate_bone(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        bone_name: str,
        start_idx: int,
        end_idx: int
    ) -> BoneInfo:
        """단일 본 길이 계산"""
        start_conf = scores[start_idx] if start_idx < len(scores) else 0
        end_conf = scores[end_idx] if end_idx < len(scores) else 0
        
        min_conf = min(start_conf, end_conf)
        is_valid = min_conf > self.confidence_threshold
        
        if is_valid:
            start_pt = keypoints[start_idx]
            end_pt = keypoints[end_idx]
            length = calculate_distance(start_pt, end_pt)
        else:
            length = 0.0
        
        return BoneInfo(
            name=bone_name,
            start_idx=start_idx,
            end_idx=end_idx,
            length=length,
            confidence=min_conf,
            is_valid=is_valid
        )
    
    def _calculate_normalization_base(
        self,
        proportions: BodyProportions,
        keypoints: np.ndarray,
        scores: np.ndarray
    ):
        """정규화 기준값 계산"""
        # 어깨 너비
        left_shoulder_idx = BODY_KEYPOINTS['left_shoulder']
        right_shoulder_idx = BODY_KEYPOINTS['right_shoulder']
        
        if (scores[left_shoulder_idx] > self.confidence_threshold and
            scores[right_shoulder_idx] > self.confidence_threshold):
            proportions.shoulder_width = calculate_distance(
                keypoints[left_shoulder_idx],
                keypoints[right_shoulder_idx]
            )
        
        # 토르소 길이 (어깨 중심 ~ 엉덩이 중심)
        left_hip_idx = BODY_KEYPOINTS['left_hip']
        right_hip_idx = BODY_KEYPOINTS['right_hip']
        
        shoulder_valid = (
            scores[left_shoulder_idx] > self.confidence_threshold and
            scores[right_shoulder_idx] > self.confidence_threshold
        )
        hip_valid = (
            scores[left_hip_idx] > self.confidence_threshold and
            scores[right_hip_idx] > self.confidence_threshold
        )
        
        if shoulder_valid and hip_valid:
            shoulder_center = (
                keypoints[left_shoulder_idx] + keypoints[right_shoulder_idx]
            ) / 2
            hip_center = (
                keypoints[left_hip_idx] + keypoints[right_hip_idx]
            ) / 2
            proportions.torso_length = calculate_distance(shoulder_center, hip_center)
    
    def _normalize_lengths(self, proportions: BodyProportions):
        """본 길이 정규화"""
        if self.normalization_base == 'shoulder_width':
            base_length = proportions.shoulder_width
        else:
            base_length = proportions.torso_length
        
        if base_length <= 0:
            return
        
        for bone_name, bone_info in proportions.bone_lengths.items():
            if bone_info.is_valid and bone_info.length > 0:
                proportions.normalized_lengths[bone_name] = (
                    bone_info.length / base_length
                )
    
    def get_key_bone_lengths(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, float]:
        """
        주요 본 길이만 반환 (간략화된 버전)
        
        Returns:
            주요 본 이름: 길이 딕셔너리
        """
        proportions = self.calculate(keypoints, scores)
        
        key_bones = [
            'left_shoulder_left_elbow',
            'left_elbow_left_wrist',
            'right_shoulder_right_elbow',
            'right_elbow_right_wrist',
            'left_hip_left_knee',
            'left_knee_left_ankle',
            'right_hip_right_knee',
            'right_knee_right_ankle',
            'left_shoulder_right_shoulder',
            'left_hip_right_hip',
        ]
        
        result = {}
        for bone_name in key_bones:
            if bone_name in proportions.bone_lengths:
                bone_info = proportions.bone_lengths[bone_name]
                if bone_info.is_valid:
                    result[bone_name] = bone_info.length
        
        result['shoulder_width'] = proportions.shoulder_width
        result['torso_length'] = proportions.torso_length
        
        return result
