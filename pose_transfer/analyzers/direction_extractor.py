"""
방향 벡터 추출 모듈

레퍼런스 이미지에서 각 본의 방향 벡터를 추출합니다.
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
from ..utils.geometry import normalize_vector, calculate_angle


@dataclass
class BoneDirection:
    """본 방향 정보"""
    name: str
    start_idx: int
    end_idx: int
    direction: np.ndarray  # 단위 방향 벡터 (2D)
    angle: float  # x축 기준 각도 (라디안)
    confidence: float
    is_valid: bool


@dataclass
class PoseDirections:
    """포즈 방향 정보"""
    bone_directions: Dict[str, BoneDirection] = field(default_factory=dict)
    
    # 주요 관절 각도
    joint_angles: Dict[str, float] = field(default_factory=dict)
    
    def get_direction(self, bone_name: str) -> Optional[np.ndarray]:
        """본 방향 벡터 반환"""
        if bone_name in self.bone_directions:
            bd = self.bone_directions[bone_name]
            if bd.is_valid:
                return bd.direction
        return None
    
    def get_angle(self, bone_name: str) -> Optional[float]:
        """본 각도 반환"""
        if bone_name in self.bone_directions:
            bd = self.bone_directions[bone_name]
            if bd.is_valid:
                return bd.angle
        return None


class DirectionExtractor:
    """
    방향 벡터 추출기
    
    키포인트에서 각 본의 방향 벡터와 관절 각도를 추출합니다.
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Args:
            confidence_threshold: 유효 키포인트 판단 임계값
        """
        self.confidence_threshold = confidence_threshold
        self._init_bone_definitions()
    
    def _init_bone_definitions(self):
        """본 정의 초기화 (BoneCalculator와 동일)"""
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
        
        # 손 본 정의
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
    
    def extract(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> PoseDirections:
        """
        키포인트에서 방향 벡터 추출
        
        Args:
            keypoints: (K, 2) 키포인트 좌표
            scores: (K,) 키포인트 신뢰도
        
        Returns:
            PoseDirections: 포즈 방향 정보
        """
        directions = PoseDirections()
        
        # Body 본 방향 추출
        for bone_name, start_idx, end_idx in self.body_bone_defs:
            bone_dir = self._extract_direction(
                keypoints, scores, bone_name, start_idx, end_idx
            )
            directions.bone_directions[bone_name] = bone_dir
        
        # Feet 본 방향 추출
        for bone_name, start_idx, end_idx in self.feet_bone_defs:
            bone_dir = self._extract_direction(
                keypoints, scores, bone_name, start_idx, end_idx
            )
            directions.bone_directions[bone_name] = bone_dir
        
        # Hand 본 방향 추출
        for bone_name, start_idx, end_idx in self.left_hand_bone_defs:
            bone_dir = self._extract_direction(
                keypoints, scores, bone_name, start_idx, end_idx
            )
            directions.bone_directions[bone_name] = bone_dir
        
        for bone_name, start_idx, end_idx in self.right_hand_bone_defs:
            bone_dir = self._extract_direction(
                keypoints, scores, bone_name, start_idx, end_idx
            )
            directions.bone_directions[bone_name] = bone_dir
        
        # 주요 관절 각도 계산
        self._calculate_joint_angles(directions, keypoints, scores)
        
        return directions
    
    def _extract_direction(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        bone_name: str,
        start_idx: int,
        end_idx: int
    ) -> BoneDirection:
        """단일 본 방향 추출"""
        start_conf = scores[start_idx] if start_idx < len(scores) else 0
        end_conf = scores[end_idx] if end_idx < len(scores) else 0
        
        min_conf = min(start_conf, end_conf)
        is_valid = min_conf > self.confidence_threshold
        
        if is_valid:
            start_pt = keypoints[start_idx]
            end_pt = keypoints[end_idx]
            diff = end_pt - start_pt
            direction = normalize_vector(diff)
            angle = np.arctan2(diff[1], diff[0])  # y, x 순서
        else:
            direction = np.zeros(2)
            angle = 0.0
        
        return BoneDirection(
            name=bone_name,
            start_idx=start_idx,
            end_idx=end_idx,
            direction=direction,
            angle=angle,
            confidence=min_conf,
            is_valid=is_valid
        )
    
    def _calculate_joint_angles(
        self,
        directions: PoseDirections,
        keypoints: np.ndarray,
        scores: np.ndarray
    ):
        """주요 관절 각도 계산"""
        # 팔꿈치 각도 (상완-하완)
        directions.joint_angles['left_elbow'] = self._calc_joint_angle(
            keypoints, scores,
            BODY_KEYPOINTS['left_shoulder'],
            BODY_KEYPOINTS['left_elbow'],
            BODY_KEYPOINTS['left_wrist']
        )
        
        directions.joint_angles['right_elbow'] = self._calc_joint_angle(
            keypoints, scores,
            BODY_KEYPOINTS['right_shoulder'],
            BODY_KEYPOINTS['right_elbow'],
            BODY_KEYPOINTS['right_wrist']
        )
        
        # 무릎 각도 (대퇴-하퇴)
        directions.joint_angles['left_knee'] = self._calc_joint_angle(
            keypoints, scores,
            BODY_KEYPOINTS['left_hip'],
            BODY_KEYPOINTS['left_knee'],
            BODY_KEYPOINTS['left_ankle']
        )
        
        directions.joint_angles['right_knee'] = self._calc_joint_angle(
            keypoints, scores,
            BODY_KEYPOINTS['right_hip'],
            BODY_KEYPOINTS['right_knee'],
            BODY_KEYPOINTS['right_ankle']
        )
        
        # 어깨 각도 (몸통-상완)
        # 몸통 중심선 기준으로 계산
        directions.joint_angles['left_shoulder'] = self._calc_shoulder_angle(
            keypoints, scores, 'left'
        )
        
        directions.joint_angles['right_shoulder'] = self._calc_shoulder_angle(
            keypoints, scores, 'right'
        )
        
        # 엉덩이 각도 (몸통-대퇴)
        directions.joint_angles['left_hip'] = self._calc_hip_angle(
            keypoints, scores, 'left'
        )
        
        directions.joint_angles['right_hip'] = self._calc_hip_angle(
            keypoints, scores, 'right'
        )
    
    def _calc_joint_angle(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        parent_idx: int,
        joint_idx: int,
        child_idx: int
    ) -> float:
        """세 점으로 관절 각도 계산"""
        if (scores[parent_idx] < self.confidence_threshold or
            scores[joint_idx] < self.confidence_threshold or
            scores[child_idx] < self.confidence_threshold):
            return np.nan
        
        v1 = keypoints[parent_idx] - keypoints[joint_idx]
        v2 = keypoints[child_idx] - keypoints[joint_idx]
        
        return calculate_angle(v1, v2)
    
    def _calc_shoulder_angle(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        side: str
    ) -> float:
        """어깨 각도 계산 (몸통 중심선 기준)"""
        if side == 'left':
            shoulder_idx = BODY_KEYPOINTS['left_shoulder']
            elbow_idx = BODY_KEYPOINTS['left_elbow']
            other_shoulder_idx = BODY_KEYPOINTS['right_shoulder']
        else:
            shoulder_idx = BODY_KEYPOINTS['right_shoulder']
            elbow_idx = BODY_KEYPOINTS['right_elbow']
            other_shoulder_idx = BODY_KEYPOINTS['left_shoulder']
        
        hip_idx = BODY_KEYPOINTS['left_hip'] if side == 'left' else BODY_KEYPOINTS['right_hip']
        
        if (scores[shoulder_idx] < self.confidence_threshold or
            scores[elbow_idx] < self.confidence_threshold or
            scores[hip_idx] < self.confidence_threshold):
            return np.nan
        
        # 몸통 방향 (어깨 -> 엉덩이)
        torso_vec = keypoints[hip_idx] - keypoints[shoulder_idx]
        # 상완 방향 (어깨 -> 팔꿈치)
        arm_vec = keypoints[elbow_idx] - keypoints[shoulder_idx]
        
        return calculate_angle(torso_vec, arm_vec)
    
    def _calc_hip_angle(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        side: str
    ) -> float:
        """엉덩이 각도 계산 (몸통 기준)"""
        if side == 'left':
            hip_idx = BODY_KEYPOINTS['left_hip']
            knee_idx = BODY_KEYPOINTS['left_knee']
            shoulder_idx = BODY_KEYPOINTS['left_shoulder']
        else:
            hip_idx = BODY_KEYPOINTS['right_hip']
            knee_idx = BODY_KEYPOINTS['right_knee']
            shoulder_idx = BODY_KEYPOINTS['right_shoulder']
        
        if (scores[hip_idx] < self.confidence_threshold or
            scores[knee_idx] < self.confidence_threshold or
            scores[shoulder_idx] < self.confidence_threshold):
            return np.nan
        
        # 몸통 방향 (엉덩이 -> 어깨)
        torso_vec = keypoints[shoulder_idx] - keypoints[hip_idx]
        # 대퇴 방향 (엉덩이 -> 무릎)
        thigh_vec = keypoints[knee_idx] - keypoints[hip_idx]
        
        return calculate_angle(torso_vec, thigh_vec)
    
    def get_key_directions(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        주요 본 방향만 반환 (간략화된 버전)
        """
        pose_dirs = self.extract(keypoints, scores)
        
        key_bones = [
            'left_shoulder_left_elbow',
            'left_elbow_left_wrist',
            'right_shoulder_right_elbow',
            'right_elbow_right_wrist',
            'left_hip_left_knee',
            'left_knee_left_ankle',
            'right_hip_right_knee',
            'right_knee_right_ankle',
        ]
        
        result = {}
        for bone_name in key_bones:
            direction = pose_dirs.get_direction(bone_name)
            if direction is not None:
                result[bone_name] = direction
        
        return result
