"""
포즈 전이 엔진

원본 이미지의 신체 비율(본 길이)을 유지하면서
레퍼런스 이미지의 포즈(방향 벡터)를 적용합니다.
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
    FACE_START_IDX,
    FACE_END_IDX,
    TOTAL_KEYPOINTS,
    get_keypoint_index
)
from ..analyzers.bone_calculator import BoneCalculator, BodyProportions
from ..analyzers.direction_extractor import DirectionExtractor, PoseDirections
from ..utils.geometry import apply_bone_transform, normalize_vector


@dataclass
class TransferConfig:
    """포즈 전이 설정"""
    confidence_threshold: float = 0.3
    use_face: bool = True
    use_hands: bool = True
    preserve_root_position: bool = True  # 원본 루트 위치 유지


@dataclass
class TransferResult:
    """포즈 전이 결과"""
    keypoints: np.ndarray  # (K, 2) 전이된 키포인트
    scores: np.ndarray  # (K,) 전이된 신뢰도
    source_bone_lengths: Dict[str, float]  # 원본 본 길이
    reference_directions: Dict[str, np.ndarray]  # 레퍼런스 방향
    transfer_log: Dict[str, str] = field(default_factory=dict)  # 각 키포인트 전이 방법


class PoseTransferEngine:
    """
    포즈 전이 엔진
    
    원본의 신체 비율 + 레퍼런스의 포즈 방향 = 전이된 포즈
    """
    
    def __init__(self, config: Optional[TransferConfig] = None):
        """
        Args:
            config: 전이 설정
        """
        self.config = config or TransferConfig()
        
        self.bone_calculator = BoneCalculator(
            confidence_threshold=self.config.confidence_threshold
        )
        self.direction_extractor = DirectionExtractor(
            confidence_threshold=self.config.confidence_threshold
        )
        
        self._init_transfer_order()
    
    def _init_transfer_order(self):
        """키포인트 전이 순서 정의 (계층적)"""
        # 루트에서부터의 전이 순서
        # 부모 -> 자식 순서로 진행해야 함
        
        self.body_transfer_order = [
            # 1. 루트 (엉덩이 중심)
            ('root', None, ['left_hip', 'right_hip']),
            
            # 2. 엉덩이 -> 어깨
            ('left_hip', 'left_hip', ['left_shoulder']),
            ('right_hip', 'right_hip', ['right_shoulder']),
            
            # 3. 어깨 -> 팔
            ('left_shoulder', 'left_shoulder', ['left_elbow']),
            ('left_elbow', 'left_elbow', ['left_wrist']),
            ('right_shoulder', 'right_shoulder', ['right_elbow']),
            ('right_elbow', 'right_elbow', ['right_wrist']),
            
            # 4. 엉덩이 -> 다리
            ('left_hip', 'left_hip', ['left_knee']),
            ('left_knee', 'left_knee', ['left_ankle']),
            ('right_hip', 'right_hip', ['right_knee']),
            ('right_knee', 'right_knee', ['right_ankle']),
            
            # 5. 발목 -> 발
            ('left_ankle', 'left_ankle', ['left_heel', 'left_big_toe']),
            ('left_big_toe', 'left_big_toe', ['left_small_toe']),
            ('right_ankle', 'right_ankle', ['right_heel', 'right_big_toe']),
            ('right_big_toe', 'right_big_toe', ['right_small_toe']),
            
            # 6. 어깨 -> 머리
            ('left_shoulder', 'left_shoulder', ['left_ear']),
            ('right_shoulder', 'right_shoulder', ['right_ear']),
            ('left_ear', 'left_ear', ['left_eye']),
            ('right_ear', 'right_ear', ['right_eye']),
            ('left_eye', 'left_eye', ['nose']),
        ]
    
    def transfer(
        self,
        source_keypoints: np.ndarray,
        source_scores: np.ndarray,
        reference_keypoints: np.ndarray,
        reference_scores: np.ndarray,
        target_image_size: Optional[Tuple[int, int]] = None
    ) -> TransferResult:
        """
        포즈 전이 수행
        
        Args:
            source_keypoints: (K, 2) 원본 키포인트 (비율 소스)
            source_scores: (K,) 원본 신뢰도
            reference_keypoints: (K, 2) 레퍼런스 키포인트 (포즈 소스)
            reference_scores: (K,) 레퍼런스 신뢰도
            target_image_size: (H, W) 타겟 이미지 크기 (스케일링용)
        
        Returns:
            TransferResult: 전이 결과
        """
        # 1. 원본에서 본 길이 추출
        source_proportions = self.bone_calculator.calculate(
            source_keypoints, source_scores
        )
        
        # 2. 레퍼런스에서 방향 벡터 추출
        reference_directions = self.direction_extractor.extract(
            reference_keypoints, reference_scores
        )
        
        # 3. 결과 배열 초기화
        num_keypoints = len(source_keypoints)
        transferred_kpts = np.zeros((num_keypoints, 2))
        transferred_scores = np.zeros(num_keypoints)
        transfer_log = {}
        
        # 4. 루트 위치 결정
        root_position = self._calculate_root_position(
            source_keypoints, source_scores,
            reference_keypoints, reference_scores
        )
        
        # 5. 계층적 전이 수행
        processed = set()
        
        for step_name, parent_name, children in self.body_transfer_order:
            for child_name in children:
                child_idx = get_keypoint_index(child_name)
                
                if step_name == 'root':
                    # 루트 키포인트는 루트 위치 기준으로 배치
                    transferred_kpts[child_idx] = self._transfer_root_child(
                        child_name, root_position,
                        source_proportions, reference_directions,
                        reference_keypoints
                    )
                    transferred_scores[child_idx] = min(
                        source_scores[child_idx],
                        reference_scores[child_idx]
                    )
                    transfer_log[child_name] = 'root_based'
                else:
                    parent_idx = get_keypoint_index(parent_name)
                    
                    if parent_idx not in processed:
                        continue
                    
                    # 부모 위치 기반으로 자식 위치 계산
                    parent_pos = transferred_kpts[parent_idx]
                    
                    child_pos, method = self._transfer_child(
                        parent_name, child_name,
                        parent_pos,
                        source_proportions, reference_directions,
                        source_keypoints, source_scores,
                        reference_keypoints, reference_scores
                    )
                    
                    transferred_kpts[child_idx] = child_pos
                    transferred_scores[child_idx] = self._calculate_transferred_score(
                        source_scores[child_idx],
                        reference_scores[child_idx],
                        method
                    )
                    transfer_log[child_name] = method
                
                processed.add(child_idx)
        
        # 6. 얼굴 키포인트 전이 (선택적)
        if self.config.use_face:
            self._transfer_face(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                transfer_log
            )
        
        # 7. 손 키포인트 전이 (선택적)
        if self.config.use_hands:
            self._transfer_hands(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                source_proportions, reference_directions,
                transfer_log
            )
        
        # 8. 결과 정리
        source_lengths = {}
        for name, info in source_proportions.bone_lengths.items():
            if info.is_valid:
                source_lengths[name] = info.length
        
        ref_dirs = {}
        for name, info in reference_directions.bone_directions.items():
            if info.is_valid:
                ref_dirs[name] = info.direction
        
        return TransferResult(
            keypoints=transferred_kpts,
            scores=transferred_scores,
            source_bone_lengths=source_lengths,
            reference_directions=ref_dirs,
            transfer_log=transfer_log
        )
    
    def _calculate_root_position(
        self,
        source_kpts: np.ndarray,
        source_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray
    ) -> np.ndarray:
        """루트 위치 계산 (엉덩이 중심)"""
        left_hip_idx = BODY_KEYPOINTS['left_hip']
        right_hip_idx = BODY_KEYPOINTS['right_hip']
        
        if self.config.preserve_root_position:
            # 원본 루트 위치 사용
            if (source_scores[left_hip_idx] > self.config.confidence_threshold and
                source_scores[right_hip_idx] > self.config.confidence_threshold):
                return (source_kpts[left_hip_idx] + source_kpts[right_hip_idx]) / 2
        
        # 레퍼런스 루트 위치 사용
        if (ref_scores[left_hip_idx] > self.config.confidence_threshold and
            ref_scores[right_hip_idx] > self.config.confidence_threshold):
            return (ref_kpts[left_hip_idx] + ref_kpts[right_hip_idx]) / 2
        
        # 폴백: 원본 사용
        return (source_kpts[left_hip_idx] + source_kpts[right_hip_idx]) / 2
    
    def _transfer_root_child(
        self,
        child_name: str,
        root_position: np.ndarray,
        source_proportions: BodyProportions,
        reference_directions: PoseDirections,
        reference_keypoints: np.ndarray
    ) -> np.ndarray:
        """루트 자식 키포인트 전이 (엉덩이)"""
        child_idx = get_keypoint_index(child_name)
        
        # 레퍼런스에서 루트 -> 자식 방향
        ref_left_hip = reference_keypoints[BODY_KEYPOINTS['left_hip']]
        ref_right_hip = reference_keypoints[BODY_KEYPOINTS['right_hip']]
        ref_root = (ref_left_hip + ref_right_hip) / 2
        
        ref_child = reference_keypoints[child_idx]
        direction = normalize_vector(ref_child - ref_root)
        
        # 원본 골반 너비의 절반
        hip_width = source_proportions.bone_lengths.get(
            'left_hip_right_hip',
            None
        )
        
        if hip_width and hip_width.is_valid:
            distance = hip_width.length / 2
        else:
            # 폴백: 레퍼런스 거리 사용
            distance = np.linalg.norm(ref_child - ref_root)
        
        return root_position + direction * distance
    
    def _transfer_child(
        self,
        parent_name: str,
        child_name: str,
        parent_pos: np.ndarray,
        source_proportions: BodyProportions,
        reference_directions: PoseDirections,
        source_kpts: np.ndarray,
        source_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """자식 키포인트 전이"""
        bone_name = f"{parent_name}_{child_name}"
        
        # 원본 본 길이
        bone_info = source_proportions.bone_lengths.get(bone_name)
        if bone_info and bone_info.is_valid:
            bone_length = bone_info.length
        else:
            # 폴백: 레퍼런스 본 길이 사용
            parent_idx = get_keypoint_index(parent_name)
            child_idx = get_keypoint_index(child_name)
            bone_length = np.linalg.norm(
                ref_kpts[child_idx] - ref_kpts[parent_idx]
            )
        
        # 레퍼런스 방향 벡터
        bone_dir = reference_directions.bone_directions.get(bone_name)
        if bone_dir and bone_dir.is_valid:
            direction = bone_dir.direction
            method = 'vector_transfer'
        else:
            # 폴백: 레퍼런스 좌표에서 직접 계산
            parent_idx = get_keypoint_index(parent_name)
            child_idx = get_keypoint_index(child_name)
            diff = ref_kpts[child_idx] - ref_kpts[parent_idx]
            direction = normalize_vector(diff)
            method = 'direct_reference'
        
        # 새 위치 계산
        new_pos = apply_bone_transform(parent_pos, direction, bone_length)
        
        return new_pos, method
    
    def _calculate_transferred_score(
        self,
        source_score: float,
        ref_score: float,
        method: str
    ) -> float:
        """전이된 키포인트의 신뢰도 계산"""
        base_score = min(source_score, ref_score)
        
        if method == 'vector_transfer':
            return base_score
        elif method == 'direct_reference':
            return base_score * 0.9
        else:
            return base_score * 0.8
    
    def _transfer_face(
        self,
        transferred_kpts: np.ndarray,
        transferred_scores: np.ndarray,
        source_kpts: np.ndarray,
        source_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        transfer_log: Dict[str, str]
    ):
        """얼굴 키포인트 전이"""
        # 코 위치를 기준으로 얼굴 전체를 상대적으로 배치
        nose_idx = BODY_KEYPOINTS['nose']
        
        if transferred_scores[nose_idx] < self.config.confidence_threshold:
            return
        
        nose_pos = transferred_kpts[nose_idx]
        
        # 원본 얼굴 스케일 계산 (눈 사이 거리 기준)
        left_eye_idx = BODY_KEYPOINTS['left_eye']
        right_eye_idx = BODY_KEYPOINTS['right_eye']
        
        source_eye_dist = np.linalg.norm(
            source_kpts[left_eye_idx] - source_kpts[right_eye_idx]
        )
        ref_eye_dist = np.linalg.norm(
            ref_kpts[left_eye_idx] - ref_kpts[right_eye_idx]
        )
        
        if ref_eye_dist > 0:
            face_scale = source_eye_dist / ref_eye_dist
        else:
            face_scale = 1.0
        
        # 레퍼런스 코 위치
        ref_nose = ref_kpts[nose_idx]
        
        # 얼굴 키포인트 전이
        for i in range(FACE_START_IDX, FACE_END_IDX + 1):
            if ref_scores[i] > self.config.confidence_threshold:
                # 레퍼런스 코 기준 상대 위치
                relative_pos = ref_kpts[i] - ref_nose
                # 스케일 적용 후 새 코 위치 기준으로 배치
                transferred_kpts[i] = nose_pos + relative_pos * face_scale
                transferred_scores[i] = min(source_scores[i], ref_scores[i])
                transfer_log[f'face_{i}'] = 'face_relative'
    
    def _transfer_hands(
        self,
        transferred_kpts: np.ndarray,
        transferred_scores: np.ndarray,
        source_kpts: np.ndarray,
        source_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        source_proportions: BodyProportions,
        reference_directions: PoseDirections,
        transfer_log: Dict[str, str]
    ):
        """손 키포인트 전이"""
        # 왼손
        self._transfer_single_hand(
            transferred_kpts, transferred_scores,
            source_kpts, source_scores,
            ref_kpts, ref_scores,
            source_proportions, reference_directions,
            is_left=True,
            transfer_log=transfer_log
        )
        
        # 오른손
        self._transfer_single_hand(
            transferred_kpts, transferred_scores,
            source_kpts, source_scores,
            ref_kpts, ref_scores,
            source_proportions, reference_directions,
            is_left=False,
            transfer_log=transfer_log
        )
    
    def _transfer_single_hand(
        self,
        transferred_kpts: np.ndarray,
        transferred_scores: np.ndarray,
        source_kpts: np.ndarray,
        source_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        source_proportions: BodyProportions,
        reference_directions: PoseDirections,
        is_left: bool,
        transfer_log: Dict[str, str]
    ):
        """단일 손 전이"""
        wrist_name = 'left_wrist' if is_left else 'right_wrist'
        wrist_idx = BODY_KEYPOINTS[wrist_name]
        hand_start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
        hand_prefix = 'left' if is_left else 'right'
        
        # 손목 위치
        wrist_pos = transferred_kpts[wrist_idx]
        
        if transferred_scores[wrist_idx] < self.config.confidence_threshold:
            return
        
        # 원본 손 스케일 계산 (손목-중지 MCP 거리)
        source_wrist = source_kpts[wrist_idx]
        source_middle_mcp = source_kpts[hand_start + 9]  # middle_mcp
        
        ref_wrist = ref_kpts[wrist_idx]
        ref_middle_mcp = ref_kpts[hand_start + 9]
        
        source_hand_scale = np.linalg.norm(source_middle_mcp - source_wrist)
        ref_hand_scale = np.linalg.norm(ref_middle_mcp - ref_wrist)
        
        if ref_hand_scale > 0:
            hand_scale = source_hand_scale / ref_hand_scale
        else:
            hand_scale = 1.0
        
        # 손 키포인트 전이
        for i in range(21):  # 손은 21개 키포인트
            kpt_idx = hand_start + i
            
            if ref_scores[kpt_idx] > self.config.confidence_threshold:
                # 레퍼런스 손목 기준 상대 위치
                relative_pos = ref_kpts[kpt_idx] - ref_wrist
                # 스케일 적용 후 새 손목 위치 기준으로 배치
                transferred_kpts[kpt_idx] = wrist_pos + relative_pos * hand_scale
                transferred_scores[kpt_idx] = min(
                    source_scores[kpt_idx], ref_scores[kpt_idx]
                )
                transfer_log[f'{hand_prefix}_hand_{i}'] = 'hand_relative'
