"""
포즈 전이 엔진 (Final: Spine-Based Logic)

- Anchor: 어깨(Shoulder) 고정
- Spine: 목(Neck) -> 골반중심(Root)으로 척추를 먼저 내려 중심 잡기
- Hips: 골반 중심에서 좌우로 엉덩이 배치 (골반 뒤틀림 방지)
- Limbs: 그 후 팔다리 전이
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from ..extractors.keypoint_constants import (
    BODY_KEYPOINTS,
    LEFT_HAND_START_IDX,
    RIGHT_HAND_START_IDX,
    FACE_START_IDX,
    FACE_END_IDX,
    get_keypoint_index
)
from ..analyzers.bone_calculator import BoneCalculator, BodyProportions
from ..analyzers.direction_extractor import DirectionExtractor, PoseDirections
from ..utils.geometry import apply_bone_transform, normalize_vector, calculate_distance

@dataclass
class TransferConfig:
    """포즈 전이 설정"""
    confidence_threshold: float = 0.3
    use_face: bool = True
    use_hands: bool = True
    enable_symmetric_fallback: bool = True

@dataclass
class TransferResult:
    """포즈 전이 결과"""
    keypoints: np.ndarray
    scores: np.ndarray
    source_bone_lengths: Dict[str, float]
    reference_directions: Dict[str, np.ndarray]
    transfer_log: Dict[str, str] = field(default_factory=dict)

class PoseTransferEngine:
    def __init__(self, config: Optional[TransferConfig] = None):
        self.config = config or TransferConfig()
        
        self.bone_calculator = BoneCalculator(
            confidence_threshold=self.config.confidence_threshold
        )
        self.direction_extractor = DirectionExtractor(
            confidence_threshold=self.config.confidence_threshold
        )
        
        self._init_transfer_order()
    
    def _init_transfer_order(self):
        # 전이 순서: 어깨 -> 팔 / (별도 로직으로 척추->골반) -> 다리
        self.body_transfer_order = [
            # 1. Anchor: 어깨 (Source 고정) - 내부 로직에서 처리됨
            
            # 2. 어깨 -> 팔 (밖으로)
            ('left_shoulder', 'left_shoulder', ['left_elbow']),
            ('left_elbow', 'left_elbow', ['left_wrist']),
            ('right_shoulder', 'right_shoulder', ['right_elbow']),
            ('right_elbow', 'right_elbow', ['right_wrist']),
            
            # [중요] 어깨->엉덩이 직접 연결 제거함 (척추 로직으로 대체)
            
            # 3. 엉덩이 -> 다리 (척추 로직으로 계산된 엉덩이에서 시작)
            ('left_hip', 'left_hip', ['left_knee']),
            ('left_knee', 'left_knee', ['left_ankle']),
            ('right_hip', 'right_hip', ['right_knee']),
            ('right_knee', 'right_knee', ['right_ankle']),
            
            # 4. 발 - 상반신만 있는 경우 스킵됨
            # (발 키포인트는 신뢰도 낮으면 자동 무효화)
            
            # 5. 머리 - 코/눈/귀는 얼굴에서 처리
            # (귀, 눈, 코는 _transfer_face에서 Source 기준으로 처리)
        ]

    def transfer(
        self,
        source_keypoints: np.ndarray,
        source_scores: np.ndarray,
        reference_keypoints: np.ndarray,
        reference_scores: np.ndarray,
        target_image_size: Optional[Tuple[int, int]] = None
    ) -> TransferResult:
        
        # 1. 정보 추출
        source_proportions = self.bone_calculator.calculate(source_keypoints, source_scores)
        reference_directions = self.direction_extractor.extract(reference_keypoints, reference_scores)
        
        # 2. 글로벌 스케일
        global_scale = self._calculate_global_scale(source_proportions, reference_keypoints, reference_scores)
        print(f"[DEBUG] Global Scale Factor (Src/Ref): {global_scale:.4f}")

        # 3. 본 길이 보정
        corrected_lengths = self._correct_bone_lengths(
            source_proportions, 
            global_scale, 
            reference_keypoints,
            self.config.enable_symmetric_fallback
        )

        # 4. 결과 초기화
        num_keypoints = len(source_keypoints)
        transferred_kpts = np.zeros((num_keypoints, 2))
        transferred_scores = np.zeros(num_keypoints)
        transfer_log = {}
        processed = set()
        
        # 5. [Phase 1] Anchor 설정 (어깨)
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        if source_scores[l_sh] > 0.3:
            transferred_kpts[l_sh] = source_keypoints[l_sh]
            transferred_scores[l_sh] = source_scores[l_sh]
            processed.add(l_sh)
            transfer_log['left_shoulder'] = 'anchor'
            
        if source_scores[r_sh] > 0.3:
            transferred_kpts[r_sh] = source_keypoints[r_sh]
            transferred_scores[r_sh] = source_scores[r_sh]
            processed.add(r_sh)
            transfer_log['right_shoulder'] = 'anchor'

        # 6. [Phase 2] 척추 기반 골반(Hip) 계산 (핵심 수정 사항)
        if l_sh in processed and r_sh in processed:
            self._transfer_torso_via_spine(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                corrected_lengths, global_scale,
                processed, transfer_log
            )
        
        # 7. [Phase 3] 나머지 부위 전이 (팔, 다리, 머리)
        for _, parent_name, children in self.body_transfer_order:
            parent_idx = get_keypoint_index(parent_name)
            
            # 부모가 처리되지 않았으면 자식도 스킵
            if parent_idx not in processed or transferred_scores[parent_idx] == 0:
                continue
            
            parent_pos = transferred_kpts[parent_idx]
            
            for child_name in children:
                child_idx = get_keypoint_index(child_name)
                
                child_pos, method = self._transfer_child(
                    parent_name, child_name, parent_pos,
                    corrected_lengths, reference_directions,
                    reference_keypoints, global_scale
                )
                
                transferred_kpts[child_idx] = child_pos
                transferred_scores[child_idx] = 0.8
                transfer_log[child_name] = method
                processed.add(child_idx)
        
        # 8. 얼굴 & 손 전이
        if self.config.use_face:
            self._transfer_face(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                transfer_log, global_scale
            )
            
        if self.config.use_hands:
            self._transfer_hands(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                corrected_lengths, reference_directions,
                transfer_log, global_scale
            )
            
        # 이미지 범위 밖 키포인트 무효화
        # Source 이미지 크기 사용
        h, w = 3500, 2500  # 여유있게 설정
        if target_image_size:
            h, w = target_image_size
        
        for i in range(len(transferred_kpts)):
            x, y = transferred_kpts[i]
            if x < 0 or x > w * 1.2 or y < 0 or y > h * 1.2:
                transferred_scores[i] = 0
        
        # 이미지 범위 밖 키포인트 무효화
        # Source 이미지 크기 사용
        h, w = 3500, 2500  # 여유있게 설정
        if target_image_size:
            h, w = target_image_size
        
        for i in range(len(transferred_kpts)):
            x, y = transferred_kpts[i]
            if x < 0 or x > w * 1.2 or y < 0 or y > h * 1.2:
                transferred_scores[i] = 0
        
        return TransferResult(
            keypoints=transferred_kpts,
            scores=transferred_scores,
            source_bone_lengths=corrected_lengths,
            reference_directions={},
            transfer_log=transfer_log
        )

    def _transfer_torso_via_spine(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, lengths, scale, processed, log):
        """어깨 중심 -> 척추 -> 골반 중심 -> 좌우 골반 순서로 계산"""
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        l_hip = BODY_KEYPOINTS['left_hip']
        r_hip = BODY_KEYPOINTS['right_hip']
        
        # 1. Source Neck (어깨 중심)
        src_neck = (trans_kpts[l_sh] + trans_kpts[r_sh]) / 2
        
        # 2. Ref Spine Vector (Ref Neck -> Ref Root)
        ref_neck = (ref_kpts[l_sh] + ref_kpts[r_sh]) / 2
        ref_root = (ref_kpts[l_hip] + ref_kpts[r_hip]) / 2
        
        spine_vec = ref_root - ref_neck
        spine_dir = normalize_vector(spine_vec)
        
        # 3. Spine Length 결정 (Source Torso 길이 or Ref * Scale)
        # Source에 hip이 없으므로 보통 Ref * Scale 사용됨
        # lengths 딕셔너리에 'spine_length' 같은건 없으므로 직접 계산
        if src_scores[l_hip] > 0.3: # Source 엉덩이 있으면 그거 씀
             src_root = (src_kpts[l_hip] + src_kpts[r_hip]) / 2
             src_neck_origin = (src_kpts[l_sh] + src_kpts[r_sh]) / 2
             spine_len = calculate_distance(src_root, src_neck_origin)
        else:
             spine_len = calculate_distance(ref_root, ref_neck) * scale
             
        # 4. New Root 위치 계산
        new_root = src_neck + spine_dir * spine_len
        
        # 5. 좌우 엉덩이 배치 (Root 기준)
        # Ref에서의 Root -> Hip 벡터를 가져와 적용
        for hip_idx, hip_name in [(l_hip, 'left_hip'), (r_hip, 'right_hip')]:
            vec = ref_kpts[hip_idx] - ref_root # Ref 중심에서 밖으로
            # 이 벡터도 스케일링 필요
            offset = vec * scale
            
            trans_kpts[hip_idx] = new_root + offset
            trans_scores[hip_idx] = 0.9
            processed.add(hip_idx)
            log[hip_name] = 'spine_calc'

    def _calculate_global_scale(self, src_props, ref_kpts, ref_scores):
        src_w = src_props.shoulder_width
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        if src_w > 0 and ref_scores[l_sh] > 0.3 and ref_scores[r_sh] > 0.3:
            ref_w = calculate_distance(ref_kpts[l_sh], ref_kpts[r_sh])
            return src_w / ref_w if ref_w > 0 else 1.0
        return 1.0

    def _correct_bone_lengths(self, src_props, scale, ref_kpts, symmetric):
        lengths = {}
        for name, info in src_props.bone_lengths.items():
            if info.is_valid: lengths[name] = info.length
            
        if symmetric:
            pairs = [
                ('left_shoulder_left_elbow', 'right_shoulder_right_elbow'),
                ('left_elbow_left_wrist', 'right_elbow_right_wrist'),
                ('left_hip_left_knee', 'right_hip_right_knee'),
                ('left_knee_left_ankle', 'right_knee_right_ankle'),
            ]
            for l, r in pairs:
                if l in lengths and r not in lengths: lengths[r] = lengths[l]
                elif r in lengths and l not in lengths: lengths[l] = lengths[r]

        # Top-Down 방식에서 필요한 다리/팔 길이 채우기
        required = [
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist')
        ]
        
        for p, c in required:
            name1 = f"{p}_{c}"
            name2 = f"{c}_{p}"
            if name1 not in lengths and name2 not in lengths:
                p_idx = get_keypoint_index(p)
                c_idx = get_keypoint_index(c)
                dist = calculate_distance(ref_kpts[p_idx], ref_kpts[c_idx]) * scale
                lengths[name1] = dist
                
        return lengths

    def _transfer_child(self, p_name, c_name, p_pos, lengths, directions, ref_kpts, scale):
        bone_name = f"{p_name}_{c_name}"
        alt_name = f"{c_name}_{p_name}"
        
        length = lengths.get(bone_name) or lengths.get(alt_name)
        
        if length is None:
            p_idx = get_keypoint_index(p_name)
            c_idx = get_keypoint_index(c_name)
            length = calculate_distance(ref_kpts[p_idx], ref_kpts[c_idx]) * scale
            method = 'ref_emergency'
        else:
            method = 'calc'
            
        p_idx = get_keypoint_index(p_name)
        c_idx = get_keypoint_index(c_name)
        
        vec = ref_kpts[c_idx] - ref_kpts[p_idx]
        direction = normalize_vector(vec)
        
        return p_pos + direction * length, method

    def _transfer_face(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, log, scale):
        # 코, 눈, 귀도 Source에서 복사 (얼굴 일관성 유지)
        head_indices = [
            BODY_KEYPOINTS['nose'],
            BODY_KEYPOINTS['left_eye'],
            BODY_KEYPOINTS['right_eye'],
            BODY_KEYPOINTS['left_ear'],
            BODY_KEYPOINTS['right_ear'],
        ]
        
        for idx in head_indices:
            if src_scores[idx] > 0.3:
                trans_kpts[idx] = src_kpts[idx]
                trans_scores[idx] = src_scores[idx]
                log[f'head_{idx}'] = 'source_head'
        
        # 얼굴 랜드마크도 Source 그대로
        for i in range(FACE_START_IDX, FACE_END_IDX + 1):
            if src_scores[i] > 0.3:
                trans_kpts[i] = src_kpts[i]
                trans_scores[i] = src_scores[i]
                log[f'face_{i}'] = 'source_face'

    def _transfer_hands(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, lengths, directions, log, scale):
        for is_left in [True, False]:
            wrist_name = 'left_wrist' if is_left else 'right_wrist'
            wrist_idx = BODY_KEYPOINTS[wrist_name]
            
            if trans_scores[wrist_idx] < 0.1: continue
            
            wrist_pos = trans_kpts[wrist_idx]
            hand_start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
            
            ref_wrist = ref_kpts[wrist_idx]
            hand_scale = scale
            
            for i in range(21):
                idx = hand_start + i
                if ref_scores[idx] > 0.3:
                    rel = ref_kpts[idx] - ref_wrist
                    trans_kpts[idx] = wrist_pos + rel * hand_scale
                    trans_scores[idx] = 0.9