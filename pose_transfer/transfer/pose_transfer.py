"""
포즈 전이 엔진 v2

주요 변경사항:
1. 얼굴: Reference 각도 적용 + Source 크기 유지
2. 어깨: Reference 각도 참조
3. 하반신: Source/Reference 유효성 기반 조건부 전이
4. 이미지 범위 밖 클리핑
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
    lower_body_confidence_threshold: float = 2.0  # 하반신 유효 판단 기준


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
        """전이 순서: 어깨 앵커 → 척추 → 팔/다리"""
        self.upper_body_order = [
            # 어깨 -> 팔
            ('left_shoulder', 'left_shoulder', ['left_elbow']),
            ('left_elbow', 'left_elbow', ['left_wrist']),
            ('right_shoulder', 'right_shoulder', ['right_elbow']),
            ('right_elbow', 'right_elbow', ['right_wrist']),
        ]
        
        self.lower_body_order = [
            # 엉덩이 -> 다리
            ('left_hip', 'left_hip', ['left_knee']),
            ('left_knee', 'left_knee', ['left_ankle']),
            ('right_hip', 'right_hip', ['right_knee']),
            ('right_knee', 'right_knee', ['right_ankle']),
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
            source_proportions, global_scale, reference_keypoints
        )

        # 4. 하반신 유효성 체크
        src_lower_valid = self._check_lower_body_valid(source_scores)
        ref_lower_valid = self._check_lower_body_valid(reference_scores)
        
        print(f"[DEBUG] Lower body valid - Source: {src_lower_valid}, Reference: {ref_lower_valid}")

        # 5. 결과 초기화
        num_keypoints = len(source_keypoints)
        transferred_kpts = np.zeros((num_keypoints, 2))
        transferred_scores = np.zeros(num_keypoints)
        transfer_log = {}
        processed = set()
        
        # 6. [Phase 1] 어깨 위치 계산 (Reference 각도 적용)
        self._transfer_shoulders_with_reference_angle(
            transferred_kpts, transferred_scores,
            source_keypoints, source_scores,
            reference_keypoints, reference_scores,
            global_scale, processed, transfer_log
        )
        
        # 7. [Phase 2] 척추/골반 계산
        self._transfer_torso_via_spine(
            transferred_kpts, transferred_scores,
            source_keypoints, source_scores,
            reference_keypoints, reference_scores,
            corrected_lengths, global_scale,
            processed, transfer_log
        )
        
        # 8. [Phase 3] 상체 (팔)
        for _, parent_name, children in self.upper_body_order:
            parent_idx = get_keypoint_index(parent_name)
            
            if parent_idx not in processed or transferred_scores[parent_idx] == 0:
                continue
            
            parent_pos = transferred_kpts[parent_idx]
            
            for child_name in children:
                child_idx = get_keypoint_index(child_name)
                
                child_pos, method = self._transfer_child(
                    parent_name, child_name, parent_pos,
                    corrected_lengths, reference_keypoints, global_scale
                )
                
                transferred_kpts[child_idx] = child_pos
                transferred_scores[child_idx] = 0.8
                transfer_log[child_name] = method
                processed.add(child_idx)
        
        # 9. [Phase 4] 하반신 (조건부)
        self._transfer_lower_body_conditional(
            transferred_kpts, transferred_scores,
            source_keypoints, source_scores,
            reference_keypoints, reference_scores,
            corrected_lengths, global_scale,
            src_lower_valid, ref_lower_valid,
            processed, transfer_log
        )
        
        # 10. 얼굴 전이 (Reference 각도 + Source 크기)
        if self.config.use_face:
            self._transfer_face_with_reference_angle(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                global_scale, transfer_log
            )
            
        # 11. 손 전이
        if self.config.use_hands:
            self._transfer_hands(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                global_scale, transfer_log
            )
        
        # 12. 이미지 범위 밖 클리핑
        if target_image_size:
            h, w = target_image_size
        else:
            # Source 이미지 기준으로 추정
            valid_pts = source_keypoints[source_scores > 0.3]
            if len(valid_pts) > 0:
                h = int(np.max(valid_pts[:, 1]) * 1.5)
                w = int(np.max(valid_pts[:, 0]) * 1.5)
            else:
                h, w = 3000, 2500
        
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

    def _check_lower_body_valid(self, scores: np.ndarray) -> bool:
        """하반신 유효성 체크: 엉덩이 + 무릎이 있어야 유효"""
        l_hip = BODY_KEYPOINTS['left_hip']
        r_hip = BODY_KEYPOINTS['right_hip']
        l_knee = BODY_KEYPOINTS['left_knee']
        r_knee = BODY_KEYPOINTS['right_knee']
        
        threshold = self.config.lower_body_confidence_threshold
        
        hip_valid = scores[l_hip] > threshold or scores[r_hip] > threshold
        knee_valid = scores[l_knee] > threshold or scores[r_knee] > threshold
        
        return hip_valid and knee_valid

    def _transfer_shoulders_with_reference_angle(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        scale, processed, log
    ):
        """어깨: Source 위치 기반 + Reference 각도 적용"""
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        # Source 어깨 중심
        src_neck = (src_kpts[l_sh] + src_kpts[r_sh]) / 2
        src_shoulder_width = calculate_distance(src_kpts[l_sh], src_kpts[r_sh])
        
        # Reference 어깨 벡터 (각도 추출)
        ref_shoulder_vec = ref_kpts[r_sh] - ref_kpts[l_sh]
        ref_shoulder_dir = normalize_vector(ref_shoulder_vec)
        
        # Source 너비 + Reference 각도로 새 어깨 위치 계산
        half_width = src_shoulder_width / 2
        
        trans_kpts[l_sh] = src_neck - ref_shoulder_dir * half_width
        trans_kpts[r_sh] = src_neck + ref_shoulder_dir * half_width
        
        trans_scores[l_sh] = src_scores[l_sh]
        trans_scores[r_sh] = src_scores[r_sh]
        
        processed.add(l_sh)
        processed.add(r_sh)
        
        log['left_shoulder'] = 'anchor_with_ref_angle'
        log['right_shoulder'] = 'anchor_with_ref_angle'

    def _transfer_torso_via_spine(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        lengths, scale, processed, log
    ):
        """척추 기반 골반 계산"""
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        l_hip = BODY_KEYPOINTS['left_hip']
        r_hip = BODY_KEYPOINTS['right_hip']
        
        # 1. 전이된 어깨 중심 (Neck)
        trans_neck = (trans_kpts[l_sh] + trans_kpts[r_sh]) / 2
        
        # 2. Reference 척추 방향
        ref_neck = (ref_kpts[l_sh] + ref_kpts[r_sh]) / 2
        ref_root = (ref_kpts[l_hip] + ref_kpts[r_hip]) / 2
        
        spine_vec = ref_root - ref_neck
        spine_dir = normalize_vector(spine_vec)
        
        # 3. Source 척추 길이
        if src_scores[l_hip] > 0.3:
            src_neck_pos = (src_kpts[l_sh] + src_kpts[r_sh]) / 2
            src_root = (src_kpts[l_hip] + src_kpts[r_hip]) / 2
            spine_len = calculate_distance(src_root, src_neck_pos)
        else:
            spine_len = calculate_distance(ref_root, ref_neck) * scale
        
        # 4. 새 골반 중심
        new_root = trans_neck + spine_dir * spine_len
        
        # 5. Reference 골반 각도 적용
        ref_hip_vec = ref_kpts[r_hip] - ref_kpts[l_hip]
        ref_hip_dir = normalize_vector(ref_hip_vec)
        
        # Source 골반 너비
        if src_scores[l_hip] > 0.3 and src_scores[r_hip] > 0.3:
            src_hip_width = calculate_distance(src_kpts[l_hip], src_kpts[r_hip])
        else:
            src_hip_width = calculate_distance(ref_kpts[l_hip], ref_kpts[r_hip]) * scale
        
        half_hip = src_hip_width / 2
        
        trans_kpts[l_hip] = new_root - ref_hip_dir * half_hip
        trans_kpts[r_hip] = new_root + ref_hip_dir * half_hip
        
        trans_scores[l_hip] = 0.9
        trans_scores[r_hip] = 0.9
        
        processed.add(l_hip)
        processed.add(r_hip)
        
        log['left_hip'] = 'spine_calc'
        log['right_hip'] = 'spine_calc'

    def _transfer_lower_body_conditional(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        lengths, scale,
        src_lower_valid, ref_lower_valid,
        processed, log
    ):
        """하반신 조건부 전이"""
        
        # Case 1: 둘 다 하반신 없음 → 스킵
        if not src_lower_valid and not ref_lower_valid:
            print("[DEBUG] Both source and reference have no valid lower body - skipping")
            return
        
        # Case 2: Source 없음 + Reference 있음 → Reference 기준으로 전이
        if not src_lower_valid and ref_lower_valid:
            print("[DEBUG] Source has no lower body, using reference with scaled proportions")
            self._transfer_lower_from_reference(
                trans_kpts, trans_scores,
                ref_kpts, ref_scores,
                lengths, scale, processed, log
            )
            return
        
        # Case 3: Source 있음 + Reference 없음 → 하반신 스킵
        if src_lower_valid and not ref_lower_valid:
            print("[DEBUG] Reference has no lower body - skipping lower body")
            return
        
        # Case 4: 둘 다 있음 → 정상 전이
        print("[DEBUG] Both have valid lower body - normal transfer")
        for _, parent_name, children in self.lower_body_order:
            parent_idx = get_keypoint_index(parent_name)
            
            if parent_idx not in processed or trans_scores[parent_idx] == 0:
                continue
            
            parent_pos = trans_kpts[parent_idx]
            
            for child_name in children:
                child_idx = get_keypoint_index(child_name)
                
                # Reference 신뢰도 체크
                if ref_scores[child_idx] < self.config.lower_body_confidence_threshold:
                    print(f"[DEBUG] Skipping {child_name} - low reference confidence")
                    continue
                
                child_pos, method = self._transfer_child(
                    parent_name, child_name, parent_pos,
                    lengths, ref_kpts, scale
                )
                
                trans_kpts[child_idx] = child_pos
                trans_scores[child_idx] = 0.8
                log[child_name] = method
                processed.add(child_idx)

    def _transfer_lower_from_reference(
        self, trans_kpts, trans_scores,
        ref_kpts, ref_scores,
        lengths, scale, processed, log
    ):
        """Reference 기준 하반신 전이 (Source에 하반신 없을 때)"""
        for _, parent_name, children in self.lower_body_order:
            parent_idx = get_keypoint_index(parent_name)
            
            if parent_idx not in processed or trans_scores[parent_idx] == 0:
                continue
            
            parent_pos = trans_kpts[parent_idx]
            
            for child_name in children:
                child_idx = get_keypoint_index(child_name)
                
                # 무릎까지만 (발목 이하는 스킵)
                if child_name in ['left_ankle', 'right_ankle']:
                    if ref_scores[child_idx] < self.config.lower_body_confidence_threshold:
                        continue
                
                if ref_scores[child_idx] < self.config.lower_body_confidence_threshold:
                    continue
                
                child_pos, method = self._transfer_child(
                    parent_name, child_name, parent_pos,
                    lengths, ref_kpts, scale
                )
                
                trans_kpts[child_idx] = child_pos
                trans_scores[child_idx] = 0.7  # 낮은 신뢰도
                log[child_name] = method + '_from_ref'
                processed.add(child_idx)

    def _transfer_face_with_reference_angle(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        scale, log
    ):
        """얼굴: Reference 각도 + Source 크기"""
        
        # 1. 코, 눈, 귀 인덱스
        nose_idx = BODY_KEYPOINTS['nose']
        l_eye = BODY_KEYPOINTS['left_eye']
        r_eye = BODY_KEYPOINTS['right_eye']
        l_ear = BODY_KEYPOINTS['left_ear']
        r_ear = BODY_KEYPOINTS['right_ear']
        
        # 2. Source 얼굴 크기 (눈 사이 거리 기준)
        src_eye_dist = calculate_distance(src_kpts[l_eye], src_kpts[r_eye])
        ref_eye_dist = calculate_distance(ref_kpts[l_eye], ref_kpts[r_eye])
        
        if ref_eye_dist > 0:
            face_scale = src_eye_dist / ref_eye_dist
        else:
            face_scale = scale
        
        # 3. 얼굴 앵커: Source 코 위치 사용
        anchor_pos = src_kpts[nose_idx].copy()
        
        # 4. Reference 코 위치
        ref_nose = ref_kpts[nose_idx]
        
        # 5. 코, 눈, 귀 전이 (Reference 상대 위치 + Source 스케일)
        head_indices = [nose_idx, l_eye, r_eye, l_ear, r_ear]
        
        for idx in head_indices:
            if ref_scores[idx] > 0.3:
                # Reference에서 코 기준 상대 위치
                rel_pos = ref_kpts[idx] - ref_nose
                # Source 스케일 적용
                trans_kpts[idx] = anchor_pos + rel_pos * face_scale
                trans_scores[idx] = min(src_scores[idx], ref_scores[idx])
                log[f'head_{idx}'] = 'face_ref_angle'
        
        # 6. 얼굴 랜드마크 (68개)
        for i in range(FACE_START_IDX, FACE_END_IDX + 1):
            if ref_scores[i] > 0.3:
                rel_pos = ref_kpts[i] - ref_nose
                trans_kpts[i] = anchor_pos + rel_pos * face_scale
                trans_scores[i] = min(src_scores[i], ref_scores[i])
                log[f'face_{i}'] = 'face_ref_angle'

    def _calculate_global_scale(self, src_props, ref_kpts, ref_scores):
        """글로벌 스케일: 어깨 너비 기준"""
        src_w = src_props.shoulder_width
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        if src_w > 0 and ref_scores[l_sh] > 0.3 and ref_scores[r_sh] > 0.3:
            ref_w = calculate_distance(ref_kpts[l_sh], ref_kpts[r_sh])
            return src_w / ref_w if ref_w > 0 else 1.0
        return 1.0

    def _correct_bone_lengths(self, src_props, scale, ref_kpts):
        """본 길이 보정 및 폴백"""
        lengths = {}
        
        for name, info in src_props.bone_lengths.items():
            if info.is_valid:
                lengths[name] = info.length
        
        # 대칭 폴백
        pairs = [
            ('left_shoulder_left_elbow', 'right_shoulder_right_elbow'),
            ('left_elbow_left_wrist', 'right_elbow_right_wrist'),
            ('left_hip_left_knee', 'right_hip_right_knee'),
            ('left_knee_left_ankle', 'right_knee_right_ankle'),
        ]
        for l, r in pairs:
            if l in lengths and r not in lengths:
                lengths[r] = lengths[l]
            elif r in lengths and l not in lengths:
                lengths[l] = lengths[r]
        
        # Reference 기반 폴백
        required = [
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist')
        ]
        
        for p, c in required:
            name = f"{p}_{c}"
            if name not in lengths:
                p_idx = get_keypoint_index(p)
                c_idx = get_keypoint_index(c)
                dist = calculate_distance(ref_kpts[p_idx], ref_kpts[c_idx]) * scale
                lengths[name] = dist
                
        return lengths

    def _transfer_child(self, p_name, c_name, p_pos, lengths, ref_kpts, scale):
        """자식 키포인트 전이"""
        bone_name = f"{p_name}_{c_name}"
        
        length = lengths.get(bone_name)
        
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

    def _transfer_hands(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        scale, log
    ):
        """손 전이"""
        for is_left in [True, False]:
            wrist_name = 'left_wrist' if is_left else 'right_wrist'
            wrist_idx = BODY_KEYPOINTS[wrist_name]
            
            if trans_scores[wrist_idx] < 0.1:
                continue
            
            wrist_pos = trans_kpts[wrist_idx]
            hand_start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
            
            ref_wrist = ref_kpts[wrist_idx]
            
            for i in range(21):
                idx = hand_start + i
                if ref_scores[idx] > 0.3:
                    rel = ref_kpts[idx] - ref_wrist
                    trans_kpts[idx] = wrist_pos + rel * scale
                    trans_scores[idx] = 0.9