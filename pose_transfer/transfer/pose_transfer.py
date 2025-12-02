"""
포즈 전이 엔진 v4

수정사항:
1. 하반신 검증: Y좌표가 이미지 높이의 90% 이하여야 유효
2. 얼굴 설정: YAML에서 face_rendering 설정 로드
3. Source 얼굴 비율 + Reference 각도 적용
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
class FacePartConfig:
    """얼굴 부위별 설정"""
    enabled: bool = True
    color: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class FaceRenderingConfig:
    """얼굴 렌더링 설정"""
    enabled: bool = True
    jawline: FacePartConfig = field(default_factory=FacePartConfig)
    left_eyebrow: FacePartConfig = field(default_factory=FacePartConfig)
    right_eyebrow: FacePartConfig = field(default_factory=FacePartConfig)
    nose: FacePartConfig = field(default_factory=FacePartConfig)
    left_eye: FacePartConfig = field(default_factory=FacePartConfig)
    right_eye: FacePartConfig = field(default_factory=FacePartConfig)
    mouth_outer: FacePartConfig = field(default_factory=FacePartConfig)
    mouth_inner: FacePartConfig = field(default_factory=FacePartConfig)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FaceRenderingConfig':
        """딕셔너리에서 설정 로드"""
        config = cls()
        if not data:
            return config
        
        config.enabled = data.get('enabled', True)
        
        parts = data.get('parts', {})
        for part_name in ['jawline', 'left_eyebrow', 'right_eyebrow', 'nose', 
                          'left_eye', 'right_eye', 'mouth_outer', 'mouth_inner']:
            if part_name in parts:
                part_data = parts[part_name]
                setattr(config, part_name, FacePartConfig(
                    enabled=part_data.get('enabled', True),
                    color=tuple(part_data.get('color', [255, 255, 255]))
                ))
        
        return config


@dataclass
class TransferConfig:
    """포즈 전이 설정"""
    confidence_threshold: float = 0.3
    use_face: bool = True
    use_hands: bool = True
    enable_symmetric_fallback: bool = True
    lower_body_confidence_threshold: float = 2.0
    lower_body_y_ratio_threshold: float = 0.85  # 이미지 높이의 85% 이하여야 유효
    face_rendering: FaceRenderingConfig = field(default_factory=FaceRenderingConfig)


@dataclass
class TransferResult:
    """포즈 전이 결과"""
    keypoints: np.ndarray
    scores: np.ndarray
    source_bone_lengths: Dict[str, float]
    reference_directions: Dict[str, np.ndarray]
    transfer_log: Dict[str, str] = field(default_factory=dict)


# 얼굴 랜드마크 인덱스 정의 (68-point face landmarks)
FACE_PARTS = {
    'jawline': list(range(0, 17)),
    'left_eyebrow': list(range(17, 22)),
    'right_eyebrow': list(range(22, 27)),
    'nose': list(range(27, 36)),
    'left_eye': list(range(36, 42)),
    'right_eye': list(range(42, 48)),
    'mouth_outer': list(range(48, 60)),
    'mouth_inner': list(range(60, 68)),
}


class PoseTransferEngine:
    def __init__(self, config: Optional[TransferConfig] = None, yaml_config: Optional[dict] = None):
        self.config = config or TransferConfig()
        
        # YAML에서 face_rendering 설정 로드
        if yaml_config and 'face_rendering' in yaml_config:
            self.config.face_rendering = FaceRenderingConfig.from_dict(yaml_config['face_rendering'])
            print(f"[DEBUG] Face rendering enabled: {self.config.face_rendering.enabled}")
            print(f"[DEBUG] Jawline enabled: {self.config.face_rendering.jawline.enabled}")
        
        self.bone_calculator = BoneCalculator(
            confidence_threshold=self.config.confidence_threshold
        )
        self.direction_extractor = DirectionExtractor(
            confidence_threshold=self.config.confidence_threshold
        )
        
        self._init_transfer_order()
    
    def _init_transfer_order(self):
        """전이 순서"""
        self.upper_body_order = [
            ('left_shoulder', 'left_shoulder', ['left_elbow']),
            ('left_elbow', 'left_elbow', ['left_wrist']),
            ('right_shoulder', 'right_shoulder', ['right_elbow']),
            ('right_elbow', 'right_elbow', ['right_wrist']),
        ]
        
        self.lower_body_order = [
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
        target_image_size: Optional[Tuple[int, int]] = None,
        source_image_size: Optional[Tuple[int, int]] = None,
        reference_image_size: Optional[Tuple[int, int]] = None
    ) -> TransferResult:
        
        # 이미지 크기 추정
        if source_image_size is None:
            valid_pts = source_keypoints[source_scores > 0.3]
            if len(valid_pts) > 0:
                src_h = int(np.max(valid_pts[:, 1]) * 1.2)
                src_w = int(np.max(valid_pts[:, 0]) * 1.2)
            else:
                src_h, src_w = 2880, 2160
            source_image_size = (src_h, src_w)
        
        if reference_image_size is None:
            valid_pts = reference_keypoints[reference_scores > 0.3]
            if len(valid_pts) > 0:
                ref_h = int(np.max(valid_pts[:, 1]) * 1.2)
                ref_w = int(np.max(valid_pts[:, 0]) * 1.2)
            else:
                ref_h, ref_w = 6192, 4128
            reference_image_size = (ref_h, ref_w)
        
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

        # 4. 하반신 유효성 체크 (신뢰도 + Y좌표 검증)
        src_lower_valid = self._check_lower_body_valid(
            source_keypoints, source_scores, source_image_size[0]
        )
        ref_lower_valid = self._check_lower_body_valid(
            reference_keypoints, reference_scores, reference_image_size[0]
        )
        
        print(f"[DEBUG] Lower body valid - Source: {src_lower_valid}, Reference: {ref_lower_valid}")

        # 5. 결과 초기화
        num_keypoints = len(source_keypoints)
        transferred_kpts = np.zeros((num_keypoints, 2))
        transferred_scores = np.zeros(num_keypoints)
        transfer_log = {}
        processed = set()
        
        # 6. 어깨 (Reference 각도 적용)
        self._transfer_shoulders_with_reference_angle(
            transferred_kpts, transferred_scores,
            source_keypoints, source_scores,
            reference_keypoints, reference_scores,
            global_scale, processed, transfer_log
        )
        
        # 7. 척추/골반
        self._transfer_torso_via_spine(
            transferred_kpts, transferred_scores,
            source_keypoints, source_scores,
            reference_keypoints, reference_scores,
            corrected_lengths, global_scale,
            processed, transfer_log
        )
        
        # 8. 팔
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
        
        # 9. 하반신 (조건부: 둘 다 유효해야 그림)
        self._transfer_lower_body_conditional(
            transferred_kpts, transferred_scores,
            source_keypoints, source_scores,
            reference_keypoints, reference_scores,
            corrected_lengths, global_scale,
            src_lower_valid, ref_lower_valid,
            processed, transfer_log
        )
        
        # 10. 얼굴 (Source 비율 + Reference 각도)
        if self.config.use_face and self.config.face_rendering.enabled:
            self._transfer_face_source_proportions(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                global_scale, transfer_log
            )
        else:
            print("[DEBUG] Face rendering disabled")
            
        # 11. 손
        if self.config.use_hands:
            self._transfer_hands(
                transferred_kpts, transferred_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                global_scale, transfer_log
            )
        
        # 12. 이미지 범위 클리핑
        if target_image_size:
            h, w = target_image_size
        else:
            h, w = source_image_size
        
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

    def _check_lower_body_valid(
        self, 
        kpts: np.ndarray, 
        scores: np.ndarray,
        image_height: int
    ) -> bool:
        """
        하반신 유효성 검증:
        1. 신뢰도 체크
        2. Y좌표가 이미지 높이의 85% 이하여야 함 (프레임 안에 제대로 있어야 함)
        """
        l_hip = BODY_KEYPOINTS['left_hip']
        r_hip = BODY_KEYPOINTS['right_hip']
        l_knee = BODY_KEYPOINTS['left_knee']
        r_knee = BODY_KEYPOINTS['right_knee']
        
        threshold = self.config.lower_body_confidence_threshold
        y_threshold = image_height * self.config.lower_body_y_ratio_threshold
        
        # 엉덩이 유효
        hip_valid = (scores[l_hip] > threshold or scores[r_hip] > threshold)
        
        # 무릎 유효 (신뢰도)
        knee_conf_valid = (scores[l_knee] > threshold or scores[r_knee] > threshold)
        
        # 무릎 Y좌표 체크 (이미지 높이의 85% 이하여야 함)
        l_knee_y = kpts[l_knee][1]
        r_knee_y = kpts[r_knee][1]
        knee_y_valid = (l_knee_y < y_threshold and l_knee_y > 0) or \
                       (r_knee_y < y_threshold and r_knee_y > 0)
        
        print(f"[DEBUG] Image height: {image_height}, Y threshold: {y_threshold:.0f}")
        print(f"[DEBUG] Knee Y: left={l_knee_y:.0f}, right={r_knee_y:.0f}")
        print(f"[DEBUG] Hip valid: {hip_valid}, Knee conf valid: {knee_conf_valid}, Knee Y valid: {knee_y_valid}")
        
        return hip_valid and knee_conf_valid and knee_y_valid

    def _transfer_shoulders_with_reference_angle(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        scale, processed, log
    ):
        """어깨: Source 위치/너비 + Reference 각도"""
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        src_neck = (src_kpts[l_sh] + src_kpts[r_sh]) / 2
        src_shoulder_width = calculate_distance(src_kpts[l_sh], src_kpts[r_sh])
        
        ref_shoulder_vec = ref_kpts[r_sh] - ref_kpts[l_sh]
        ref_shoulder_dir = normalize_vector(ref_shoulder_vec)
        
        half_width = src_shoulder_width / 2
        
        trans_kpts[l_sh] = src_neck - ref_shoulder_dir * half_width
        trans_kpts[r_sh] = src_neck + ref_shoulder_dir * half_width
        
        trans_scores[l_sh] = src_scores[l_sh]
        trans_scores[r_sh] = src_scores[r_sh]
        
        processed.add(l_sh)
        processed.add(r_sh)
        
        log['left_shoulder'] = 'anchor_ref_angle'
        log['right_shoulder'] = 'anchor_ref_angle'

    def _transfer_torso_via_spine(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        lengths, scale, processed, log
    ):
        """척추 기반 골반"""
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        l_hip = BODY_KEYPOINTS['left_hip']
        r_hip = BODY_KEYPOINTS['right_hip']
        
        trans_neck = (trans_kpts[l_sh] + trans_kpts[r_sh]) / 2
        
        ref_neck = (ref_kpts[l_sh] + ref_kpts[r_sh]) / 2
        ref_root = (ref_kpts[l_hip] + ref_kpts[r_hip]) / 2
        
        spine_vec = ref_root - ref_neck
        spine_dir = normalize_vector(spine_vec)
        
        if src_scores[l_hip] > 0.3:
            src_neck_pos = (src_kpts[l_sh] + src_kpts[r_sh]) / 2
            src_root = (src_kpts[l_hip] + src_kpts[r_hip]) / 2
            spine_len = calculate_distance(src_root, src_neck_pos)
        else:
            spine_len = calculate_distance(ref_root, ref_neck) * scale
        
        new_root = trans_neck + spine_dir * spine_len
        
        ref_hip_vec = ref_kpts[r_hip] - ref_kpts[l_hip]
        ref_hip_dir = normalize_vector(ref_hip_vec)
        
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
        
        # 둘 중 하나라도 유효하지 않으면 → 하반신 스킵 (골반까지만)
        if not src_lower_valid or not ref_lower_valid:
            print("[DEBUG] Lower body SKIPPED - invalid in source or reference")
            return
        
        print("[DEBUG] Lower body TRANSFER - both valid")
        for _, parent_name, children in self.lower_body_order:
            parent_idx = get_keypoint_index(parent_name)
            
            if parent_idx not in processed or trans_scores[parent_idx] == 0:
                continue
            
            parent_pos = trans_kpts[parent_idx]
            
            for child_name in children:
                child_idx = get_keypoint_index(child_name)
                
                if ref_scores[child_idx] < self.config.lower_body_confidence_threshold:
                    continue
                
                child_pos, method = self._transfer_child(
                    parent_name, child_name, parent_pos,
                    lengths, ref_kpts, scale
                )
                
                trans_kpts[child_idx] = child_pos
                trans_scores[child_idx] = 0.8
                log[child_name] = method
                processed.add(child_idx)

    def _transfer_face_source_proportions(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        scale, log
    ):
        """얼굴: Source 비율 유지 + Reference 각도만 적용"""
        nose_idx = BODY_KEYPOINTS['nose']
        l_eye = BODY_KEYPOINTS['left_eye']
        r_eye = BODY_KEYPOINTS['right_eye']
        l_ear = BODY_KEYPOINTS['left_ear']
        r_ear = BODY_KEYPOINTS['right_ear']
        
        src_eye_vec = src_kpts[r_eye] - src_kpts[l_eye]
        ref_eye_vec = ref_kpts[r_eye] - ref_kpts[l_eye]
        
        src_eye_dist = np.linalg.norm(src_eye_vec)
        ref_eye_dist = np.linalg.norm(ref_eye_vec)
        
        if src_eye_dist < 1 or ref_eye_dist < 1:
            return
        
        src_angle = np.arctan2(src_eye_vec[1], src_eye_vec[0])
        ref_angle = np.arctan2(ref_eye_vec[1], ref_eye_vec[0])
        rotation_angle = ref_angle - src_angle
        
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        anchor_pos = src_kpts[nose_idx].copy()
        src_nose = src_kpts[nose_idx]
        
        # 코, 눈, 귀 전이
        head_indices = [nose_idx, l_eye, r_eye, l_ear, r_ear]
        
        for idx in head_indices:
            if src_scores[idx] > 0.3:
                rel_pos = src_kpts[idx] - src_nose
                rotated_pos = rotation_matrix @ rel_pos
                trans_kpts[idx] = anchor_pos + rotated_pos
                trans_scores[idx] = src_scores[idx]
                log[f'head_{idx}'] = 'src_proportion_rotated'
        
        # 얼굴 랜드마크 (68개)
        src_face_nose = src_kpts[FACE_START_IDX + 30]
        
        for i in range(FACE_START_IDX, FACE_END_IDX + 1):
            part_idx = i - FACE_START_IDX
            part_name = self._get_face_part_name(part_idx)
            
            # 부위별 활성화 체크
            if part_name and not self._is_face_part_enabled(part_name):
                trans_scores[i] = 0
                continue
            
            if src_scores[i] > 0.3:
                rel_pos = src_kpts[i] - src_face_nose
                rotated_pos = rotation_matrix @ rel_pos
                trans_kpts[i] = anchor_pos + rotated_pos
                trans_scores[i] = src_scores[i]
                log[f'face_{i}'] = 'src_proportion_rotated'

    def _get_face_part_name(self, local_idx: int) -> Optional[str]:
        """얼굴 랜드마크 인덱스 → 부위 이름"""
        for part_name, indices in FACE_PARTS.items():
            if local_idx in indices:
                return part_name
        return None

    def _is_face_part_enabled(self, part_name: str) -> bool:
        """얼굴 부위 활성화 여부"""
        if not self.config.face_rendering.enabled:
            return False
        
        part_config = getattr(self.config.face_rendering, part_name, None)
        if part_config:
            return part_config.enabled
        return True

    def _calculate_global_scale(self, src_props, ref_kpts, ref_scores):
        src_w = src_props.shoulder_width
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        if src_w > 0 and ref_scores[l_sh] > 0.3 and ref_scores[r_sh] > 0.3:
            ref_w = calculate_distance(ref_kpts[l_sh], ref_kpts[r_sh])
            return src_w / ref_w if ref_w > 0 else 1.0
        return 1.0

    def _correct_bone_lengths(self, src_props, scale, ref_kpts):
        lengths = {}
        
        for name, info in src_props.bone_lengths.items():
            if info.is_valid:
                lengths[name] = info.length
        
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