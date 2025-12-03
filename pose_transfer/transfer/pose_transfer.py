"""
포즈 전이 엔진 (Final Fix)
- 하반신 검증 강화 (이미지 경계 마진 체크)
- 얼굴 부위별 렌더링 설정 완벽 적용
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
    enabled: bool = True
    color: Tuple[int, int, int] = (255, 255, 255)

@dataclass
class FaceRenderingConfig:
    enabled: bool = True
    parts: Dict[str, FacePartConfig] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FaceRenderingConfig':
        config = cls()
        if not data: return config
        
        config.enabled = data.get('enabled', True)
        parts_data = data.get('parts', {})
        
        # 기본 부위 정의
        default_parts = ['jawline', 'left_eyebrow', 'right_eyebrow', 'nose', 
                         'left_eye', 'right_eye', 'mouth_outer', 'mouth_inner']
        
        config.parts = {}
        for part in default_parts:
            p_data = parts_data.get(part, {})
            config.parts[part] = FacePartConfig(
                enabled=p_data.get('enabled', True),
                color=tuple(p_data.get('color', [255, 255, 255]))
            )
        return config

@dataclass
class TransferConfig:
    confidence_threshold: float = 0.3
    use_face: bool = True
    use_hands: bool = True
    enable_symmetric_fallback: bool = True
    # 하반신 검증용
    lower_body_confidence_threshold: float = 0.5  # 점수 기준 (너무 높으면 안됨)
    lower_body_margin_ratio: float = 0.05         # 이미지 바닥에서 5% 이내면 무효

    face_rendering: FaceRenderingConfig = field(default_factory=FaceRenderingConfig)

@dataclass
class TransferResult:
    keypoints: np.ndarray
    scores: np.ndarray
    source_bone_lengths: Dict[str, float]
    reference_directions: Dict[str, np.ndarray]
    transfer_log: Dict[str, str] = field(default_factory=dict)

# 얼굴 랜드마크 인덱스 매핑
FACE_PARTS_IDX = {
    'jawline': range(0, 17),
    'left_eyebrow': range(17, 22),
    'right_eyebrow': range(22, 27),
    'nose': range(27, 36),
    'left_eye': range(36, 42),
    'right_eye': range(42, 48),
    'mouth_outer': range(48, 60),
    'mouth_inner': range(60, 68),
}

class PoseTransferEngine:
    def __init__(self, config: Optional[TransferConfig] = None, yaml_config: Optional[dict] = None):
        self.config = config or TransferConfig()
        
        # YAML 설정 주입 (얼굴 설정)
        if yaml_config and 'face_rendering' in yaml_config:
            self.config.face_rendering = FaceRenderingConfig.from_dict(yaml_config['face_rendering'])
        
        self.bone_calculator = BoneCalculator(confidence_threshold=self.config.confidence_threshold)
        self.direction_extractor = DirectionExtractor(confidence_threshold=self.config.confidence_threshold)
        self._init_transfer_order()
    
    def _init_transfer_order(self):
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
        source_image_size: Optional[Tuple[int, int]] = None,
        reference_image_size: Optional[Tuple[int, int]] = None,
        target_image_size: Optional[Tuple[int, int]] = None
    ) -> TransferResult:
        
        # 1. 이미지 크기 확정 (없으면 키포인트 기준 추정)
        if source_image_size is None:
            max_y = np.max(source_keypoints[:, 1])
            source_image_size = (int(max_y * 1.1), int(np.max(source_keypoints[:, 0])))
        if reference_image_size is None:
            max_y = np.max(reference_keypoints[:, 1])
            reference_image_size = (int(max_y * 1.1), int(np.max(reference_keypoints[:, 0])))

        # 2. 하반신 유효성 검사 (Ghost Legs 제거 핵심)
        src_lower_valid = self._check_lower_body_valid(source_keypoints, source_scores, source_image_size[0])
        ref_lower_valid = self._check_lower_body_valid(reference_keypoints, reference_scores, reference_image_size[0])
        
        print(f"[DEBUG] Lower Body Check: Src={src_lower_valid}, Ref={ref_lower_valid}")

        # 3. 기본 정보 추출 및 스케일 계산
        source_proportions = self.bone_calculator.calculate(source_keypoints, source_scores)
        reference_directions = self.direction_extractor.extract(reference_keypoints, reference_scores)
        global_scale = self._calculate_global_scale(source_proportions, reference_keypoints, reference_scores)
        
        corrected_lengths = self._correct_bone_lengths(source_proportions, global_scale, reference_keypoints)

        # 4. 전이 시작
        num_kpts = len(source_keypoints)
        trans_kpts = np.zeros((num_kpts, 2))
        trans_scores = np.zeros(num_kpts)
        transfer_log = {}
        processed = set()

        # (A) 어깨 (Anchor)
        self._transfer_shoulders(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, global_scale, processed, transfer_log)

        # (B) 척추 -> 골반
        self._transfer_torso(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, global_scale, processed, transfer_log)

        # (C) 팔 (상반신)
        self._transfer_chain(self.upper_body_order, trans_kpts, trans_scores, corrected_lengths, reference_keypoints, global_scale, processed, transfer_log, reference_scores)

        # (D) 다리 (하반신 - 조건부)
        if src_lower_valid and ref_lower_valid:
            self._transfer_chain(self.lower_body_order, trans_kpts, trans_scores, corrected_lengths, reference_keypoints, global_scale, processed, transfer_log, reference_scores)
        else:
            print("[DEBUG] 하반신 전이 스킵 (Source 또는 Reference 하반신 누락)")

        # (E) 얼굴 (설정 적용)
        if self.config.use_face:
            self._transfer_face(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, global_scale, transfer_log)

        # (F) 손
        if self.config.use_hands:
            self._transfer_hands(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, global_scale, transfer_log)

        # 5. 클리핑
        if target_image_size:
            h, w = target_image_size
            for i in range(num_kpts):
                x, y = trans_kpts[i]
                if x < 0 or x > w or y < 0 or y > h:
                    trans_scores[i] = 0

        return TransferResult(trans_kpts, trans_scores, corrected_lengths, {}, transfer_log)

    def _check_lower_body_valid(self, kpts, scores, img_h):
        # 1. 엉덩이/무릎 점수 체크
        indices = [BODY_KEYPOINTS['left_knee'], BODY_KEYPOINTS['right_knee']]
        max_score = max([scores[i] for i in indices])
        
        if max_score < self.config.lower_body_confidence_threshold:
            return False

        # 2. Y좌표 경계 체크 (이미지 바닥 5% 이내면 무효)
        # 엉덩이, 무릎이 이미지 끝에 너무 붙어있으면 모델이 억지로 추정한 것임
        margin = img_h * self.config.lower_body_margin_ratio
        limit = img_h - margin
        
        l_knee_y = kpts[BODY_KEYPOINTS['left_knee']][1]
        r_knee_y = kpts[BODY_KEYPOINTS['right_knee']][1]
        
        # 무릎 중 하나라도 경계를 넘으면(너무 아래면) False
        if (l_knee_y > limit and scores[BODY_KEYPOINTS['left_knee']] > 0.1) or \
           (r_knee_y > limit and scores[BODY_KEYPOINTS['right_knee']] > 0.1):
            return False
            
        return True

    def _transfer_shoulders(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, scale, processed, log):
        # 어깨 Anchor + Ref 각도
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        
        src_center = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
        src_width = calculate_distance(s_kpts[l_sh], s_kpts[r_sh])
        
        ref_vec = r_kpts[r_sh] - r_kpts[l_sh]
        ref_dir = normalize_vector(ref_vec)
        
        t_kpts[l_sh] = src_center - ref_dir * (src_width / 2)
        t_kpts[r_sh] = src_center + ref_dir * (src_width / 2)
        t_scores[l_sh] = t_scores[r_sh] = 0.9
        
        processed.add(l_sh); processed.add(r_sh)
        log['shoulder'] = 'anchor'

    def _transfer_torso(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, r_scores, scale, processed, log):
        # 척추 -> 골반
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
        
        t_neck = (t_kpts[l_sh] + t_kpts[r_sh]) / 2
        r_neck = (r_kpts[l_sh] + r_kpts[r_sh]) / 2
        r_root = (r_kpts[l_hip] + r_kpts[r_hip]) / 2
        
        spine_vec = r_root - r_neck
        spine_dir = normalize_vector(spine_vec)
        
        # Source 엉덩이가 있으면 Source 길이, 없으면 Ref * Scale
        if s_scores[l_hip] > 0.3:
            s_root = (s_kpts[l_hip] + s_kpts[r_hip]) / 2
            s_neck = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
            spine_len = calculate_distance(s_root, s_neck)
        else:
            spine_len = calculate_distance(r_root, r_neck) * scale
            
        new_root = t_neck + spine_dir * spine_len
        
        # 골반 배치
        r_hip_vec = r_kpts[r_hip] - r_kpts[l_hip]
        r_hip_dir = normalize_vector(r_hip_vec)
        
        # 골반 너비
        if s_scores[l_hip] > 0.3:
            hip_width = calculate_distance(s_kpts[l_hip], s_kpts[r_hip])
        else:
            hip_width = calculate_distance(r_kpts[l_hip], r_kpts[r_hip]) * scale
            
        t_kpts[l_hip] = new_root - r_hip_dir * (hip_width / 2)
        t_kpts[r_hip] = new_root + r_hip_dir * (hip_width / 2)
        t_scores[l_hip] = t_scores[r_hip] = 0.9
        processed.add(l_hip); processed.add(r_hip)

    def _transfer_chain(self, order, t_kpts, t_scores, lengths, r_kpts, scale, processed, log, r_scores):
        for _, p_name, children in order:
            p_idx = get_keypoint_index(p_name)
            if p_idx not in processed: continue
            
            p_pos = t_kpts[p_idx]
            for c_name in children:
                c_idx = get_keypoint_index(c_name)
                # Ref 점수 체크
                if r_scores[c_idx] < 0.1: continue
                
                # 길이 & 방향
                bone = f"{p_name}_{c_name}"
                alt = f"{c_name}_{p_name}"
                length = lengths.get(bone) or lengths.get(alt) or \
                         calculate_distance(r_kpts[p_idx], r_kpts[c_idx]) * scale
                
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                direct = normalize_vector(vec)
                
                t_kpts[c_idx] = p_pos + direct * length
                t_scores[c_idx] = 0.8
                processed.add(c_idx)
                log[c_name] = 'chain'

    def _transfer_face(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, scale, log):
        # 얼굴 회전 및 부위별 On/Off
        if not self.config.face_rendering.enabled: return

        nose = BODY_KEYPOINTS['nose']
        l_eye, r_eye = BODY_KEYPOINTS['left_eye'], BODY_KEYPOINTS['right_eye']
        
        # 회전 매트릭스 계산
        s_vec = s_kpts[r_eye] - s_kpts[l_eye]
        r_vec = r_kpts[r_eye] - r_kpts[l_eye]
        angle = np.arctan2(r_vec[1], r_vec[0]) - np.arctan2(s_vec[1], s_vec[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        anchor = s_kpts[nose]
        
        # 1. 68 랜드마크
        src_face_nose = s_kpts[FACE_START_IDX + 30] # 코끝
        for i in range(FACE_START_IDX, FACE_END_IDX + 1):
            local_idx = i - FACE_START_IDX
            part_name = self._get_face_part_name(local_idx)
            
            # YAML 설정 체크 (여기가 핵심!)
            part_config = self.config.face_rendering.parts.get(part_name)
            if part_config and not part_config.enabled:
                t_scores[i] = 0.0 # 렌더링 안 함
                continue
                
            if s_scores[i] > 0.3:
                rel = s_kpts[i] - src_face_nose
                t_kpts[i] = anchor + rot_mat @ rel
                t_scores[i] = s_scores[i]
                log[f'face_{i}'] = 'rot'

        # 2. COCO 기본 얼굴 키포인트 (눈, 귀, 코)
        head_parts = [nose, l_eye, r_eye, BODY_KEYPOINTS['left_ear'], BODY_KEYPOINTS['right_ear']]
        for idx in head_parts:
             if s_scores[idx] > 0.3:
                 rel = s_kpts[idx] - s_kpts[nose]
                 t_kpts[idx] = anchor + rot_mat @ rel
                 t_scores[idx] = s_scores[idx]

    def _transfer_hands(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, r_scores, scale, log):
        for is_left in [True, False]:
            w_name = 'left_wrist' if is_left else 'right_wrist'
            w_idx = BODY_KEYPOINTS[w_name]
            if t_scores[w_idx] == 0: continue
            
            start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
            ref_w = r_kpts[w_idx]
            
            for i in range(21):
                idx = start + i
                if r_scores[idx] > 0.2:
                    rel = r_kpts[idx] - ref_w
                    t_kpts[idx] = t_kpts[w_idx] + rel * scale
                    t_scores[idx] = 0.9

    def _get_face_part_name(self, idx):
        for name, r in FACE_PARTS_IDX.items():
            if idx in r: return name
        return None
    
    def _calculate_global_scale(self, src_props, ref_kpts, ref_scores):
        src_w = src_props.shoulder_width
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        if src_w > 0 and ref_scores[l_sh] > 0.3:
            ref_w = calculate_distance(ref_kpts[l_sh], ref_kpts[r_sh])
            return src_w / ref_w if ref_w > 0 else 1.0
        return 1.0
    
    def _correct_bone_lengths(self, props, scale, ref_kpts):
        lengths = {}
        for n, i in props.bone_lengths.items():
            if i.is_valid: lengths[n] = i.length
        # 대칭 등 추가 로직 생략 (기존 유지)
        return lengths