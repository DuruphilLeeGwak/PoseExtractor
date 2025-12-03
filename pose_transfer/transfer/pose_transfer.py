"""
포즈 전이 엔진 v6: 프레임 밖 키포인트 사전 절삭

핵심 변경:
1. 전이 전에 Source/Reference 원본 키포인트의 경계/신뢰도 체크
2. 경계(하단 10%) 또는 저신뢰(<2.0) 키포인트는 전이에서 제외
3. 계층적 절삭: 부모가 절삭되면 자식도 자동 절삭
"""
import numpy as np
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass, field

from ..extractors.keypoint_constants import (
    BODY_KEYPOINTS,
    FEET_KEYPOINTS,
    LEFT_HAND_START_IDX,
    RIGHT_HAND_START_IDX,
    FACE_START_IDX,
    FACE_END_IDX,
    get_keypoint_index
)
from ..analyzers.bone_calculator import BoneCalculator
from ..analyzers.direction_extractor import DirectionExtractor
from ..utils.geometry import normalize_vector, calculate_distance


# ============================================================
# 설정 클래스
# ============================================================
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
        if not data:
            return config
        config.enabled = data.get('enabled', True)
        parts_data = data.get('parts', {})
        for part in ['jawline', 'left_eyebrow', 'right_eyebrow', 'nose',
                     'left_eye', 'right_eye', 'mouth_outer', 'mouth_inner']:
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
    
    # 하반신 절삭 설정 (핵심!)
    lower_body_confidence_threshold: float = 2.0  # 이 미만이면 저신뢰로 판단
    lower_body_margin_ratio: float = 0.10  # 이미지 하단 10% = 경계 영역
    
    # 화면 밖 허용 범위 (최종 클리핑용)
    visibility_margin: float = 0.15
    
    face_rendering: FaceRenderingConfig = field(default_factory=FaceRenderingConfig)


@dataclass
class TransferResult:
    keypoints: np.ndarray
    scores: np.ndarray
    source_bone_lengths: Dict[str, float]
    reference_directions: Dict[str, np.ndarray]
    transfer_log: Dict[str, str] = field(default_factory=dict)
    
    def to_json(self) -> Dict:
        from ..utils.io import convert_to_openpose_format
        valid_pts = self.keypoints[self.scores > 0.1]
        if len(valid_pts) > 0:
            h = int(np.max(valid_pts[:, 1]) * 1.2)
            w = int(np.max(valid_pts[:, 0]) * 1.2)
        else:
            h, w = 1000, 1000
        return convert_to_openpose_format(
            self.keypoints[np.newaxis, ...],
            self.scores[np.newaxis, ...],
            (h, w)
        )


# 하반신 계층 구조 (부모 -> 자식들)
LOWER_BODY_HIERARCHY = {
    'left_hip': ['left_knee'],
    'right_hip': ['right_knee'],
    'left_knee': ['left_ankle'],
    'right_knee': ['right_ankle'],
    'left_ankle': ['left_big_toe', 'left_small_toe', 'left_heel'],
    'right_ankle': ['right_big_toe', 'right_small_toe', 'right_heel'],
}

# 얼굴 파트 인덱스
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
        
        # YAML 설정 로드
        if yaml_config:
            if 'face_rendering' in yaml_config:
                self.config.face_rendering = FaceRenderingConfig.from_dict(yaml_config['face_rendering'])
            if 'transfer' in yaml_config:
                tc = yaml_config['transfer']
                self.config.visibility_margin = tc.get('visibility_margin', 0.15)
                self.config.lower_body_confidence_threshold = tc.get('lower_body_confidence_threshold', 2.0)
                self.config.lower_body_margin_ratio = tc.get('lower_body_margin_ratio', 0.10)
                print(f"[CONFIG] lower_body_confidence_threshold: {self.config.lower_body_confidence_threshold}")
                print(f"[CONFIG] lower_body_margin_ratio: {self.config.lower_body_margin_ratio}")
        
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

    # ============================================================
    # 핵심: 원본 키포인트 경계/신뢰도 체크 (전이 전에 호출)
    # ============================================================
    def _get_invalid_indices(self, kpts: np.ndarray, scores: np.ndarray, img_h: int, label: str) -> Set[int]:
        """
        절삭해야 할 키포인트 인덱스 집합 반환
        
        절삭 조건:
        1. Y좌표가 이미지 하단 경계(90%)를 넘음
        2. 신뢰도가 threshold 미만
        3. 부모가 절삭되면 자식도 절삭 (계층적)
        """
        invalid = set()
        boundary_y = img_h * (1 - self.config.lower_body_margin_ratio)  # 90% 지점
        conf_thresh = self.config.lower_body_confidence_threshold
        
        print(f"\n[{label}] 경계 체크 (boundary_y={boundary_y:.0f}, conf_thresh={conf_thresh})")
        
        # 체크할 하반신 부위 (순서 중요: 부모 먼저)
        lower_body_parts = [
            ('left_hip', BODY_KEYPOINTS['left_hip']),
            ('right_hip', BODY_KEYPOINTS['right_hip']),
            ('left_knee', BODY_KEYPOINTS['left_knee']),
            ('right_knee', BODY_KEYPOINTS['right_knee']),
            ('left_ankle', BODY_KEYPOINTS['left_ankle']),
            ('right_ankle', BODY_KEYPOINTS['right_ankle']),
        ]
        
        # 발 키포인트
        feet_parts = [
            ('left_big_toe', FEET_KEYPOINTS['left_big_toe']),
            ('left_small_toe', FEET_KEYPOINTS['left_small_toe']),
            ('left_heel', FEET_KEYPOINTS['left_heel']),
            ('right_big_toe', FEET_KEYPOINTS['right_big_toe']),
            ('right_small_toe', FEET_KEYPOINTS['right_small_toe']),
            ('right_heel', FEET_KEYPOINTS['right_heel']),
        ]
        
        for part_name, idx in lower_body_parts + feet_parts:
            y = kpts[idx][1]
            conf = scores[idx]
            
            # 절삭 조건
            over_boundary = y >= boundary_y
            low_conf = conf < conf_thresh
            
            if over_boundary or low_conf:
                invalid.add(idx)
                
                # 자식들도 모두 절삭 (계층적 절삭)
                self._invalidate_children(part_name, invalid)
                
                reason = []
                if over_boundary:
                    reason.append(f"Y={y:.0f}>=경계{boundary_y:.0f}")
                if low_conf:
                    reason.append(f"conf={conf:.2f}<{conf_thresh}")
                print(f"  [{label}] {part_name} (idx={idx}) 무효: {', '.join(reason)}")
        
        return invalid
    
    def _invalidate_children(self, parent_name: str, invalid: Set[int]):
        """부모가 무효화되면 모든 자식도 재귀적으로 무효화"""
        if parent_name not in LOWER_BODY_HIERARCHY:
            return
        
        for child_name in LOWER_BODY_HIERARCHY[parent_name]:
            if child_name in BODY_KEYPOINTS:
                child_idx = BODY_KEYPOINTS[child_name]
            elif child_name in FEET_KEYPOINTS:
                child_idx = FEET_KEYPOINTS[child_name]
            else:
                continue
            
            invalid.add(child_idx)
            self._invalidate_children(child_name, invalid)

    # ============================================================
    # 메인 전이 함수
    # ============================================================
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
        
        # 1. 이미지 크기 추정
        if source_image_size is None:
            valid_pts = source_keypoints[source_scores > 0.3]
            src_h = int(np.max(valid_pts[:, 1]) * 1.05) if len(valid_pts) > 0 else 1000
            src_w = int(np.max(valid_pts[:, 0]) * 1.05) if len(valid_pts) > 0 else 1000
            source_image_size = (src_h, src_w)
        
        if reference_image_size is None:
            valid_pts = reference_keypoints[reference_scores > 0.3]
            ref_h = int(np.max(valid_pts[:, 1]) * 1.05) if len(valid_pts) > 0 else 1000
            ref_w = int(np.max(valid_pts[:, 0]) * 1.05) if len(valid_pts) > 0 else 1000
            reference_image_size = (ref_h, ref_w)
        
        src_h, src_w = source_image_size
        ref_h, ref_w = reference_image_size
        
        print(f"\n[DEBUG] Source size: {src_h}x{src_w}, Reference size: {ref_h}x{ref_w}")

        # ★★★ 핵심: 전이 전에 원본 키포인트 유효성 체크 ★★★
        print("\n" + "="*60)
        print("[STEP 1] 원본 키포인트 유효성 체크")
        print("="*60)
        
        src_invalid = self._get_invalid_indices(source_keypoints, source_scores, src_h, "Source")
        ref_invalid = self._get_invalid_indices(reference_keypoints, reference_scores, ref_h, "Reference")
        
        # 최종 무효 집합 (둘 중 하나라도 무효면 전이하지 않음)
        final_invalid = src_invalid | ref_invalid
        print(f"\n[최종 무효 인덱스] {sorted(final_invalid)}")

        # 2. 정보 추출
        source_proportions = self.bone_calculator.calculate(source_keypoints, source_scores)
        global_scale = self._calculate_global_scale(source_proportions, reference_keypoints, reference_scores)
        corrected_lengths = self._correct_bone_lengths(source_proportions, global_scale, reference_keypoints)
        
        print(f"\n[DEBUG] Global scale: {global_scale:.4f}")

        # 3. 전이 시작
        print("\n" + "="*60)
        print("[STEP 2] 전이 실행")
        print("="*60)
        
        num_kpts = len(source_keypoints)
        trans_kpts = np.zeros((num_kpts, 2))
        trans_scores = np.zeros(num_kpts)
        transfer_log = {}
        processed = set()

        # (A) 어깨 (항상 전이)
        self._transfer_shoulders(trans_kpts, trans_scores, source_keypoints, source_scores,
                                 reference_keypoints, global_scale, processed, transfer_log)

        # (B) 척추 -> 골반 (무효 체크 포함)
        self._transfer_torso(trans_kpts, trans_scores, source_keypoints, source_scores,
                            reference_keypoints, reference_scores, global_scale, 
                            processed, transfer_log, final_invalid)

        # (C) 팔 (상체 - 무효 체크 없음)
        self._transfer_chain(self.upper_body_order, trans_kpts, trans_scores,
                            corrected_lengths, reference_keypoints, global_scale,
                            processed, transfer_log, reference_scores, set())

        # (D) 하반신 (무효 체크 포함!)
        self._transfer_chain(self.lower_body_order, trans_kpts, trans_scores,
                            corrected_lengths, reference_keypoints, global_scale,
                            processed, transfer_log, reference_scores, final_invalid)
        
        # (E) 발 (무효 체크 포함!)
        self._transfer_feet(trans_kpts, trans_scores, reference_keypoints, reference_scores,
                           global_scale, processed, transfer_log, final_invalid)

        # (F) 얼굴
        if self.config.use_face and self.config.face_rendering.enabled:
            self._transfer_face(trans_kpts, trans_scores, source_keypoints, source_scores,
                               reference_keypoints, global_scale, transfer_log)

        # (G) 손
        if self.config.use_hands:
            self._transfer_hands(trans_kpts, trans_scores, source_keypoints, source_scores,
                                reference_keypoints, reference_scores, global_scale, transfer_log)

        # (H) 최종 클리핑 (혹시 남은 화면 밖 키포인트 제거)
        self._final_clipping(trans_kpts, trans_scores, src_h, src_w)

        return TransferResult(trans_kpts, trans_scores, corrected_lengths, {}, transfer_log)

    # ============================================================
    # 전이 함수들
    # ============================================================
    def _transfer_shoulders(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, scale, processed, log):
        """어깨: Source 위치/너비 + Reference 각도"""
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        
        src_center = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
        src_width = calculate_distance(s_kpts[l_sh], s_kpts[r_sh])
        
        ref_vec = r_kpts[r_sh] - r_kpts[l_sh]
        ref_dir = normalize_vector(ref_vec)
        
        t_kpts[l_sh] = src_center - ref_dir * (src_width / 2)
        t_kpts[r_sh] = src_center + ref_dir * (src_width / 2)
        t_scores[l_sh] = t_scores[r_sh] = max(s_scores[l_sh], s_scores[r_sh])
        
        processed.add(l_sh)
        processed.add(r_sh)
        log['shoulder'] = 'anchor_ref_angle'
        print(f"  [전이] 어깨 완료")

    def _transfer_torso(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, r_scores, 
                        scale, processed, log, invalid: Set[int]):
        """척추 -> 골반 (무효 체크 포함)"""
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
        
        # 골반이 둘 다 무효면 스킵
        if l_hip in invalid and r_hip in invalid:
            print(f"  [스킵] 골반 무효 - torso 전이 생략")
            return
        
        t_neck = (t_kpts[l_sh] + t_kpts[r_sh]) / 2
        r_neck = (r_kpts[l_sh] + r_kpts[r_sh]) / 2
        r_root = (r_kpts[l_hip] + r_kpts[r_hip]) / 2
        
        spine_vec = r_root - r_neck
        spine_dir = normalize_vector(spine_vec)
        
        # Source 척추 길이 사용 (가능하면)
        if s_scores[l_hip] > 0.3 and s_scores[r_hip] > 0.3:
            s_neck = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
            s_root = (s_kpts[l_hip] + s_kpts[r_hip]) / 2
            spine_len = calculate_distance(s_root, s_neck)
        else:
            spine_len = calculate_distance(r_root, r_neck) * scale
        
        new_root = t_neck + spine_dir * spine_len
        
        # 골반 방향
        r_hip_vec = r_kpts[r_hip] - r_kpts[l_hip]
        r_hip_dir = normalize_vector(r_hip_vec)
        
        if s_scores[l_hip] > 0.3 and s_scores[r_hip] > 0.3:
            hip_width = calculate_distance(s_kpts[l_hip], s_kpts[r_hip])
        else:
            hip_width = calculate_distance(r_kpts[l_hip], r_kpts[r_hip]) * scale
        
        t_kpts[l_hip] = new_root - r_hip_dir * (hip_width / 2)
        t_kpts[r_hip] = new_root + r_hip_dir * (hip_width / 2)
        
        # 무효가 아닌 것만 점수 부여
        if l_hip not in invalid:
            t_scores[l_hip] = 0.9
            processed.add(l_hip)
        if r_hip not in invalid:
            t_scores[r_hip] = 0.9
            processed.add(r_hip)
        
        log['torso'] = 'spine_calc'
        print(f"  [전이] 골반 완료 (l_hip valid: {l_hip not in invalid}, r_hip valid: {r_hip not in invalid})")

    def _transfer_chain(self, order, t_kpts, t_scores, lengths, r_kpts, scale, 
                        processed, log, r_scores, invalid: Set[int]):
        """체인 전이 (무효 체크 포함)"""
        for _, p_name, children in order:
            p_idx = get_keypoint_index(p_name)
            
            # 부모가 처리되지 않았거나 무효면 스킵
            if p_idx not in processed:
                continue
            if p_idx in invalid:
                print(f"  [스킵] {p_name} 무효")
                continue
            
            p_pos = t_kpts[p_idx]
            
            for c_name in children:
                c_idx = get_keypoint_index(c_name)
                
                # 자식이 무효면 스킵
                if c_idx in invalid:
                    print(f"  [스킵] {c_name} 무효")
                    continue
                
                # Reference에서 신뢰도 체크
                if r_scores[c_idx] < 0.1:
                    continue
                
                # 본 길이 계산
                bone = f"{p_name}_{c_name}"
                length = lengths.get(bone) or calculate_distance(r_kpts[p_idx], r_kpts[c_idx]) * scale
                
                # 방향
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                direction = normalize_vector(vec)
                
                t_kpts[c_idx] = p_pos + direction * length
                t_scores[c_idx] = 0.8
                processed.add(c_idx)
                log[c_name] = 'chain'
                print(f"  [전이] {c_name} 완료")

    def _transfer_feet(self, t_kpts, t_scores, r_kpts, r_scores, scale, 
                       processed, log, invalid: Set[int]):
        """발 키포인트 전이 (무효 체크 포함)"""
        feet_connections = [
            ('left_ankle', ['left_heel', 'left_big_toe']),
            ('right_ankle', ['right_heel', 'right_big_toe']),
            ('left_big_toe', ['left_small_toe']),
            ('right_big_toe', ['right_small_toe']),
        ]
        
        for p_name, children in feet_connections:
            if p_name in BODY_KEYPOINTS:
                p_idx = BODY_KEYPOINTS[p_name]
            else:
                p_idx = FEET_KEYPOINTS[p_name]
            
            if p_idx not in processed or p_idx in invalid:
                continue
            
            p_pos = t_kpts[p_idx]
            
            for c_name in children:
                c_idx = FEET_KEYPOINTS[c_name]
                
                if c_idx in invalid:
                    continue
                
                if r_scores[c_idx] < 0.1:
                    continue
                
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                t_kpts[c_idx] = p_pos + vec * scale
                t_scores[c_idx] = 0.7
                processed.add(c_idx)

    def _transfer_face(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, scale, log):
        """얼굴: Source 비율 + Reference 각도"""
        if not self.config.face_rendering.enabled:
            return
        
        nose = BODY_KEYPOINTS['nose']
        l_eye, r_eye = BODY_KEYPOINTS['left_eye'], BODY_KEYPOINTS['right_eye']
        
        # 회전 각도 계산
        s_vec = s_kpts[r_eye] - s_kpts[l_eye]
        r_vec = r_kpts[r_eye] - r_kpts[l_eye]
        
        if np.linalg.norm(s_vec) < 1 or np.linalg.norm(r_vec) < 1:
            return
        
        angle = np.arctan2(r_vec[1], r_vec[0]) - np.arctan2(s_vec[1], s_vec[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        anchor = s_kpts[nose]
        
        # 머리 키포인트
        for idx in [nose, l_eye, r_eye, BODY_KEYPOINTS['left_ear'], BODY_KEYPOINTS['right_ear']]:
            if s_scores[idx] > 0.3:
                rel = s_kpts[idx] - s_kpts[nose]
                t_kpts[idx] = anchor + rot_mat @ rel
                t_scores[idx] = s_scores[idx]
        
        # 얼굴 68점
        src_face_nose = s_kpts[FACE_START_IDX + 30]
        for i in range(FACE_START_IDX, FACE_END_IDX + 1):
            local_idx = i - FACE_START_IDX
            part_name = self._get_face_part_name(local_idx)
            
            if part_name and not self.config.face_rendering.parts.get(part_name, FacePartConfig()).enabled:
                t_scores[i] = 0.0
                continue
            
            if s_scores[i] > 0.3:
                rel = s_kpts[i] - src_face_nose
                t_kpts[i] = anchor + rot_mat @ rel
                t_scores[i] = s_scores[i]

    def _transfer_hands(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, r_scores, scale, log):
        """손 전이"""
        for is_left in [True, False]:
            w_name = 'left_wrist' if is_left else 'right_wrist'
            w_idx = BODY_KEYPOINTS[w_name]
            
            if t_scores[w_idx] < 0.1:
                continue
            
            start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
            ref_w = r_kpts[w_idx]
            
            for i in range(21):
                idx = start + i
                if r_scores[idx] > 0.2:
                    rel = r_kpts[idx] - ref_w
                    t_kpts[idx] = t_kpts[w_idx] + rel * scale
                    t_scores[idx] = 0.9

    def _final_clipping(self, kpts, scores, img_h, img_w):
        """최종 화면 밖 클리핑"""
        limit_y = img_h * (1 + self.config.visibility_margin)
        limit_x = img_w * (1 + self.config.visibility_margin)
        
        clipped_count = 0
        for i in range(len(kpts)):
            x, y = kpts[i]
            if scores[i] > 0 and (x < -img_w * 0.3 or x > limit_x or y < -img_h * 0.1 or y > limit_y):
                scores[i] = 0.0
                clipped_count += 1
        
        if clipped_count > 0:
            print(f"  [최종 클리핑] {clipped_count}개 키포인트 제거")

    # ============================================================
    # 유틸리티
    # ============================================================
    def _get_face_part_name(self, idx):
        for name, r in FACE_PARTS_IDX.items():
            if idx in r:
                return name
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
        for n, info in props.bone_lengths.items():
            if info.is_valid:
                lengths[n] = info.length
        
        # 대칭 보정
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
        
        return lengths