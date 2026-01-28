"""
PoseTransferEngine Module (v26 - Rigid Face Rotation)

변경사항:
- [Critical] Face 5와 Face 68의 분리된 로직 폐기 -> 통합 강체 회전(Rigid Rotation) 도입
- [Fix] 얼굴 동기화 실패 해결: 얼굴 전체를 '목(Neck)' 기준으로 통째로 회전시킴
- [Fix] 얼굴 축소 문제 해결: Src의 랜드마크 간 거리를 100% 보존 (Identity 유지)
- Ref의 고개 각도(Neck->Nose Angle)를 완벽하게 반영
"""
import numpy as np
from typing import Dict, Tuple, Optional, List

from ..analyzers.bone_calculator import BoneCalculator
from ..analyzers.direction_extractor import DirectionExtractor
from ..utils.geometry import calculate_distance, rotate_point
from ..extractors.keypoint_constants import BODY_KEYPOINTS
from ..logic.keypoint_generator import KeypointGenerator

from .config import TransferConfig, TransferResult, FaceRenderingConfig
from .logic import BodyTransfer, FaceTransfer, HandTransfer


class PoseTransferEngine:
    def __init__(self, config: Optional[TransferConfig] = None, yaml_config: Optional[dict] = None):
        self.config = config or TransferConfig()
        self.yaml_config = yaml_config or {}
        self._face_transfer_debug = {}  # Face Transfer 디버그 정보 저장
        
        if yaml_config:
            if 'face_rendering' in yaml_config:
                self.config.face_rendering = FaceRenderingConfig.from_dict(yaml_config['face_rendering'])
                self.config.use_face = yaml_config['face_rendering'].get('enabled', True)
        
        self.bone_calculator = BoneCalculator(confidence_threshold=self.config.confidence_threshold)
        self.direction_extractor = DirectionExtractor(confidence_threshold=self.config.confidence_threshold)
        self.keypoint_generator = KeypointGenerator(confidence_threshold=self.config.confidence_threshold)
        
        self.body_logic = BodyTransfer()
        self.face_logic = FaceTransfer(self.config)
        self.hand_logic = HandTransfer()

    def transfer(
        self,
        source_keypoints: np.ndarray, source_scores: np.ndarray,
        reference_keypoints: np.ndarray, reference_scores: np.ndarray,
        source_image_size: Optional[Tuple[int, int]] = None,
        reference_image_size: Optional[Tuple[int, int]] = None,
        target_image_size: Optional[Tuple[int, int]] = None,
        alignment_case: Optional[str] = None,
        source_depths: Optional[np.ndarray] = None,
        reference_depths: Optional[np.ndarray] = None,
        depth_z_scale: Optional[float] = None,
    ) -> TransferResult:
        
        print("\n" + "="*70)
        print("🔍 [DEBUG] PoseTransferEngine v30 (Dual Scaling: Torso-Width, Legs-Length)")
        print("="*70)
        
        # 0. Ref 기준으로 Src 누락 키포인트 생성
        source_keypoints, source_scores = self.keypoint_generator.generate_missing_keypoints(
            source_keypoints, source_scores, reference_keypoints, reference_scores
        )
        
        # 1. 기본 설정
        if source_image_size is None:
            max_y = np.max(source_keypoints[:, 1])
            source_image_size = (int(max_y * 1.1), int(np.max(source_keypoints[:, 0])))

        # 2. 하반신 유효성 체크
        ref_lower_valid = True
        if alignment_case in ['F_H', 'H_H']:
            ref_lower_valid = False
        else:
            ref_knee_score = min(
                reference_scores[BODY_KEYPOINTS['left_knee']], 
                reference_scores[BODY_KEYPOINTS['right_knee']]
            )
            if ref_knee_score < 0.1:
                ref_lower_valid = False
        
        # 3. 데이터 추출 & 스케일 계산
        source_proportions = self.bone_calculator.calculate(source_keypoints, source_scores)
        global_scale = self._calculate_global_scale(source_proportions, reference_keypoints, reference_scores)
        
        # 뼈 길이 보정
        corrected_lengths = self._correct_bone_lengths(
            source_proportions, 
            global_scale, 
            reference_keypoints, 
            reference_scores
        )
        
        hand_scale_ratio = self._calculate_hand_scale_ratio(source_keypoints, source_scores, reference_keypoints, reference_scores)
        
        # ----------------------------------------------------------------------
        # 4. 결과 배열 초기화
        # ----------------------------------------------------------------------
        num_kpts = len(source_keypoints)
        trans_kpts = np.zeros((num_kpts, 2))
        trans_scores = np.zeros(num_kpts)
        transfer_log = {'face_parts': []}
        processed = set()

        # ══════════════════════════════════════════════════════════════
        # STEP 1: Body Transfer (Torso Width Scaling)
        # ══════════════════════════════════════════════════════════════
        print("\n🏃 Body Transfer...")
        
        # [핵심 수정] 몸통 비율(Torso Ratio)을 '높이'가 아닌 '어깨 너비' 기준으로 계산
        ls, rs = 5, 6 
        src_width = 0
        ref_width = 0
        
        if source_scores[ls] > 0.1 and source_scores[rs] > 0.1:
            src_width = np.linalg.norm(source_keypoints[ls] - source_keypoints[rs])
            
        if reference_scores[ls] > 0.1 and reference_scores[rs] > 0.1:
            ref_width = np.linalg.norm(reference_keypoints[ls] - reference_keypoints[rs])
            
        # 어깨 없으면 힙 사용
        if src_width == 0 or ref_width == 0:
            lh, rh = 11, 12
            if source_scores[lh] > 0.1 and source_scores[rh] > 0.1:
                src_width = np.linalg.norm(source_keypoints[lh] - source_keypoints[rh])
            if reference_scores[lh] > 0.1 and reference_scores[rh] > 0.1:
                ref_width = np.linalg.norm(reference_keypoints[lh] - reference_keypoints[rh])

        # 최종 비율
        if ref_width > 0:
            torso_ratio = src_width / ref_width
            torso_ratio = np.clip(torso_ratio, 0.5, 2.0)
            print(f"   📐 Torso Scale Strategy: Width-Based (Src={src_width:.1f} / Ref={ref_width:.1f} -> {torso_ratio:.3f})")
        else:
            torso_ratio = 1.0
            print(f"   ⚠️ Torso Scale Strategy: Default (Width calc failed)")

        # 기존 로직 수행
        self.body_logic.transfer_shoulders(
            trans_kpts, trans_scores, source_keypoints, source_scores,
            reference_keypoints, torso_ratio=torso_ratio, processed=processed,
            log=transfer_log, r_scores=reference_scores,
        )
        
        self.body_logic.transfer_torso(
            trans_kpts, trans_scores, source_keypoints, source_scores, 
            reference_keypoints, hand_scale_ratio, processed, transfer_log
        )
        
        # Upper Body Chain
        self.body_logic.transfer_chain(
            trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, 
            hand_scale_ratio, processed, transfer_log, is_lower=False,
            src_proportions=source_proportions, ref_proportions=self.bone_calculator.calculate(reference_keypoints, reference_scores),
            src_depths=source_depths, ref_depths=reference_depths, depth_z_scale=depth_z_scale
        )
        
        if getattr(self.config, 'enable_upper_ratio_tuning', True):
            self.body_logic.fine_tune_upper_ratio(
                trans_kpts, trans_scores, source_keypoints, source_scores,
                reference_keypoints, reference_scores, processed, transfer_log,
                source_depths=source_depths, depth_z_scale=depth_z_scale
            )
        
        # Lower Body Chain
        if ref_lower_valid:
            self.body_logic.transfer_chain(
                trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, 
                hand_scale_ratio, processed, transfer_log, is_lower=True,
                src_proportions=source_proportions, ref_proportions=self.bone_calculator.calculate(reference_keypoints, reference_scores),
                src_depths=source_depths, ref_depths=reference_depths, depth_z_scale=depth_z_scale
            )
            
            self.body_logic.transfer_feet(
                trans_kpts, trans_scores, source_keypoints, source_scores,
                corrected_lengths, reference_keypoints, reference_scores,
                hand_scale_ratio, processed, transfer_log
            )
            
            if getattr(self.config, 'enable_lower_ratio_tuning', True):
                self.body_logic.fine_tune_lower_ratio(
                    trans_kpts, trans_scores, source_keypoints, source_scores,
                    reference_keypoints, reference_scores, processed, transfer_log,
                    source_depths=source_depths, depth_z_scale=depth_z_scale
                )

        # ══════════════════════════════════════════════════════════════
        # STEP 1.5 ~ 4: 나머지 로직 (기존과 동일)
        # ══════════════════════════════════════════════════════════════
        self._fill_missing_from_reference(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, hand_scale_ratio, processed, transfer_log)
        
        # Face
        self._transfer_ears_from_ref(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, processed, transfer_log)
        self._transfer_body_face(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, body_scale=torso_ratio)
        if self.config.use_face:
            self._transfer_face_landmarks(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, source_depths, reference_depths, depth_z_scale, face_scale=self._face_transfer_debug.get('face_scale', 1.0))
        
        # Hands
        if self.config.use_hands:
            self._transfer_hands(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, hand_scale_ratio, transfer_log)
        
        # Global Resize
        current_face_scale = self._face_transfer_debug.get('face_scale', 1.0)
        if current_face_scale > 0 and abs(current_face_scale - 1.0) > 0.05:
            resize_factor = 1.0 / current_face_scale
            resize_factor = np.clip(resize_factor, 0.5, 1.5)
            
            ls, rs = 5, 6
            if trans_scores[ls] > 0.1 and trans_scores[rs] > 0.1:
                pivot = (trans_kpts[ls] + trans_kpts[rs]) / 2.0
            else:
                pivot = np.mean(trans_kpts[trans_scores>0.1], axis=0) if np.any(trans_scores>0.1) else np.array([0.,0.])
            
            mask = trans_scores > 0.0
            trans_kpts[mask] = pivot + (trans_kpts[mask] - pivot) * resize_factor

        return TransferResult(trans_kpts, trans_scores, corrected_lengths, {}, transfer_log)
    
    def _transfer_body_face(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        body_scale=1.0
    ):
        """
        Body Face 전이 - Neck Length matches Face Scale (Best Neck Balance)
        """
        # [상수 정의]
        NOSE = 0
        LEFT_EYE, RIGHT_EYE = 1, 2
        LEFT_EAR, RIGHT_EAR = 3, 4
        LS, RS = 5, 6
        LEFT_HIP, RIGHT_HIP = 11, 12
        
        # 0. 필수 키포인트 확인
        if trans_scores[LS] < 0.1 or trans_scores[RS] < 0.1:
            print("   ⚠️ [Body Face] Trans 어깨가 없어 전이 불가")
            return

        # =========================================================
        # 1. Scale Calculation
        # =========================================================
        def _estimate_face_width(kpts, scores):
            if scores[LEFT_EAR] > 0.1 and scores[RIGHT_EAR] > 0.1:
                return np.linalg.norm(kpts[RIGHT_EAR] - kpts[LEFT_EAR])
            if scores[LEFT_EYE] > 0.1 and scores[RIGHT_EYE] > 0.1:
                return np.linalg.norm(kpts[RIGHT_EYE] - kpts[LEFT_EYE]) * 2.2
            return 100.0

        src_face_width = _estimate_face_width(src_kpts, src_scores)
        src_neck = (src_kpts[LS] + src_kpts[RS]) / 2.0
        
        if (src_scores[LEFT_HIP] > 0.1 and src_scores[RIGHT_HIP] > 0.1):
            src_hip_center = (src_kpts[LEFT_HIP] + src_kpts[RIGHT_HIP]) / 2.0 
        else:
            src_hip_center = src_neck + np.array([0, 300])
        src_torso_height = np.linalg.norm(src_hip_center - src_neck)
        src_face_torso_ratio = src_face_width / src_torso_height if src_torso_height > 0 else 0.3
        
        trans_neck = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
        if (trans_scores[LEFT_HIP] > 0.1 and trans_scores[RIGHT_HIP] > 0.1):
            trans_hip_center = (trans_kpts[LEFT_HIP] + trans_kpts[RIGHT_HIP]) / 2.0
        else:
            trans_hip_center = trans_neck + np.array([0, 300])
        trans_torso_height = np.linalg.norm(trans_hip_center - trans_neck)
        trans_face_scale_target = src_face_torso_ratio * trans_torso_height
        
        ref_face_width = _estimate_face_width(ref_kpts, ref_scores)
        width_scale = trans_face_scale_target / ref_face_width if ref_face_width > 0 else 1.0

        def _get_vertical_ear_dist(kpts, scores, neck_pt):
            dists = []
            if scores[LEFT_EAR] > 0.1: dists.append(abs(kpts[LEFT_EAR][1] - neck_pt[1]))
            if scores[RIGHT_EAR] > 0.1: dists.append(abs(kpts[RIGHT_EAR][1] - neck_pt[1]))
            if not dists: return None
            return np.mean(dists)

        ref_neck = (ref_kpts[LS] + ref_kpts[RS]) / 2.0
        src_ear_v_dist = _get_vertical_ear_dist(src_kpts, src_scores, src_neck)
        ref_ear_v_dist = _get_vertical_ear_dist(ref_kpts, ref_scores, ref_neck)
        ear_scale = src_ear_v_dist / ref_ear_v_dist if (src_ear_v_dist and ref_ear_v_dist and ref_ear_v_dist > 1.0) else None

        face_scale = 1.0
        is_side_view = False
        if width_scale > body_scale * 1.5:
            is_side_view = True
            face_scale = ear_scale if ear_scale is not None else body_scale
        else:
            face_scale = width_scale
            is_side_view = False

        if face_scale < 0.95:
            face_scale = face_scale * 0.65 + 1.0 * 0.35

        print(f"   🔄 Final Face Scale: {face_scale:.3f} (SideView={is_side_view})")
        self._face_transfer_debug = {'src_face_torso_ratio': src_face_torso_ratio, 'face_scale': face_scale}

        # =========================================================
        # 2. Rotation & Positioning
        # =========================================================
        def _angle(vec): return np.arctan2(vec[1], vec[0])
        def _normalize(vec):
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-6 else np.zeros_like(vec)

        rotation_rad = 0.0
        if (src_scores[LEFT_EYE] > 0.1 and src_scores[RIGHT_EYE] > 0.1 and 
            ref_scores[LEFT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1):
            src_eye_vec = src_kpts[RIGHT_EYE] - src_kpts[LEFT_EYE]
            ref_eye_vec = ref_kpts[RIGHT_EYE] - ref_kpts[LEFT_EYE]
            rotation_rad = _angle(ref_eye_vec) - _angle(src_eye_vec)
        
        c, s = np.cos(rotation_rad), np.sin(rotation_rad)
        rot_mat = np.array([[c, -s], [s, c]])

        # [STEP 1] Eyes Placement (Anchor)
        trans_eye_center = None
        
        if ref_scores[LEFT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1:
            ref_eye_c = (ref_kpts[LEFT_EYE] + ref_kpts[RIGHT_EYE]) / 2.0
            ref_neck_to_eye = ref_eye_c - ref_neck
            
            target_dist = np.linalg.norm(ref_neck_to_eye)
            
            if not is_side_view and src_scores[LEFT_EYE] > 0.1 and src_scores[RIGHT_EYE] > 0.1:
                src_eye_c = (src_kpts[LEFT_EYE] + src_kpts[RIGHT_EYE]) / 2.0
                src_dist = np.linalg.norm(src_eye_c - src_neck)
                
                # [핵심 수정] 목 길이는 Face Scale을 따름 (머리 크기에 비례한 목 길이)
                target_dist = src_dist * face_scale

            ref_dir = _normalize(ref_neck_to_eye)
            trans_eye_center = trans_neck + ref_dir * target_dist
            
            if src_scores[LEFT_EYE] > 0.1 and src_scores[RIGHT_EYE] > 0.1:
                src_eye_c = (src_kpts[LEFT_EYE] + src_kpts[RIGHT_EYE]) / 2.0
                
                vec_l = src_kpts[LEFT_EYE] - src_eye_c
                vec_l_rot = np.dot(rot_mat, vec_l) * face_scale
                trans_kpts[LEFT_EYE] = trans_eye_center + vec_l_rot
                trans_scores[LEFT_EYE] = min(src_scores[LEFT_EYE], ref_scores[LEFT_EYE])
                
                vec_r = src_kpts[RIGHT_EYE] - src_eye_c
                vec_r_rot = np.dot(rot_mat, vec_r) * face_scale
                trans_kpts[RIGHT_EYE] = trans_eye_center + vec_r_rot
                trans_scores[RIGHT_EYE] = min(src_scores[RIGHT_EYE], ref_scores[RIGHT_EYE])
        
        # [STEP 2] Ears & Nose
        if trans_eye_center is not None:
            src_eye_c = (src_kpts[LEFT_EYE] + src_kpts[RIGHT_EYE]) / 2.0
            target_parts = [LEFT_EAR, RIGHT_EAR, NOSE]
            
            for part_idx in target_parts:
                if src_scores[part_idx] > 0.1:
                    src_vec = src_kpts[part_idx] - src_eye_c
                    rotated_vec = np.dot(rot_mat, src_vec) * face_scale
                    
                    trans_kpts[part_idx] = trans_eye_center + rotated_vec
                    
                    ref_score = ref_scores[part_idx] if ref_scores[part_idx] > 0 else 0.5
                    trans_scores[part_idx] = min(src_scores[part_idx], ref_score)

        print(f"   ✅ Body Face Transferred: Neck scaled by Face ({face_scale:.2f})")
    def _transfer_ears_from_ref(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        processed, transfer_log
    ):
        """
        귀를 ref 기반으로 전이 (어깨-귀 거리/각도 반영)
        
        원리:
        1. Ref의 어깨-귀 비율을 Trans에 적용
        2. Ref의 어깨-귀 방향(각도)을 Trans에 적용
        3. 이를 통해 얼굴 방향(정면/측면)이 ref를 따르게 함
        """
        LS, RS = 5, 6
        LEFT_EAR, RIGHT_EAR = 3, 4
        
        # Trans 어깨 위치 (이미 Body Transfer에서 계산됨)
        if trans_scores[LS] < 0.1 or trans_scores[RS] < 0.1:
            print("   ⚠️ [Ears] Trans 어깨가 없어 귀 전이 불가")
            return
            
        trans_left_shoulder = trans_kpts[LS]
        trans_right_shoulder = trans_kpts[RS]
        trans_shoulder_width = np.linalg.norm(trans_right_shoulder - trans_left_shoulder)
        
        # Ref 어깨-귀 벡터
        if ref_scores[LS] > 0.1 and ref_scores[RS] > 0.1:
            ref_left_shoulder = ref_kpts[LS]
            ref_right_shoulder = ref_kpts[RS]
            ref_shoulder_width = np.linalg.norm(ref_right_shoulder - ref_left_shoulder)
            
            # 왼쪽 귀
            if ref_scores[LEFT_EAR] > 0.1 and ref_shoulder_width > 0:
                ref_vec = ref_kpts[LEFT_EAR] - ref_left_shoulder
                ref_ratio = np.linalg.norm(ref_vec) / ref_shoulder_width
                ref_dir = ref_vec / np.linalg.norm(ref_vec) if np.linalg.norm(ref_vec) > 0 else ref_vec
                
                # Trans에 적용 (ref 방향 + trans 스케일)
                trans_distance = ref_ratio * trans_shoulder_width
                trans_kpts[LEFT_EAR] = trans_left_shoulder + ref_dir * trans_distance
                trans_scores[LEFT_EAR] = min(ref_scores[LEFT_EAR], 0.9)
                processed.add(LEFT_EAR)
                transfer_log['face_parts'].append('left_ear(ref_based)')
                print(f"   👂 Left Ear: Ref ratio={ref_ratio:.3f}, Trans distance={trans_distance:.1f}px")
            
            # 오른쪽 귀
            if ref_scores[RIGHT_EAR] > 0.1 and ref_shoulder_width > 0:
                ref_vec = ref_kpts[RIGHT_EAR] - ref_right_shoulder
                ref_ratio = np.linalg.norm(ref_vec) / ref_shoulder_width
                ref_dir = ref_vec / np.linalg.norm(ref_vec) if np.linalg.norm(ref_vec) > 0 else ref_vec
                
                # Trans에 적용
                trans_distance = ref_ratio * trans_shoulder_width
                trans_kpts[RIGHT_EAR] = trans_right_shoulder + ref_dir * trans_distance
                trans_scores[RIGHT_EAR] = min(ref_scores[RIGHT_EAR], 0.9)
                processed.add(RIGHT_EAR)
                transfer_log['face_parts'].append('right_ear(ref_based)')
                print(f"   👂 Right Ear: Ref ratio={ref_ratio:.3f}, Trans distance={trans_distance:.1f}px")

    

    def _transfer_face_landmarks(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        source_depths=None, reference_depths=None, depth_z_scale=1.0,
        face_scale=None
    ):
        """
        Face Landmarks 전이 (23~90: 68개 상세 얼굴 랜드마크)
        
        전략:
        - Trans 코 위치를 기준으로 Ref의 각도(방향) 적용
        - Src의 얼굴 내부 비율(코 기준 거리)을 유지
        - 스케일은 Trans/Src 어깨비로 보정
        """
        NOSE = 0
        LEFT_EYE, RIGHT_EYE = 1, 2
        LEFT_EAR, RIGHT_EAR = 3, 4
        LS, RS = 5, 6
        
        face_landmarks_indices = list(range(23, 91))
        
        # 1. Trans 코 위치 확인 (이미 _transfer_body_face에서 배치됨)
        if trans_scores[NOSE] < 0.1:
            print("   ⚠️ [Face Landmarks] Trans 코가 없어 전이 불가")
            return
        
        trans_nose = trans_kpts[NOSE]
        
        # 2. Global scale 계산 (어깨 너비 기준: Src 비율 유지 + Trans 크기 보정)
        if trans_scores[LS] > 0.1 and trans_scores[RS] > 0.1:
            trans_shoulder_width = np.linalg.norm(trans_kpts[RS] - trans_kpts[LS])
        else:
            trans_shoulder_width = 1.0
        
        if src_scores[LS] > 0.1 and src_scores[RS] > 0.1:
            src_shoulder_width = np.linalg.norm(src_kpts[RS] - src_kpts[LS])
        else:
            src_shoulder_width = 1.0
        
        # Use face_scale from _transfer_body_face if available (better accuracy)
        if face_scale is not None:
            global_scale = face_scale
            print(f"\n🎯 [Face Landmarks] Using face_scale from body_face: {global_scale:.3f}")
        else:
            # Fallback: calculate from shoulder width
            global_scale = trans_shoulder_width / src_shoulder_width if src_shoulder_width > 0 else 1.0
            print(f"\n🎯 [Face Landmarks] Calculated global_scale: {global_scale:.3f}")
        

        # 3. Ref에서 코-랜드마크 3D 벡터 가져와서 Trans에 적용 (depth가 있으면 3D, 없으면 기존 2D)
        if ref_scores[NOSE] < 0.1:
            print("   ⚠️ [Face Landmarks] Ref 코가 없어 전이 불가")
            return

        ref_nose = ref_kpts[NOSE]
        use_3d = reference_depths is not None and source_depths is not None and len(reference_depths) > NOSE and len(source_depths) > NOSE

        # Face 5점(코/눈/귀) 정합: Ref → Trans 유사변환(s,R,t)
        def _compute_similarity(src_pts, dst_pts):
            src_mean = np.mean(src_pts, axis=0)
            dst_mean = np.mean(dst_pts, axis=0)
            src_centered = src_pts - src_mean
            dst_centered = dst_pts - dst_mean

            src_var = np.sum(src_centered ** 2)
            if src_var < 1e-8:
                return 1.0, np.eye(2), dst_mean - src_mean

            cov = dst_centered.T @ src_centered / len(src_pts)
            U, S, Vt = np.linalg.svd(cov)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            scale = np.trace(np.diag(S)) / (src_var / len(src_pts))
            t = dst_mean - scale * (R @ src_mean)
            return scale, R, t

        ref_anchor = []
        trans_anchor = []
        for idx in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]:
            if ref_scores[idx] > 0.1 and trans_scores[idx] > 0.1:
                ref_anchor.append(ref_kpts[idx])
                trans_anchor.append(trans_kpts[idx])

        use_similarity = len(ref_anchor) >= 2
        if use_similarity:
            ref_anchor = np.array(ref_anchor, dtype=np.float32)
            trans_anchor = np.array(trans_anchor, dtype=np.float32)
            sim_scale, sim_R, sim_t = _compute_similarity(ref_anchor, trans_anchor)

        # --- 3D z축 jawline~nose 비율 정규화 및 clamp ---
        face_landmarks_count = 0
        # src/참조 jawline~nose z차 계산
        def get_jaw_nose_z(kpts, depths, jaw_idx, nose_idx):
            if depths is None or len(depths) <= max(jaw_idx, nose_idx):
                return None
            return float(depths[jaw_idx] - depths[nose_idx])

        # jawline 기준점: 턱 중앙(8번)
        JAW_CHIN_IDX = 23 + 8
        src_jaw_z = get_jaw_nose_z(src_kpts, source_depths, JAW_CHIN_IDX, NOSE) if use_3d else None
        ref_jaw_z = get_jaw_nose_z(ref_kpts, reference_depths, JAW_CHIN_IDX, NOSE) if use_3d else None
        # clamp 계수(최대 1.5배)
        z_clamp_ratio = 1.5

        for idx in face_landmarks_indices:
            if ref_scores[idx] > 0.1:
                if use_3d and idx < len(reference_depths) and NOSE < len(reference_depths):
                    # 3D 벡터 계산
                    ref_vec_3d = np.array([
                        ref_kpts[idx][0] - ref_nose[0],
                        ref_kpts[idx][1] - ref_nose[1],
                        (reference_depths[idx] - reference_depths[NOSE]) * depth_z_scale
                    ], dtype=np.float32)
                    # jawline~nose z비율 정규화 및 clamp
                    if idx == JAW_CHIN_IDX and src_jaw_z is not None and ref_jaw_z is not None and abs(ref_jaw_z) > 1e-6:
                        # src/ref z비율로 보정, clamp
                        z_ratio = np.clip(abs(src_jaw_z) / abs(ref_jaw_z), 1.0 / z_clamp_ratio, z_clamp_ratio)
                        ref_vec_3d[2] = ref_vec_3d[2] * z_ratio
                    # trans nose 기준 2D 위치에 3D 벡터의 xy만 적용 (z는 별도 저장/활용 가능)
                    trans_pos = np.array([
                        trans_nose[0] + ref_vec_3d[0] * global_scale,
                        trans_nose[1] + ref_vec_3d[1] * global_scale
                    ], dtype=np.float32)
                elif use_similarity:
                    sim_pos = sim_scale * (sim_R @ ref_kpts[idx]) + sim_t
                    vec_from_nose = sim_pos - trans_nose
                    trans_pos = trans_nose + vec_from_nose * (global_scale / sim_scale if sim_scale > 1e-6 else global_scale)
                else:
                    # 폴백: 이전 방식 (ref 방향 + global_scale)
                    ref_nose_lm_vec = ref_kpts[idx] - ref_nose
                    ref_nose_lm_dist = np.linalg.norm(ref_nose_lm_vec)
                    if ref_nose_lm_dist < 1e-6:
                        continue
                    ref_dir = ref_nose_lm_vec / ref_nose_lm_dist
                    trans_pos = trans_nose + ref_dir * (ref_nose_lm_dist * global_scale)

                trans_kpts[idx][0] = trans_pos[0]
                trans_kpts[idx][1] = trans_pos[1]
                trans_scores[idx] = ref_scores[idx]
                face_landmarks_count += 1

        # Jawline 보정 제거 (face_scale이 이미 적용됨)
        if face_landmarks_count > 0:
            print(f"   ✅ Face Landmarks Transferred: {face_landmarks_count} keypoints (face_scale={global_scale:.3f})")


    def _transfer_face_rigid(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores
    ):
        """
        (DEPRECATED - 하위 호환성을 위해 남김)
        _transfer_body_face()와 _transfer_face_landmarks()를 순차 호출
        """
        self._transfer_body_face(trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores)
        # ⭐ FIX: Pass face_scale from _transfer_body_face
        self._transfer_face_landmarks(
            trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores,
            face_scale=self._face_transfer_debug.get('face_scale', 1.0)
        )

    def _transfer_hands(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, hand_scale_ratio, transfer_log=None):
        """
        손 전이: REF 구조 우선 + SRC 크기 비율 반영
        
        전략:
        1. REF 손이 있으면: REF 구조 사용
            - SRC 손이 있으면 SRC/REF 평균 손 크기 비율로 스케일
            - SRC 손이 없으면 hand_scale_ratio 사용
        2. REF 손이 없고 SRC만 있으면: SRC 사용 (fallback)
        """
        print("\n" + "="*60)
        print("👋 _transfer_hands() - REF pose + SRC size")
        print("="*60)
        print(f"   hand_scale_ratio (ref→src): {hand_scale_ratio:.3f}")

        if transfer_log is not None:
            transfer_log['hand_scale_ratio'] = float(hand_scale_ratio)
            transfer_log.setdefault('hand_debug', [])
        
        LW, RW = 9, 10
        # Left(91-111), Right(112-132)

        def _hand_mean(kpts, scores, wrist_idx, start, end, thr=0.2):
            if wrist_idx >= len(scores) or scores[wrist_idx] <= thr:
                return None, 0
            wrist = kpts[wrist_idx]
            dists = []
            for idx in range(start, end):
                if idx < len(scores) and scores[idx] > thr:
                    dists.append(np.linalg.norm(kpts[idx] - wrist))
            if len(dists) == 0:
                return None, 0
            return float(np.mean(dists)), len(dists)

        def _torso_length(kpts, scores, thr=0.2):
            l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
            l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
            if (scores[l_sh] > thr and scores[r_sh] > thr and scores[l_hip] > thr and scores[r_hip] > thr):
                neck = (kpts[l_sh] + kpts[r_sh]) / 2.0
                root = (kpts[l_hip] + kpts[r_hip]) / 2.0
                return float(np.linalg.norm(neck - root))
            return None

        hands = [
            (LW, 91, 112, "Left"),
            (RW, 112, 133, "Right")
        ]

        # Precompute src/ref hand means for both sides
        src_means = {}
        ref_means = {}
        for wrist_idx, start, end, side in hands:
            src_mean, src_cnt = _hand_mean(src_kpts, src_scores, wrist_idx, start, end)
            ref_mean, ref_cnt = _hand_mean(ref_kpts, ref_scores, wrist_idx, start, end)
            src_means[side] = (src_mean, src_cnt)
            ref_means[side] = (ref_mean, ref_cnt)

        # Src base: max of available hand sizes
        src_base_candidates = [m for m, c in src_means.values() if m is not None and c > 5]
        src_base = float(max(src_base_candidates)) if len(src_base_candidates) > 0 else None

        # Ref hand/torso ratios
        ref_torso = _torso_length(ref_kpts, ref_scores)
        ref_ratios = {}
        for side, (ref_mean, ref_cnt) in ref_means.items():
            if ref_mean is not None and ref_cnt > 5 and ref_torso is not None and ref_torso > 1e-6:
                ref_ratios[side] = float(ref_mean / ref_torso)
        ref_ratio_max = max(ref_ratios.values()) if len(ref_ratios) > 0 else None
        
        for wrist_idx, start, end, side in hands:
            print(f"\n   [{side}] Checking wrist...")
            print(f"      wrist_idx={wrist_idx}")
            print(f"      trans_wrist_score={trans_scores[wrist_idx]:.3f}")

            hand_debug = {
                'side': side,
                'wrist_idx': int(wrist_idx),
                'trans_wrist_score': float(trans_scores[wrist_idx])
            }
            
            if trans_scores[wrist_idx] < 0.1:
                print(f"      ❌ trans wrist score < 0.1, SKIP")
                hand_debug.update({
                    'status': 'skip',
                    'reason': 'trans_wrist_score<0.1'
                })
                if transfer_log is not None:
                    transfer_log['hand_debug'].append(hand_debug)
                continue
            
            trans_wrist = trans_kpts[wrist_idx]
            
            # 전략: src 손 우선, 없으면 ref 손 사용
            src_hand_count = sum(1 for i in range(start, min(end, len(src_scores))) if src_scores[i] > 0.2)
            ref_hand_count = sum(1 for i in range(start, min(end, len(ref_scores))) if ref_scores[i] > 0.2)
            
            print(f"      src_hand_count={src_hand_count}/21")
            print(f"      ref_hand_count={ref_hand_count}/21")
            hand_debug['src_hand_count'] = int(src_hand_count)
            hand_debug['ref_hand_count'] = int(ref_hand_count)
            
            if ref_hand_count > 5:
                # REF 손 구조 사용 + SRC 크기 비율 반영
                ref_wrist = ref_kpts[wrist_idx]
                scale = hand_scale_ratio
                scale_source = f"hand_scale_ratio={hand_scale_ratio:.3f}"

                # New strategy: ref hand/torso ratio normalized to src max hand size
                ref_mean, ref_cnt = ref_means.get(side, (None, 0))
                if src_base is not None and ref_mean is not None and ref_cnt > 5 and ref_ratio_max is not None and ref_ratio_max > 1e-6:
                    ref_ratio = ref_ratios.get(side, None)
                    if ref_ratio is not None and ref_ratio > 0:
                        target_hand_size = src_base * (ref_ratio / ref_ratio_max)
                        scale = float(target_hand_size / ref_mean)
                        scale_source = "ref hand/torso ratio normalized to src max"
                        hand_debug['src_hand_base'] = float(src_base)
                        hand_debug['ref_torso_len'] = float(ref_torso) if ref_torso is not None else None
                        hand_debug['ref_hand_ratio'] = float(ref_ratio)
                        hand_debug['ref_hand_ratio_max'] = float(ref_ratio_max)
                        hand_debug['target_hand_size'] = float(target_hand_size)

                # Fallback: mean(src/ref hand size)
                if scale_source.startswith("hand_scale_ratio") and src_hand_count > 5:
                    src_wrist = src_kpts[wrist_idx]
                    src_dists = []
                    ref_dists = []
                    for idx in range(start, end):
                        if (
                            idx < len(src_scores) and idx < len(ref_scores)
                            and src_scores[idx] > 0.2 and ref_scores[idx] > 0.2
                        ):
                            src_dists.append(np.linalg.norm(src_kpts[idx] - src_wrist))
                            ref_dists.append(np.linalg.norm(ref_kpts[idx] - ref_wrist))
                    if len(ref_dists) >= 3 and np.mean(ref_dists) > 1e-6:
                        src_mean = float(np.mean(src_dists)) if len(src_dists) > 0 else 0.0
                        ref_mean = float(np.mean(ref_dists))
                        scale = float(src_mean / ref_mean)
                        scale_source = "mean(src/ref hand size)"
                        hand_debug['src_hand_mean_dist'] = src_mean
                        hand_debug['ref_hand_mean_dist'] = ref_mean
                        hand_debug['pairwise_count'] = int(min(len(src_dists), len(ref_dists)))

                print(f"      → Using REFERENCE (pose) + SRC size ({scale_source}, scale={scale:.3f})")
                print(f"      ref_wrist={ref_wrist}")
                print(f"      trans_wrist={trans_wrist}")

                transferred = 0
                for idx in range(start, end):
                    if idx < len(ref_scores) and ref_scores[idx] > 0.2:
                        rel = ref_kpts[idx] - ref_wrist
                        scaled_rel = rel * scale
                        trans_kpts[idx] = trans_wrist + scaled_rel
                        trans_scores[idx] = ref_scores[idx] * 0.9
                        transferred += 1
                        if transferred <= 3:
                            print(f"         idx={idx}: rel_length={np.linalg.norm(rel):.1f}, scaled_length={np.linalg.norm(scaled_rel):.1f}")
                print(f"      ✅ Used REFERENCE (scaled by {scale:.3f}): {transferred}/21 keypoints")

                hand_debug.update({
                    'status': 'ok',
                    'strategy': 'reference',
                    'scale': float(scale),
                    'scale_source': scale_source,
                    'transferred': int(transferred)
                })

            elif src_hand_count > 5:
                # REF 손이 없으면 SRC 사용 (fallback)
                src_wrist = src_kpts[wrist_idx]
                transferred = 0
                for idx in range(start, end):
                    if idx < len(src_scores) and src_scores[idx] > 0.2:
                        rel = src_kpts[idx] - src_wrist
                        trans_kpts[idx] = trans_wrist + rel
                        trans_scores[idx] = src_scores[idx]
                        transferred += 1
                print(f"      ✅ Used SOURCE (fallback): {transferred}/21 keypoints")
                hand_debug.update({
                    'status': 'ok',
                    'strategy': 'source_fallback',
                    'scale': 1.0,
                    'scale_source': 'source_relative',
                    'transferred': int(transferred)
                })

            else:
                print(f"      ⚠️ No valid hand in both src and ref (src={src_hand_count}, ref={ref_hand_count})")
                hand_debug.update({
                    'status': 'skip',
                    'reason': 'no_valid_hand'
                })

            if transfer_log is not None:
                transfer_log['hand_debug'].append(hand_debug)

    def _check_lower_body_valid(self, kpts, scores, img_h):
        # ... (기존 로직 유지)
        return True
    
    def _fill_missing_from_reference(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores,
        global_scale, processed, log
    ):
        """
        Ref에 있지만 Trans에 누락된 키포인트를 채웁니다.
        
        전략:
        1. ref_score > threshold이고 trans_score == 0인 키포인트 찾기
        2. 부모 관절이 trans에 있으면 부모로부터 ref 방향/거리 * global_scale로 생성
        3. 부모도 없으면 ref 위치 그대로 사용 (절대 위치)
        
        사용 사례:
        - src는 상반신만, ref는 전신 → ref 하반신을 src 비율로 생성
        - src는 손 없음, ref는 손 있음 → ref 손을 src 비율로 생성
        """
        print("\n🔄 Fill Missing Keypoints from Reference...")
        
        threshold = 0.3
        
        # Body 키포인트에 대한 부모 관계 (간단한 체인 구조)
        # 형식: {자식_idx: 부모_idx}
        parent_map = {
            # Upper body
            7: 5,   # left_elbow -> left_shoulder
            9: 7,   # left_wrist -> left_elbow
            8: 6,   # right_elbow -> right_shoulder
            10: 8,  # right_wrist -> right_elbow
            # Lower body
            13: 11, # left_knee -> left_hip
            15: 13, # left_ankle -> left_knee
            14: 12, # right_knee -> right_hip
            16: 14, # right_ankle -> right_knee
            # Feet
            17: 15, # left_big_toe -> left_ankle
            18: 15, # left_small_toe -> left_ankle
            19: 15, # left_heel -> left_ankle
            20: 16, # right_big_toe -> right_ankle
            21: 16, # right_small_toe -> right_ankle
            22: 16, # right_heel -> right_ankle
        }
        
        filled_count = 0
        threshold = 0.3
        
        # Body 키포인트 (0~22) 처리
        for idx in range(min(23, len(ref_scores), len(trans_scores))):
            # ref에는 있지만 trans에는 없는 키포인트
            if ref_scores[idx] > threshold and trans_scores[idx] < 0.01:
                # 전략 1: 부모 관절 기반 생성
                parent_idx = parent_map.get(idx)
                if parent_idx is not None and trans_scores[parent_idx] > 0.1:
                    # 부모가 trans에 존재
                    parent_trans = trans_kpts[parent_idx]
                    parent_ref = ref_kpts[parent_idx]
                    
                    # Ref에서 부모->자식 벡터
                    ref_vec = ref_kpts[idx] - parent_ref
                    ref_dist = calculate_distance(parent_ref, ref_kpts[idx])
                    
                    # Src에서 해당 뼈 길이 (있으면)
                    if idx < len(src_scores) and src_scores[idx] > threshold and parent_idx < len(src_scores) and src_scores[parent_idx] > threshold:
                        src_dist = calculate_distance(src_kpts[parent_idx], src_kpts[idx])
                        target_dist = src_dist
                        source_info = "src_bone_length"
                    else:
                        # Src에 없으면 ref 거리 * global_scale
                        target_dist = ref_dist * global_scale
                        source_info = f"ref_bone*scale({global_scale:.2f})"
                    
                    # 방향은 ref 따라가기, 거리는 src 비율
                    if ref_dist > 1e-6:
                        direction = ref_vec / ref_dist
                        trans_kpts[idx] = parent_trans + direction * target_dist
                        trans_scores[idx] = 0.7  # 생성된 키포인트는 낮은 신뢰도
                        filled_count += 1
                        print(f"   ✅ Filled #{idx}: parent={parent_idx}, dist={target_dist:.1f} ({source_info})")
                
                # 전략 2: 부모가 없거나 부모도 trans에 없으면 ref 절대 위치 사용
                elif ref_scores[idx] > 0.5:  # 높은 신뢰도만
                    trans_kpts[idx] = ref_kpts[idx].copy()
                    trans_scores[idx] = 0.6
                    filled_count += 1
                    print(f"   ⚠️ Filled #{idx}: using ref absolute position (no parent in trans)")
        
        # 얼굴 키포인트 (23~90): use_face=True일 때만 채움
        if self.config.use_face:
            for idx in range(23, min(91, len(ref_scores), len(trans_scores))):
                if ref_scores[idx] > 0.5 and trans_scores[idx] < 0.01:
                    # Src에 없는 얼굴 키포인트는 ref 그대로 (얼굴은 절대 위치가 의미 있음)
                    trans_kpts[idx] = ref_kpts[idx].copy()
                    trans_scores[idx] = ref_scores[idx] * 0.9  # ref 점수 상속
                    filled_count += 1
        
        # 손 키포인트는 _transfer_hands에서 src 우선/ref fallback으로 처리됨
        
        print(f"   📊 Total filled: {filled_count} keypoints")
    
    def _calculate_torso_ratio(self, src_kpts, src_scores, ref_kpts, ref_scores):
        """몸통 길이(torso length) 기반 비율 계산.
        
        몸통 길이 = 어깨 중심 ~ 골반 중심 거리
        - 시점(정면/측면)에 거의 독립적
        - 어깨 넓이보다 훨씬 안정적
        
        Returns:
            src_torso / ref_torso (기본값 1.0)
        """
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
        
        # Src torso length
        if (src_scores[l_sh] > 0.3 and src_scores[r_sh] > 0.3 and
            src_scores[l_hip] > 0.3 and src_scores[r_hip] > 0.3):
            src_neck = (src_kpts[l_sh] + src_kpts[r_sh]) / 2
            src_root = (src_kpts[l_hip] + src_kpts[r_hip]) / 2
            src_torso = calculate_distance(src_neck, src_root)
        else:
            print(f"   ⚠️ Src torso 계산 불가 (어깨/골반 누락), torso_ratio=1.0 사용")
            return 1.0
        
        # Ref torso length
        if (ref_scores[l_sh] > 0.3 and ref_scores[r_sh] > 0.3 and
            ref_scores[l_hip] > 0.3 and ref_scores[r_hip] > 0.3):
            ref_neck = (ref_kpts[l_sh] + ref_kpts[r_sh]) / 2
            ref_root = (ref_kpts[l_hip] + ref_kpts[r_hip]) / 2
            ref_torso = calculate_distance(ref_neck, ref_root)
        else:
            print(f"   ⚠️ Ref torso 계산 불가 (어깨/골반 누락), torso_ratio=1.0 사용")
            return 1.0
        
        if ref_torso < 1e-6:
            return 1.0
        
        ratio = src_torso / ref_torso
        # Clamp: 극단적인 값 방지 (0.5~8.0 범위)
        # 이전 0.85~1.25는 너무 제한적이어서 어깨가 비정상적으로 좁아짐
        min_ratio, max_ratio = 0.5, 8.0
        clamped = float(np.clip(ratio, min_ratio, max_ratio))
        print(f"   📏 Torso Ratio: src={src_torso:.1f}, ref={ref_torso:.1f} → {ratio:.3f} (clamped={clamped:.3f})")
        return clamped
    
    def _calculate_global_scale(
        self,
        src_proportions,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray
    ) -> float:
        """
        [수정됨] Global Scale Strategy: Fixed to 1.0
        Ref와의 비율 계산을 무시하고, Src의 원본 절대 크기를 그대로 사용합니다.
        """
        print(f"\n📏 Global Scale Strategy: Fixed to 1.0 (Source Size Preservation)")
        return 1.0
    
    def _calculate_hand_scale_ratio(self, src_kpts, src_scores, ref_kpts, ref_scores):
        """
        손 전이용 스케일 계산 (ref → src 비율)
        KeypointGenerator와 동일한 로직
        우선순위: 어깨 너비 > 골반 너비 > 1.0
        """
        threshold = self.config.confidence_threshold
        
        left_shoulder = BODY_KEYPOINTS['left_shoulder']
        right_shoulder = BODY_KEYPOINTS['right_shoulder']
        left_hip = BODY_KEYPOINTS['left_hip']
        right_hip = BODY_KEYPOINTS['right_hip']
        
        # 1순위: 어깨 너비
        if (src_scores[left_shoulder] > threshold and src_scores[right_shoulder] > threshold and
            ref_scores[left_shoulder] > threshold and ref_scores[right_shoulder] > threshold):
            
            src_shoulder_width = calculate_distance(src_kpts[left_shoulder], src_kpts[right_shoulder])
            ref_shoulder_width = calculate_distance(ref_kpts[left_shoulder], ref_kpts[right_shoulder])
            
            if ref_shoulder_width > 1e-6:
                return src_shoulder_width / ref_shoulder_width
        
        # 2순위: 골반 너비
        if (src_scores[left_hip] > threshold and src_scores[right_hip] > threshold and
            ref_scores[left_hip] > threshold and ref_scores[right_hip] > threshold):
            
            src_hip_width = calculate_distance(src_kpts[left_hip], src_kpts[right_hip])
            ref_hip_width = calculate_distance(ref_kpts[left_hip], ref_kpts[right_hip])
            
            if ref_hip_width > 1e-6:
                return src_hip_width / ref_hip_width
        
        # 기본값
        return 1.0
    
    def _correct_bone_lengths(
        self,
        src_props,
        global_scale: float,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        [수정됨] 뼈 길이 보정 - 다리 원근감 유지하되, 과도한 축소 방지 (Damped Perspective)
        """
        # 1. 기본 스케일링 (Src 길이 * Global Scale)
        corrected = {}
        for name, info in src_props.bone_lengths.items():
            corrected[name] = info.length * global_scale

        # 2. 대칭 쌍 정의
        
        # A. 원근감(Perspective)을 반영할 부위 (팔, 다리)
        perspective_pairs = [
            ('left_shoulder_left_elbow', 'right_shoulder_right_elbow'),
            ('left_elbow_left_wrist', 'right_elbow_right_wrist'),
            ('left_hip_left_knee', 'right_hip_right_knee'),
            ('left_knee_left_ankle', 'right_knee_right_ankle')
        ]
        
        # B. 원근감을 무시하고 Src 평균만 쓸 부위 (발)
        average_only_pairs = [
            ('left_ankle_left_big_toe', 'right_ankle_right_big_toe'),
            ('left_ankle_left_heel', 'right_ankle_right_heel'),
            ('left_ankle_left_small_toe', 'right_ankle_right_small_toe'),
            ('left_big_toe_left_small_toe', 'right_big_toe_right_small_toe') 
        ]
        
        # 헬퍼 함수
        def get_bone_dist(kpts, scores, bone_name):
            try:
                # 간단한 인덱스 매핑
                if 'shoulder_left_elbow' in bone_name: s,e=5,7
                elif 'shoulder_right_elbow' in bone_name: s,e=6,8
                elif 'elbow_left_wrist' in bone_name: s,e=7,9
                elif 'elbow_right_wrist' in bone_name: s,e=8,10
                elif 'hip_left_knee' in bone_name: s,e=11,13
                elif 'hip_right_knee' in bone_name: s,e=12,14
                elif 'knee_left_ankle' in bone_name: s,e=13,15
                elif 'knee_right_ankle' in bone_name: s,e=14,16
                else: return 0.0
                
                if scores[s] > 0.1 and scores[e] > 0.1:
                    return np.linalg.norm(kpts[s] - kpts[e])
            except:
                pass
            return 0.0

        print(f"\n⚖️ [Symmetric Correction] Perspective with Damping (Legs Protected)")
        
        # (A) 팔/다리: 원근감 적용 (단, 다리는 최소 길이 보장)
        for l_name, r_name in perspective_pairs:
            l_len = corrected.get(l_name, 0)
            r_len = corrected.get(r_name, 0)
            
            src_avg = 0.0
            if l_len > 0 and r_len > 0: src_avg = (l_len + r_len) / 2
            elif l_len > 0: src_avg = l_len
            elif r_len > 0: src_avg = r_len
            
            if src_avg <= 0: continue

            # Ref 비율 계산
            ref_l_dist = get_bone_dist(ref_kpts, ref_scores, l_name)
            ref_r_dist = get_bone_dist(ref_kpts, ref_scores, r_name)
            
            l_ratio, r_ratio = 1.0, 1.0
            if ref_l_dist > 0 and ref_r_dist > 0:
                ref_avg = (ref_l_dist + ref_r_dist) / 2
                
                raw_l_ratio = ref_l_dist / ref_avg
                raw_r_ratio = ref_r_dist / ref_avg
                
                # 🛡️ [Leg Protection] 다리는 너무 짧아지지 않게 제한 (Damping)
                if 'knee' in l_name or 'ankle' in l_name:
                    # 최소 0.9 (90%) ~ 최대 1.15 (115%)
                    # 원근감은 살리되(0.9), 숏다리는 방지
                    l_ratio = np.clip(raw_l_ratio, 0.90, 1.15)
                    r_ratio = np.clip(raw_r_ratio, 0.90, 1.15)
                    print(f"   🛡️ Leg Damping ({l_name.split('_')[1]}): Ratio Clamped {raw_l_ratio:.2f}->{l_ratio:.2f}")
                else:
                    # 팔은 자유롭게 (0.8 ~ 1.2)
                    l_ratio = np.clip(raw_l_ratio, 0.8, 1.2)
                    r_ratio = np.clip(raw_r_ratio, 0.8, 1.2)

            if l_len > 0: corrected[l_name] = src_avg * l_ratio
            if r_len > 0: corrected[r_name] = src_avg * r_ratio

        # (B) 발: 비율 고정 (Src 평균)
        for l_name, r_name in average_only_pairs:
            l_len = corrected.get(l_name, 0)
            r_len = corrected.get(r_name, 0)
            
            src_avg = 0.0
            if l_len > 0 and r_len > 0: src_avg = (l_len + r_len) / 2
            elif l_len > 0: src_avg = l_len
            elif r_len > 0: src_avg = r_len
            
            if src_avg > 0:
                corrected[l_name] = src_avg
                corrected[r_name] = src_avg
                
        print(f"   🔹 Feet: Fixed to Src Average Length")

        return corrected