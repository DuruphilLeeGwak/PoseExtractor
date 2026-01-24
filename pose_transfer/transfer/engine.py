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
    ) -> TransferResult:
        
        print("\n" + "="*70)
        print("🔍 [DEBUG] PoseTransferEngine v26 (Rigid Face Rotation)")
        print("="*70)
        
        # 0. Ref 기준으로 Src 누락 키포인트 생성
        print("\n[Transfer Stage] 누락된 키포인트 생성...")
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
        
        # 3. 데이터 추출
        source_proportions = self.bone_calculator.calculate(source_keypoints, source_scores)
        global_scale = self._calculate_global_scale(source_proportions, reference_keypoints, reference_scores)
        corrected_lengths = self._correct_bone_lengths(source_proportions, global_scale, reference_keypoints)
        
        # 손 전이용 스케일 계산 (ref → src 비율)
        hand_scale_ratio = self._calculate_hand_scale_ratio(source_keypoints, source_scores, reference_keypoints, reference_scores)
        
        print(f"\n📏 Global Scale: {global_scale:.3f}")
        print(f"👋 Hand Scale Ratio (ref→src): {hand_scale_ratio:.3f}")

        # 4. 결과 초기화
        num_kpts = len(source_keypoints)
        trans_kpts = np.zeros((num_kpts, 2))
        trans_scores = np.zeros(num_kpts)
        transfer_log = {
            'face_parts': []  # For ear and face tracking
        }
        processed = set()

        # ══════════════════════════════════════════════════════════════
        # STEP 1: Body Transfer (Absolute Logic - v12 Base)
        # ══════════════════════════════════════════════════════════════
        print("\n🏃 Body Transfer...")
        
        # Upper Body (torso_ratio 기반)
        torso_ratio = self._calculate_torso_ratio(source_keypoints, source_scores, reference_keypoints, reference_scores)
        self.body_logic.transfer_shoulders(
            trans_kpts,
            trans_scores,
            source_keypoints,
            source_scores,
            reference_keypoints,
            torso_ratio=torso_ratio,
            processed=processed,
            log=transfer_log,
            r_scores=reference_scores,
        )
        self.body_logic.transfer_torso(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, hand_scale_ratio, processed, transfer_log)
        self.body_logic.transfer_chain(trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, hand_scale_ratio, processed, transfer_log, is_lower=False)
        # Fine-tune upper limb ratios before hand transfer
        self.body_logic.fine_tune_upper_ratio(
            trans_kpts, trans_scores,
            source_keypoints, source_scores,
            reference_keypoints, reference_scores,
            processed, transfer_log
        )
        
        # Lower Body
        if ref_lower_valid:
            self.body_logic.transfer_chain(trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, hand_scale_ratio, processed, transfer_log, is_lower=True)
            self.body_logic.transfer_feet(
                trans_kpts, trans_scores,
                source_keypoints, source_scores,
                corrected_lengths,
                reference_keypoints, reference_scores,
                hand_scale_ratio,
                processed, transfer_log
            )
            # Fine-tune lower limb ratios after feet transfer
            self.body_logic.fine_tune_lower_ratio(
                trans_kpts, trans_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                processed, transfer_log
            )

        # ══════════════════════════════════════════════════════════════
        # STEP 1.5: Fill Missing Keypoints from Reference
        # ══════════════════════════════════════════════════════════════
        # ref에 있지만 trans에 누락된 키포인트를 ref 기준으로 채우기
        # (src < ref 키포인트일 때 ref 포즈 따라가기)
        self._fill_missing_from_reference(
            trans_kpts, trans_scores,
            source_keypoints, source_scores,
            reference_keypoints, reference_scores,
            hand_scale_ratio, processed, transfer_log
        )

        # ══════════════════════════════════════════════════════════════
        # STEP 2: Face Transfer
        # ══════════════════════════════════════════════════════════════
        print(f"\n🔍 [DEBUG] self.config.use_face = {getattr(self.config, 'use_face', 'ATTRIBUTE_NOT_FOUND')}")
        print("\n👤 Face Transfer...")
        
        # Body Face (코, 눈, 귀)는 항상 처리 (use_face 설정 무관)
        # 귀는 ref 방향 따르도록 먼저 배치
        self._transfer_ears_from_ref(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, processed, transfer_log)
        # Body Face (코, 눈)를 ref 각도에 맞춰 회전
        self._transfer_body_face(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores)
        
        # Face Landmarks (23-90)는 use_face 설정에 따라 선택적 처리
        if self.config.use_face:
            print("   ✅ Face Landmarks enabled")
            self._transfer_face_landmarks(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores)
        else:
            print("   ⏭️  Face Landmarks disabled (use_face=False)")

        # ══════════════════════════════════════════════════════════════
        # STEP 3: Hand Transfer
        # ══════════════════════════════════════════════════════════════
        if self.config.use_hands:
            print("\n✋ Hand Transfer...")
            self._transfer_hands(
                trans_kpts, trans_scores,
                source_keypoints, source_scores,
                reference_keypoints, reference_scores,
                hand_scale_ratio,
                transfer_log=transfer_log
            )

        print("\n" + "="*70)
        print("✅ Transfer Complete")
        print("="*70 + "\n")

        # Face Transfer 디버그 정보를 transfer_log에 추가
        if self._face_transfer_debug:
            transfer_log['face_transfer_debug'] = self._face_transfer_debug

        return TransferResult(trans_kpts, trans_scores, corrected_lengths, {}, transfer_log)

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

    def _transfer_body_face(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores
    ):
        """
        Body Face 전이 - 어깨 기준 Ref 구조 적용 방식
        
        전략:
        1. 양쪽 어깨를 pivot으로 Ref의 face 구조(귀-눈-코) 적용
        2. Face 스케일: Src의 (어깨-귀 거리)/(몸통 높이) 비율을 Trans에 반영
        3. Ref의 어깨→귀→눈→코 각도와 거리를 그대로 사용
        
        효과:
        - Src 신체 비율 보존 (얼굴 크기 vs 몸통 크기)
        - Ref 얼굴 각도 및 구조 100% 반영
        - 좌우 독립 적용으로 비대칭 포즈도 정확
        """
        NOSE = 0
        LEFT_EYE, RIGHT_EYE = 1, 2
        LEFT_EAR, RIGHT_EAR = 3, 4
        LS, RS = 5, 6
        LEFT_HIP, RIGHT_HIP = 11, 12
        
        # 0. 필수 키포인트 확인
        if trans_scores[LS] < 0.1 or trans_scores[RS] < 0.1:
            print("   ⚠️ [Body Face] Trans 어깨가 없어 전이 불가")
            return
        
        if ref_scores[LS] < 0.1 or ref_scores[RS] < 0.1:
            print("   ⚠️ [Body Face] Ref 어깨가 없어 전이 불가")
            return
        
        if src_scores[LS] < 0.1 or src_scores[RS] < 0.1:
            print("   ⚠️ [Body Face] Src 어깨가 없어 전이 불가")
            return

        # 1. Src의 얼굴-몸통 비율 계산 (디버그/폴백용)
        # (얼굴 폭) / (몸통 높이)
        def _estimate_face_width(kpts, scores):
            # 귀가 있으면 귀-귀 거리 사용
            if scores[LEFT_EAR] > 0.1 and scores[RIGHT_EAR] > 0.1:
                return np.linalg.norm(kpts[RIGHT_EAR] - kpts[LEFT_EAR])
            # 귀가 없으면 눈-눈 거리로 대체 (얼굴 폭의 근사치)
            if scores[LEFT_EYE] > 0.1 and scores[RIGHT_EYE] > 0.1:
                return np.linalg.norm(kpts[RIGHT_EYE] - kpts[LEFT_EYE]) * 2.2
            return 100.0

        src_face_width = _estimate_face_width(src_kpts, src_scores)
        
        # Src 몸통 높이 = 어깨 중점 → 엉덩이 중점
        src_neck = (src_kpts[LS] + src_kpts[RS]) / 2.0
        src_hip_center = (src_kpts[LEFT_HIP] + src_kpts[RIGHT_HIP]) / 2.0 if (src_scores[LEFT_HIP] > 0.1 and src_scores[RIGHT_HIP] > 0.1) else src_neck + np.array([0, 300])
        src_torso_height = np.linalg.norm(src_hip_center - src_neck)
        
        # Src 얼굴-몸통 비율
        src_face_torso_ratio = src_face_width / src_torso_height if src_torso_height > 0 else 0.3
        
        print(f"   📏 Src: 얼굴폭≈{src_face_width:.1f}px, 몸통 높이={src_torso_height:.1f}px")
        print(f"   📊 Src: 얼굴/몸통 비율={src_face_torso_ratio:.3f}")

        # 2. Trans 몸통 높이
        trans_neck = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
        trans_hip_center = (trans_kpts[LEFT_HIP] + trans_kpts[RIGHT_HIP]) / 2.0 if (trans_scores[LEFT_HIP] > 0.1 and trans_scores[RIGHT_HIP] > 0.1) else trans_neck + np.array([0, 300])
        trans_torso_height = np.linalg.norm(trans_hip_center - trans_neck)
        
        # Trans에서 얼굴이 차지해야 할 크기
        trans_face_scale_target = src_face_torso_ratio * trans_torso_height
        
        print(f"   📏 Trans: 몸통 높이={trans_torso_height:.1f}px")
        print(f"   🎯 Trans: 목표 얼굴 크기={trans_face_scale_target:.1f}px (src 비율 적용)")

        # 3. Ref 얼굴 폭
        ref_face_width = _estimate_face_width(ref_kpts, ref_scores)
        
        # 4. Face 스케일 계산
        face_scale = trans_face_scale_target / ref_face_width if ref_face_width > 0 else 1.0
        
        print(f"   📏 Ref: 얼굴폭≈{ref_face_width:.1f}px")
        print(f"   🔄 Face Scale: {face_scale:.3f} (ref → trans)")
        
        # 디버그 정보 저장
        self._face_transfer_debug = {
            'src_face_torso_ratio': src_face_torso_ratio,
            'src_face_width': src_face_width,
            'src_torso_height': src_torso_height,
            'trans_torso_height': trans_torso_height,
            'trans_face_scale_target': trans_face_scale_target,
            'ref_face_width': ref_face_width,
            'face_scale': face_scale
        }

        # 5. 어깨 중점(neck)을 pivot으로 Face 배치
        # 방향(회전)은 ref, 비율(거리)은 src를 사용
        trans_neck = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
        ref_neck = (ref_kpts[LS] + ref_kpts[RS]) / 2.0
        src_neck = (src_kpts[LS] + src_kpts[RS]) / 2.0

        # 회전 계산 (눈 우선, 없으면 귀)
        def _angle(vec):
            return np.arctan2(vec[1], vec[0])

        rot = 0.0
        if (src_scores[LEFT_EYE] > 0.1 and src_scores[RIGHT_EYE] > 0.1 and
            ref_scores[LEFT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1):
            src_eye_vec = src_kpts[RIGHT_EYE] - src_kpts[LEFT_EYE]
            ref_eye_vec = ref_kpts[RIGHT_EYE] - ref_kpts[LEFT_EYE]
            rot = _angle(ref_eye_vec) - _angle(src_eye_vec)
        elif (src_scores[LEFT_EAR] > 0.1 and src_scores[RIGHT_EAR] > 0.1 and
              ref_scores[LEFT_EAR] > 0.1 and ref_scores[RIGHT_EAR] > 0.1):
            src_ear_vec = src_kpts[RIGHT_EAR] - src_kpts[LEFT_EAR]
            ref_ear_vec = ref_kpts[RIGHT_EAR] - ref_kpts[LEFT_EAR]
            rot = _angle(ref_ear_vec) - _angle(src_ear_vec)

        c, s = np.cos(rot), np.sin(rot)
        def _rotate(vec):
            return np.array([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c])

        # 스케일: 좌우/상하 분리 (어깨 너비 / 몸통 높이 기준)
        src_shoulder_width = np.linalg.norm(src_kpts[RS] - src_kpts[LS])
        trans_shoulder_width = np.linalg.norm(trans_kpts[RS] - trans_kpts[LS])
        ref_shoulder_width = np.linalg.norm(ref_kpts[RS] - ref_kpts[LS])
        if src_shoulder_width > 1e-6:
            face_scale_x = trans_shoulder_width / src_shoulder_width
        else:
            face_scale_x = 1.0

        # 목 길이: src 몸 비율 + ref 어깨/목 비율 반영
        if src_scores[NOSE] > 0.1:
            src_neck_len = np.linalg.norm(src_kpts[NOSE] - src_neck)
        else:
            src_neck_len = np.linalg.norm(src_kpts[LEFT_EYE] - src_neck) if src_scores[LEFT_EYE] > 0.1 else 0.0

        # 목 길이: src 비율만 사용 (어깨 너비 기준 스케일)
        if src_neck_len > 1e-6 and src_shoulder_width > 1e-6:
            neck_scale = trans_shoulder_width / src_shoulder_width
        else:
            neck_scale = 1.0

        # Anchor 보정은 사용하지 않고, 어깨 스케일에만 종속
        anchor_pull = 1.0

        # 세로 스케일은 어깨 스케일과 동일하게 유지
        face_scale_y = neck_scale

        def _scale_vec(vec, dist):
            base = vec * (dist * anchor_pull)
            return np.array([base[0] * face_scale_x, base[1] * face_scale_y])

        # 좌/우 거리 비율 blending (귀 전용)
        ear_alpha = 0.5
        src_left_ear_dist = np.linalg.norm(src_kpts[LEFT_EAR] - src_neck) if src_scores[LEFT_EAR] > 0.1 else None
        src_right_ear_dist = np.linalg.norm(src_kpts[RIGHT_EAR] - src_neck) if src_scores[RIGHT_EAR] > 0.1 else None
        ref_left_ear_dist = np.linalg.norm(ref_kpts[LEFT_EAR] - ref_neck) if ref_scores[LEFT_EAR] > 0.1 else None
        ref_right_ear_dist = np.linalg.norm(ref_kpts[RIGHT_EAR] - ref_neck) if ref_scores[RIGHT_EAR] > 0.1 else None

        blended_left_ear_dist = None
        blended_right_ear_dist = None
        if (src_left_ear_dist and src_right_ear_dist and ref_left_ear_dist and ref_right_ear_dist):
            src_sum = src_left_ear_dist + src_right_ear_dist
            ref_sum = ref_left_ear_dist + ref_right_ear_dist
            if src_sum > 1e-6 and ref_sum > 1e-6:
                src_ratio = src_left_ear_dist / src_sum
                ref_ratio = ref_left_ear_dist / ref_sum
                blended_ratio = ear_alpha * ref_ratio + (1.0 - ear_alpha) * src_ratio
                blended_left_ear_dist = src_sum * blended_ratio
                blended_right_ear_dist = src_sum * (1.0 - blended_ratio)
        
        # 왼쪽 귀
        if src_scores[LEFT_EAR] > 0.1 and ref_scores[LEFT_EAR] > 0.1:
            src_dist = blended_left_ear_dist if blended_left_ear_dist is not None else np.linalg.norm(src_kpts[LEFT_EAR] - src_neck)
            ref_vec = ref_kpts[LEFT_EAR] - ref_neck
            ref_len = np.linalg.norm(ref_vec)
            ref_dir = ref_vec / ref_len if ref_len > 1e-6 else ref_vec
            trans_neck_to_lear_vec = _scale_vec(ref_dir, src_dist)
            trans_lear = trans_neck + trans_neck_to_lear_vec

            trans_kpts[LEFT_EAR][0] = trans_lear[0]
            trans_kpts[LEFT_EAR][1] = trans_lear[1]
            trans_scores[LEFT_EAR] = min(src_scores[LEFT_EAR], ref_scores[LEFT_EAR])
            print(f"   👂 L-Ear: Neck 기준 벡터=({trans_neck_to_lear_vec[0]:.1f}, {trans_neck_to_lear_vec[1]:.1f})")

            self._face_transfer_debug['left_ear_vec'] = trans_neck_to_lear_vec.tolist()
        elif ref_scores[LEFT_EAR] > 0.1:
            ref_neck_to_lear_vec = ref_kpts[LEFT_EAR] - ref_neck
            trans_neck_to_lear_vec = ref_neck_to_lear_vec * face_scale
            trans_lear = trans_neck + trans_neck_to_lear_vec

            trans_kpts[LEFT_EAR][0] = trans_lear[0]
            trans_kpts[LEFT_EAR][1] = trans_lear[1]
            trans_scores[LEFT_EAR] = min(src_scores[LEFT_EAR], ref_scores[LEFT_EAR])
            print(f"   👂 L-Ear(fallback): Neck 기준 벡터=({trans_neck_to_lear_vec[0]:.1f}, {trans_neck_to_lear_vec[1]:.1f})")

            self._face_transfer_debug['left_ear_vec'] = trans_neck_to_lear_vec.tolist()
        
        # 왼쪽 눈
        if src_scores[LEFT_EYE] > 0.1 and ref_scores[LEFT_EYE] > 0.1:
            src_dist = np.linalg.norm(src_kpts[LEFT_EYE] - src_neck)
            ref_vec = ref_kpts[LEFT_EYE] - ref_neck
            ref_len = np.linalg.norm(ref_vec)
            ref_dir = ref_vec / ref_len if ref_len > 1e-6 else ref_vec
            trans_neck_to_leye_vec = _scale_vec(ref_dir, src_dist)
            trans_leye = trans_neck + trans_neck_to_leye_vec

            trans_kpts[LEFT_EYE][0] = trans_leye[0]
            trans_kpts[LEFT_EYE][1] = trans_leye[1]
            trans_scores[LEFT_EYE] = min(src_scores[LEFT_EYE], ref_scores[LEFT_EYE])
            print(f"   👁️ L-Eye: Neck 기준 벡터=({trans_neck_to_leye_vec[0]:.1f}, {trans_neck_to_leye_vec[1]:.1f})")

            self._face_transfer_debug['left_eye_vec'] = trans_neck_to_leye_vec.tolist()
        elif ref_scores[LEFT_EYE] > 0.1:
            ref_neck_to_leye_vec = ref_kpts[LEFT_EYE] - ref_neck
            trans_neck_to_leye_vec = ref_neck_to_leye_vec * face_scale
            trans_leye = trans_neck + trans_neck_to_leye_vec

            trans_kpts[LEFT_EYE][0] = trans_leye[0]
            trans_kpts[LEFT_EYE][1] = trans_leye[1]
            trans_scores[LEFT_EYE] = min(src_scores[LEFT_EYE], ref_scores[LEFT_EYE])
            print(f"   👁️ L-Eye(fallback): Neck 기준 벡터=({trans_neck_to_leye_vec[0]:.1f}, {trans_neck_to_leye_vec[1]:.1f})")

            self._face_transfer_debug['left_eye_vec'] = trans_neck_to_leye_vec.tolist()

        # 오른쪽 귀
        if src_scores[RIGHT_EAR] > 0.1 and ref_scores[RIGHT_EAR] > 0.1:
            src_dist = blended_right_ear_dist if blended_right_ear_dist is not None else np.linalg.norm(src_kpts[RIGHT_EAR] - src_neck)
            ref_vec = ref_kpts[RIGHT_EAR] - ref_neck
            ref_len = np.linalg.norm(ref_vec)
            ref_dir = ref_vec / ref_len if ref_len > 1e-6 else ref_vec
            trans_neck_to_rear_vec = _scale_vec(ref_dir, src_dist)
            trans_rear = trans_neck + trans_neck_to_rear_vec

            trans_kpts[RIGHT_EAR][0] = trans_rear[0]
            trans_kpts[RIGHT_EAR][1] = trans_rear[1]
            trans_scores[RIGHT_EAR] = min(src_scores[RIGHT_EAR], ref_scores[RIGHT_EAR])
            print(f"   👂 R-Ear: Neck 기준 벡터=({trans_neck_to_rear_vec[0]:.1f}, {trans_neck_to_rear_vec[1]:.1f})")

            self._face_transfer_debug['right_ear_vec'] = trans_neck_to_rear_vec.tolist()
        elif ref_scores[RIGHT_EAR] > 0.1:
            ref_neck_to_rear_vec = ref_kpts[RIGHT_EAR] - ref_neck
            trans_neck_to_rear_vec = ref_neck_to_rear_vec * face_scale
            trans_rear = trans_neck + trans_neck_to_rear_vec

            trans_kpts[RIGHT_EAR][0] = trans_rear[0]
            trans_kpts[RIGHT_EAR][1] = trans_rear[1]
            trans_scores[RIGHT_EAR] = min(src_scores[RIGHT_EAR], ref_scores[RIGHT_EAR])
            print(f"   👂 R-Ear(fallback): Neck 기준 벡터=({trans_neck_to_rear_vec[0]:.1f}, {trans_neck_to_rear_vec[1]:.1f})")

            self._face_transfer_debug['right_ear_vec'] = trans_neck_to_rear_vec.tolist()
        
        # 오른쪽 눈
        if src_scores[RIGHT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1:
            src_dist = np.linalg.norm(src_kpts[RIGHT_EYE] - src_neck)
            ref_vec = ref_kpts[RIGHT_EYE] - ref_neck
            ref_len = np.linalg.norm(ref_vec)
            ref_dir = ref_vec / ref_len if ref_len > 1e-6 else ref_vec
            trans_neck_to_reye_vec = _scale_vec(ref_dir, src_dist)
            trans_reye = trans_neck + trans_neck_to_reye_vec

            trans_kpts[RIGHT_EYE][0] = trans_reye[0]
            trans_kpts[RIGHT_EYE][1] = trans_reye[1]
            trans_scores[RIGHT_EYE] = min(src_scores[RIGHT_EYE], ref_scores[RIGHT_EYE])
            print(f"   👁️ R-Eye: Neck 기준 벡터=({trans_neck_to_reye_vec[0]:.1f}, {trans_neck_to_reye_vec[1]:.1f})")

            self._face_transfer_debug['right_eye_vec'] = trans_neck_to_reye_vec.tolist()
        elif ref_scores[RIGHT_EYE] > 0.1:
            ref_neck_to_reye_vec = ref_kpts[RIGHT_EYE] - ref_neck
            trans_neck_to_reye_vec = ref_neck_to_reye_vec * face_scale
            trans_reye = trans_neck + trans_neck_to_reye_vec

            trans_kpts[RIGHT_EYE][0] = trans_reye[0]
            trans_kpts[RIGHT_EYE][1] = trans_reye[1]
            trans_scores[RIGHT_EYE] = min(src_scores[RIGHT_EYE], ref_scores[RIGHT_EYE])
            print(f"   👁️ R-Eye(fallback): Neck 기준 벡터=({trans_neck_to_reye_vec[0]:.1f}, {trans_neck_to_reye_vec[1]:.1f})")

            self._face_transfer_debug['right_eye_vec'] = trans_neck_to_reye_vec.tolist()

        # 코: 양쪽 눈의 중점에서 Ref 비율로 배치
        if trans_scores[LEFT_EYE] > 0.1 and trans_scores[RIGHT_EYE] > 0.1:
            trans_eye_center = (trans_kpts[LEFT_EYE] + trans_kpts[RIGHT_EYE]) / 2.0

            # Src 눈 중점 → 코 벡터 (비율 유지)
            if (src_scores[LEFT_EYE] > 0.1 and src_scores[RIGHT_EYE] > 0.1 and src_scores[NOSE] > 0.1 and
                ref_scores[LEFT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1 and ref_scores[NOSE] > 0.1):
                src_eye_center = (src_kpts[LEFT_EYE] + src_kpts[RIGHT_EYE]) / 2.0
                src_dist = np.linalg.norm(src_kpts[NOSE] - src_eye_center)

                ref_eye_center = (ref_kpts[LEFT_EYE] + ref_kpts[RIGHT_EYE]) / 2.0
                ref_vec = ref_kpts[NOSE] - ref_eye_center
                ref_len = np.linalg.norm(ref_vec)
                ref_dir = ref_vec / ref_len if ref_len > 1e-6 else ref_vec

                trans_eyecenter_to_nose_vec = _scale_vec(ref_dir, src_dist)
                trans_nose = trans_eye_center + trans_eyecenter_to_nose_vec

                trans_kpts[NOSE][0] = trans_nose[0]
                trans_kpts[NOSE][1] = trans_nose[1]
                trans_scores[NOSE] = min(src_scores[NOSE], ref_scores[NOSE])
                print(f"   👃 Nose: 눈 중점 기준 벡터=({trans_eyecenter_to_nose_vec[0]:.1f}, {trans_eyecenter_to_nose_vec[1]:.1f})")

                self._face_transfer_debug['nose_vec'] = trans_eyecenter_to_nose_vec.tolist()
            elif ref_scores[LEFT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1 and ref_scores[NOSE] > 0.1:
                ref_eye_center = (ref_kpts[LEFT_EYE] + ref_kpts[RIGHT_EYE]) / 2.0
                ref_eyecenter_to_nose_vec = ref_kpts[NOSE] - ref_eye_center
                trans_eyecenter_to_nose_vec = ref_eyecenter_to_nose_vec * face_scale
                trans_nose = trans_eye_center + trans_eyecenter_to_nose_vec

                trans_kpts[NOSE][0] = trans_nose[0]
                trans_kpts[NOSE][1] = trans_nose[1]
                trans_scores[NOSE] = min(src_scores[NOSE], ref_scores[NOSE])
                print(f"   👃 Nose(fallback): 눈 중점 기준 벡터=({trans_eyecenter_to_nose_vec[0]:.1f}, {trans_eyecenter_to_nose_vec[1]:.1f})")

                self._face_transfer_debug['nose_vec'] = trans_eyecenter_to_nose_vec.tolist()
        
        print(f"   ✅ Body Face Transferred: 5 keypoints (Neck pivot)")
        print(f"   ℹ️ Pivot: 어깨 중점(Neck) → 측면 포즈도 안정적")
        print(f"   ℹ️ 양쪽 눈/귀 모두 Neck 기준으로 배치")
        print(f"   ℹ️ Ref 구조 100% 반영, Src 얼굴/몸통 비율 유지")

    def _transfer_face_landmarks(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores
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
        
        global_scale = trans_shoulder_width / src_shoulder_width if src_shoulder_width > 0 else 1.0
        
        # 3. Ref에서 코-랜드마크 벡터 가져와서 Trans에 적용
        if ref_scores[NOSE] < 0.1:
            print("   ⚠️ [Face Landmarks] Ref 코가 없어 전이 불가")
            return
        
        ref_nose = ref_kpts[NOSE]
        
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

        # Ref 구조를 5점 정합으로 전이
        face_landmarks_count = 0
        for idx in face_landmarks_indices:
            if ref_scores[idx] > 0.1:
                if use_similarity:
                    trans_pos = sim_scale * (sim_R @ ref_kpts[idx]) + sim_t
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

        # Jawline 보정: src 비율(좌우/상하) 기반으로 jawline을 재스케일
        FACE_START = 23
        jawline_idx = [FACE_START + i for i in range(0, 17)]
        left_brow_idx = [FACE_START + i for i in range(17, 22)]
        right_brow_idx = [FACE_START + i for i in range(22, 27)]

        def _mean_point(kpts, scores, indices):
            pts = [kpts[i] for i in indices if i < len(scores) and scores[i] > 0.1]
            return np.mean(pts, axis=0) if len(pts) else None

        # Src 기준 비율 계산 (jaw/ear, brow-chin/ear)
        src_jaw_left = src_kpts[jawline_idx[0]] if src_scores[jawline_idx[0]] > 0.1 else None
        src_jaw_right = src_kpts[jawline_idx[-1]] if src_scores[jawline_idx[-1]] > 0.1 else None
        src_jaw_chin = src_kpts[jawline_idx[8]] if src_scores[jawline_idx[8]] > 0.1 else None
        src_brow_center = _mean_point(src_kpts, src_scores, left_brow_idx + right_brow_idx)
        src_ear_left = src_kpts[LEFT_EAR] if src_scores[LEFT_EAR] > 0.1 else None
        src_ear_right = src_kpts[RIGHT_EAR] if src_scores[RIGHT_EAR] > 0.1 else None

        # Trans 기준 현재 비율 계산
        trans_jaw_left = trans_kpts[jawline_idx[0]] if trans_scores[jawline_idx[0]] > 0.1 else None
        trans_jaw_right = trans_kpts[jawline_idx[-1]] if trans_scores[jawline_idx[-1]] > 0.1 else None
        trans_jaw_chin = trans_kpts[jawline_idx[8]] if trans_scores[jawline_idx[8]] > 0.1 else None
        trans_brow_center = _mean_point(trans_kpts, trans_scores, left_brow_idx + right_brow_idx)
        trans_ear_left = trans_kpts[LEFT_EAR] if trans_scores[LEFT_EAR] > 0.1 else None
        trans_ear_right = trans_kpts[RIGHT_EAR] if trans_scores[RIGHT_EAR] > 0.1 else None

        src_jaw_width = np.linalg.norm(src_jaw_right - src_jaw_left) if (src_jaw_left is not None and src_jaw_right is not None) else None
        src_ear_width = np.linalg.norm(src_ear_right - src_ear_left) if (src_ear_left is not None and src_ear_right is not None) else None
        src_brow_chin = np.linalg.norm(src_jaw_chin - src_brow_center) if (src_brow_center is not None and src_jaw_chin is not None) else None

        trans_jaw_width = np.linalg.norm(trans_jaw_right - trans_jaw_left) if (trans_jaw_left is not None and trans_jaw_right is not None) else None
        trans_ear_width = np.linalg.norm(trans_ear_right - trans_ear_left) if (trans_ear_left is not None and trans_ear_right is not None) else None
        trans_brow_chin = np.linalg.norm(trans_jaw_chin - trans_brow_center) if (trans_brow_center is not None and trans_jaw_chin is not None) else None

        # 목표 jawline 크기: trans 귀폭에 src 비율을 반영
        target_jaw_width = None
        target_brow_chin = None
        if src_jaw_width and src_ear_width and trans_ear_width and src_ear_width > 1e-6:
            target_jaw_width = trans_ear_width * (src_jaw_width / src_ear_width)
        if src_brow_chin and src_ear_width and trans_ear_width and src_ear_width > 1e-6:
            target_brow_chin = trans_ear_width * (src_brow_chin / src_ear_width)

        scale_x = (target_jaw_width / trans_jaw_width) if (target_jaw_width and trans_jaw_width and trans_jaw_width > 1e-6) else 1.0
        scale_y = (target_brow_chin / trans_brow_chin) if (target_brow_chin and trans_brow_chin and trans_brow_chin > 1e-6) else 1.0

        # 과도한 변형 방지
        scale_x = float(np.clip(scale_x, 0.5, 2.0))
        scale_y = float(np.clip(scale_y, 0.5, 2.0))

        # Jawline pivot: 귀 중점 + 코 기준 (5점 기반 위치)
        if trans_scores[LEFT_EAR] > 0.1 and trans_scores[RIGHT_EAR] > 0.1:
            ear_mid = (trans_kpts[LEFT_EAR] + trans_kpts[RIGHT_EAR]) / 2.0
        else:
            ear_mid = _mean_point(trans_kpts, trans_scores, jawline_idx) or trans_nose

        jaw_pivot = np.array([ear_mid[0], trans_nose[1]], dtype=np.float32)

        if scale_x != 1.0 or scale_y != 1.0:
            for idx in jawline_idx:
                if idx < len(trans_scores) and trans_scores[idx] > 0.1:
                    v = trans_kpts[idx] - jaw_pivot
                    v = np.array([v[0] * scale_x, v[1] * scale_y])
                    trans_kpts[idx] = jaw_pivot + v

            # Jawline 보정에 맞춰 내부 파트도 미세 보정
            inner_pivot = trans_nose
            for idx in face_landmarks_indices:
                if idx in jawline_idx:
                    continue
                if idx < len(trans_scores) and trans_scores[idx] > 0.1:
                    v = trans_kpts[idx] - inner_pivot
                    v = np.array([v[0] * scale_x, v[1] * scale_y])
                    trans_kpts[idx] = inner_pivot + v
        
        if face_landmarks_count > 0:
            print(f"   ✅ Face Landmarks Transferred: {face_landmarks_count} keypoints (ref 구조)")
            print(f"   🔧 Jawline scale: x={scale_x:.3f}, y={scale_y:.3f} (src ratio applied)")

    def _transfer_face_rigid(
        self, trans_kpts, trans_scores,
        src_kpts, src_scores, ref_kpts, ref_scores
    ):
        """
        (DEPRECATED - 하위 호환성을 위해 남김)
        _transfer_body_face()와 _transfer_face_landmarks()를 순차 호출
        """
        self._transfer_body_face(trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores)
        self._transfer_face_landmarks(trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores)

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
    
    def _calculate_global_scale(self, src_props, ref_kpts, ref_scores):
        """Fallback: torso 대신 어깨 기반 스케일 (하위 호환)."""
        src_w = src_props.shoulder_width
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        if src_w > 0 and ref_scores[l_sh] > 0.3:
            ref_w = calculate_distance(ref_kpts[l_sh], ref_kpts[r_sh])
            return src_w / ref_w if ref_w > 0 else 1.0
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
    
    def _correct_bone_lengths(self, props, scale, ref_kpts):
        lengths = {}
        for n, i in props.bone_lengths.items():
            if i.is_valid: lengths[n] = i.length
        return lengths