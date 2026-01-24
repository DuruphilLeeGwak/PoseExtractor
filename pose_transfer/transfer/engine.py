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
        
        # Lower Body
        if ref_lower_valid:
            self.body_logic.transfer_chain(trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, hand_scale_ratio, processed, transfer_log, is_lower=True)
            self.body_logic.transfer_feet(trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, hand_scale_ratio, processed, transfer_log)

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
            self._transfer_hands(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores, hand_scale_ratio)

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

        # 1. Src의 얼굴-몸통 비율 계산
        # (어깨-귀 거리 평균) / (몸통 높이)
        src_ls_to_lear = np.linalg.norm(src_kpts[LEFT_EAR] - src_kpts[LS]) if src_scores[LEFT_EAR] > 0.1 else 0
        src_rs_to_rear = np.linalg.norm(src_kpts[RIGHT_EAR] - src_kpts[RS]) if src_scores[RIGHT_EAR] > 0.1 else 0
        src_shoulder_to_ear_avg = (src_ls_to_lear + src_rs_to_rear) / 2.0 if (src_ls_to_lear > 0 or src_rs_to_rear > 0) else 100.0
        
        # Src 몸통 높이 = 어깨 중점 → 엉덩이 중점
        src_neck = (src_kpts[LS] + src_kpts[RS]) / 2.0
        src_hip_center = (src_kpts[LEFT_HIP] + src_kpts[RIGHT_HIP]) / 2.0 if (src_scores[LEFT_HIP] > 0.1 and src_scores[RIGHT_HIP] > 0.1) else src_neck + np.array([0, 300])
        src_torso_height = np.linalg.norm(src_hip_center - src_neck)
        
        # Src 얼굴-몸통 비율
        src_face_torso_ratio = src_shoulder_to_ear_avg / src_torso_height if src_torso_height > 0 else 0.3
        
        print(f"   📏 Src: 어깨-귀 평균={src_shoulder_to_ear_avg:.1f}px, 몸통 높이={src_torso_height:.1f}px")
        print(f"   📊 Src: 얼굴/몸통 비율={src_face_torso_ratio:.3f}")

        # 2. Trans 몸통 높이
        trans_neck = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
        trans_hip_center = (trans_kpts[LEFT_HIP] + trans_kpts[RIGHT_HIP]) / 2.0 if (trans_scores[LEFT_HIP] > 0.1 and trans_scores[RIGHT_HIP] > 0.1) else trans_neck + np.array([0, 300])
        trans_torso_height = np.linalg.norm(trans_hip_center - trans_neck)
        
        # Trans에서 얼굴이 차지해야 할 크기
        trans_face_scale_target = src_face_torso_ratio * trans_torso_height
        
        print(f"   📏 Trans: 몸통 높이={trans_torso_height:.1f}px")
        print(f"   🎯 Trans: 목표 얼굴 크기={trans_face_scale_target:.1f}px (src 비율 적용)")

        # 3. Ref의 어깨-귀 거리 평균
        ref_ls_to_lear = np.linalg.norm(ref_kpts[LEFT_EAR] - ref_kpts[LS]) if ref_scores[LEFT_EAR] > 0.1 else 0
        ref_rs_to_rear = np.linalg.norm(ref_kpts[RIGHT_EAR] - ref_kpts[RS]) if ref_scores[RIGHT_EAR] > 0.1 else 0
        ref_shoulder_to_ear_avg = (ref_ls_to_lear + ref_rs_to_rear) / 2.0 if (ref_ls_to_lear > 0 or ref_rs_to_rear > 0) else 100.0
        
        # 4. Face 스케일 계산
        face_scale = trans_face_scale_target / ref_shoulder_to_ear_avg if ref_shoulder_to_ear_avg > 0 else 1.0
        
        print(f"   📏 Ref: 어깨-귀 평균={ref_shoulder_to_ear_avg:.1f}px")
        print(f"   🔄 Face Scale: {face_scale:.3f} (ref → trans)")
        
        # 디버그 정보 저장
        self._face_transfer_debug = {
            'src_face_torso_ratio': src_face_torso_ratio,
            'src_shoulder_to_ear_avg': src_shoulder_to_ear_avg,
            'src_torso_height': src_torso_height,
            'trans_torso_height': trans_torso_height,
            'trans_face_scale_target': trans_face_scale_target,
            'ref_shoulder_to_ear_avg': ref_shoulder_to_ear_avg,
            'face_scale': face_scale
        }

        # 5. 어깨 중점(neck)을 pivot으로 Face 배치
        # 측면 포즈에서도 안정적으로 작동
        trans_neck = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
        ref_neck = (ref_kpts[LS] + ref_kpts[RS]) / 2.0
        
        # 왼쪽 귀
        if ref_scores[LEFT_EAR] > 0.1:
            ref_neck_to_lear_vec = ref_kpts[LEFT_EAR] - ref_neck
            trans_neck_to_lear_vec = ref_neck_to_lear_vec * face_scale
            trans_lear = trans_neck + trans_neck_to_lear_vec
            
            trans_kpts[LEFT_EAR][0] = trans_lear[0]
            trans_kpts[LEFT_EAR][1] = trans_lear[1]
            trans_scores[LEFT_EAR] = min(src_scores[LEFT_EAR], ref_scores[LEFT_EAR])
            print(f"   👂 L-Ear: Neck 기준 벡터=({trans_neck_to_lear_vec[0]:.1f}, {trans_neck_to_lear_vec[1]:.1f})")
            
            self._face_transfer_debug['left_ear_vec'] = trans_neck_to_lear_vec.tolist()
        
        # 왼쪽 눈
        if ref_scores[LEFT_EYE] > 0.1:
            ref_neck_to_leye_vec = ref_kpts[LEFT_EYE] - ref_neck
            trans_neck_to_leye_vec = ref_neck_to_leye_vec * face_scale
            trans_leye = trans_neck + trans_neck_to_leye_vec
            
            trans_kpts[LEFT_EYE][0] = trans_leye[0]
            trans_kpts[LEFT_EYE][1] = trans_leye[1]
            trans_scores[LEFT_EYE] = min(src_scores[LEFT_EYE], ref_scores[LEFT_EYE])
            print(f"   👁️ L-Eye: Neck 기준 벡터=({trans_neck_to_leye_vec[0]:.1f}, {trans_neck_to_leye_vec[1]:.1f})")
            
            self._face_transfer_debug['left_eye_vec'] = trans_neck_to_leye_vec.tolist()

        # 오른쪽 귀
        if ref_scores[RIGHT_EAR] > 0.1:
            ref_neck_to_rear_vec = ref_kpts[RIGHT_EAR] - ref_neck
            trans_neck_to_rear_vec = ref_neck_to_rear_vec * face_scale
            trans_rear = trans_neck + trans_neck_to_rear_vec
            
            trans_kpts[RIGHT_EAR][0] = trans_rear[0]
            trans_kpts[RIGHT_EAR][1] = trans_rear[1]
            trans_scores[RIGHT_EAR] = min(src_scores[RIGHT_EAR], ref_scores[RIGHT_EAR])
            print(f"   👂 R-Ear: Neck 기준 벡터=({trans_neck_to_rear_vec[0]:.1f}, {trans_neck_to_rear_vec[1]:.1f})")
            
            self._face_transfer_debug['right_ear_vec'] = trans_neck_to_rear_vec.tolist()
        
        # 오른쪽 눈
        if ref_scores[RIGHT_EYE] > 0.1:
            ref_neck_to_reye_vec = ref_kpts[RIGHT_EYE] - ref_neck
            trans_neck_to_reye_vec = ref_neck_to_reye_vec * face_scale
            trans_reye = trans_neck + trans_neck_to_reye_vec
            
            trans_kpts[RIGHT_EYE][0] = trans_reye[0]
            trans_kpts[RIGHT_EYE][1] = trans_reye[1]
            trans_scores[RIGHT_EYE] = min(src_scores[RIGHT_EYE], ref_scores[RIGHT_EYE])
            print(f"   👁️ R-Eye: Neck 기준 벡터=({trans_neck_to_reye_vec[0]:.1f}, {trans_neck_to_reye_vec[1]:.1f})")
            
            self._face_transfer_debug['right_eye_vec'] = trans_neck_to_reye_vec.tolist()

        # 코: 양쪽 눈의 중점에서 Ref 비율로 배치
        if trans_scores[LEFT_EYE] > 0.1 and trans_scores[RIGHT_EYE] > 0.1:
            trans_eye_center = (trans_kpts[LEFT_EYE] + trans_kpts[RIGHT_EYE]) / 2.0
            
            # Ref에서 눈 중점 → 코 벡터
            if ref_scores[LEFT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1 and ref_scores[NOSE] > 0.1:
                ref_eye_center = (ref_kpts[LEFT_EYE] + ref_kpts[RIGHT_EYE]) / 2.0
                ref_eyecenter_to_nose_vec = ref_kpts[NOSE] - ref_eye_center
                trans_eyecenter_to_nose_vec = ref_eyecenter_to_nose_vec * face_scale
                trans_nose = trans_eye_center + trans_eyecenter_to_nose_vec
                
                trans_kpts[NOSE][0] = trans_nose[0]
                trans_kpts[NOSE][1] = trans_nose[1]
                trans_scores[NOSE] = min(src_scores[NOSE], ref_scores[NOSE])
                print(f"   👃 Nose: 눈 중점 기준 벡터=({trans_eyecenter_to_nose_vec[0]:.1f}, {trans_eyecenter_to_nose_vec[1]:.1f})")
                
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
        - Trans 코 위치를 기준으로 Ref의 랜드마크 구조 적용
        - Ref의 얼굴 내부 비율 100% 사용
        """
        NOSE = 0
        LS, RS = 5, 6
        
        face_landmarks_indices = list(range(23, 91))
        
        # 1. Trans 코 위치 확인 (이미 _transfer_body_face에서 배치됨)
        if trans_scores[NOSE] < 0.1:
            print("   ⚠️ [Face Landmarks] Trans 코가 없어 전이 불가")
            return
        
        trans_nose = trans_kpts[NOSE]
        
        # 2. Global scale 계산 (어깨 너비 기준)
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
        
        face_landmarks_count = 0
        for idx in face_landmarks_indices:
            if ref_scores[idx] > 0.1:
                # Ref에서 코-랜드마크 벡터
                ref_nose_lm_vec = ref_kpts[idx] - ref_nose
                ref_nose_lm_dist = np.linalg.norm(ref_nose_lm_vec)
                ref_nose_lm_angle = np.arctan2(ref_nose_lm_vec[1], ref_nose_lm_vec[0])
                
                # Trans에 적용 (global_scale로 스케일)
                trans_nose_lm_dist = ref_nose_lm_dist * global_scale
                trans_lm_x = trans_nose[0] + trans_nose_lm_dist * np.cos(ref_nose_lm_angle)
                trans_lm_y = trans_nose[1] + trans_nose_lm_dist * np.sin(ref_nose_lm_angle)
                
                trans_kpts[idx][0] = trans_lm_x
                trans_kpts[idx][1] = trans_lm_y
                trans_scores[idx] = ref_scores[idx]
                face_landmarks_count += 1
        
        if face_landmarks_count > 0:
            print(f"   ✅ Face Landmarks Transferred: {face_landmarks_count} keypoints (ref 구조)")

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

    def _transfer_hands(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, hand_scale_ratio):
        """
        손 전이: SRC 우선, SRC 없으면 REF 사용 (hand_scale_ratio 적용)
        
        전략:
        1. SRC에 손이 있으면: SRC 사용 (신체 비율 유지)
        2. SRC에 없고 REF에만 있으면: REF + hand_scale_ratio
        """
        print("\n" + "="*60)
        print("👋 _transfer_hands() - SRC priority")
        print("="*60)
        print(f"   hand_scale_ratio (ref→src): {hand_scale_ratio:.3f}")
        
        LW, RW = 9, 10
        # Left(91-111), Right(112-132)
        
        hands = [
            (LW, 91, 112, "Left"),
            (RW, 112, 133, "Right")
        ]
        
        for wrist_idx, start, end, side in hands:
            print(f"\n   [{side}] Checking wrist...")
            print(f"      wrist_idx={wrist_idx}")
            print(f"      trans_wrist_score={trans_scores[wrist_idx]:.3f}")
            
            if trans_scores[wrist_idx] < 0.1:
                print(f"      ❌ trans wrist score < 0.1, SKIP")
                continue
            
            trans_wrist = trans_kpts[wrist_idx]
            
            # 전략: src 손 우선, 없으면 ref 손 사용
            src_hand_count = sum(1 for i in range(start, min(end, len(src_scores))) if src_scores[i] > 0.2)
            ref_hand_count = sum(1 for i in range(start, min(end, len(ref_scores))) if ref_scores[i] > 0.2)
            
            print(f"      src_hand_count={src_hand_count}/21")
            print(f"      ref_hand_count={ref_hand_count}/21")
            
            if src_hand_count > 5:
                # SRC 손 사용 (우선순위 1)
                src_wrist = src_kpts[wrist_idx]
                transferred = 0
                for idx in range(start, end):
                    if idx < len(src_scores) and src_scores[idx] > 0.2:
                        rel = src_kpts[idx] - src_wrist
                        trans_kpts[idx] = trans_wrist + rel
                        trans_scores[idx] = src_scores[idx]
                        transferred += 1
                print(f"      ✅ Used SOURCE: {transferred}/21 keypoints")
            
            elif ref_hand_count > 5:
                # REF 손 사용 (hand_scale_ratio 적용)
                print(f"      → Using REFERENCE (src_hand_count={src_hand_count} <= 5)")
                ref_wrist = ref_kpts[wrist_idx]
                print(f"      ref_wrist={ref_wrist}")
                print(f"      trans_wrist={trans_wrist}")
                print(f"      applying hand_scale_ratio={hand_scale_ratio:.3f}")
                
                transferred = 0
                for idx in range(start, end):
                    if idx < len(ref_scores) and ref_scores[idx] > 0.2:
                        rel = ref_kpts[idx] - ref_wrist
                        scaled_rel = rel * hand_scale_ratio
                        trans_kpts[idx] = trans_wrist + scaled_rel
                        trans_scores[idx] = ref_scores[idx] * 0.9
                        transferred += 1
                        if transferred <= 3:
                            print(f"         idx={idx}: rel_length={np.linalg.norm(rel):.1f}, scaled_length={np.linalg.norm(scaled_rel):.1f}")
                print(f"      ✅ Used REFERENCE (scaled by {hand_scale_ratio:.3f}): {transferred}/21 keypoints")
            
            else:
                print(f"      ⚠️ No valid hand in both src and ref (src={src_hand_count}, ref={ref_hand_count})")

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