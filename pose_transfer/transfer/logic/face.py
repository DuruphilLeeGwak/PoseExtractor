"""
Face Transfer Logic Module (Refactored v2.0)

역할:
- 얼굴 부위(0~4) 및 상세 랜드마크(23~90) 전이 담당
- Body Scale을 기반으로 얼굴 크기(Face Scale) 정밀 계산
- Reference의 회전(Rotation) 및 표정 구조 유지
"""
import numpy as np
from typing import Dict, Any, Set, Optional, Tuple

class FaceTransfer:
    def __init__(self, config):
        self.config = config
        # 계산된 변환 정보(Scale, Pivot 등)를 저장하여 랜드마크 전이 시 재사용
        self._transform_cache = {}

    def transfer_structure(
        self,
        trans_kpts: np.ndarray,
        trans_scores: np.ndarray,
        src_kpts: np.ndarray,
        src_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        processed: Set[int],
        log: Dict[str, Any],
        body_scale: float = 1.0
    ):
        """
        얼굴 기본 구조(0~4: 코, 눈, 귀) 전이 및 변환 행렬 계산
        """
        NOSE = 0
        LEFT_EYE, RIGHT_EYE = 1, 2
        LEFT_EAR, RIGHT_EAR = 3, 4
        LS, RS = 5, 6 # 어깨
        
        # 0. 선행 조건: Trans 어깨가 생성되어 있어야 함 (Pivot 계산용)
        if trans_scores[LS] < 0.1 or trans_scores[RS] < 0.1:
            return

        # ---------------------------------------------------------
        # 1. Scale Calculation (Src 얼굴 비율 vs Trans 어깨)
        # ---------------------------------------------------------
        
        # Src 얼굴 너비 추정
        if src_scores[LEFT_EAR] > 0.1 and src_scores[RIGHT_EAR] > 0.1:
            src_face_width = np.linalg.norm(src_kpts[RIGHT_EAR] - src_kpts[LEFT_EAR])
        elif src_scores[LEFT_EYE] > 0.1 and src_scores[RIGHT_EYE] > 0.1:
            src_face_width = np.linalg.norm(src_kpts[RIGHT_EYE] - src_kpts[LEFT_EYE]) * 2.2
        else:
            src_face_width = 100.0

        # Src 어깨 너비
        if src_scores[LS] > 0.1 and src_scores[RS] > 0.1:
            src_shoulder_width = np.linalg.norm(src_kpts[RS] - src_kpts[LS])
        else:
            src_shoulder_width = 1.0
        
        # Trans 어깨 너비 (이미 BodyTransfer에서 생성됨)
        trans_neck = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
        trans_shoulder_width = np.linalg.norm(trans_kpts[RS] - trans_kpts[LS])

        # Body Ratio (Trans 어깨 / Src 어깨) -> 몸이 얼마나 커졌는지
        body_ratio = trans_shoulder_width / src_shoulder_width if src_shoulder_width > 0 else 1.0
        
        # Ref 얼굴 너비
        if ref_scores[LEFT_EAR] > 0.1 and ref_scores[RIGHT_EAR] > 0.1:
            ref_face_width = np.linalg.norm(ref_kpts[RIGHT_EAR] - ref_kpts[LEFT_EAR])
        elif ref_scores[LEFT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1:
            ref_face_width = np.linalg.norm(ref_kpts[RIGHT_EYE] - ref_kpts[LEFT_EYE]) * 2.2
        else:
            ref_face_width = 100.0

        # [핵심] Face Scale 결정
        # Src 좌표계에서 바디가 생성되었으므로, 얼굴도 Src 비율 유지
        # Src 얼굴 크기를 그대로 사용 (body와 일관성 유지)
        
        # Src 얼굴/어깨 비율 유지
        face_to_shoulder_ratio = src_face_width / src_shoulder_width if src_shoulder_width > 0 else 0.5
        
        # Trans에서의 목표 얼굴 너비 = Trans 어깨 너비 * Src 비율
        target_face_width = trans_shoulder_width * face_to_shoulder_ratio
        
        # Face Scale = Src 목표 얼굴 / Ref 얼굴 (Ref 얼굴을 Src 크기로 스케일링)
        face_scale = target_face_width / ref_face_width if ref_face_width > 0 else 1.0
        face_scale = np.clip(face_scale, 0.3, 3.0)  # 안전 범위

        # ---------------------------------------------------------
        # 2. Pivot Calculation (목 -> 눈 중심)
        # ---------------------------------------------------------
        
        # Src 목-눈 거리 추정
        src_neck = (src_kpts[LS] + src_kpts[RS]) / 2.0
        if src_scores[LEFT_EYE] > 0.1 and src_scores[RIGHT_EYE] > 0.1:
            src_eye_c = (src_kpts[LEFT_EYE] + src_kpts[RIGHT_EYE]) / 2.0
            src_neck_eye_dist = np.linalg.norm(src_eye_c - src_neck)
        else:
            # 눈이 없으면 어깨 너비의 절반 정도로 추정
            src_neck_eye_dist = src_shoulder_width * 0.5 

        # Trans에서의 목표 목-눈 거리 (몸 커진 비율 반영)
        target_neck_eye_dist = src_neck_eye_dist * body_ratio

        # Ref의 목-눈 방향 벡터 (Ref의 고개 각도 반영)
        ref_neck = (ref_kpts[LS] + ref_kpts[RS]) / 2.0
        if ref_scores[LEFT_EYE] > 0.1 and ref_scores[RIGHT_EYE] > 0.1:
            ref_eye_c = (ref_kpts[LEFT_EYE] + ref_kpts[RIGHT_EYE]) / 2.0
            ref_dir_vec = ref_eye_c - ref_neck
            ref_dir_norm = np.linalg.norm(ref_dir_vec)
            # 방향 벡터 정규화
            ref_dir = ref_dir_vec / ref_dir_norm if ref_dir_norm > 0 else np.array([0, -1])
            ref_pivot_origin = ref_eye_c # Ref 기준점은 눈 중심
        else:
            ref_dir = np.array([0, -1]) # 수직 위
            ref_pivot_origin = ref_kpts[NOSE] # 눈 없으면 코 기준

        # 최종 Pivot (Trans에서의 눈 중심 위치)
        pivot = trans_neck + ref_dir * target_neck_eye_dist

        # ---------------------------------------------------------
        # 3. Cache Transform & Apply
        # ---------------------------------------------------------
        
        # 랜드마크 전이를 위해 캐시에 저장
        self._transform_cache = {
            'scale': face_scale,
            'pivot': pivot,         # Trans 상의 기준점 (눈 중심)
            'ref_origin': ref_pivot_origin # Ref 상의 기준점 (눈 중심)
        }
        
        # 키포인트 0~4 배치
        for idx in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]:
            if ref_scores[idx] > 0.1:
                # Ref에서의 상대 벡터 (Ref눈중심 -> Ref부위)
                rel_vec = ref_kpts[idx] - ref_pivot_origin
                
                # Scale 적용 (회전은 Ref 좌표계에 이미 포함됨)
                scaled_vec = rel_vec * face_scale
                
                # Trans Pivot에 더하기
                trans_kpts[idx] = pivot + scaled_vec
                trans_scores[idx] = ref_scores[idx]
                processed.add(idx)

        # 로그 기록
        if log is not None:
            log.setdefault('face_transfer_debug', {}).update({
                'face_scale': face_scale,
                'body_ratio': body_ratio,
                'target_face_width': target_face_width
            })

    def transfer_landmarks(
        self,
        trans_kpts: np.ndarray,
        trans_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        processed: Set[int]
    ):
        """
        얼굴 상세 랜드마크(23~90) 전이
        - transfer_structure에서 계산된 캐시 사용
        """
        if not self._transform_cache:
            return

        scale = self._transform_cache['scale']
        pivot = self._transform_cache['pivot']
        ref_origin = self._transform_cache['ref_origin']

        for idx in range(23, 91):
            if idx < len(ref_scores) and ref_scores[idx] > 0.1:
                rel_vec = ref_kpts[idx] - ref_origin
                trans_kpts[idx] = pivot + (rel_vec * scale)
                trans_scores[idx] = ref_scores[idx]
                processed.add(idx)

    def transfer_ears_fallback(
        self,
        trans_kpts: np.ndarray,
        trans_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        processed: Set[int]
    ):
        """
        귀 전이 Fallback (눈/코가 없어서 구조 전이에 실패했을 때)
        - 어깨를 기준으로 귀 위치 추정
        """
        LS, RS = 5, 6
        LEFT_EAR, RIGHT_EAR = 3, 4
        
        # 이미 귀가 전이되었으면 패스
        if trans_scores[LEFT_EAR] > 0.1 and trans_scores[RIGHT_EAR] > 0.1:
            return

        # 어깨가 없으면 포기
        if trans_scores[LS] < 0.1 or trans_scores[RS] < 0.1:
            return

        trans_width = np.linalg.norm(trans_kpts[RS] - trans_kpts[LS])

        for ear_idx, shoulder_idx in [(LEFT_EAR, LS), (RIGHT_EAR, RS)]:
            # 이미 처리된 귀는 패스
            if trans_scores[ear_idx] > 0.1: continue

            if ref_scores[ear_idx] > 0.1 and ref_scores[shoulder_idx] > 0.1:
                # Ref: 어깨 -> 귀 벡터
                ref_vec = ref_kpts[ear_idx] - ref_kpts[shoulder_idx]
                
                # Ref 어깨 너비 대비 비율 계산
                ref_sh_width = np.linalg.norm(ref_kpts[RS] - ref_kpts[LS])
                ratio = np.linalg.norm(ref_vec) / ref_sh_width if ref_sh_width > 0 else 0.3
                
                # 방향은 Ref 유지
                norm = np.linalg.norm(ref_vec)
                dir_vec = ref_vec / norm if norm > 0 else ref_vec
                
                # Trans 적용
                trans_kpts[ear_idx] = trans_kpts[shoulder_idx] + dir_vec * (trans_width * ratio)
                trans_scores[ear_idx] = ref_scores[ear_idx] * 0.8
                processed.add(ear_idx)