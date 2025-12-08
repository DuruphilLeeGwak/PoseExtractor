import numpy as np
from ...extractors.keypoint_constants import BODY_KEYPOINTS, FACE_START_IDX, FACE_END_IDX
from ...utils.geometry import calculate_distance, normalize_vector

FACE_PARTS_IDX = {
    'jawline': range(0, 17), 'left_eyebrow': range(17, 22), 'right_eyebrow': range(22, 27),
    'nose': range(27, 36), 'left_eye': range(36, 42), 'right_eye': range(42, 48),
    'mouth_outer': range(48, 60), 'mouth_inner': range(60, 68),
}

class FaceTransfer:
    def __init__(self, config):
        self.config = config

    def transfer(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, r_scores, log):
        """
        얼굴 전이 v5 (Pure Source Shape + Reference Rotation)
        - 형태(Shape): Source 얼굴의 상대 좌표를 그대로 사용 (Identity 보존)
        - 각도(Angle): Ref 양쪽 눈 각도와 Source 양쪽 눈 각도의 차이만큼 Source를 회전
        - 위치(Pos): Source 목 길이와 Ref 목 방향을 결합한 Anchor에 배치
        """
        print("\n" + "="*60)
        print("👤 [DEBUG] FaceTransfer.transfer() - v5 (Src Identity + Ref Angle)")
        print("="*60)
        
        if not self.config.face_rendering.enabled:
            print("   ❌ face_rendering disabled")
            return
        
        # 주요 키포인트 인덱스
        nose = BODY_KEYPOINTS['nose']
        l_eye = BODY_KEYPOINTS['left_eye']
        r_eye = BODY_KEYPOINTS['right_eye']
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        # ============================================================
        # 1. 앵커 계산: Source 목 길이 유지 + Ref 목 방향 적용 (기존 유지)
        # ============================================================
        s_sh_center = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
        s_neck_len = calculate_distance(s_kpts[nose], s_sh_center)
        
        r_sh_center = (r_kpts[l_sh] + r_kpts[r_sh]) / 2
        r_neck_vec = r_kpts[nose] - r_sh_center
        r_neck_dir = normalize_vector(r_neck_vec)
        
        t_sh_center = (t_kpts[l_sh] + t_kpts[r_sh]) / 2
        
        # 앵커: Trans 어깨에서 Ref 방향으로, Source 길이만큼 이동
        target_neck_len = max(s_neck_len, 20.0) 
        anchor = t_sh_center + r_neck_dir * target_neck_len
        
        print(f"\n📍 Anchor Calculation:")
        print(f"   Src Neck Length: {s_neck_len:.1f}")
        print(f"   Ref Neck Dir: ({r_neck_dir[0]:.2f}, {r_neck_dir[1]:.2f})")
        print(f"   New Anchor: ({anchor[0]:.1f}, {anchor[1]:.1f})")
        
        # ============================================================
        # 2. 회전 각도 계산 (Rotation Angle Calculation)
        # ============================================================
        # Source 눈 각도 (수평선 기준)
        s_eye_vec = s_kpts[r_eye] - s_kpts[l_eye]
        s_angle = np.arctan2(s_eye_vec[1], s_eye_vec[0])
        
        # Reference 눈 각도 (수평선 기준)
        r_eye_vec = r_kpts[r_eye] - r_kpts[l_eye]
        r_angle = np.arctan2(r_eye_vec[1], r_eye_vec[0])
        
        # 회전해야 할 양 (Delta)
        delta_angle = r_angle - s_angle
        
        print(f"\n📐 Rotation Analysis:")
        print(f"   Src Angle: {np.degrees(s_angle):.1f}°")
        print(f"   Ref Angle: {np.degrees(r_angle):.1f}°")
        print(f"   >>> Delta Rotation: {np.degrees(delta_angle):.1f}°")
        
        # 회전 행렬 (Rotation Matrix)
        cos_a = np.cos(delta_angle)
        sin_a = np.sin(delta_angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # Source 얼굴 중심 (회전축)
        # (68 랜드마크가 없는 경우 COCO Nose 사용)
        ref_face_nose_idx = FACE_START_IDX + 30
        src_face_center = s_kpts[ref_face_nose_idx] if s_scores[ref_face_nose_idx] > 0.3 else s_kpts[nose]
        
        # ============================================================
        # 3. 전체 얼굴 전이 (Source 형태 + 회전 적용)
        # ============================================================
        transferred_count = 0
        
        # 68 랜드마크 + COCO Head Parts 통합 처리
        # 주의: COCO Parts(눈,코,귀)도 함께 회전시켜야 함
        all_face_indices = list(range(FACE_START_IDX, FACE_END_IDX + 1)) + \
                           [nose, l_eye, r_eye, BODY_KEYPOINTS['left_ear'], BODY_KEYPOINTS['right_ear']]
        
        for i in all_face_indices:
            # 설정 체크 (68 랜드마크인 경우)
            if i >= FACE_START_IDX:
                local_idx = i - FACE_START_IDX
                part_name = self._get_part_name(local_idx)
                part_config = self.config.face_rendering.parts.get(part_name)
                if part_config and not part_config.enabled:
                    t_scores[i] = 0.0
                    continue
            
            # Source 점수가 유효한 경우에만 전이 (Source 형태를 쓰므로)
            if s_scores[i] > 0.1:
                # 1. Source 중심 기준 상대 좌표 계산
                rel_vec = s_kpts[i] - src_face_center
                
                # 2. 회전 적용 (Rotate)
                rotated_vec = np.dot(rotation_matrix, rel_vec)
                
                # 3. Anchor 위치에 배치
                t_kpts[i] = anchor + rotated_vec
                
                # 점수는 Source 점수 혹은 Ref 점수 중 높은 것 사용 (또는 Source 유지)
                t_scores[i] = s_scores[i]
                
                if i >= FACE_START_IDX:
                    log[f'face_{i}'] = 'src_rotated'
                    transferred_count += 1
            else:
                # Source가 없으면 전이 불가 (Ref 형태를 쓰지 않기로 했으므로)
                t_scores[i] = 0.0

        print(f"   ✅ Transferred {transferred_count} face keypoints using Source Identity + Ref Angle")

    def _get_part_name(self, idx):
        for name, r in FACE_PARTS_IDX.items():
            if idx in r: return name
        return None