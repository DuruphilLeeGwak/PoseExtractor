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
        # 3D depth 사용 여부 (기본값 False로 안정성 우선)
        self.use_3d_depth = getattr(config.face_rendering, 'use_3d_depth', False)

    def transfer(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, r_scores, log, 
                 src_depth=None, ref_depth=None):
        """
        얼굴 전이 v6 (2D Body Anchor + Optional 3D Face Offset)
        
        핵심 개선:
        - Face center를 항상 COCO nose로 통일 (2D 좌표계)
        - Anchor는 2D body keypoint 기반으로 안정적 계산
        - Depth는 선택적 보정으로만 사용 (use_3d_depth=True일 때)
        
        Args:
            t_kpts, t_scores: Transfer 결과 (출력)
            s_kpts, s_scores: Source keypoints
            r_kpts, r_scores: Reference keypoints
            log: 디버그 로그
            src_depth: Source depth map (optional, HxW numpy array)
            ref_depth: Reference depth map (optional, HxW numpy array)
        """
        print("\n" + "="*60)
        print("👤 [DEBUG] FaceTransfer.transfer() - v6 (Unified 2D Coord)")
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
        # 1. 2D Body Anchor 계산 (기존 로직 유지)
        # ============================================================
        s_sh_center = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
        s_neck_len = calculate_distance(s_kpts[nose], s_sh_center)
        
        r_sh_center = (r_kpts[l_sh] + r_kpts[r_sh]) / 2
        r_neck_vec = r_kpts[nose] - r_sh_center
        r_neck_dir = normalize_vector(r_neck_vec)
        
        t_sh_center = (t_kpts[l_sh] + t_kpts[r_sh]) / 2
        
        # 기본 2D Anchor
        target_neck_len = max(s_neck_len, 20.0)
        anchor_2d = t_sh_center + r_neck_dir * target_neck_len
        
        print(f"\n📍 2D Anchor Calculation:")
        print(f"   Src Neck Length: {s_neck_len:.1f}")
        print(f"   Ref Neck Dir: ({r_neck_dir[0]:.2f}, {r_neck_dir[1]:.2f})")
        print(f"   Base 2D Anchor: ({anchor_2d[0]:.1f}, {anchor_2d[1]:.1f})")
        
        # ============================================================
        # 2. Optional Depth Offset (3D 보정)
        # ============================================================
        anchor = anchor_2d.copy()
        depth_offset_applied = False
        
        if self.use_3d_depth and src_depth is not None and ref_depth is not None:
            try:
                # COCO nose 위치에서 depth 값 추출
                h, w = src_depth.shape
                
                # Source nose depth
                src_nose_y = int(np.clip(s_kpts[nose][1], 0, h - 1))
                src_nose_x = int(np.clip(s_kpts[nose][0], 0, w - 1))
                src_nose_depth = src_depth[src_nose_y, src_nose_x]
                
                # Reference nose depth
                ref_nose_y = int(np.clip(r_kpts[nose][1], 0, h - 1))
                ref_nose_x = int(np.clip(r_kpts[nose][0], 0, w - 1))
                ref_nose_depth = ref_depth[ref_nose_y, ref_nose_x]
                
                # Depth 차이를 Y축 오프셋으로 변환
                # (카메라 정면 가정: Z축 멀어지면 위로, 가까우면 아래로)
                depth_diff = ref_nose_depth - src_nose_depth
                z_scale = 0.5  # 튜닝 파라미터
                
                depth_offset = np.array([0, depth_diff * z_scale])
                anchor = anchor_2d + depth_offset
                depth_offset_applied = True
                
                print(f"\n🌊 Depth Offset:")
                print(f"   Src Nose Depth: {src_nose_depth:.3f}")
                print(f"   Ref Nose Depth: {ref_nose_depth:.3f}")
                print(f"   Depth Diff: {depth_diff:.3f}")
                print(f"   Y Offset: {depth_offset[1]:.1f}")
                print(f"   Final Anchor: ({anchor[0]:.1f}, {anchor[1]:.1f})")
                
            except Exception as e:
                print(f"   ⚠️ Depth offset 실패 (fallback to 2D): {e}")
                anchor = anchor_2d
        
        if not depth_offset_applied:
            print(f"   ℹ️ Using 2D Anchor only (3D depth: {'disabled' if not self.use_3d_depth else 'unavailable'})")
        
        # ============================================================
        # 3. Face Center 통일 (항상 COCO Nose 사용)
        # ============================================================
        # CRITICAL FIX: 68 landmarks와 좌표계를 통일하기 위해 항상 COCO nose 사용
        src_face_center = s_kpts[nose]  # 2D body keypoint
        
        print(f"\n🎯 Face Center:")
        print(f"   Using COCO Nose (2D): ({src_face_center[0]:.1f}, {src_face_center[1]:.1f})")
        
        # ============================================================
        # 4. 회전 각도 계산 (2D Eye Angle)
        # ============================================================
        s_eye_vec = s_kpts[r_eye] - s_kpts[l_eye]
        s_angle = np.arctan2(s_eye_vec[1], s_eye_vec[0])
        
        r_eye_vec = r_kpts[r_eye] - r_kpts[l_eye]
        r_angle = np.arctan2(r_eye_vec[1], r_eye_vec[0])
        
        delta_angle = r_angle - s_angle
        
        print(f"\n📐 Rotation Analysis:")
        print(f"   Src Angle: {np.degrees(s_angle):.1f}°")
        print(f"   Ref Angle: {np.degrees(r_angle):.1f}°")
        print(f"   >>> Delta Rotation: {np.degrees(delta_angle):.1f}°")
        
        # 회전 행렬
        cos_a = np.cos(delta_angle)
        sin_a = np.sin(delta_angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # ============================================================
        # 5. 전체 얼굴 전이 (68 landmarks + COCO head parts)
        # ============================================================
        transferred_count = 0
        
        # 68 랜드마크 + COCO Head Parts 통합 처리
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
            
            # Source 점수가 유효한 경우에만 전이
            if s_scores[i] > 0.1:
                # 1. Source 중심(COCO nose) 기준 상대 좌표 계산
                rel_vec = s_kpts[i] - src_face_center
                
                # 2. 회전 적용
                rotated_vec = np.dot(rotation_matrix, rel_vec)
                
                # 3. Anchor 위치에 배치
                t_kpts[i] = anchor + rotated_vec
                
                # 4. 점수는 Source 점수 유지
                t_scores[i] = s_scores[i]
                
                if i >= FACE_START_IDX:
                    log[f'face_{i}'] = 'src_rotated_v6'
                    transferred_count += 1
            else:
                # Source가 없으면 전이 불가
                t_scores[i] = 0.0

        mode_str = "2D+3D" if depth_offset_applied else "2D-only"
        print(f"   ✅ Transferred {transferred_count} face keypoints ({mode_str} mode)")
        log['face_mode'] = mode_str

    def _get_part_name(self, idx):
        for name, r in FACE_PARTS_IDX.items():
            if idx in r: return name
        return None