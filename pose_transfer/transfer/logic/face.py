import numpy as np
from ...extractors.keypoint_constants import BODY_KEYPOINTS, FACE_START_IDX, FACE_END_IDX
from ...utils.geometry import calculate_distance

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
        얼굴 전이 v2
        - REF 얼굴 형태(각도) 사용
        - SRC 얼굴 크기로 스케일링
        - REF 비율 기준으로 얼굴 위치 계산 (NEW!)
        """
        print("\n" + "="*60)
        print("👤 [DEBUG] FaceTransfer.transfer() - v2")
        print("="*60)
        
        if not self.config.face_rendering.enabled:
            print("   ❌ face_rendering disabled")
            return
        
        nose = BODY_KEYPOINTS['nose']
        l_eye = BODY_KEYPOINTS['left_eye']
        r_eye = BODY_KEYPOINTS['right_eye']
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        # ============================================================
        # 1. 얼굴 크기 비율 계산 (SRC / REF)
        # ============================================================
        s_eye_dist = calculate_distance(s_kpts[l_eye], s_kpts[r_eye])
        r_eye_dist = calculate_distance(r_kpts[l_eye], r_kpts[r_eye])
        
        face_scale = s_eye_dist / r_eye_dist if r_eye_dist > 0 else 1.0
        
        print(f"\n📐 Face Scale Calculation:")
        print(f"   src_eye_distance: {s_eye_dist:.1f}px")
        print(f"   ref_eye_distance: {r_eye_dist:.1f}px")
        print(f"   face_scale (src/ref): {face_scale:.3f}")
        
        # ============================================================
        # 2. Y축 회전 분석 (정보 출력용)
        # ============================================================
        y_rotation_indicator = r_eye_dist / s_eye_dist
        print(f"\n📐 Y-axis Rotation Analysis:")
        print(f"   eye_distance_ratio (ref/src): {y_rotation_indicator:.3f}")
        
        if y_rotation_indicator < 0.85:
            estimated_y_angle = np.degrees(np.arccos(min(y_rotation_indicator, 1.0)))
            print(f"   ✅ REF is turned ~{estimated_y_angle:.0f}° sideways")
        else:
            print(f"   ℹ️ Both faces appear roughly frontal")
        
        # ============================================================
        # 3. 앵커 위치 계산 (NEW: REF 비율 기준)
        # ============================================================
        # REF에서 코-어깨중심 오프셋 계산
        ref_sh_center = (r_kpts[l_sh] + r_kpts[r_sh]) / 2
        ref_nose_offset = r_kpts[nose] - ref_sh_center
        
        # TRANS 어깨 중심 가져오기
        trans_sh_center = (t_kpts[l_sh] + t_kpts[r_sh]) / 2
        
        # REF 오프셋을 face_scale 적용하여 TRANS 어깨 중심에 배치
        anchor = trans_sh_center + ref_nose_offset * face_scale
        
        print(f"\n📍 Anchor Calculation (NEW):")
        print(f"   ref_shoulder_center: ({ref_sh_center[0]:.1f}, {ref_sh_center[1]:.1f})")
        print(f"   ref_nose: ({r_kpts[nose][0]:.1f}, {r_kpts[nose][1]:.1f})")
        print(f"   ref_nose_offset: ({ref_nose_offset[0]:.1f}, {ref_nose_offset[1]:.1f})")
        print(f"   trans_shoulder_center: ({trans_sh_center[0]:.1f}, {trans_sh_center[1]:.1f})")
        print(f"   anchor (trans_sh_center + offset*scale): ({anchor[0]:.1f}, {anchor[1]:.1f})")
        print(f"   (OLD anchor would be src_nose: ({s_kpts[nose][0]:.1f}, {s_kpts[nose][1]:.1f}))")
        
        # ============================================================
        # 4. REF 얼굴 중심점 (68-landmark 코 끝)
        # ============================================================
        ref_face_nose_idx = FACE_START_IDX + 30
        ref_face_nose = r_kpts[ref_face_nose_idx] if r_scores[ref_face_nose_idx] > 0.3 else r_kpts[nose]
        
        # SRC 얼굴 중심 (fallback용)
        src_face_nose = s_kpts[ref_face_nose_idx] if s_scores[ref_face_nose_idx] > 0.3 else s_kpts[nose]
        
        # ============================================================
        # 5. 68 랜드마크 전이 (REF 형태 + SRC 크기)
        # ============================================================
        transferred = 0
        skipped_no_ref = 0
        
        for i in range(FACE_START_IDX, FACE_END_IDX + 1):
            local_idx = i - FACE_START_IDX
            part_name = self._get_part_name(local_idx)
            
            # YAML 설정 체크
            part_config = self.config.face_rendering.parts.get(part_name)
            if part_config and not part_config.enabled:
                t_scores[i] = 0.0
                continue
            
            # REF 키포인트가 유효한지 확인
            if r_scores[i] > 0.3:
                # REF 형태 사용
                rel = r_kpts[i] - ref_face_nose
                scaled_rel = rel * face_scale
                t_kpts[i] = anchor + scaled_rel
                t_scores[i] = r_scores[i]
                log[f'face_{i}'] = 'ref_shape'
                transferred += 1
            elif s_scores[i] > 0.3:
                # REF에 없으면 SRC 사용 (fallback)
                rel = s_kpts[i] - src_face_nose
                t_kpts[i] = anchor + rel
                t_scores[i] = s_scores[i]
                log[f'face_{i}'] = 'src_fallback'
                transferred += 1
            else:
                skipped_no_ref += 1
        
        print(f"\n📊 68 Landmark Transfer Result:")
        print(f"   transferred: {transferred}/68")
        print(f"   skipped (no valid ref/src): {skipped_no_ref}")
        print(f"   ✅ REF shape + SRC scale")
        print(f"   ✅ Anchor based on REF offset ratio")

        # ============================================================
        # 6. COCO 기본 얼굴 키포인트 (nose, eyes, ears)
        # ============================================================
        ref_nose_pos = r_kpts[nose]
        
        head_parts = [
            (nose, 'nose'),
            (l_eye, 'left_eye'),
            (r_eye, 'right_eye'),
            (BODY_KEYPOINTS['left_ear'], 'left_ear'),
            (BODY_KEYPOINTS['right_ear'], 'right_ear')
        ]
        
        print(f"\n📊 COCO Head Keypoints:")
        for idx, name in head_parts:
            if r_scores[idx] > 0.3:
                rel = r_kpts[idx] - ref_nose_pos
                scaled_rel = rel * face_scale
                t_kpts[idx] = anchor + scaled_rel
                t_scores[idx] = r_scores[idx]
                print(f"   ✅ {name}: REF shape, pos=({t_kpts[idx][0]:.1f}, {t_kpts[idx][1]:.1f})")
            elif s_scores[idx] > 0.3:
                rel = s_kpts[idx] - s_kpts[nose]
                t_kpts[idx] = anchor + rel
                t_scores[idx] = s_scores[idx]
                print(f"   ⚠️ {name}: SRC fallback")
        
        print(f"\n" + "="*60)

    def _get_part_name(self, idx):
        for name, r in FACE_PARTS_IDX.items():
            if idx in r: return name
        return None