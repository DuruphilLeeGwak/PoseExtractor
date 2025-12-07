import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

# 상수 정의
LOWER_INDICES = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles

class BodyType(Enum):
    FULL = "full"
    HALF = "half"  # 상반신 (UPPER → HALF로 변경)

class AlignmentCase(Enum):
    """
    정렬 케이스: {SRC}_{REF} 형식
    F = Full (전신)
    H = Half (상반신)
    """
    F_F = "F_F"  # Full → Full (전신 → 전신)
    F_H = "F_H"  # Full → Half (전신 → 상반신)
    H_F = "H_F"  # Half → Full (상반신 → 전신)
    H_H = "H_H"  # Half → Half (상반신 → 상반신)

class AlignManager:
    def __init__(self, config):
        self.config = config

    def determine_case(self, src_kpts, src_scores, ref_kpts, ref_scores, 
                       src_img_size=None, ref_img_size=None):
        """
        src/ref의 Body Type을 판별하고 케이스 결정
        """
        print("\n🔍 [DEBUG] AlignManager.determine_case()")
        
        src_type = self._get_type(src_scores, src_kpts, "src", src_img_size)
        ref_type = self._get_type(ref_scores, ref_kpts, "ref", ref_img_size)
        
        # Case 결정: {SRC}_{REF}
        if src_type == BodyType.FULL and ref_type == BodyType.FULL:
            case = AlignmentCase.F_F
        elif src_type == BodyType.FULL and ref_type == BodyType.HALF:
            case = AlignmentCase.F_H
        elif src_type == BodyType.HALF and ref_type == BodyType.FULL:
            case = AlignmentCase.H_F
        else:  # HALF → HALF
            case = AlignmentCase.H_H
        
        print(f"   Result: {src_type.value} → {ref_type.value} = Case {case.value}")
        
        return src_type, ref_type, case

    def _get_type(self, scores, kpts, label, img_size=None):
        """
        Body Type 판별 (v3: 해부학적 순서 검증)
        
        전신(FULL) 판별 조건 (모두 충족해야 함):
        1. 하반신 키포인트 점수 >= threshold 개수 충족
        2. 해부학적 순서: hip.y < knee.y < ankle.y (한쪽이라도)
        3. knee/ankle 점수가 ghost_threshold 이상 (한쪽이라도)
        
        하나라도 실패하면 상반신(HALF)
        """
        print(f"\n   🔍 [DEBUG] _get_type({label})")
        
        # 설정값 가져오기
        kpt_threshold = getattr(self.config, 'kpt_threshold', 0.1)
        min_valid = getattr(self.config, 'full_body_min_valid_lower', 4)
        ghost_score_threshold = getattr(self.config, 'ghost_score_threshold', 2.0)
        
        # ============================================================
        # [조건 1] 하반신 키포인트 점수 체크 (기존 로직)
        # ============================================================
        print(f"      [조건 1] 하반신 점수 체크 (threshold={kpt_threshold})")
        
        valid_count = 0
        for i in LOWER_INDICES:
            if i < len(scores):
                score = scores[i]
                is_valid = score > kpt_threshold
                if is_valid:
                    valid_count += 1
        
        score_check_pass = valid_count >= min_valid
        print(f"         valid_count: {valid_count} >= {min_valid}? {'PASS ✅' if score_check_pass else 'FAIL ❌'}")
        
        if not score_check_pass:
            print(f"      → HALF (점수 체크 실패)")
            return BodyType.HALF
        
        # ============================================================
        # [조건 2] 해부학적 순서 검증 (핵심!)
        # 정상 전신: hip.y < knee.y < ankle.y
        # Ghost Leg: 이 순서가 깨짐 (DWPose가 추측한 경우)
        # ============================================================
        print(f"\n      [조건 2] 해부학적 순서 검증 (hip.y < knee.y < ankle.y)")
        
        # 골반이 없는 경우도 처리
        l_hip_y = kpts[11][1] if 11 < len(kpts) and scores[11] > kpt_threshold else None
        r_hip_y = kpts[12][1] if 12 < len(kpts) and scores[12] > kpt_threshold else None
        l_knee_y = kpts[13][1] if 13 < len(kpts) and scores[13] > kpt_threshold else None
        r_knee_y = kpts[14][1] if 14 < len(kpts) and scores[14] > kpt_threshold else None
        l_ankle_y = kpts[15][1] if 15 < len(kpts) and scores[15] > kpt_threshold else None
        r_ankle_y = kpts[16][1] if 16 < len(kpts) and scores[16] > kpt_threshold else None
        
        def check_leg_order(hip_y, knee_y, ankle_y, side):
            """한쪽 다리의 해부학적 순서 검증"""
            if hip_y is None or knee_y is None:
                print(f"         {side}: hip 또는 knee 없음 → 검증 불가")
                return False
            
            # hip.y < knee.y 체크 (무릎이 골반보다 아래에 있어야 함)
            hip_knee_ok = hip_y < knee_y
            
            # ankle이 있으면 knee.y < ankle.y도 체크
            if ankle_y is not None:
                knee_ankle_ok = knee_y < ankle_y
                order_ok = hip_knee_ok and knee_ankle_ok
                print(f"         {side}: hip.y({hip_y:.0f}) < knee.y({knee_y:.0f}) < ankle.y({ankle_y:.0f})?")
                print(f"                hip<knee: {hip_knee_ok}, knee<ankle: {knee_ankle_ok} → {'OK ✅' if order_ok else 'FAIL ❌'}")
            else:
                order_ok = hip_knee_ok
                print(f"         {side}: hip.y({hip_y:.0f}) < knee.y({knee_y:.0f})? {hip_knee_ok} → {'OK ✅' if order_ok else 'FAIL ❌'}")
            
            return order_ok
        
        left_order_ok = check_leg_order(l_hip_y, l_knee_y, l_ankle_y, "Left ")
        right_order_ok = check_leg_order(r_hip_y, r_knee_y, r_ankle_y, "Right")
        
        anatomy_check_pass = left_order_ok or right_order_ok
        print(f"         At least one leg OK? {'PASS ✅' if anatomy_check_pass else 'FAIL ❌'}")
        
        if not anatomy_check_pass:
            print(f"      → HALF (해부학적 순서 검증 실패 - Ghost Leg)")
            return BodyType.HALF
        
        # ============================================================
        # [조건 3] Ghost Score 검증
        # DWPose 정상 점수: 3.0 ~ 8.0
        # 추측 점수: 0.5 ~ 2.0
        # knee/ankle 점수가 너무 낮으면 Ghost
        # ============================================================
        print(f"\n      [조건 3] Ghost Score 검증 (threshold={ghost_score_threshold})")
        
        knee_scores = []
        ankle_scores = []
        
        if 13 < len(scores): knee_scores.append(scores[13])
        if 14 < len(scores): knee_scores.append(scores[14])
        if 15 < len(scores): ankle_scores.append(scores[15])
        if 16 < len(scores): ankle_scores.append(scores[16])
        
        max_knee_score = max(knee_scores) if knee_scores else 0
        max_ankle_score = max(ankle_scores) if ankle_scores else 0
        
        print(f"         knee scores: {[f'{s:.2f}' for s in knee_scores]}, max={max_knee_score:.2f}")
        print(f"         ankle scores: {[f'{s:.2f}' for s in ankle_scores]}, max={max_ankle_score:.2f}")
        
        # knee 또는 ankle 중 하나라도 threshold 이상이어야 함
        ghost_check_pass = max_knee_score >= ghost_score_threshold or max_ankle_score >= ghost_score_threshold
        print(f"         max(knee, ankle) >= {ghost_score_threshold}? {'PASS ✅' if ghost_check_pass else 'FAIL ❌'}")
        
        if not ghost_check_pass:
            print(f"      → HALF (Ghost Score 검증 실패 - 낮은 신뢰도)")
            return BodyType.HALF
        
        # ============================================================
        # 모든 조건 통과 → FULL
        # ============================================================
        print(f"      → FULL (모든 조건 통과)")
        return BodyType.FULL

    def calc_scale(self, src_face_size, ref_face_size):
        """
        얼굴 크기 기반 스케일 팩터 계산
        H_H, H_F 등에서 Src 얼굴 크기에 1:1로 맞추기 위함
        """
        if not self.config.face_scale_enabled or ref_face_size < 1:
            return 1.0
            
        # 스케일 계산
        scale = src_face_size / ref_face_size
        
        # [수정] 스케일 클리핑 범위 완화 (0.2 ~ 5.0)
        # H_H에서 얼굴이 너무 작아지는 것 방지, 또는 너무 커져서 캔버스 확장 유도
        scale = np.clip(scale, 0.2, 5.0)
        
        return scale

    def align_coordinates(self, kpts, scores, case, src_person_bbox, src_face_bbox, face_bbox_func):
        """
        좌표(kpts) 이동(Shift)을 통한 정렬
        전제: kpts는 이미 calc_scale()을 통해 Src와 크기(비율)가 맞춰진 상태임
        """
        print("\n" + "="*60)
        print(f"🔍 [DEBUG] AlignManager.align_coordinates(Case {case.value})")
        print("="*60)
        
        aligned_kpts = kpts.copy()
        
        # Case F_F (전신 -> 전신): 기존대로 발(Bottom) 기준 정렬
        if case == AlignmentCase.F_F:
            print("\n🦶 Case F_F: Bottom-based alignment")
            
            src_bottom = src_person_bbox.bbox[3]
            print(f"   src_person_bbox: {src_person_bbox.bbox}")
            print(f"   src_bottom (y2): {src_bottom}")
            
            feet_idx = [15, 16, 17, 18, 19, 20, 21, 22]
            valid_y = []
            print(f"\n   feet_idx to check: {feet_idx}")
            for i in feet_idx:
                if i < len(scores) and scores[i] > 0.1:
                    valid_y.append(kpts[i][1])
                    print(f"      idx={i}: score={scores[i]:.3f}, y={kpts[i][1]:.1f} ✅")
                else:
                    score = scores[i] if i < len(scores) else 0
                    print(f"      idx={i}: score={score:.3f} ❌")
            
            trans_bottom = max(valid_y) if valid_y else 0
            print(f"   trans_bottom (max): {trans_bottom}")
            
            if trans_bottom > 0:
                shift_y = src_bottom - trans_bottom
                aligned_kpts[:, 1] += shift_y
                print(f"   ✅ shift_y = {src_bottom} - {trans_bottom} = {shift_y:.1f}")
            else:
                print(f"   ❌ trans_bottom = 0, NO SHIFT")
        
        # Case H_F, H_H, F_H: 얼굴 중심 기준 강력 정렬
        else:
            print(f"\n👤 Case {case.value}: Face Center Alignment (Src 기준)")
            
            # 1. Src 이미지의 얼굴 중심
            src_cx, src_cy = src_face_bbox.center
            
            # 2. 현재 전이된 키포인트(Trans)의 얼굴 중심 계산
            # (이미 스케일링이 적용된 상태의 좌표에서 계산)
            trans_face_info = face_bbox_func(kpts, scores)
            trans_cx, trans_cy = trans_face_info.center
            
            print(f"   Src Face Center: ({src_cx:.1f}, {src_cy:.1f})")
            print(f"   Trans Face Center: ({trans_cx:.1f}, {trans_cy:.1f})")
            
            # 3. 이동량 계산 (Src 중심 - Trans 중심)
            shift_x = src_cx - trans_cx
            shift_y = src_cy - trans_cy
            
            # 4. 전체 키포인트 이동
            aligned_kpts[:, 0] += shift_x
            aligned_kpts[:, 1] += shift_y
            
            print(f"   ✅ Shift Applied: x={shift_x:.1f}, y={shift_y:.1f}")
            
            # H_F의 경우 다리가 길어져서 화면 아래로 내려갈 수 있음.
            # 이는 이후 CanvasManager가 처리함.
        
        # 정렬 후 하반신 위치 출력
        print(f"\n📊 After Alignment - Lower Body Status:")
        lower_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        lower_indices = [11, 12, 13, 14, 15, 16]
        for name, idx in zip(lower_names, lower_indices):
            if idx < len(scores):
                score = scores[idx]
                pos = aligned_kpts[idx]
                status = "✅" if score > 0.1 else "❌"
                print(f"   {status} {name:15} (idx={idx}): score={score:.3f}, pos={pos}")
        
        print("="*60)
        return aligned_kpts