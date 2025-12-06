import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

# 상수 정의
LOWER_INDICES = [11, 12, 13, 14, 15, 16] # hips, knees, ankles

class BodyType(Enum):
    FULL = "full"
    UPPER = "upper"

class AlignmentCase(Enum):
    A = "A" # Full -> Full
    B = "B" # Upper -> Upper
    C = "C" # Full -> Upper
    D = "D" # Upper -> Full

class AlignManager:
    def __init__(self, config):
        self.config = config

    def determine_case(self, src_kpts, src_scores, ref_kpts, ref_scores):
        print("\n🔍 [DEBUG] AlignManager.determine_case()")
        
        src_type = self._get_type(src_scores, "src")
        ref_type = self._get_type(ref_scores, "ref")
        
        if src_type == BodyType.FULL and ref_type == BodyType.FULL: case = AlignmentCase.A
        elif src_type == BodyType.UPPER and ref_type == BodyType.UPPER: case = AlignmentCase.B
        elif src_type == BodyType.FULL and ref_type == BodyType.UPPER: case = AlignmentCase.C
        else: case = AlignmentCase.D
        
        print(f"   Result: {src_type.value} -> {ref_type.value} = Case {case.value}")
        
        return src_type, ref_type, case

    def _get_type(self, scores, label):
        print(f"\n   🔍 [DEBUG] _get_type({label})")
        print(f"      LOWER_INDICES: {LOWER_INDICES}")
        print(f"      kpt_threshold: {self.config.kpt_threshold}")
        print(f"      full_body_min_valid_lower: {self.config.full_body_min_valid_lower}")
        
        valid_count = 0
        for i in LOWER_INDICES:
            if i < len(scores):
                score = scores[i]
                is_valid = score > self.config.kpt_threshold
                status = "✅" if is_valid else "❌"
                print(f"      idx={i}: score={score:.3f} {status}")
                if is_valid:
                    valid_count += 1
        
        result = BodyType.FULL if valid_count >= self.config.full_body_min_valid_lower else BodyType.UPPER
        print(f"      valid_count: {valid_count} -> {result.value}")
        
        return result

    def calc_scale(self, src_face_size, ref_face_size):
        if not self.config.face_scale_enabled or ref_face_size < 1:
            return 1.0
        scale = np.clip(src_face_size / ref_face_size, 0.3, 3.0)
        return 1.0 if abs(scale - 1.0) < 0.05 else scale

    def align_coordinates(self, kpts, scores, case, src_person_bbox, src_face_bbox, face_bbox_func):
        """
        좌표(kpts) 자체를 이동(Shift)시켜 정렬
        """
        print("\n" + "="*60)
        print(f"🔍 [DEBUG] AlignManager.align_coordinates(Case {case.value})")
        print("="*60)
        
        aligned_kpts = kpts.copy()
        
        # 입력 키포인트 하반신 상태 확인
        print("\n📊 Input Keypoints - Lower Body Status:")
        lower_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        lower_indices = [11, 12, 13, 14, 15, 16]
        for name, idx in zip(lower_names, lower_indices):
            if idx < len(scores):
                score = scores[idx]
                pos = kpts[idx]
                status = "✅" if score > 0.1 else "❌"
                print(f"   {status} {name:15} (idx={idx:2d}): score={score:.3f}, pos={pos}")
        
        if case == AlignmentCase.A:
            print("\n🦶 Case A: Bottom-based alignment")
            
            # 발바닥(Bottom) 기준 정렬
            src_bottom = src_person_bbox.bbox[3]
            print(f"   src_person_bbox: {src_person_bbox.bbox}")
            print(f"   src_bottom (y2): {src_bottom}")
            
            # Transferred의 가장 낮은 Y좌표
            feet_idx = [15, 16, 17, 18, 19, 20, 21, 22]
            print(f"\n   feet_idx to check: {feet_idx}")
            
            valid_y = []
            for i in feet_idx:
                if i < len(scores):
                    score = scores[i]
                    y = kpts[i][1]
                    is_valid = score > 0.1
                    status = "✅" if is_valid else "❌"
                    print(f"      idx={i}: score={score:.3f}, y={y:.1f} {status}")
                    if is_valid:
                        valid_y.append(y)
            
            trans_bottom = max(valid_y) if valid_y else 0
            print(f"\n   valid_y list: {valid_y}")
            print(f"   trans_bottom (max): {trans_bottom}")
            
            if trans_bottom > 0:
                shift_y = src_bottom - trans_bottom
                aligned_kpts[:, 1] += shift_y
                print(f"   ✅ shift_y = {src_bottom} - {trans_bottom} = {shift_y:.1f}")
            else:
                print(f"   ❌ trans_bottom = 0, NO SHIFT APPLIED!")
                
        else:
            print(f"\n👤 Case {case.value}: Face-based alignment")
            
            # 얼굴 중심 기준 정렬 (B, C, D)
            src_cx, src_cy = src_face_bbox.center
            print(f"   src_face_bbox: {src_face_bbox.bbox}")
            print(f"   src_face_center: ({src_cx:.1f}, {src_cy:.1f})")
            
            trans_face_info = face_bbox_func(kpts, scores)
            trans_cx, trans_cy = trans_face_info.center
            print(f"   trans_face_bbox: {trans_face_info.bbox}")
            print(f"   trans_face_center: ({trans_cx:.1f}, {trans_cy:.1f})")
            
            shift_x = src_cx - trans_cx
            shift_y = src_cy - trans_cy
            
            aligned_kpts[:, 0] += shift_x
            aligned_kpts[:, 1] += shift_y
            print(f"   ✅ shift: ({shift_x:.1f}, {shift_y:.1f})")
        
        # 정렬 후 하반신 상태 확인
        print("\n📊 After Alignment - Lower Body Status:")
        for name, idx in zip(lower_names, lower_indices):
            if idx < len(scores):
                score = scores[idx]
                pos = aligned_kpts[idx]
                status = "✅" if score > 0.1 else "❌"
                print(f"   {status} {name:15} (idx={idx:2d}): score={score:.3f}, pos={pos}")
        
        print("\n" + "="*60)
            
        return aligned_kpts