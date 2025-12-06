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
        src_type = self._get_type(src_scores)
        ref_type = self._get_type(ref_scores)
        
        if src_type == BodyType.FULL and ref_type == BodyType.FULL: case = AlignmentCase.A
        elif src_type == BodyType.UPPER and ref_type == BodyType.UPPER: case = AlignmentCase.B
        elif src_type == BodyType.FULL and ref_type == BodyType.UPPER: case = AlignmentCase.C
        else: case = AlignmentCase.D
        
        return src_type, ref_type, case

    def _get_type(self, scores):
        valid = sum(1 for i in LOWER_INDICES if i < len(scores) and scores[i] > self.config.kpt_threshold)
        return BodyType.FULL if valid >= self.config.full_body_min_valid_lower else BodyType.UPPER

    def calc_scale(self, src_face_size, ref_face_size):
        if not self.config.face_scale_enabled or ref_face_size < 1:
            return 1.0
        scale = np.clip(src_face_size / ref_face_size, 0.3, 3.0)
        return 1.0 if abs(scale - 1.0) < 0.05 else scale

    def align_coordinates(self, kpts, scores, case, src_person_bbox, src_face_bbox, face_bbox_func):
        """
        좌표(kpts) 자체를 이동(Shift)시켜 정렬
        """
        aligned_kpts = kpts.copy()
        
        if case == AlignmentCase.A:
            # 발바닥(Bottom) 기준 정렬
            src_bottom = src_person_bbox.bbox[3]
            
            # Transferred의 가장 낮은 Y좌표
            feet_idx = [15, 16, 17, 18, 19, 20, 21, 22]
            valid_y = [kpts[i][1] for i in feet_idx if i < len(scores) and scores[i] > 0.1]
            trans_bottom = max(valid_y) if valid_y else 0
            
            if trans_bottom > 0:
                shift_y = src_bottom - trans_bottom
                aligned_kpts[:, 1] += shift_y
                print(f"   🦶 [Align A] Shift Y: {shift_y:.1f}")
                
        else:
            # 얼굴 중심 기준 정렬 (B, C, D)
            src_cx, src_cy = src_face_bbox.center
            
            trans_face_info = face_bbox_func(kpts, scores)
            trans_cx, trans_cy = trans_face_info.center
            
            shift_x = src_cx - trans_cx
            shift_y = src_cy - trans_cy
            
            aligned_kpts[:, 0] += shift_x
            aligned_kpts[:, 1] += shift_y
            print(f"   👤 [Align {case.value}] Shift: ({shift_x:.1f}, {shift_y:.1f})")
            
        return aligned_kpts