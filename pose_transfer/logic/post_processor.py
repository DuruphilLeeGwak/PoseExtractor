import numpy as np
from .align_manager import AlignmentCase, LOWER_INDICES
# 경로 주의: constants는 상위 extractors에 있음
from ..extractors.keypoint_constants import BODY_KEYPOINTS, FEET_KEYPOINTS

class PostProcessor:
    def __init__(self, config):
        self.config = config

    def process_by_case(self, kpts, scores, case, src_scores):
        new_scores = scores.copy()
        # Case D: 상반신 -> 전신 (하반신 제거)
        if case == AlignmentCase.D:
            for idx in LOWER_INDICES:
                if idx < len(src_scores) and src_scores[idx] < self.config.kpt_threshold:
                    if idx < len(new_scores): new_scores[idx] = 0.0
            if FEET_KEYPOINTS:
                for idx in FEET_KEYPOINTS.values():
                    if idx < len(src_scores) and src_scores[idx] < self.config.kpt_threshold:
                        if idx < len(new_scores): new_scores[idx] = 0.0
        return kpts, new_scores

    def apply_head_padding(self, kpts, scores):
        nose = BODY_KEYPOINTS.get('nose', 0)
        neck = BODY_KEYPOINTS.get('left_shoulder', 5)
        
        if scores[nose] <= 0.1: return 50.0 
        
        head_len = np.linalg.norm(kpts[nose] - kpts[neck])
        padding_px = head_len * 1.5 * self.config.head_padding_ratio
        return max(20.0, padding_px)

    def finalize_canvas(self, kpts, scores, head_pad):
        valid = kpts[scores > self.config.kpt_threshold]
        if len(valid) == 0: return kpts, (100, 100)
        
        min_xy = valid.min(axis=0)
        max_xy = valid.max(axis=0)
        
        base_pad = self.config.crop_padding_px
        
        w = int((max_xy[0] - min_xy[0]) + base_pad * 2)
        h = int((max_xy[1] - min_xy[1]) + base_pad * 2 + head_pad)
        
        final_kpts = kpts.copy()
        final_kpts[:, 0] -= (min_xy[0] - base_pad)
        final_kpts[:, 1] -= (min_xy[1] - base_pad - head_pad)
        
        print(f"   ✂️ [Final Crop] Canvas: {w}x{h}, HeadPad: {head_pad:.1f}")
        return final_kpts, (h, w)