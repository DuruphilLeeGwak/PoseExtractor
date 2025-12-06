import numpy as np
from .align_manager import AlignmentCase, LOWER_INDICES
from ..extractors.keypoint_constants import BODY_KEYPOINTS, FEET_KEYPOINTS

class PostProcessor:
    def __init__(self, config):
        self.config = config

    def process_by_case(self, kpts, scores, case, src_scores):
        new_scores = scores.copy()
        # Case D: 상반신 -> 전신 (하반신 제거) - 기존 유지
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
        """
        [Step 10] 최종 캔버스 크기 결정 및 좌표 이동 (데이터 보존 최우선)
        """
        # 1. 유효한 모든 키포인트의 범위(BBox) 계산
        valid_mask = (scores > 0.01) # 점수가 조금이라도 있으면 살림
        if not np.any(valid_mask): return kpts, (100, 100)
        
        valid_kpts = kpts[valid_mask]
        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)
        
        base_pad = self.config.crop_padding_px
        
        # 2. 캔버스 크기 계산 (모든 점을 포함하도록)
        # 너비 = (최대X - 최소X) + 패딩
        # 높이 = (최대Y - 최소Y) + 패딩 + 머리패딩
        content_w = max_x - min_x
        content_h = max_y - min_y
        
        final_w = int(content_w + base_pad * 2)
        final_h = int(content_h + base_pad * 2 + head_pad)
        
        # 3. 좌표 이동 (Shift)
        # 모든 점이 (패딩, 패딩+머리패딩) 위치에서 시작하도록 이동
        # 즉, min_x가 0이 되도록 빼고, 패딩을 더함
        shift_x = -min_x + base_pad
        shift_y = -min_y + base_pad + head_pad
        
        final_kpts = kpts.copy()
        final_kpts[:, 0] += shift_x
        final_kpts[:, 1] += shift_y
        
        print(f"   ✂️ [Final Crop] Content: {content_w:.0f}x{content_h:.0f} -> Canvas: {final_w}x{final_h}")
        print(f"      Shift: ({shift_x:.1f}, {shift_y:.1f}) (Top-Left)")
        
        return final_kpts, (final_h, final_w)