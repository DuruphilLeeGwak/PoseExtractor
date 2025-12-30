"""
Post Processor (Simplified v2.0)

변경사항:
- Case 기반 처리를 단순화된 Boolean 기반으로 변경
"""
import numpy as np
from ..extractors.keypoint_constants import BODY_KEYPOINTS, FEET_KEYPOINTS

# 하반신 인덱스
LOWER_INDICES = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles

class PostProcessor:
    """후처리 프로세서 (단순화)"""
    
    def __init__(self, config):
        self.config = config

    def process(
        self, 
        kpts: np.ndarray, 
        scores: np.ndarray, 
        src_scores: np.ndarray,
        ref_scores: np.ndarray,
        should_transfer_lower: bool
    ) -> tuple:
        """
        후처리 수행
        
        Args:
            kpts: 전이된 키포인트
            scores: 전이된 점수
            src_scores: Source 원본 점수
            ref_scores: Reference 원본 점수
            should_transfer_lower: 하반신 전이 여부
        
        Returns:
            (processed_kpts, processed_scores)
        
        Note:
            Source에 없던 하반신 키포인트가 생성된 경우 제거
            (예: 상반신 Source → 전신 Reference)
        """
        new_scores = scores.copy()
        
        # Source에 하반신이 없었는데 전이에서 생성된 경우 제거
        if not should_transfer_lower or not self._src_has_lower_body(src_scores):
            print("\n[PostProcessor] Removing lower body (Source doesn't have it)")
            for idx in LOWER_INDICES:
                if idx < len(src_scores) and src_scores[idx] < self.config.kpt_threshold:
                    if idx < len(new_scores): 
                        new_scores[idx] = 0.0
            
            # 발 키포인트도 제거
            if FEET_KEYPOINTS:
                for idx in FEET_KEYPOINTS.values():
                    if idx < len(src_scores) and src_scores[idx] < self.config.kpt_threshold:
                        if idx < len(new_scores): 
                            new_scores[idx] = 0.0
        
        return kpts, new_scores
    
    def _src_has_lower_body(self, src_scores: np.ndarray, threshold: float = 0.1) -> bool:
        """Source에 유효한 하반신이 있는지 확인"""
        valid_count = 0
        for idx in LOWER_INDICES:
            if idx < len(src_scores) and src_scores[idx] > threshold:
                valid_count += 1
        return valid_count >= 2  # 최소 2개 이상의 하반신 키포인트

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
        content_w = max_x - min_x
        content_h = max_y - min_y
        
        final_w = int(content_w + base_pad * 2)
        final_h = int(content_h + base_pad * 2 + head_pad)
        
        # 3. 좌표 이동 (Shift)
        shift_x = -min_x + base_pad
        shift_y = -min_y + base_pad + head_pad
        
        final_kpts = kpts.copy()
        final_kpts[:, 0] += shift_x
        final_kpts[:, 1] += shift_y
        
        print(f"   ✂️ [Final Crop] Content: {content_w:.0f}x{content_h:.0f} -> Canvas: {final_w}x{final_h}")
        print(f"      Shift: ({shift_x:.1f}, {shift_y:.1f}) (Top-Left)")
        
        return final_kpts, (final_h, final_w)