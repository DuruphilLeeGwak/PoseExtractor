"""
Canvas Manager Module (Final Fix: Variable Name Error Resolved)
- Fix: 'fixed_pad' NameError 해결
- Config의 canvas_padding_ratio를 반영하여 안전한 여백 확보
- 흰색 패딩(BORDER_CONSTANT) 적용
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class CanvasManager:
    def __init__(self, config):
        self.config = config

    def expand_canvas_to_fit(
        self, 
        source_image: np.ndarray, 
        keypoints: np.ndarray, 
        scores: np.ndarray,
        head_pad_px: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        
        h, w = source_image.shape[:2]
        
        # 1. 유효 키포인트 범위(BBox) 계산
        valid_mask = scores > 0.01
        if not np.any(valid_mask):
            return source_image, keypoints, (h, w)
            
        valid_kpts = keypoints[valid_mask]
        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)
        
        # 2. 여백(Padding) 계산
        # (A) 고정 픽셀 패딩 (변수명 통일: fixed_pad)
        fixed_pad = self.config.crop_padding_px
        
        # (B) 비율 패딩
        # 현재 콘텐츠 크기(또는 원본 크기)의 N% 만큼 여유를 둠
        ratio = self.config.canvas_padding_ratio
        ratio_pad_w = int(w * ratio)
        ratio_pad_h = int(h * ratio)
        
        # 최종 필요한 캔버스 경계 (키포인트 + 여백)
        req_x1 = int(min_x - fixed_pad - ratio_pad_w)
        req_y1 = int(min_y - fixed_pad - ratio_pad_h - head_pad_px) # 머리 위는 head_pad 추가
        req_x2 = int(max_x + fixed_pad + ratio_pad_w)
        req_y2 = int(max_y + fixed_pad + ratio_pad_h)
        
        # 3. 원본 이미지 대비 부족한 부분 계산
        # 왼쪽/위쪽이 0보다 작으면 패딩 필요
        pad_l = max(0, -req_x1)
        pad_t = max(0, -req_y1)
        
        # 오른쪽/아래쪽이 이미지 크기보다 크면 패딩 필요
        pad_r = max(0, req_x2 - w)
        pad_b = max(0, req_y2 - h)
        
        # 4. 패딩이 필요 없다면 원본 반환
        if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
            return source_image, keypoints, (h, w)
            
        # 5. 이미지 확장 (흰색 패딩)
        print(f"   🖼️ [Canvas Expansion] Padding (White): T={pad_t}, B={pad_b}, L={pad_l}, R={pad_r}")
        
        padded_image = cv2.copyMakeBorder(
            source_image, pad_t, pad_b, pad_l, pad_r, 
            cv2.BORDER_CONSTANT, value=(255, 255, 255) # White
        )
        
        # 6. 키포인트 이동 (Shift)
        shifted_kpts = keypoints.copy()
        shifted_kpts[:, 0] += pad_l
        shifted_kpts[:, 1] += pad_t
        
        new_h, new_w = padded_image.shape[:2]
        return padded_image, shifted_kpts, (new_h, new_w)