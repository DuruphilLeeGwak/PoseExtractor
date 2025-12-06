"""
Canvas Manager Module (DEBUG VERSION)
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
        
        print("\n" + "="*60)
        print("🔍 [DEBUG] CanvasManager.expand_canvas_to_fit()")
        print("="*60)
        
        h, w = source_image.shape[:2]
        print(f"\n📐 Source Image: {w}x{h}")
        print(f"   head_pad_px: {head_pad_px}")
        print(f"   crop_padding_px: {self.config.crop_padding_px}")
        print(f"   canvas_padding_ratio: {self.config.canvas_padding_ratio}")
        
        # 1. 유효 키포인트 범위(BBox) 계산
        valid_mask = scores > 0.01
        valid_count = np.sum(valid_mask)
        print(f"\n📊 Valid Keypoints (score > 0.01): {valid_count}")
        
        if not np.any(valid_mask):
            print("   ❌ No valid keypoints!")
            return source_image, keypoints, (h, w)
        
        # 하반신 키포인트 상태 확인
        print("\n📊 Lower Body Keypoints Status:")
        lower_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        lower_indices = [11, 12, 13, 14, 15, 16]
        for name, idx in zip(lower_names, lower_indices):
            if idx < len(scores):
                score = scores[idx]
                pos = keypoints[idx]
                is_valid = score > 0.01
                in_bounds = 0 <= pos[0] <= w and 0 <= pos[1] <= h
                status = "✅" if is_valid else "❌"
                bounds = "📍" if in_bounds else "⚠️ OUT"
                print(f"   {status} {name:15}: score={score:.3f}, pos=({pos[0]:.1f}, {pos[1]:.1f}) {bounds}")
            
        valid_kpts = keypoints[valid_mask]
        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)
        
        print(f"\n📏 Valid Keypoints BBox:")
        print(f"   min: ({min_x:.1f}, {min_y:.1f})")
        print(f"   max: ({max_x:.1f}, {max_y:.1f})")
        print(f"   size: {max_x - min_x:.1f} x {max_y - min_y:.1f}")
        
        # 2. 여백(Padding) 계산
        fixed_pad = self.config.crop_padding_px
        ratio = self.config.canvas_padding_ratio
        ratio_pad_w = int(w * ratio)
        ratio_pad_h = int(h * ratio)
        
        print(f"\n📦 Padding Calculation:")
        print(f"   fixed_pad: {fixed_pad}")
        print(f"   ratio_pad: ({ratio_pad_w}, {ratio_pad_h})")
        
        # 최종 필요한 캔버스 경계
        req_x1 = int(min_x - fixed_pad - ratio_pad_w)
        req_y1 = int(min_y - fixed_pad - ratio_pad_h - head_pad_px)
        req_x2 = int(max_x + fixed_pad + ratio_pad_w)
        req_y2 = int(max_y + fixed_pad + ratio_pad_h)
        
        print(f"\n📐 Required Canvas Bounds:")
        print(f"   req_x1: {req_x1}, req_y1: {req_y1}")
        print(f"   req_x2: {req_x2}, req_y2: {req_y2}")
        
        # 3. 원본 이미지 대비 부족한 부분 계산
        pad_l = max(0, -req_x1)
        pad_t = max(0, -req_y1)
        pad_r = max(0, req_x2 - w)
        pad_b = max(0, req_y2 - h)
        
        print(f"\n🔲 Padding Needed:")
        print(f"   Left:   {pad_l} (req_x1={req_x1} < 0? {req_x1 < 0})")
        print(f"   Top:    {pad_t} (req_y1={req_y1} < 0? {req_y1 < 0})")
        print(f"   Right:  {pad_r} (req_x2={req_x2} > w={w}? {req_x2 > w})")
        print(f"   Bottom: {pad_b} (req_y2={req_y2} > h={h}? {req_y2 > h})")
        
        # 4. 패딩이 필요 없다면 원본 반환
        if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
            print("\n   ✅ No padding needed, returning original")
            return source_image, keypoints, (h, w)
            
        # 5. 이미지 확장 (흰색 패딩)
        print(f"\n   🖼️ Expanding Canvas: T={pad_t}, B={pad_b}, L={pad_l}, R={pad_r}")
        
        padded_image = cv2.copyMakeBorder(
            source_image, pad_t, pad_b, pad_l, pad_r, 
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        # 6. 키포인트 이동 (Shift)
        shifted_kpts = keypoints.copy()
        shifted_kpts[:, 0] += pad_l
        shifted_kpts[:, 1] += pad_t
        
        new_h, new_w = padded_image.shape[:2]
        
        print(f"\n📐 Final Canvas: {new_w}x{new_h}")
        print(f"   Keypoint Shift: (+{pad_l}, +{pad_t})")
        
        # 시프트 후 하반신 위치 확인
        print("\n📊 After Shift - Lower Body Positions:")
        for name, idx in zip(lower_names, lower_indices):
            if idx < len(scores):
                score = scores[idx]
                pos = shifted_kpts[idx]
                is_valid = score > 0.01
                in_bounds = 0 <= pos[0] <= new_w and 0 <= pos[1] <= new_h
                status = "✅" if is_valid else "❌"
                bounds = "📍" if in_bounds else "⚠️ OUT"
                print(f"   {status} {name:15}: pos=({pos[0]:.1f}, {pos[1]:.1f}) {bounds}")
        
        print("\n" + "="*60)
        
        return padded_image, shifted_kpts, (new_h, new_w)