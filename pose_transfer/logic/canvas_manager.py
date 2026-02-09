"""
Canvas Manager Module (Refactored v5.5 - Strict Size Check)

위치: pose_transfer/logic/canvas_manager.py
변경사항:
- [Fix] 키포인트가 이미지 내부에 있으면 절대 확장하지 않도록 수정 (불필요한 패딩 방지)
- [Fix] load_image_safe 헬퍼 유지
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union

class CanvasManager:
    def __init__(self, config):
        self.config = config

    def load_image_safe(self, image_source: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image_source, np.ndarray): return image_source.copy()
        path_str = str(image_source)
        img = cv2.imread(path_str)
        if img is None:
            try:
                img_array = np.fromfile(path_str, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except: pass
        if img is None: raise FileNotFoundError(f"Failed to load: {path_str}")
        return img

    def expand_canvas_to_fit(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        padding_ratio: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        키포인트가 이미지 밖으로 나갔을 때만 캔버스를 확장함.
        """
        h, w = image.shape[:2]
        
        # 유효한 키포인트만 필터링
        valid_mask = scores > 0.1
        if not np.any(valid_mask):
            return image, keypoints, (h, w)
            
        valid_kpts = keypoints[valid_mask]
        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)
        
        # [Critical Fix] 엄격한 검사: 모든 포인트가 안에 있다면 원본 반환
        # padding_ratio가 0이면 여유공간 계산 안함
        if min_x >= 0 and min_y >= 0 and max_x <= w and max_y <= h:
            if padding_ratio <= 0:
                # print("   🖼️ Canvas: Fit perfectly. No expansion.")
                return image, keypoints, (h, w)

        # 확장 필요성 계산
        # 이미지 경계와 키포인트 경계 중 더 넓은 범위를 잡음
        current_min_x = min(0, min_x)
        current_min_y = min(0, min_y)
        current_max_x = max(w, max_x)
        current_max_y = max(h, max_y)
        
        # 기본 확장량 계산 (키포인트가 이미지 밖으로 나간 양)
        base_pad_left = abs(current_min_x) if current_min_x < 0 else 0
        base_pad_top = abs(current_min_y) if current_min_y < 0 else 0
        base_pad_right = (current_max_x - w) if current_max_x > w else 0
        base_pad_bottom = (current_max_y - h) if current_max_y > h else 0
        
        # [수정] 패딩된 방향에만 추가 여유 패딩 적용
        # 가로 패딩 = 이미지 가로(w)의 %, 세로 패딩 = 이미지 세로(h)의 %
        extra_pad_h = int(w * padding_ratio)  # 가로방향 여유 (좌/우)
        extra_pad_v = int(h * padding_ratio)  # 세로방향 여유 (상/하)
        
        # 패딩된 방향에만 추가 여유 적용
        pad_left = int(base_pad_left + extra_pad_h) if base_pad_left > 0 else 0
        pad_top = int(base_pad_top + extra_pad_v) if base_pad_top > 0 else 0
        pad_right = int(base_pad_right + extra_pad_h) if base_pad_right > 0 else 0
        pad_bottom = int(base_pad_bottom + extra_pad_v) if base_pad_bottom > 0 else 0
        
        # 최종 확인: 패딩이 0이면 원본 반환
        if pad_left == 0 and pad_top == 0 and pad_right == 0 and pad_bottom == 0:
            return image, keypoints, (h, w)
            
        # 캔버스 확장 (검은색 배경)
        expanded_img = cv2.copyMakeBorder(
            image, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, 
            value=(0, 0, 0)
        )
        
        # 좌표 보정
        offset = np.array([pad_left, pad_top])
        adjusted_kpts = keypoints.copy()
        adjusted_kpts[valid_mask] += offset
        
        new_h, new_w = expanded_img.shape[:2]
        print(f"   🖼️ Canvas Expanded: {w}x{h} -> {new_w}x{new_h} (Pad: L{pad_left}, T{pad_top}, R{pad_right}, B{pad_bottom})")
        
        return expanded_img, adjusted_kpts, (new_h, new_w)

    def crop_to_keypoints(self, image, keypoints, scores, padding_ratio=0.1):
        return image, keypoints, image.shape[:2]