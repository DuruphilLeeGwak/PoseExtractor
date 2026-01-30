"""
Canvas Manager Module (Refactored v2.1 - Helper Added)

위치: pose_transfer/logic/canvas_manager.py
역할:
- 전이된 키포인트가 원본 이미지 범위를 벗어날 경우, 캔버스를 자동으로 확장(Padding)
- 이미지 로딩 헬퍼(load_image_safe) 제공
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union

class CanvasManager:
    def __init__(self, config):
        self.config = config

    def load_image_safe(self, image_source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        [Helper] 경로 또는 이미지를 받아 안전하게 np.ndarray로 반환
        """
        if isinstance(image_source, np.ndarray):
            return image_source.copy()
        
        # 경로인 경우
        path_str = str(image_source)
        img = cv2.imread(path_str)
        
        if img is None:
            # 한글 경로 등으로 인해 실패했을 경우 numpy로 읽기 시도
            try:
                img_array = np.fromfile(path_str, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception:
                pass
                
        if img is None:
            raise FileNotFoundError(f"Failed to load image from: {path_str}")
            
        return img

    def expand_canvas_to_fit(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        padding_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        키포인트가 이미지 밖으로 나갔다면 캔버스를 확장
        """
        h, w = image.shape[:2]
        
        # 유효한 키포인트만 필터링
        valid_mask = scores > 0.1
        if not np.any(valid_mask):
            return image, keypoints, (h, w)
            
        valid_kpts = keypoints[valid_mask]
        
        # 현재 키포인트의 범위 (Bounding Box)
        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)
        
        # 이미지 범위와 비교하여 확장 필요 여부 계산
        new_min_x = min(0, min_x)
        new_min_y = min(0, min_y)
        new_max_x = max(w, max_x)
        new_max_y = max(h, max_y)
        
        # 여유 공간(Padding) 추가
        pad_w = int((new_max_x - new_min_x) * padding_ratio)
        pad_h = int((new_max_y - new_min_y) * padding_ratio)
        
        # 최종 캔버스 좌표
        final_min_x = int(new_min_x - pad_w)
        final_min_y = int(new_min_y - pad_h)
        final_max_x = int(new_max_x + pad_w)
        final_max_y = int(new_max_y + pad_h)
        
        # 확장할 크기 계산
        # 왼쪽/위쪽으로 확장이 필요한 경우 (음수 좌표 발생 시)
        pad_left = abs(final_min_x) if final_min_x < 0 else 0
        pad_top = abs(final_min_y) if final_min_y < 0 else 0
        
        # 오른쪽/아래쪽으로 확장이 필요한 경우
        pad_right = (final_max_x - w) if final_max_x > w else 0
        pad_bottom = (final_max_y - h) if final_max_y > h else 0
        
        # 확장이 필요 없다면 원본 반환
        if pad_left == 0 and pad_top == 0 and pad_right == 0 and pad_bottom == 0:
            return image, keypoints, (h, w)
            
        # 캔버스 확장 (검은색 배경)
        expanded_img = cv2.copyMakeBorder(
            image, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, 
            value=(0, 0, 0)
        )
        
        # 키포인트 좌표 보정 (왼쪽/위쪽 패딩만큼 이동)
        offset = np.array([pad_left, pad_top])
        adjusted_kpts = keypoints.copy()
        adjusted_kpts[valid_mask] += offset
        
        new_h, new_w = expanded_img.shape[:2]
        
        print(f"   🖼️ Canvas Expanded: {w}x{h} -> {new_w}x{new_h} (Pad: L{pad_left}, T{pad_top}, R{pad_right}, B{pad_bottom})")
        
        return expanded_img, adjusted_kpts, (new_h, new_w)

    def crop_to_keypoints(self, image, keypoints, scores, padding_ratio=0.1):
        """(Optional) 키포인트 중심으로 크롭"""
        return image, keypoints, image.shape[:2]