"""
스켈레톤 렌더링 모듈 (Updated: 동적 스케일링 적용)

이미지 해상도에 비례하여 선 두께와 점 크기를 조절합니다.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List

from ..extractors.keypoint_constants import (
    BODY_KEYPOINTS,
    FEET_KEYPOINTS,
    BODY_COLORS,
    FACE_COLOR,
    LEFT_HAND_COLOR,
    RIGHT_HAND_COLOR,
    FACE_START_IDX,
    FACE_END_IDX,
    LEFT_HAND_START_IDX,
    RIGHT_HAND_START_IDX,
    get_body_bone_indices,
    get_feet_bone_indices,
    get_hand_bone_indices,
    get_face_bone_indices
)


class SkeletonRenderer:
    """
    스켈레톤 렌더러
    """
    
    def __init__(
        self,
        line_thickness: int = 6,      # 기준 해상도(1000px)에서의 기본 두께
        point_radius: int = 4,        # 기준 해상도(1000px)에서의 점 크기
        kpt_threshold: float = 0.3,
        draw_face: bool = True,
        draw_hands: bool = True,
        face_line_thickness: int = 3, # 기준 해상도에서의 얼굴 선 두께
        hand_line_thickness: int = 3, # 기준 해상도에서의 손 선 두께
        reference_resolution: int = 1000 # 기준 해상도 (긴 변 기준)
    ):
        self.base_line_thickness = line_thickness
        self.base_point_radius = point_radius
        self.kpt_threshold = kpt_threshold
        self.draw_face = draw_face
        self.draw_hands = draw_hands
        self.base_face_thickness = face_line_thickness
        self.base_hand_thickness = hand_line_thickness
        self.reference_resolution = reference_resolution
        
        # 본 인덱스 초기화
        self.body_bones = get_body_bone_indices()
        self.feet_bones = get_feet_bone_indices()
        self.left_hand_bones = get_hand_bone_indices(is_left=True)
        self.right_hand_bones = get_hand_bone_indices(is_left=False)
        self.face_bones = get_face_bone_indices()
    
    def _get_scaled_value(self, image_shape: Tuple[int, ...], base_value: int) -> int:
        """
        이미지 크기에 비례하여 값 스케일링
        """
        h, w = image_shape[:2]
        max_dim = max(h, w)
        
        # 기준 해상도 대비 현재 이미지 비율 계산
        scale = max_dim / self.reference_resolution
        
        # 최소 1픽셀은 보장
        return max(1, int(base_value * scale))

    def render(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        background_color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        스켈레톤 렌더링
        """
        if background_color is not None:
            canvas = np.full(image.shape, background_color, dtype=np.uint8)
        else:
            canvas = image.copy()
        
        # 현재 이미지 크기에 맞는 두께 계산
        body_thick = self._get_scaled_value(canvas.shape, self.base_line_thickness)
        face_thick = self._get_scaled_value(canvas.shape, self.base_face_thickness)
        hand_thick = self._get_scaled_value(canvas.shape, self.base_hand_thickness)
        
        # 1. Body 본 그리기
        self._draw_bones(
            canvas, keypoints, scores,
            self.body_bones, BODY_COLORS,
            body_thick
        )
        
        # 2. Feet 본 그리기
        self._draw_bones(
            canvas, keypoints, scores,
            self.feet_bones, BODY_COLORS,
            body_thick
        )
        
        # 3. 얼굴 그리기
        if self.draw_face:
            self._draw_bones(
                canvas, keypoints, scores,
                self.face_bones, [FACE_COLOR] * len(self.face_bones),
                face_thick
            )
        
        # 4. 손 그리기
        if self.draw_hands:
            self._draw_bones(
                canvas, keypoints, scores,
                self.left_hand_bones, [LEFT_HAND_COLOR] * len(self.left_hand_bones),
                hand_thick
            )
            self._draw_bones(
                canvas, keypoints, scores,
                self.right_hand_bones, [RIGHT_HAND_COLOR] * len(self.right_hand_bones),
                hand_thick
            )
        
        # 5. 키포인트 그리기
        self._draw_keypoints(canvas, keypoints, scores)
        
        return canvas
    
    def render_skeleton_only(
        self,
        image_shape: Tuple[int, int, int],
        keypoints: np.ndarray,
        scores: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """검은 배경에 스켈레톤만 렌더링"""
        # 이미지 쉐이프가 (H, W)인 경우 (H, W, 3)으로 보정
        if len(image_shape) == 2:
            image_shape = (image_shape[0], image_shape[1], 3)
            
        canvas = np.full(image_shape, background_color, dtype=np.uint8)
        return self.render(canvas, keypoints, scores)
    
    def _draw_bones(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        bone_indices: List[Tuple[int, int]],
        colors: List[Tuple[int, int, int]],
        thickness: int
    ):
        """본 그리기"""
        for i, (start_idx, end_idx) in enumerate(bone_indices):
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue
            
            if (scores[start_idx] < self.kpt_threshold or 
                scores[end_idx] < self.kpt_threshold):
                continue
            
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))
            
            color = colors[i % len(colors)]
            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    def _draw_keypoints(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray
    ):
        """키포인트 그리기"""
        # 현재 이미지 크기에 맞는 반지름 계산
        radius = self._get_scaled_value(canvas.shape, self.base_point_radius)
        
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score < self.kpt_threshold:
                continue
            
            center = tuple(kpt.astype(int))
            
            # 부위별 색상
            if i < 23:  # Body + Feet
                color = (255, 255, 255)
            elif i < 91:  # Face
                color = FACE_COLOR
            elif i < 112:  # Left Hand
                color = LEFT_HAND_COLOR
            else:  # Right Hand
                color = RIGHT_HAND_COLOR
            
            cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
            # 테두리는 반지름의 1/4 정도 (최소 1px)
            stroke = max(1, radius // 4)
            cv2.circle(canvas, center, radius, (255, 255, 255), stroke, cv2.LINE_AA)

# 편의 함수
def render_skeleton(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_shape: Tuple[int, int, int],
    kpt_threshold: float = 0.3
) -> np.ndarray:
    renderer = SkeletonRenderer(kpt_threshold=kpt_threshold)
    return renderer.render_skeleton_only(image_shape, keypoints, scores)