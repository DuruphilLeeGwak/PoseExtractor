"""
스켈레톤 렌더링 모듈

키포인트를 이미지로 시각화합니다.
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
    
    키포인트를 이미지에 시각화합니다.
    """
    
    def __init__(
        self,
        line_thickness: int = 4,
        point_radius: int = 4,
        kpt_threshold: float = 0.3,
        draw_face: bool = True,
        draw_hands: bool = True,
        face_line_thickness: int = 2,
        hand_line_thickness: int = 2
    ):
        """
        Args:
            line_thickness: Body 선 굵기
            point_radius: 키포인트 반지름
            kpt_threshold: 키포인트 표시 임계값
            draw_face: 얼굴 그리기 여부
            draw_hands: 손 그리기 여부
            face_line_thickness: 얼굴 선 굵기
            hand_line_thickness: 손 선 굵기
        """
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.kpt_threshold = kpt_threshold
        self.draw_face = draw_face
        self.draw_hands = draw_hands
        self.face_line_thickness = face_line_thickness
        self.hand_line_thickness = hand_line_thickness
        
        # 본 인덱스 초기화
        self.body_bones = get_body_bone_indices()
        self.feet_bones = get_feet_bone_indices()
        self.left_hand_bones = get_hand_bone_indices(is_left=True)
        self.right_hand_bones = get_hand_bone_indices(is_left=False)
        self.face_bones = get_face_bone_indices()
    
    def render(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        background_color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        스켈레톤 렌더링
        
        Args:
            image: 배경 이미지 (None이면 검은 배경)
            keypoints: (K, 2) 키포인트
            scores: (K,) 신뢰도
            background_color: 배경색 (None이면 원본 이미지 사용)
        
        Returns:
            렌더링된 이미지
        """
        if background_color is not None:
            canvas = np.full(image.shape, background_color, dtype=np.uint8)
        else:
            canvas = image.copy()
        
        # 1. Body 본 그리기
        self._draw_bones(
            canvas, keypoints, scores,
            self.body_bones, BODY_COLORS,
            self.line_thickness
        )
        
        # 2. Feet 본 그리기
        self._draw_bones(
            canvas, keypoints, scores,
            self.feet_bones, BODY_COLORS,
            self.line_thickness
        )
        
        # 3. 얼굴 그리기
        if self.draw_face:
            self._draw_bones(
                canvas, keypoints, scores,
                self.face_bones, [FACE_COLOR] * len(self.face_bones),
                self.face_line_thickness
            )
        
        # 4. 손 그리기
        if self.draw_hands:
            self._draw_bones(
                canvas, keypoints, scores,
                self.left_hand_bones, [LEFT_HAND_COLOR] * len(self.left_hand_bones),
                self.hand_line_thickness
            )
            self._draw_bones(
                canvas, keypoints, scores,
                self.right_hand_bones, [RIGHT_HAND_COLOR] * len(self.right_hand_bones),
                self.hand_line_thickness
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
        """
        검은 배경에 스켈레톤만 렌더링 (ControlNet용)
        
        Args:
            image_shape: (H, W, C) 이미지 형태
            keypoints: (K, 2) 키포인트
            scores: (K,) 신뢰도
            background_color: 배경색
        
        Returns:
            스켈레톤 이미지
        """
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
        """본(뼈대) 그리기"""
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
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score < self.kpt_threshold:
                continue
            
            center = tuple(kpt.astype(int))
            
            # 부위별 색상
            if i < 23:  # Body + Feet
                color = (255, 255, 255)  # 흰색
            elif i < 91:  # Face
                color = FACE_COLOR
            elif i < 112:  # Left Hand
                color = LEFT_HAND_COLOR
            else:  # Right Hand
                color = RIGHT_HAND_COLOR
            
            cv2.circle(canvas, center, self.point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, self.point_radius, (255, 255, 255), 1, cv2.LINE_AA)
    
    def render_comparison(
        self,
        image: np.ndarray,
        source_keypoints: np.ndarray,
        source_scores: np.ndarray,
        transferred_keypoints: np.ndarray,
        transferred_scores: np.ndarray
    ) -> np.ndarray:
        """
        원본 vs 전이 비교 렌더링
        
        Returns:
            좌: 원본, 우: 전이된 스켈레톤
        """
        h, w = image.shape[:2]
        
        # 원본 스켈레톤 (파란색 계열)
        left_canvas = image.copy()
        self._draw_bones_custom(
            left_canvas, source_keypoints, source_scores,
            self.body_bones + self.feet_bones,
            (255, 100, 100),  # 파란색
            self.line_thickness
        )
        
        # 전이 스켈레톤 (빨간색 계열)
        right_canvas = np.zeros_like(image)
        self._draw_bones_custom(
            right_canvas, transferred_keypoints, transferred_scores,
            self.body_bones + self.feet_bones,
            (100, 100, 255),  # 빨간색
            self.line_thickness
        )
        
        # 좌우 결합
        comparison = np.hstack([left_canvas, right_canvas])
        
        return comparison
    
    def _draw_bones_custom(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        bone_indices: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness: int
    ):
        """단일 색상으로 본 그리기"""
        for start_idx, end_idx in bone_indices:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue
            
            if (scores[start_idx] < self.kpt_threshold or 
                scores[end_idx] < self.kpt_threshold):
                continue
            
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))
            
            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)


def render_skeleton(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_shape: Tuple[int, int, int],
    kpt_threshold: float = 0.3
) -> np.ndarray:
    """
    편의 함수: 스켈레톤 렌더링
    """
    renderer = SkeletonRenderer(kpt_threshold=kpt_threshold)
    return renderer.render_skeleton_only(image_shape, keypoints, scores)
