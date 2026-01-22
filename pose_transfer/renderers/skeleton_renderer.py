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
        draw_neck: bool = False,      # 가상 목 생성 여부 (기본 비활성화)
        face_line_thickness: int = 3, # 기준 해상도에서의 얼굴 선 두께
        hand_line_thickness: int = 3, # 기준 해상도에서의 손 선 두께
        reference_resolution: int = 1000 # 기준 해상도 (긴 변 기준)
    ):
        self.base_line_thickness = line_thickness
        self.base_point_radius = point_radius
        self.kpt_threshold = kpt_threshold
        self.draw_face = draw_face
        self.draw_hands = draw_hands
        self.draw_neck = draw_neck    # 가상 목 그리기 활성화 여부
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
        background_color: Optional[Tuple[int, int, int]] = None,
        occluded_indices: Optional[set] = None,  # -1 레이어: 가려진 키포인트
        out_of_frame_indices: Optional[set] = None  # -2 레이어: 프레임 밖 키포인트
    ) -> np.ndarray:
        """
        스켈레톤 렌더링
        """
        if occluded_indices is None:
            occluded_indices = set()
        if out_of_frame_indices is None:
            out_of_frame_indices = set()
        
        # 🔍 디버그: occluded_indices 확인
        if occluded_indices:
            print(f"[RENDERER] occluded_indices 전달됨: {sorted(occluded_indices)[:10]}{'...' if len(occluded_indices) > 10 else ''} (총 {len(occluded_indices)}개)")
        
        if background_color is not None:
            canvas = np.full(image.shape, background_color, dtype=np.uint8)
        else:
            canvas = image.copy()
        
        # 현재 이미지 크기에 맞는 두께 계산
        body_thick = self._get_scaled_value(canvas.shape, self.base_line_thickness)
        face_thick = self._get_scaled_value(canvas.shape, self.base_face_thickness)
        hand_thick = self._get_scaled_value(canvas.shape, self.base_hand_thickness)
        
        # 0. 가상 Neck 연결 그리기 (nose -> neck -> shoulders)
        # 주의: 어깨 중점이 반드시 목은 아님 (정면이 아닌 각도에서는 부정확)
        # 기본적으로 비활성화 (draw_neck=False)
        if self.draw_neck:
            self._draw_virtual_neck(canvas, keypoints, scores, body_thick, occluded_indices, out_of_frame_indices)
        
        # 1. Body 본 그리기
        self._draw_bones(
            canvas, keypoints, scores,
            self.body_bones, BODY_COLORS,
            body_thick, occluded_indices, out_of_frame_indices
        )
        
        # 2. Feet 본 그리기
        self._draw_bones(
            canvas, keypoints, scores,
            self.feet_bones, BODY_COLORS,
            body_thick, occluded_indices, out_of_frame_indices
        )
        
        # 3. 얼굴 그리기
        if self.draw_face:
            self._draw_bones(
                canvas, keypoints, scores,
                self.face_bones, [FACE_COLOR] * len(self.face_bones),
                face_thick, occluded_indices, out_of_frame_indices
            )
        
        # 4. 손 그리기
        if self.draw_hands:
            self._draw_bones(
                canvas, keypoints, scores,
                self.left_hand_bones, [LEFT_HAND_COLOR] * len(self.left_hand_bones),
                hand_thick, occluded_indices, out_of_frame_indices
            )
            self._draw_bones(
                canvas, keypoints, scores,
                self.right_hand_bones, [RIGHT_HAND_COLOR] * len(self.right_hand_bones),
                hand_thick, occluded_indices, out_of_frame_indices
            )
        
        # 5. 키포인트 그리기
        self._draw_keypoints(canvas, keypoints, scores, out_of_frame_indices, occluded_indices)
        
        return canvas
    
    def render_skeleton_only(
        self,
        image_shape: Tuple[int, int, int],
        keypoints: np.ndarray,
        scores: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 0, 0),
        occluded_indices: Optional[set] = None,  # -1 레이어
        out_of_frame_indices: Optional[set] = None  # -2 레이어
    ) -> np.ndarray:
        """검은 배경에 스켈레톤만 렌더링"""
        # 이미지 쉐이프가 (H, W)인 경우 (H, W, 3)으로 보정
        if len(image_shape) == 2:
            image_shape = (image_shape[0], image_shape[1], 3)
            
        canvas = np.full(image_shape, background_color, dtype=np.uint8)
        return self.render(canvas, keypoints, scores, None, occluded_indices, out_of_frame_indices)
    
    def _draw_bones(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        bone_indices: List[Tuple[int, int]],
        colors: List[Tuple[int, int, int]],
        thickness: int,
        occluded_indices: set = None,  # 가려진 키포인트 집합
        out_of_frame_indices: set = None  # 프레임 밖 키포인트 집합
    ):
        """본 그리기 (가려진 부위는 투명도 50%, 프레임 밖은 그리지 않음)"""
        if occluded_indices is None:
            occluded_indices = set()
        if out_of_frame_indices is None:
            out_of_frame_indices = set()
        
        for i, (start_idx, end_idx) in enumerate(bone_indices):
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue
            
            # 🚫 프레임 밖 키포인트가 포함된 본은 그리지 않음
            if start_idx in out_of_frame_indices or end_idx in out_of_frame_indices:
                continue
            
            if (scores[start_idx] < self.kpt_threshold or 
                scores[end_idx] < self.kpt_threshold):
                continue
            
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))
            
            color = colors[i % len(colors)]
            
            # 가려진 키포인트면 투명도 50%로 그리기
            is_occluded = start_idx in occluded_indices or end_idx in occluded_indices
            
            if is_occluded:
                # 알파 블렌딩을 위한 오버레이 생성
                overlay = canvas.copy()
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
                # 50% 투명도 적용 (canvas를 직접 업데이트)
                canvas[:] = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)
            else:
                # 정상 라인만 그리기
                cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    def _draw_virtual_neck(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        thickness: int,
        occluded_indices: set = None,
        out_of_frame_indices: set = None
    ):
        """
        가상 neck 연결 그리기 (COCO-WholeBody용)
        
        ⚠️ 주의: 이 방법은 정면 각도에서만 정확합니다
        COCO-WholeBody에는 neck 키포인트가 없으므로
        양쪽 어깨의 중점을 가상 neck으로 사용하여
        nose -> neck 중심축을 그립니다.
        
        각도가 정면이 아닐 경우 어깨 중점이 실제 목 위치와 다를 수 있습니다.
        """
        if occluded_indices is None:
            occluded_indices = set()
        if out_of_frame_indices is None:
            out_of_frame_indices = set()
        
        nose_idx = BODY_KEYPOINTS['nose']  # 0
        l_shoulder_idx = BODY_KEYPOINTS['left_shoulder']  # 5
        r_shoulder_idx = BODY_KEYPOINTS['right_shoulder']  # 6
        
        # 필수 키포인트 체크
        if (nose_idx >= len(keypoints) or 
            l_shoulder_idx >= len(keypoints) or 
            r_shoulder_idx >= len(keypoints)):
            return
        
        # 프레임 밖 체크
        if (nose_idx in out_of_frame_indices or
            l_shoulder_idx in out_of_frame_indices or
            r_shoulder_idx in out_of_frame_indices):
            return
        
        # 신뢰도 체크
        if (scores[nose_idx] < self.kpt_threshold or
            scores[l_shoulder_idx] < self.kpt_threshold or
            scores[r_shoulder_idx] < self.kpt_threshold):
            return
        
        # 가상 neck 위치 = 양쪽 어깨의 중점
        neck_pos = (keypoints[l_shoulder_idx] + keypoints[r_shoulder_idx]) / 2
        
        # nose -> neck 라인 그리기
        nose_pt = tuple(keypoints[nose_idx].astype(int))
        neck_pt = tuple(neck_pos.astype(int))
        
        # 폐색 체크 (nose나 어깨 중 하나라도 가려졌으면)
        is_occluded = (nose_idx in occluded_indices or 
                      l_shoulder_idx in occluded_indices or 
                      r_shoulder_idx in occluded_indices)
        
        # 중심축 색상 (body 색상 사용)
        color = BODY_COLORS[0]  # 첫 번째 body 색상
        
        if is_occluded:
            # 알파 블렌딩 (50% 투명도)
            overlay = canvas.copy()
            cv2.line(overlay, nose_pt, neck_pt, color, thickness, cv2.LINE_AA)
            canvas[:] = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)
        else:
            # 정상 라인
            cv2.line(canvas, nose_pt, neck_pt, color, thickness, cv2.LINE_AA)
    
    def _draw_keypoints(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        out_of_frame_indices: set = None,  # 프레임 밖 키포인트 집합
        occluded_indices: set = None  # 가려진 키포인트 집합
    ):
        """키포인트 그리기 (가려진 부위는 투명도 50%)"""
        if out_of_frame_indices is None:
            out_of_frame_indices = set()
        if occluded_indices is None:
            occluded_indices = set()
        
        # 현재 이미지 크기에 맞는 반지름 계산
        radius = self._get_scaled_value(canvas.shape, self.base_point_radius)
        h, w = canvas.shape[:2]  # 이미지 크기
        
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score < self.kpt_threshold:
                continue
            
            # 🚫 프레임 밖 키포인트는 렌더링하지 않음
            # GhostFilter의 -2 레이어 마킹을 신뢰 (렌더링 시점 재계산 불필요)
            if i in out_of_frame_indices:
                continue
            
            x, y = int(kpt[0]), int(kpt[1])  # 키포인트 좌표
            
            # 가려진 키포인트인지 확인
            is_occluded = i in occluded_indices
            
            center = tuple(kpt.astype(int))  # 키포인트 중심 좌표
            
            # 부위별 색상
            if i < 23:  # Body + Feet
                color = (255, 255, 255)
            elif i < 91:  # Face
                color = FACE_COLOR
            elif i < 112:  # Left Hand
                color = LEFT_HAND_COLOR
            else:  # Right Hand
                color = RIGHT_HAND_COLOR
            
            # 가려진 키포인트는 투명도 50%로 그리기
            if is_occluded:
                # 알파 블렌딩을 위한 오버레이
                overlay = canvas.copy()
                cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
                stroke = max(1, radius // 4)
                cv2.circle(overlay, center, radius, (255, 255, 255), stroke, cv2.LINE_AA)
                # 50% 투명도 적용 (canvas를 직접 업데이트)
                canvas[:] = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)
            else:
                # 정상 포인트
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