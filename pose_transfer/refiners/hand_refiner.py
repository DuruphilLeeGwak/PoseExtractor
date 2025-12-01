"""
손 키포인트 정밀화 모듈

작은 손 영역을 업스케일하여 키포인트 추출 정확도를 높입니다.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict

from ..extractors.keypoint_constants import (
    BODY_KEYPOINTS,
    LEFT_HAND_START_IDX,
    RIGHT_HAND_START_IDX
)
from ..utils.geometry import expand_bbox


class HandRefiner:
    """
    손 키포인트 정밀화기
    
    손 영역이 너무 작으면 업스케일 후 재추출하여
    키포인트 정확도를 높입니다.
    """
    
    def __init__(
        self,
        min_hand_size: int = 48,
        max_scale_factor: float = 4.0,
        roi_expand_ratio: float = 1.5,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            min_hand_size: 최소 손 크기 (픽셀)
            max_scale_factor: 최대 업스케일 배율
            roi_expand_ratio: ROI 확장 비율
            confidence_threshold: 유효 키포인트 판단 임계값
        """
        self.min_hand_size = min_hand_size
        self.max_scale_factor = max_scale_factor
        self.roi_expand_ratio = roi_expand_ratio
        self.confidence_threshold = confidence_threshold
    
    def estimate_hand_roi(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        is_left: bool,
        image_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        손 영역(ROI) 추정
        
        손목 위치와 팔 방향을 기반으로 손 영역을 추정합니다.
        
        Args:
            keypoints: (K, 2) 전체 키포인트
            scores: (K,) 키포인트 신뢰도
            is_left: 왼손 여부
            image_shape: (H, W) 이미지 크기
        
        Returns:
            (x1, y1, x2, y2) 바운딩 박스 또는 None
        """
        img_h, img_w = image_shape
        
        # 손목과 팔꿈치 인덱스
        wrist_name = 'left_wrist' if is_left else 'right_wrist'
        elbow_name = 'left_elbow' if is_left else 'right_elbow'
        
        wrist_idx = BODY_KEYPOINTS[wrist_name]
        elbow_idx = BODY_KEYPOINTS[elbow_name]
        
        # 손목 신뢰도 확인
        if scores[wrist_idx] < self.confidence_threshold:
            return None
        
        wrist = keypoints[wrist_idx]
        
        # 팔 방향 계산 (팔꿈치 -> 손목)
        if scores[elbow_idx] > self.confidence_threshold:
            elbow = keypoints[elbow_idx]
            forearm_vec = wrist - elbow
            forearm_length = np.linalg.norm(forearm_vec)
            
            # 손 크기 추정 (하완 길이의 약 40%)
            hand_size = forearm_length * 0.4
        else:
            # 팔꿈치 없으면 기본 크기 사용
            hand_size = self.min_hand_size * 1.5
        
        # 손 방향으로 ROI 중심 이동
        if scores[elbow_idx] > self.confidence_threshold:
            direction = forearm_vec / (forearm_length + 1e-6)
            roi_center = wrist + direction * (hand_size * 0.5)
        else:
            roi_center = wrist
        
        # 바운딩 박스 생성
        half_size = hand_size * self.roi_expand_ratio / 2
        
        x1 = max(0, int(roi_center[0] - half_size))
        y1 = max(0, int(roi_center[1] - half_size))
        x2 = min(img_w, int(roi_center[0] + half_size))
        y2 = min(img_h, int(roi_center[1] + half_size))
        
        # 유효한 ROI인지 확인
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        
        return (x1, y1, x2, y2)
    
    def check_needs_upscale(
        self,
        roi: Tuple[int, int, int, int]
    ) -> Tuple[bool, float]:
        """
        업스케일 필요 여부 확인
        
        Returns:
            (필요 여부, 권장 스케일 팩터)
        """
        x1, y1, x2, y2 = roi
        roi_size = min(x2 - x1, y2 - y1)
        
        if roi_size >= self.min_hand_size:
            return False, 1.0
        
        scale_factor = self.min_hand_size / roi_size
        scale_factor = min(scale_factor, self.max_scale_factor)
        
        return True, scale_factor
    
    def crop_and_upscale(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int],
        scale_factor: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        ROI 크롭 및 업스케일
        
        Returns:
            upscaled_crop: 업스케일된 이미지
            transform_info: 좌표 변환 정보
        """
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        
        new_h = int((y2 - y1) * scale_factor)
        new_w = int((x2 - x1) * scale_factor)
        
        upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        transform_info = {
            'roi': roi,
            'scale_factor': scale_factor,
            'offset': (x1, y1)
        }
        
        return upscaled, transform_info
    
    def transform_keypoints_back(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        transform_info: Dict,
        is_left: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        업스케일 좌표를 원본 좌표로 역변환
        
        Args:
            keypoints: (21, 2) 손 키포인트 (업스케일 이미지 기준)
            scores: (21,) 신뢰도
            transform_info: 변환 정보
            is_left: 왼손 여부
        
        Returns:
            원본 좌표계의 키포인트, 신뢰도
        """
        scale = transform_info['scale_factor']
        offset_x, offset_y = transform_info['offset']
        
        # 스케일 역변환 + 오프셋 적용
        original_kpts = keypoints / scale
        original_kpts[:, 0] += offset_x
        original_kpts[:, 1] += offset_y
        
        return original_kpts, scores
    
    def refine_hand(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        extractor,  # DWPoseExtractor
        is_left: bool
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        손 키포인트 정밀화
        
        Args:
            image: 원본 이미지
            keypoints: (K, 2) 전체 키포인트
            scores: (K,) 신뢰도
            extractor: DWPose 추출기
            is_left: 왼손 여부
        
        Returns:
            refined_keypoints: 정밀화된 손 키포인트 (21, 2)
            refined_scores: 정밀화된 신뢰도 (21,)
            was_refined: 정밀화 수행 여부
        """
        hand_start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
        
        # 기존 손 키포인트
        original_hand_kpts = keypoints[hand_start:hand_start + 21]
        original_hand_scores = scores[hand_start:hand_start + 21]
        
        # ROI 추정
        img_h, img_w = image.shape[:2]
        roi = self.estimate_hand_roi(keypoints, scores, is_left, (img_h, img_w))
        
        if roi is None:
            return original_hand_kpts, original_hand_scores, False
        
        # 업스케일 필요 여부 확인
        needs_upscale, scale_factor = self.check_needs_upscale(roi)
        
        if not needs_upscale:
            return original_hand_kpts, original_hand_scores, False
        
        # 크롭 및 업스케일
        upscaled_crop, transform_info = self.crop_and_upscale(
            image, roi, scale_factor
        )
        
        # 업스케일 이미지에서 재추출
        try:
            all_kpts, all_scores = extractor.extract(upscaled_crop)
            
            if len(all_kpts) == 0:
                return original_hand_kpts, original_hand_scores, False
            
            # 첫 번째 인물의 손 키포인트
            new_hand_kpts = all_kpts[0][hand_start:hand_start + 21]
            new_hand_scores = all_scores[0][hand_start:hand_start + 21]
            
            # 원본 좌표로 역변환
            refined_kpts, refined_scores = self.transform_keypoints_back(
                new_hand_kpts, new_hand_scores, transform_info, is_left
            )
            
            # 새 키포인트가 더 좋은지 확인
            original_valid = np.sum(original_hand_scores > self.confidence_threshold)
            refined_valid = np.sum(refined_scores > self.confidence_threshold)
            
            if refined_valid > original_valid:
                return refined_kpts, refined_scores, True
            else:
                return original_hand_kpts, original_hand_scores, False
                
        except Exception as e:
            print(f"Hand refinement failed: {e}")
            return original_hand_kpts, original_hand_scores, False
    
    def refine_both_hands(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        extractor
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, bool]]:
        """
        양손 키포인트 정밀화
        
        Returns:
            keypoints: 정밀화된 전체 키포인트
            scores: 정밀화된 신뢰도
            refinement_info: {'left': bool, 'right': bool}
        """
        result_kpts = keypoints.copy()
        result_scores = scores.copy()
        refinement_info = {'left': False, 'right': False}
        
        # 왼손 정밀화
        left_kpts, left_scores, left_refined = self.refine_hand(
            image, keypoints, scores, extractor, is_left=True
        )
        if left_refined:
            result_kpts[LEFT_HAND_START_IDX:LEFT_HAND_START_IDX + 21] = left_kpts
            result_scores[LEFT_HAND_START_IDX:LEFT_HAND_START_IDX + 21] = left_scores
            refinement_info['left'] = True
        
        # 오른손 정밀화
        right_kpts, right_scores, right_refined = self.refine_hand(
            image, keypoints, scores, extractor, is_left=False
        )
        if right_refined:
            result_kpts[RIGHT_HAND_START_IDX:RIGHT_HAND_START_IDX + 21] = right_kpts
            result_scores[RIGHT_HAND_START_IDX:RIGHT_HAND_START_IDX + 21] = right_scores
            refinement_info['right'] = True
        
        return result_kpts, result_scores, refinement_info
