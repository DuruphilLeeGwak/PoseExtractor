"""
다중 인물 필터링 모듈

이미지에 여러 사람이 있을 때, 중심에 가깝고 가장 큰 인물을 선택합니다.
"""
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

from ..utils.geometry import (
    calculate_bbox,
    calculate_bbox_area,
    calculate_distance
)


@dataclass
class PersonScore:
    """인물 스코어 정보"""
    index: int
    area_score: float
    center_score: float
    total_score: float
    bbox: Tuple[float, float, float, float]
    center: np.ndarray
    valid_keypoint_count: int


class PersonFilter:
    """
    다중 인물 중 주요 인물 선택 필터
    
    선정 기준:
    1. 키포인트 바운딩 박스 면적 (클수록 높은 점수)
    2. 이미지 중심까지의 거리 (가까울수록 높은 점수)
    """
    
    def __init__(
        self,
        area_weight: float = 0.6,
        center_weight: float = 0.4,
        min_keypoints: int = 5,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            area_weight: 면적 가중치 (0~1)
            center_weight: 중심 거리 가중치 (0~1)
            min_keypoints: 최소 유효 키포인트 수
            confidence_threshold: 유효 키포인트 판단 임계값
        """
        self.area_weight = area_weight
        self.center_weight = center_weight
        self.min_keypoints = min_keypoints
        self.confidence_threshold = confidence_threshold
    
    def select_main_person(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, int, Optional[PersonScore]]:
        """
        주요 인물 1명 선택
        
        Args:
            keypoints: (N, K, 2) 모든 인물의 키포인트
            scores: (N, K) 모든 인물의 신뢰도
            image_size: (height, width) 이미지 크기
        
        Returns:
            selected_keypoints: (K, 2) 선택된 인물의 키포인트
            selected_scores: (K,) 선택된 인물의 신뢰도
            selected_index: 선택된 인물 인덱스
            score_info: 스코어 상세 정보 (디버그용)
        """
        if len(keypoints) == 0:
            return np.array([]), np.array([]), -1, None
        
        if len(keypoints) == 1:
            return keypoints[0], scores[0], 0, None
        
        # 이미지 중심 및 최대 거리 계산
        img_h, img_w = image_size
        img_center = np.array([img_w / 2, img_h / 2])
        max_diagonal = np.sqrt(img_w**2 + img_h**2)
        max_area = img_w * img_h
        
        # 각 인물별 스코어 계산
        person_scores: List[PersonScore] = []
        
        for idx in range(len(keypoints)):
            person_kpts = keypoints[idx]
            person_scrs = scores[idx]
            
            # 유효한 키포인트 필터링
            valid_mask = person_scrs > self.confidence_threshold
            valid_kpts = person_kpts[valid_mask]
            
            # 최소 키포인트 수 미달
            if len(valid_kpts) < self.min_keypoints:
                person_scores.append(PersonScore(
                    index=idx,
                    area_score=0,
                    center_score=0,
                    total_score=0,
                    bbox=(0, 0, 0, 0),
                    center=np.array([0, 0]),
                    valid_keypoint_count=len(valid_kpts)
                ))
                continue
            
            # 바운딩 박스 계산
            bbox = calculate_bbox(valid_kpts)
            area = calculate_bbox_area(bbox)
            
            # 키포인트 중심점
            kpt_center = np.mean(valid_kpts, axis=0)
            dist_to_center = calculate_distance(kpt_center, img_center)
            
            # 정규화된 스코어 계산
            area_score = area / max_area if max_area > 0 else 0
            center_score = 1 - (dist_to_center / max_diagonal) if max_diagonal > 0 else 0
            
            # 가중 합산
            total_score = (
                self.area_weight * area_score +
                self.center_weight * center_score
            )
            
            person_scores.append(PersonScore(
                index=idx,
                area_score=area_score,
                center_score=center_score,
                total_score=total_score,
                bbox=bbox,
                center=kpt_center,
                valid_keypoint_count=len(valid_kpts)
            ))
        
        # 최고 스코어 인물 선택
        if not person_scores:
            return keypoints[0], scores[0], 0, None
        
        best_person = max(person_scores, key=lambda p: p.total_score)
        selected_idx = best_person.index
        
        return (
            keypoints[selected_idx],
            scores[selected_idx],
            selected_idx,
            best_person
        )
    
    def get_all_scores(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        image_size: Tuple[int, int]
    ) -> List[PersonScore]:
        """
        모든 인물의 스코어 계산 (디버그/시각화용)
        """
        if len(keypoints) == 0:
            return []
        
        img_h, img_w = image_size
        img_center = np.array([img_w / 2, img_h / 2])
        max_diagonal = np.sqrt(img_w**2 + img_h**2)
        max_area = img_w * img_h
        
        person_scores: List[PersonScore] = []
        
        for idx in range(len(keypoints)):
            person_kpts = keypoints[idx]
            person_scrs = scores[idx]
            
            valid_mask = person_scrs > self.confidence_threshold
            valid_kpts = person_kpts[valid_mask]
            
            if len(valid_kpts) < self.min_keypoints:
                person_scores.append(PersonScore(
                    index=idx,
                    area_score=0,
                    center_score=0,
                    total_score=0,
                    bbox=(0, 0, 0, 0),
                    center=np.array([0, 0]),
                    valid_keypoint_count=len(valid_kpts)
                ))
                continue
            
            bbox = calculate_bbox(valid_kpts)
            area = calculate_bbox_area(bbox)
            kpt_center = np.mean(valid_kpts, axis=0)
            dist_to_center = calculate_distance(kpt_center, img_center)
            
            area_score = area / max_area if max_area > 0 else 0
            center_score = 1 - (dist_to_center / max_diagonal) if max_diagonal > 0 else 0
            total_score = (
                self.area_weight * area_score +
                self.center_weight * center_score
            )
            
            person_scores.append(PersonScore(
                index=idx,
                area_score=area_score,
                center_score=center_score,
                total_score=total_score,
                bbox=bbox,
                center=kpt_center,
                valid_keypoint_count=len(valid_kpts)
            ))
        
        return person_scores


def filter_main_person(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_size: Tuple[int, int],
    area_weight: float = 0.6,
    center_weight: float = 0.4,
    min_keypoints: int = 5,
    confidence_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    편의 함수: 주요 인물 필터링
    
    Args:
        keypoints: (N, K, 2) 모든 인물의 키포인트
        scores: (N, K) 모든 인물의 신뢰도
        image_size: (height, width)
        area_weight: 면적 가중치
        center_weight: 중심 거리 가중치
        min_keypoints: 최소 유효 키포인트 수
        confidence_threshold: 유효 키포인트 판단 임계값
    
    Returns:
        keypoints: (K, 2) 선택된 인물의 키포인트
        scores: (K,) 선택된 인물의 신뢰도
        selected_index: 선택된 인물 인덱스
    """
    filter = PersonFilter(
        area_weight=area_weight,
        center_weight=center_weight,
        min_keypoints=min_keypoints,
        confidence_threshold=confidence_threshold
    )
    
    kpts, scrs, idx, _ = filter.select_main_person(
        keypoints, scores, image_size
    )
    
    return kpts, scrs, idx
