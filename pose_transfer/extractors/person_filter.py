"""다중 인물 중 "메인 인물"을 선택하는 필터.

이 모듈의 목적
    - 포즈 추출기(DWPose/RTMLib)가 한 프레임에서 여러 사람(N명)을 반환할 때,
        파이프라인이 "어느 사람"을 기준으로 진행할지 일관되게 결정합니다.

선정 전략(휴리스틱)
    1) 키포인트 바운딩 박스 면적이 큰 인물 (화면에서 크게 잡힌 인물)
    2) 이미지 중심에 가까운 인물 (주 피사체가 중앙에 있는 경우가 많다는 가정)

중요한 가정/주의
    - 이 필터는 신원(얼굴 유사도), 포즈 유사도, 추적 ID 같은 고급 기준은 사용하지 않습니다.
        즉, "중앙에 작게 있는 주인공" vs "가장자리에서 크게 잡힌 사람" 같은 상황에서는
        원하는 사람이 아닐 수도 있습니다.
    - "유효 키포인트"는 `confidence_threshold` 초과인 점만 사용합니다.
        threshold가 너무 높으면 bbox가 작아져(점이 적어져) 잘못 선택될 수 있고,
        너무 낮으면 노이즈 점까지 포함되어 bbox가 부풀어 잘못 선택될 수 있습니다.
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
                주요 인물 1명 선택.

                동작 개요
                    - 각 인물 i에 대해, score가 충분히 높은 키포인트만 모아 bbox를 만들고
                        (bbox 면적이 클수록 +, bbox 중심이 이미지 중심에 가까울수록 +) 점수를 계산합니다.
                    - 최종적으로 total_score가 가장 높은 인물을 선택합니다.
        
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
        # 방어 코드: 입력이 비었으면 선택 불가
        if len(keypoints) == 0:
            return np.array([]), np.array([]), -1, None
        
        # 1명만 있으면 고민할 필요 없이 그대로 사용
        if len(keypoints) == 1:
            return keypoints[0], scores[0], 0, None
        
        # 이미지 중심 및 정규화 상수 계산
        # - center_score는 "중심까지의 거리"를 0~1로 정규화해서 사용합니다.
        # - area_score는 "bbox 면적"을 0~1로 정규화해서 사용합니다.
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
            # - confidence_threshold 초과인 점만 bbox/중심 계산에 사용
            valid_mask = person_scrs > self.confidence_threshold
            valid_kpts = person_kpts[valid_mask]
            
            # 최소 키포인트 수 미달이면 "사람으로 간주하기 어렵다"고 보고 0점 처리
            # (후속 단계에서 max(total_score)로 뽑힐 가능성을 낮춤)
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
            
            # 바운딩 박스 계산 (valid_kpts만으로)
            bbox = calculate_bbox(valid_kpts)
            area = calculate_bbox_area(bbox)
            
            # 키포인트 중심점(평균). bbox 중심이 아니라 "점의 평균"을 사용합니다.
            kpt_center = np.mean(valid_kpts, axis=0)
            dist_to_center = calculate_distance(kpt_center, img_center)
            
            # 정규화된 스코어 계산
            # - area_score: 0~1 (클수록 좋음)
            # - center_score: 0~1 (중앙에 가까울수록 1)
            area_score = area / max_area if max_area > 0 else 0
            center_score = 1 - (dist_to_center / max_diagonal) if max_diagonal > 0 else 0
            
            # 가중 합산
            # - area_weight, center_weight는 합이 1일 필요는 없지만(상대 가중치),
            #   일반적으로 0~1 범위에서 합이 1이 되도록 쓰는 것을 권장합니다.
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
        # - 동점일 때는 Python의 max가 먼저 등장한 항목을 선택할 수 있으니
        #   완벽한 결정 규칙이 필요하면 tie-breaker를 추가해야 합니다.
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
                모든 인물의 스코어 계산 (디버그/시각화용).

                사용처 예시
                    - 디버그 이미지에 각 인물 bbox/점수 오버레이
                    - 특정 프레임에서 "왜 0번이 아니라 1번 인물이 선택됐는지" 확인
        """
        if len(keypoints) == 0:
            return []
        
        # select_main_person과 동일한 정규화 스킴을 사용해야 결과가 일관됩니다.
        img_h, img_w = image_size
        img_center = np.array([img_w / 2, img_h / 2])
        max_diagonal = np.sqrt(img_w**2 + img_h**2)
        max_area = img_w * img_h
        
        person_scores: List[PersonScore] = []
        
        for idx in range(len(keypoints)):
            person_kpts = keypoints[idx]
            person_scrs = scores[idx]
            
            # 유효 키포인트만 사용
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
    # NOTE: 내장 함수명 `filter`와 이름이 겹치지 않도록 변수명을 명확히 합니다.
    person_filter = PersonFilter(
        area_weight=area_weight,
        center_weight=center_weight,
        min_keypoints=min_keypoints,
        confidence_threshold=confidence_threshold
    )
    
    kpts, scrs, idx, _ = person_filter.select_main_person(
        keypoints, scores, image_size
    )
    
    return kpts, scrs, idx
