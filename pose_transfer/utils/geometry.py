"""
기하학 연산 유틸리티
"""
import numpy as np
from typing import Tuple, Optional, List


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """두 점 사이의 유클리드 거리 계산"""
    return np.linalg.norm(p1 - p2)


def calculate_center(points: np.ndarray) -> np.ndarray:
    """점들의 중심점 계산 (유효한 점만 사용)"""
    valid_points = points[~np.isnan(points).any(axis=1)]
    if len(valid_points) == 0:
        return np.array([np.nan, np.nan])
    return np.mean(valid_points, axis=0)


def calculate_bbox(points: np.ndarray) -> Tuple[float, float, float, float]:
    """
    키포인트들의 바운딩 박스 계산
    
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    valid_points = points[~np.isnan(points).any(axis=1)]
    if len(valid_points) == 0:
        return (0, 0, 0, 0)
    
    x_min, y_min = valid_points.min(axis=0)
    x_max, y_max = valid_points.max(axis=0)
    return (x_min, y_min, x_max, y_max)


def calculate_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """바운딩 박스 면적 계산"""
    x_min, y_min, x_max, y_max = bbox
    return max(0, (x_max - x_min) * (y_max - y_min))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """벡터 정규화 (단위 벡터로 변환)"""
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 벡터 사이의 각도 계산 (라디안)"""
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot_product)


def rotate_point(point: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    """점을 중심 기준으로 회전"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    translated = point - center
    rotated = np.array([
        translated[0] * cos_a - translated[1] * sin_a,
        translated[0] * sin_a + translated[1] * cos_a
    ])
    return rotated + center


def mirror_point_horizontal(point: np.ndarray, center_x: float) -> np.ndarray:
    """수평 대칭 (좌우 반전)"""
    mirrored = point.copy()
    mirrored[0] = 2 * center_x - point[0]
    return mirrored


def scale_point(point: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """점을 중심 기준으로 스케일링"""
    return center + (point - center) * scale


def interpolate_point(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    """두 점 사이 선형 보간"""
    return p1 + (p2 - p1) * t


def get_bone_vector(start: np.ndarray, end: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    본 벡터와 길이 계산
    
    Returns:
        (방향 벡터, 길이)
    """
    diff = end - start
    length = np.linalg.norm(diff)
    direction = normalize_vector(diff)
    return direction, length


def apply_bone_transform(
    start: np.ndarray,
    direction: np.ndarray,
    length: float
) -> np.ndarray:
    """본 변환 적용 (시작점 + 방향 * 길이)"""
    return start + direction * length


def calculate_centroid(keypoints: np.ndarray, scores: np.ndarray, 
                       threshold: float = 0.3) -> np.ndarray:
    """
    신뢰도 가중 중심점 계산
    """
    valid_mask = scores > threshold
    if not valid_mask.any():
        return np.array([np.nan, np.nan])
    
    valid_kpts = keypoints[valid_mask]
    valid_scores = scores[valid_mask]
    
    weights = valid_scores / valid_scores.sum()
    centroid = np.average(valid_kpts, axis=0, weights=weights)
    return centroid


def point_in_bbox(point: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
    """점이 바운딩 박스 안에 있는지 확인"""
    x_min, y_min, x_max, y_max = bbox
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def expand_bbox(bbox: Tuple[float, float, float, float], 
                ratio: float) -> Tuple[float, float, float, float]:
    """바운딩 박스 확장"""
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    new_w = w * ratio
    new_h = h * ratio
    
    return (
        cx - new_w / 2,
        cy - new_h / 2,
        cx + new_w / 2,
        cy + new_h / 2
    )
