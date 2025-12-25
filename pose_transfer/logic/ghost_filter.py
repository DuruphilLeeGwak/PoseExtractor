"""
Ghost Filter v4.6 - 무관용 원칙 (Zero Tolerance)

변경사항:
- [Revert] v4.5의 '골반/관절 보호(Tolerance)' 로직 전면 폐기 (유령 부활 원인 제거)
- [Critical] 하단/경계에 닿은 모든 키포인트(골반 포함) 즉시 삭제
- [Chain] 부모(골반)가 삭제되면 자식(무릎/발)도 즉시 사망하는 '연쇄 삭제' 강화
- [Overlap] 손-발 중첩 감지 활성화
"""
import numpy as np
from typing import Tuple, Optional, Dict, Set, List
from dataclasses import dataclass

from ..extractors.keypoint_constants import (
    BODY_BONES, FEET_BONES, HAND_BONES,
    FACE_START_IDX, FACE_END_IDX,
    LEFT_HAND_START_IDX, LEFT_HAND_END_IDX,
    RIGHT_HAND_START_IDX, RIGHT_HAND_END_IDX,
    get_keypoint_index
)


@dataclass
class GhostFilterConfig:
    """Ghost Filter 설정"""
    enabled: bool = True
    
    # [A] 프레임 이탈 체크
    check_bounds: bool = True
    bounds_margin: float = 0.05
    
    # [B] 경계값 감지
    check_boundary_values: bool = True
    boundary_tolerance: float = 5.0
    
    # [C] 예측 가능 키포인트 허용 (최소한의 마진만 허용)
    allow_predictable: bool = True
    predictable_margin: float = 0.10  # [축소] 0.20 -> 0.10 (더 엄격하게)
    min_confidence_for_prediction: float = 0.5  # [상향] 0.3 -> 0.5 (확실한 놈만 살림)
    
    # [D] 클러스터링 체크
    check_clustering: bool = True
    min_cluster_spread: float = 20.0
    
    # [E] 계층적 연결성 검사 (Consistency Check)
    check_consistency: bool = True
    
    # [F] 하단 강제 절삭
    hard_bottom_check: bool = True
    hard_bottom_threshold: float = 0.95
    
    # [G] 스마트 손-발 중첩 방지
    check_hand_foot_overlap: bool = True
    overlap_radius: float = 150.0
    hip_confidence_threshold: float = 0.4
    
    confidence_threshold: float = 0.1


@dataclass
class FilterResult:
    """필터링 결과"""
    filtered_scores: np.ndarray
    removed_indices: Set[int]
    removal_reasons: Dict[int, str]


class GhostFilter:
    """통합 Ghost Filter v4.6"""
    
    def __init__(self, config: Optional[GhostFilterConfig] = None):
        self.config = config or GhostFilterConfig()
        self._build_adjacency_map()
        
        # 족보 규칙 (자식 -> 부모)
        self.hierarchy_rules = {
            # Feet -> Ankle
            17: 15, 18: 15, 19: 15, 
            20: 16, 21: 16, 22: 16,
            
            # Fingers -> Wrist
            **{i: 9 for i in range(91, 112)},
            **{i: 10 for i in range(112, 133)},
            
            # Ankle -> Knee
            15: 13, 16: 14, 
            
            # Knee -> Hip (핵심 고리)
            13: 11, 14: 12  
        }
    
    def _build_adjacency_map(self):
        self.adjacency: Dict[int, List[int]] = {}
        all_bones = BODY_BONES + FEET_BONES
        for start_name, end_name in all_bones:
            try:
                s, e = get_keypoint_index(start_name), get_keypoint_index(end_name)
                for u, v in [(s, e), (e, s)]:
                    if u not in self.adjacency: self.adjacency[u] = []
                    self.adjacency[u].append(v)
            except: pass
            
    def filter_single(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray, 
        image_size: Tuple[int, int]
    ) -> FilterResult:
        if not self.config.enabled:
            return FilterResult(scores.copy(), set(), {})
        
        h, w = image_size
        filtered_scores = scores.copy()
        removed_indices = set()
        removal_reasons = {}
        
        def remove(idx, reason):
            if idx not in removed_indices:
                filtered_scores[idx] = 0.0
                removed_indices.add(idx)
                removal_reasons[idx] = reason

        # =========================================================
        # [Step 1] 하단 강제 절삭 (예외 없음)
        # =========================================================
        if self.config.hard_bottom_check:
            limit_y = h * self.config.hard_bottom_threshold
            # 모든 키포인트 검사 (골반 포함!)
            for idx in range(len(keypoints)):
                if scores[idx] > 0.01:
                    if keypoints[idx][1] > limit_y:
                        remove(idx, f"hard_bottom(y>{limit_y:.0f})")

        # =========================================================
        # [Step 2] 경계값 감지 (예외 없음)
        # =========================================================
        if self.config.check_boundary_values:
            tol = self.config.boundary_tolerance
            for i in range(len(keypoints)):
                if scores[i] < self.config.confidence_threshold: continue
                x, y = keypoints[i]
                # 상하좌우 모든 경계 체크
                if x <= tol or x >= w-tol or y <= tol:
                    remove(i, f"boundary_val({x:.0f},{y:.0f})")

        # =========================================================
        # [Step 3] 클러스터링 체크
        # =========================================================
        if self.config.check_clustering:
            for start, end, name in [(91, 111, "LHand"), (112, 132, "RHand")]:
                points = [keypoints[i] for i in range(start, end+1) if scores[i] > 0.1]
                indices = [i for i in range(start, end+1) if scores[i] > 0.1]
                if len(points) > 5:
                    pts = np.array(points)
                    spread = np.sqrt(np.std(pts[:,0])**2 + np.std(pts[:,1])**2)
                    if spread < self.config.min_cluster_spread:
                        for idx in indices: remove(idx, f"clustered_{name}")

        # =========================================================
        # [Step 4] 스마트 손-발 중첩 방지
        # =========================================================
        if self.config.check_hand_foot_overlap:
            hand_indices = [9, 10] + list(range(91, 113))
            left_leg_parts = [13, 15] + [17, 18, 19] 
            right_leg_parts = [14, 16] + [20, 21, 22] 
            
            active_hands = []
            for h_idx in hand_indices:
                if h_idx not in removed_indices and h_idx < len(scores) and scores[h_idx] > 0.1:
                    active_hands.append(keypoints[h_idx])
            
            if active_hands:
                active_hands = np.array(active_hands)
                
                # Hip 점수가 0점이면 이미 '죽은' 것이므로 valid=False
                l_hip_valid = (filtered_scores[11] > self.config.hip_confidence_threshold) if 11 < len(scores) else False
                for idx in left_leg_parts:
                    if idx not in removed_indices and idx < len(scores) and scores[idx] > 0.05:
                        if l_hip_valid: continue
                        if np.min(np.linalg.norm(active_hands - keypoints[idx], axis=1)) < self.config.overlap_radius:
                            remove(idx, "hand_overlap_ghost(L)")

                r_hip_valid = (filtered_scores[12] > self.config.hip_confidence_threshold) if 12 < len(scores) else False
                for idx in right_leg_parts:
                    if idx not in removed_indices and idx < len(scores) and scores[idx] > 0.05:
                        if r_hip_valid: continue
                        if np.min(np.linalg.norm(active_hands - keypoints[idx], axis=1)) < self.config.overlap_radius:
                            remove(idx, "hand_overlap_ghost(R)")

        # =========================================================
        # [Step 5] 계층적 연결성 검사 (Chain Kill)
        # =========================================================
        if self.config.check_consistency:
            for _ in range(3):
                for child, parent in self.hierarchy_rules.items():
                    if child < len(filtered_scores) and filtered_scores[child] > 0.01:
                        # 부모가 원래 없거나(score<threshold) OR 방금 삭제되었으면(filtered_scores==0)
                        parent_dead = False
                        if parent < len(filtered_scores):
                            if filtered_scores[parent] < self.config.confidence_threshold: parent_dead = True
                        
                        if parent_dead:
                            p_name = str(parent)
                            remove(child, f"orphan_node(parent_{p_name}_dead)")

        # =========================================================
        # [Step 6] 프레임 이탈 체크 (엄격 모드)
        # =========================================================
        if self.config.check_bounds:
            margin_w = w * self.config.bounds_margin
            margin_h = h * self.config.bounds_margin
            for i in range(len(keypoints)):
                if i in removed_indices or scores[i] < self.config.confidence_threshold: continue
                x, y = keypoints[i]
                if (margin_w < x < w-margin_w and margin_h < y < h-margin_h): continue
                
                is_predictable = False
                if self.config.allow_predictable and i in self.adjacency:
                    for adj in self.adjacency[i]:
                        # 인접 관절이 '살아있어야만' 인정
                        if adj < len(scores) and scores[adj] > self.config.min_confidence_for_prediction:
                            if adj not in removed_indices: # [중요] 삭제된 관절은 예측 근거가 될 수 없음
                                is_predictable = True
                                break
                
                if not is_predictable:
                    remove(i, f"out_of_bounds({x:.0f},{y:.0f})")
                        
        return FilterResult(filtered_scores, removed_indices, removal_reasons)

    def compute_intersection(self, src_scores, ref_scores, threshold):
        return (src_scores > threshold) & (ref_scores > threshold)
    
    def apply_intersection_mask(self, kpts, scores, mask):
        s = scores.copy()
        s[~mask] = 0.0
        return kpts, s

def create_ghost_filter(**kwargs):
    config = GhostFilterConfig(**kwargs)
    return GhostFilter(config)

def filter_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_size: Tuple[int, int],
    bounds_margin: float = 0.05,
    predictable_margin: float = 0.15
) -> np.ndarray:
    ghost_filter = create_ghost_filter(
        bounds_margin=bounds_margin,
        predictable_margin=predictable_margin
    )
    result = ghost_filter.filter_single(keypoints, scores, image_size)
    return result.filtered_scores