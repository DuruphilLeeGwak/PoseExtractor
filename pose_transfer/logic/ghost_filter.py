"""
Ghost Filter v4.2 - 손/발 할루시네이션 완벽 제거 (Final)

변경사항:
- [Critical] v3.1 구버전에서 v4.2 최신 로직으로 복구
- [New] 손가락 족보 검사: 손목(Wrist, idx 9/10)이 없으면 손가락(91~132) 자동 삭제
- [New] 스마트 중첩 방지: 손 근처에 발이 찍히면 발 삭제 (요가 자세 제외)
- [New] 하단/경계 강제 절삭: 바닥과 벽에 붙은 노이즈 제거
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
    
    # [C] 예측 가능 키포인트 허용
    allow_predictable: bool = True
    predictable_margin: float = 0.20
    min_confidence_for_prediction: float = 0.3
    
    # [D] 클러스터링 체크
    check_clustering: bool = True
    min_cluster_spread: float = 20.0
    
    # [E] 계층적 연결성 검사 (Consistency Check) [핵심]
    # 부모(손목/발목)가 없으면 자식(손가락/발)을 삭제
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
    """통합 Ghost Filter v4.2"""
    
    def __init__(self, config: Optional[GhostFilterConfig] = None):
        self.config = config or GhostFilterConfig()
        self._build_adjacency_map()
        
        # [핵심] 족보 규칙: 자식 -> 부모 매핑
        self.hierarchy_rules = {
            # --- 발 (Feet -> Ankle) ---
            17: 15, 18: 15, 19: 15, # Left Foot -> LAnkle
            20: 16, 21: 16, 22: 16, # Right Foot -> RAnkle
            
            # --- 손 (Fingers -> Wrist) ---
            # 왼쪽 손가락 (91~111) -> 왼쪽 손목 (9)
            **{i: 9 for i in range(91, 112)},
            
            # 오른쪽 손가락 (112~132) -> 오른쪽 손목 (10)
            **{i: 10 for i in range(112, 133)}
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
        # [0] 스마트 손-발 중첩 방지 (Hand-Foot Overlap)
        # =========================================================
        if self.config.check_hand_foot_overlap:
            hand_indices = [9, 10] + list(range(91, 113))
            left_leg_parts = [13, 15] + [17, 18, 19] 
            right_leg_parts = [14, 16] + [20, 21, 22] 
            
            active_hands = []
            for h_idx in hand_indices:
                if h_idx < len(scores) and scores[h_idx] > 0.1:
                    active_hands.append(keypoints[h_idx])
            
            if active_hands:
                active_hands = np.array(active_hands)
                
                # Left Leg Check
                l_hip_valid = scores[11] > self.config.hip_confidence_threshold if 11 < len(scores) else False
                for idx in left_leg_parts:
                    if idx < len(scores) and scores[idx] > 0.05:
                        if l_hip_valid: continue # 골반 있으면 봐줌
                        if np.min(np.linalg.norm(active_hands - keypoints[idx], axis=1)) < self.config.overlap_radius:
                            remove(idx, "hand_overlap_ghost(L)")

                # Right Leg Check
                r_hip_valid = scores[12] > self.config.hip_confidence_threshold if 12 < len(scores) else False
                for idx in right_leg_parts:
                    if idx < len(scores) and scores[idx] > 0.05:
                        if r_hip_valid: continue
                        if np.min(np.linalg.norm(active_hands - keypoints[idx], axis=1)) < self.config.overlap_radius:
                            remove(idx, "hand_overlap_ghost(R)")

        # =========================================================
        # [1] 계층적 연결성 검사 (Consistency Check) - 손가락 포함
        # =========================================================
        if self.config.check_consistency:
            for _ in range(2): 
                for child, parent in self.hierarchy_rules.items():
                    if (child < len(filtered_scores) and filtered_scores[child] > 0.01 and 
                        parent < len(filtered_scores) and filtered_scores[parent] < self.config.confidence_threshold):
                        
                        parent_name = "Wrist" if parent in [9, 10] else "Ankle"
                        remove(child, f"orphan_node({parent_name}_{parent}_missing)")

        # =========================================================
        # [2] 하단 강제 절삭 (Hard Bottom Cut)
        # =========================================================
        if self.config.hard_bottom_check:
            limit_y = h * self.config.hard_bottom_threshold
            for idx in range(len(keypoints)):
                if scores[idx] > 0.01:
                    if keypoints[idx][1] > limit_y:
                        remove(idx, f"hard_bottom(y>{limit_y:.0f})")

        # =========================================================
        # [3] 기존 필터 (경계값, 클러스터링, 프레임 이탈)
        # =========================================================
        if self.config.check_boundary_values:
            tol = self.config.boundary_tolerance
            for i in range(len(keypoints)):
                if scores[i] < self.config.confidence_threshold: continue
                x, y = keypoints[i]
                if x <= tol or x >= w-tol or y <= tol:
                    remove(i, f"boundary_val({x:.0f},{y:.0f})")

        if self.config.check_clustering:
            for start, end, name in [(91, 111, "LHand"), (112, 132, "RHand")]:
                points = [keypoints[i] for i in range(start, end+1) if scores[i] > 0.1]
                indices = [i for i in range(start, end+1) if scores[i] > 0.1]
                if len(points) > 5:
                    pts = np.array(points)
                    spread = np.sqrt(np.std(pts[:,0])**2 + np.std(pts[:,1])**2)
                    if spread < self.config.min_cluster_spread:
                        for idx in indices: remove(idx, f"clustered_{name}")

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
                        if adj < len(scores) and scores[adj] > self.config.min_confidence_for_prediction:
                            if adj not in removed_indices:
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

# 편의 함수
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