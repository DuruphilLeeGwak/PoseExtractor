"""
Ghost Filter v4.4 - 완전한 하반신 족보 (Full Leg Hierarchy)

변경사항:
- [Critical] Src 이미지의 '골반 없는 유령 다리' 해결
- [Fix] 족보 규칙(hierarchy_rules)에 '무릎->골반', '발목->무릎' 관계 추가
- 효과: 골반(Hip)이 0점이면 무릎(Knee), 발목(Ankle), 발(Foot)이 연쇄적으로 모두 삭제됨
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
    """통합 Ghost Filter v4.4"""
    
    def __init__(self, config: Optional[GhostFilterConfig] = None):
        self.config = config or GhostFilterConfig()
        self._build_adjacency_map()
        
        # [핵심 수정] 완전한 족보 규칙 (자식 -> 부모)
        self.hierarchy_rules = {
            # --- 1. 발가락 -> 발목 ---
            17: 15, 18: 15, 19: 15, 
            20: 16, 21: 16, 22: 16,
            
            # --- 2. 손가락 -> 손목 ---
            **{i: 9 for i in range(91, 112)},
            **{i: 10 for i in range(112, 133)},
            
            # --- 3. [NEW] 발목 -> 무릎 ---
            15: 13, # LAnkle -> LKnee
            16: 14, # RAnkle -> RKnee
            
            # --- 4. [NEW] 무릎 -> 골반 ---
            13: 11, # LKnee -> LHip
            14: 12  # RKnee -> RHip
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

        # [Step 1] 하단 강제 절삭
        if self.config.hard_bottom_check:
            limit_y = h * self.config.hard_bottom_threshold
            for idx in range(len(keypoints)):
                if scores[idx] > 0.01:
                    if keypoints[idx][1] > limit_y:
                        remove(idx, f"hard_bottom(y>{limit_y:.0f})")

        # [Step 2] 경계값 감지
        if self.config.check_boundary_values:
            tol = self.config.boundary_tolerance
            for i in range(len(keypoints)):
                if scores[i] < self.config.confidence_threshold: continue
                x, y = keypoints[i]
                if x <= tol or x >= w-tol or y <= tol:
                    remove(i, f"boundary_val({x:.0f},{y:.0f})")

        # [Step 3] 클러스터링 체크
        if self.config.check_clustering:
            for start, end, name in [(91, 111, "LHand"), (112, 132, "RHand")]:
                points = [keypoints[i] for i in range(start, end+1) if scores[i] > 0.1]
                indices = [i for i in range(start, end+1) if scores[i] > 0.1]
                if len(points) > 5:
                    pts = np.array(points)
                    spread = np.sqrt(np.std(pts[:,0])**2 + np.std(pts[:,1])**2)
                    if spread < self.config.min_cluster_spread:
                        for idx in indices: remove(idx, f"clustered_{name}")

        # [Step 4] 스마트 손-발 중첩 방지
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
                
                l_hip_valid = scores[11] > self.config.hip_confidence_threshold if 11 < len(scores) else False
                for idx in left_leg_parts:
                    if idx not in removed_indices and idx < len(scores) and scores[idx] > 0.05:
                        if l_hip_valid: continue
                        if np.min(np.linalg.norm(active_hands - keypoints[idx], axis=1)) < self.config.overlap_radius:
                            remove(idx, "hand_overlap_ghost(L)")

                r_hip_valid = scores[12] > self.config.hip_confidence_threshold if 12 < len(scores) else False
                for idx in right_leg_parts:
                    if idx not in removed_indices and idx < len(scores) and scores[idx] > 0.05:
                        if r_hip_valid: continue
                        if np.min(np.linalg.norm(active_hands - keypoints[idx], axis=1)) < self.config.overlap_radius:
                            remove(idx, "hand_overlap_ghost(R)")

        # [Step 5] 계층적 연결성 검사 (The Chain Reaction)
        # 이제 무릎->골반 규칙이 추가되어, 골반이 없으면 무릎도 삭제됩니다.
        if self.config.check_consistency:
            for _ in range(3): # 연쇄 작용(Hip->Knee->Ankle->Foot)을 위해 3회 반복
                for child, parent in self.hierarchy_rules.items():
                    if child < len(filtered_scores) and filtered_scores[child] > 0.01:
                        # 부모가 죽었으면 자식도 죽음
                        if (parent < len(filtered_scores) and filtered_scores[parent] < self.config.confidence_threshold):
                            
                            p_name = "Joint"
                            if parent in [11, 12]: p_name = "Hip"
                            elif parent in [13, 14]: p_name = "Knee"
                            elif parent in [15, 16]: p_name = "Ankle"
                            elif parent in [9, 10]: p_name = "Wrist"
                            
                            remove(child, f"orphan_node({p_name}_{parent}_dead)")

        # [Step 6] 프레임 이탈 체크
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