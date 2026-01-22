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
from dataclasses import dataclass, field

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

    # [D-2] 손 폐색/환각(예측) 억제
    # 목적: 손이 가려졌는데도 손가락을 '예측'으로 그리는 문제를 줄이되,
    #       손이 작게 보이는 경우(멀리 있는 손)까지 과하게 제거하지 않도록
    #       "손목 주변에 손가락이 모여있는지"(기하)로 판단합니다.
    #
    # - 손가락(91~132)만 제거 대상이며 손목(9/10)은 유지합니다.
    # - elbow/wrist가 유효할 때 forearm 길이를 기준으로 반경을 잡습니다.
    check_hand_occlusion: bool = True
    # (legacy alias) 기존 파이프라인 파라미터 호환용
    check_hand_presence: bool = True

    hand_finger_min_confidence: float = 0.1
    hand_min_finger_points: int = 6

    # 손목 주변 "정상 반경": max(min_radius_px, forearm * radius_ratio)
    hand_wrist_radius_ratio: float = 0.8
    hand_wrist_min_radius_px: float = 60.0

    # 정상으로 인정되는 "근접 비율": 근접점/활성점 >= min_near_ratio
    hand_min_near_ratio: float = 0.8

    # 멀리 튄 점(outlier)이 많으면 환각으로 판단
    hand_far_outlier_ratio: float = 1.6
    hand_max_far_points: int = 1
    
    # [D-3] 발 폐색/환각 억제 (발 전용 기준)
    # 발은 3개 키포인트만 있고 좁은 영역에 모여있는 게 정상이므로
    # 손과 다른 기준을 적용합니다.
    foot_min_distance_std: float = 10.0  # 발은 좁은 영역 (손: 30.0)
    foot_min_avg_confidence: float = 4.0  # 발은 높은 신뢰도 요구 (손: 2.0)
    
    # [E] 계층적 연결성 검사 (Consistency Check)
    check_consistency: bool = True
    
    # [F] 하단 강제 절삭 (발 키포인트 보존 위해 비활성화)
    hard_bottom_check: bool = False  # ⚠️ True로 설정 시 이미지 하단 키포인트 무조건 제거
    hard_bottom_threshold: float = 0.95
    
    # [G] 스마트 손-발 중첩 방지
    check_hand_foot_overlap: bool = True
    overlap_radius: float = 150.0
    hip_confidence_threshold: float = 0.4
    
    confidence_threshold: float = 0.1

    # Debug
    debug_hand_removals: bool = False
    debug_hand_print_limit: int = 80
    debug_hand_summary_only: bool = False
    debug_hand_include_wrists: bool = True
    debug_hand_print_to_console: bool = True


@dataclass
class FilterResult:
    """필터링 결과"""
    filtered_scores: np.ndarray          # 필터링된 신뢰도 배열 (제거된 키포인트는 0)
    removed_indices: Set[int]            # 완전히 제거된 키포인트 인덱스 집합
    removal_reasons: Dict[int, str]      # {인덱스: 제거사유} 매핑
    debug_lines: List[str] = field(default_factory=list)  # 디버그 메시지 리스트
    occluded_indices: Set[int] = field(default_factory=set)  # -1 레이어: 가려진 키포인트 (이미지 안)
    out_of_frame_indices: Set[int] = field(default_factory=set)  # -2 레이어: 프레임 밖 키포인트


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
        keypoints: np.ndarray,   # (133, 2) 키포인트 좌표 [x, y]
        scores: np.ndarray,      # (133,) 신뢰도 점수
        image_size: Tuple[int, int]  # (height, width)
    ) -> FilterResult:
        if not self.config.enabled:
            return FilterResult(scores.copy(), set(), {}, [], set(), set())
        
        h: int = image_size[0]  # 이미지 높이 (픽셀)
        w: int = image_size[1]  # 이미지 폭 (픽셀)
        filtered_scores: np.ndarray = scores.copy()  # 수정될 신뢰도 배열
        removed_indices: Set[int] = set()  # 완전 제거된 인덱스들
        removal_reasons: Dict[int, str] = {}  # 제거 사유 기록
        debug_lines: List[str] = []  # 디버그 메시지들
        occluded_indices: Set[int] = set()  # -1 레이어: 가려진 키포인트
        out_of_frame_indices: Set[int] = set()  # -2 레이어: 프레임 밖 키포인트

        def explain_reason(r: str) -> str:
            """제거 사유(reason)에 대한 짧은 한국어 설명을 반환합니다."""
            key = (r or "").split("(", 1)[0]
            if key.startswith("boundary_val"):
                return "경계(0/가장자리) 좌표로 찍힌 감지 실패 패턴으로 제거"
            if key.startswith("hard_bottom"):
                return "이미지 하단 영역으로 판정되어 제거(hard bottom)"
            if key.startswith("clustered_LHand") or key.startswith("clustered_RHand"):
                return "손 키포인트가 한 점에 뭉쳐 감지 실패로 판단되어 제거"
            if key.startswith("occluded_LHand") or key.startswith("occluded_RHand"):
                return "-1 레이어로 마킹됨 (가려진 것으로 판단, 제거 안 됨!)"
            if key.startswith("hand_overlap_ghost"):
                return "손-발 중첩으로 유령(오검출) 가능성이 높아 제거"
            if key.startswith("orphan_node"):
                return "부모 관절이 죽어서(점수 0) 연쇄적으로 제거(chain kill)"
            if key.startswith("out_of_bounds"):
                return "프레임 밖으로 나가 예측 불가로 제거"
            return "필터 규칙에 의해 제거"
        
        debug = bool(getattr(self.config, "debug_hand_removals", False))
        summary_only = bool(getattr(self.config, "debug_hand_summary_only", False))
        include_wrists = bool(getattr(self.config, "debug_hand_include_wrists", True))
        print_to_console = bool(getattr(self.config, "debug_hand_print_to_console", True))
        debug_limit = int(getattr(self.config, "debug_hand_print_limit", 80))
        debug_printed = 0
        if include_wrists:
            hand_dbg_indices = set([9, 10] + list(range(91, 133)))
        else:
            hand_dbg_indices = set(range(91, 133))

        def remove(idx, reason):
            if idx not in removed_indices:
                filtered_scores[idx] = 0.0
                removed_indices.add(idx)
                removal_reasons[idx] = reason

                nonlocal debug_printed
                if (not summary_only) and debug and idx in hand_dbg_indices and debug_printed < debug_limit:
                    debug_printed += 1
                    # 디버그 출력(최소/핵심 정보)
                    x, y = keypoints[idx]
                    note = explain_reason(reason)
                    line = (
                        f"[HAND] 제거 idx={idx} score={scores[idx]:.3f} "
                        f"xy=({x:.1f},{y:.1f}) 사유={reason} | 설명={note}"
                    )
                    debug_lines.append(line)
                    if print_to_console:
                        print(f"   [GhostFilter]{line}")

        # =========================================================
        # [Step 1] 하단 강제 절삭 - 비활성화됨
        # =========================================================
        # ❌ 제거됨: 프레임 이탈은 렌더링 시점에서 처리
        # if self.config.hard_bottom_check:
        #     ...

        # =========================================================
        # [Step 2] 더미 좌표 및 프레임 이탈 처리
        # =========================================================
        # A) 더미 좌표: DWPose 감지 실패 시 경계값(0, 0) 등에 점을 찍는 패턴 제거
        # B) 프레임 이탈: 이미지 경계 밖 키포인트를 -2 레이어로 마킹
        # C) 해부학적 비정상: 발이 상체에 있는 등의 물리적 불가능 감지
        
        # 신체 부위 매핑 (idx → 부위명)
        def get_body_part_name(idx: int) -> str:
            if idx == 0: return "Nose"
            elif 1 <= idx <= 4: return "Face(eyes/ears)"
            elif 5 <= idx <= 10: return "Arms"
            elif 11 <= idx <= 16: return "Legs"
            elif 17 <= idx <= 19: return "LFoot"
            elif 20 <= idx <= 22: return "RFoot"
            elif 23 <= idx <= 90: return "Face"
            elif 91 <= idx <= 111: return "LHand"
            elif 112 <= idx <= 132: return "RHand"
            else: return "Unknown"
        
        # 해부학적 비정상 감지 (발이 상체에 있는 경우)
        def is_anatomically_abnormal(idx: int, y: float) -> bool:
            # 발(idx 17-22)이 y < 2000인 상체 위치에 있으면 비정상
            if 17 <= idx <= 22 and y < h * 0.7:  # 이미지 높이의 70% 이상은 하체
                return True
            return False
        
        if self.config.check_boundary_values or self.config.check_bounds:
            tol: float = self.config.boundary_tolerance  # 경계 허용 오차 (픽셀)
            margin: float = self.config.bounds_margin if self.config.check_bounds else 0.01
            
            # 🔍 주요 키포인트 좌표 디버그 출력
            if debug and print_to_console:
                key_points = {
                    9: "LWrist", 10: "RWrist",
                    11: "LHip", 12: "RHip",
                    13: "LKnee", 14: "RKnee",
                    15: "LAnkle", 16: "RAnkle",
                    17: "LBigToe", 18: "LSmallToe", 19: "LHeel",
                    20: "RBigToe", 21: "RSmallToe", 22: "RHeel"
                }
                print(f"\n   🔍 [Step2 디버그] 프레임 크기: {w}x{h}, margin: {margin*100:.1f}% ({int(w*margin)}px x {int(h*margin)}px)")
                print(f"   ⚠️  더미 좌표 판정 기준: 경계 ±{tol:.0f}px 이내 (y >= {h-tol:.0f}이면 하단 더미)")
                for idx, name in key_points.items():
                    if idx < len(keypoints) and scores[idx] >= self.config.confidence_threshold:
                        x, y = keypoints[idx]
                        # 해부학적 비정상 감지 (발이 상체에)
                        anomaly = ""
                        if idx >= 20 and idx <= 22 and y < 2000:  # RFoot가 상체에
                            anomaly = " ⚠️ 할루시네이션(발이 상체 위치)"
                        elif idx >= 17 and idx <= 19 and y < 2000:  # LFoot가 상체에
                            anomaly = " ⚠️ 할루시네이션(발이 상체 위치)"
                        print(f"   [Step2] idx={idx:2d} {name:10s} xy=({x:7.1f},{y:7.1f}) score={scores[idx]:.3f}{anomaly}")
            
            for idx in range(len(keypoints)):
                if scores[idx] < self.config.confidence_threshold:
                    continue
                
                x: float = keypoints[idx][0]  # x 좌표
                y: float = keypoints[idx][1]  # y 좌표
                
                # --- [판정 1] 더미 좌표인가? ---
                # DWPose 감지 실패 시 (0, y) 또는 (w-1, y) 같은 경계값에 점을 찍음
                # ✅ 하단 경계는 더 너그럽게 (80px) - 정상적인 발이 하단에 있을 수 있음
                bottom_tol = 80.0  # 하단은 80px까지 허용 (발 보존)
                is_dummy_coord: bool = False
                dummy_reason: str = ""
                if self.config.check_boundary_values:
                    if x <= tol:
                        is_dummy_coord = True
                        dummy_reason = f"x<={tol:.0f} (좌측경계)"
                    elif x >= w - tol:
                        is_dummy_coord = True
                        dummy_reason = f"x>={w-tol:.0f} (우측경계)"
                    elif y <= tol:
                        is_dummy_coord = True
                        dummy_reason = f"y<={tol:.0f} (상단경계)"
                    elif y >= h - bottom_tol:
                        # 하단 경계 - 신뢰도가 낮을 때만 더미로 판정
                        if scores[idx] < 4.0:  # 신뢰도 4.0 이상이면 정상 발로 간주
                            is_dummy_coord = True
                            dummy_reason = f"y>={h-bottom_tol:.0f} (하단경계, 실제y={y:.1f}, conf={scores[idx]:.2f}<4.0)"
                
                # --- [판정 2] 프레임 밖인가? ---
                # 일반적인 프레임 이탈 (x < 0, x >= w 등)
                is_out_of_frame: bool = False
                out_reason: str = ""
                if self.config.check_bounds:
                    margin_px_w = int(w * margin)
                    margin_px_h = int(h * margin)
                    if x < -margin_px_w:
                        is_out_of_frame = True
                        out_reason = f"x<{-margin_px_w} (좌측밖)"
                    elif x >= w + margin_px_w:
                        is_out_of_frame = True
                        out_reason = f"x>={w+margin_px_w} (우측밖)"
                    elif y < -margin_px_h:
                        is_out_of_frame = True
                        out_reason = f"y<{-margin_px_h} (상단밖)"
                    elif y >= h + margin_px_h:
                        is_out_of_frame = True
                        out_reason = f"y>={h+margin_px_h} (하단밖, 실제y={y:.1f})"
                
                # --- [판정 3] 해부학적 비정상인가? ---
                # 발이 상체에 있는 등의 물리적 불가능 감지
                is_anatomical_error: bool = is_anatomically_abnormal(idx, y)
                if is_anatomical_error:
                    is_out_of_frame = True  # 해부학적 비정상은 프레임 밖으로 처리
                    out_reason = f"anatomical_error(발이 상체 y={y:.0f}, 정상범위>={h*0.7:.0f})"
                
                # --- [판정 4] 하체 발(foot)이 boundary 근처 + 낮은 confidence인가? ---
                # ✅ 발목(15,16)은 제외! 발목은 항상 하단에 있고 높은 신뢰도 가짐
                # ✅ 무릎(13,14)도 제외! 무릎은 하단보다 위에 있음
                # ✅ 발(17-22)만 체크: 하단 경계 + 매우 낮은 신뢰도일 때만 더미로 판정
                is_weak_leg: bool = False
                weak_leg_reason: str = ""
                if idx in [17, 18, 19, 20, 21, 22]:  # 발만 체크 (발목/무릎 제외)
                    boundary_near_threshold = h - 50  # 하단 50px 이내 (더 엄격)
                    # 신뢰도 4.5 미만이면서 하단 50px 이내에 있으면 더미로 판정
                    # (정상 발은 보통 6.0 이상 신뢰도)
                    if y >= boundary_near_threshold and scores[idx] < 4.5:
                        is_weak_leg = True
                        is_dummy_coord = True  # 더미로 처리
                        weak_leg_reason = f"weak_foot(y={y:.1f}>={boundary_near_threshold:.0f}, conf={scores[idx]:.2f}<4.5)"
                        dummy_reason = weak_leg_reason
                
                # 더미도 아니고 프레임 밖도 아니면 스킵
                if not is_dummy_coord and not is_out_of_frame:
                    continue
                
                # --- [판정 3] 부모가 유효한가? ---
                parent_idx: Optional[int] = self.hierarchy_rules.get(idx)  # 부모 인덱스 (없으면 None)
                parent_valid: bool = False  # 부모가 유효한지 플래그
                
                if parent_idx is not None:  # 부모가 존재하면
                    # 부모가 아직 제거되지 않았고 신뢰도가 충분한지 확인
                    parent_valid = (
                        parent_idx not in removed_indices and
                        filtered_scores[parent_idx] >= self.config.confidence_threshold
                    )
                
                # ========================================
                # [처리 로직]
                # ========================================
                
                # 🔴 케이스 A: 더미 + 부모 없음 → 완전 제거
                if is_dummy_coord and not parent_valid:
                    remove(idx, f"dummy_no_parent({x:.0f},{y:.0f})")
                
                # 🟡 케이스 B: 프레임 밖 또는 (더미 + 부모 있음) → -2 레이어 마킹
                elif is_out_of_frame or (is_dummy_coord and parent_valid):
                    out_of_frame_indices.add(idx)  # -2 레이어 마킹
                    # filtered_scores[idx]는 그대로 유지 (제거하지 않음!)
                    if debug and print_to_console:
                        body_part = get_body_part_name(idx)
                        if is_out_of_frame:
                            reason_detail = f"out_of_frame({out_reason})"
                        else:
                            reason_detail = f"dummy({dummy_reason})+parent_valid(idx={parent_idx})"
                        print(f"   [GhostFilter][Step2] idx={idx:3d} [{body_part:15s}] -2레이어 마킹 xy=({x:.1f},{y:.1f}) reason={reason_detail}")
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
        # [Step 3.5] 폐색/환각 억제 (기하 기반)
        # =========================================================
        # 키포인트가 가려졌는데도 '예측'으로 생성되는 경우,
        # 해당 키포인트를 -1 레이어(occluded)로 마킹합니다.
        # 
        # 현재 구현: 손 키포인트(91-132)에 대해서만 처리
        # - 작은 손(멀리 있음)은 전체 스케일이 작아도 손목 주변에 점들이 모이는 특성
        # - 가려져 환각이 생기면 점들이 손목에서 멀리 튀거나(outlier) 방향성이 깨지는 경향
        # =========================================================
        if self.config.check_hand_occlusion or self.config.check_hand_presence:
            thr = float(self.config.hand_finger_min_confidence)
            min_pts = int(self.config.hand_min_finger_points)
            radius_ratio = float(self.config.hand_wrist_radius_ratio)
            min_radius = float(self.config.hand_wrist_min_radius_px)
            min_near_ratio = float(self.config.hand_min_near_ratio)
            far_ratio = float(self.config.hand_far_outlier_ratio)
            max_far = int(self.config.hand_max_far_points)
            min_avg_conf = float(getattr(self.config, 'hand_min_avg_confidence', 3.0))
            min_dist_std = float(getattr(self.config, 'hand_min_distance_std', 50.0))
            foot_min_dist_std = float(getattr(self.config, 'foot_min_distance_std', 10.0))
            foot_min_avg_conf = float(getattr(self.config, 'foot_min_avg_confidence', 4.0))

            # 신체 부위 정의: (name, parent_idx, anchor_idx, start_idx, end_idx, min_radius_override)
            # - Hands: elbow-wrist-fingers (21 points each)
            # - Feet: knee-ankle-toes (3 points each)
            # - Face: neck-nose-face_points (68 points)
            # - LArm/RArm: shoulder-elbow-wrist (3 points each)
            sides = [
                ("LHand", 7, 9, 91, 111, None),     # 왼손: 팔꿈치-손목-손가락
                ("RHand", 8, 10, 112, 132, None),   # 오른손: 팔꿈치-손목-손가락
                ("LFoot", 13, 15, 17, 19, None),    # 왼발: 무릎-발목-발가락
                ("RFoot", 14, 16, 20, 22, None),    # 오른발: 무릎-발목-발가락
                ("Face", 1, 0, 23, 90, 80.0),       # 얼굴: 목-코-얼굴포인트 (작은 영역)
                ("LArm", 5, 7, 5, 9, None),         # 왼팔: 왼어깨-왼팔꿈치-왼손목
                ("RArm", 6, 8, 6, 10, None),        # 오른팔: 오른어깨-오른팔꿈치-오른손목
                ("LLeg", 11, 13, 11, 15, None),     # 왼다리: 왼엉덩이-왼무릎-왼발목
                ("RLeg", 12, 14, 12, 16, None),     # 오른다리: 오른엉덩이-오른무릎-오른발목
            ]
            for name, parent_idx, anchor_idx, start, end, min_radius_override in sides:
                if anchor_idx >= len(keypoints) or anchor_idx >= len(filtered_scores):
                    continue
                
                # 부위별로 다른 총 키포인트 개수
                total_pts = end - start + 1
                # 부위별 최소 포인트 요구량 조정 (손:6/21, 발:2/3, 얼굴:20/68, 팔/다리:2)
                adjusted_min_pts = min(min_pts, max(2, total_pts // 3))
                
                # anchor 키포인트 (손목/발목/코/팔꿈치) 신뢰도 체크
                if filtered_scores[anchor_idx] < 0.3:
                    if debug:
                        line = (
                            f"[BODY][Step3.5] {name} verdict=SKIP(anchor<0.3) "
                            f"anchor_score={float(filtered_scores[anchor_idx]):.3f} thr={thr:.2f}"
                        )
                        debug_lines.append(line)
                        if print_to_console:
                            print(f"   [GhostFilter]{line}")
                    continue
                anchor = keypoints[anchor_idx]

                # parent optional but improves scale (forearm/shin length)
                if parent_idx < len(filtered_scores) and filtered_scores[parent_idx] >= 0.3:
                    parent = keypoints[parent_idx]
                    limb_length = float(np.linalg.norm(anchor - parent))
                else:
                    limb_length = 0.0

                # Determine expected radius around anchor
                # 부위별 min_radius 조정 (얼굴은 작은 영역)
                effective_min_radius = min_radius_override if min_radius_override else min_radius
                base_radius = max(effective_min_radius, limb_length * radius_ratio) if limb_length > 1.0 else effective_min_radius
                far_radius = max(base_radius * 1.5, limb_length * far_ratio) if limb_length > 1.0 else base_radius * 1.5

                active_idxs = []
                near_count = 0
                far_count = 0
                active_details = []
                distances = []  # 거리 분산 계산용
                confidence_sum = 0.0  # 평균 신뢰도 계산용

                for idx in range(start, end + 1):
                    if idx >= len(filtered_scores) or idx in removed_indices:
                        continue
                    # 프레임 밖 키포인트는 제외 (Step 3.4에서 이미 마킹됨)
                    if idx in out_of_frame_indices:
                        continue
                    c = filtered_scores[idx]
                    if c < thr:
                        continue
                    active_idxs.append(idx)
                    d = float(np.linalg.norm(keypoints[idx] - anchor))
                    active_details.append((idx, float(c), float(d)))
                    distances.append(d)
                    confidence_sum += float(c)
                    if d <= base_radius:
                        near_count += 1
                    if d >= far_radius:
                        far_count += 1

                if len(active_idxs) < adjusted_min_pts:
                    if debug:
                        line = (
                            f"[BODY][Step3.5] {name} verdict=SKIP(active<min_pts) "
                            f"active={len(active_idxs)}/{total_pts} min_pts={adjusted_min_pts} thr={thr:.2f} "
                            f"anchor_score={float(filtered_scores[anchor_idx]):.3f}"
                        )
                        debug_lines.append(line)
                        if print_to_console:
                            print(f"   [GhostFilter]{line}")
                    continue

                near_ratio = near_count / max(1, len(active_idxs))
                avg_confidence = confidence_sum / max(1, len(active_idxs))
                dist_std = float(np.std(distances)) if len(distances) > 1 else 0.0
                
                # 🔍 추가 지표: 좌표 클러스터링 감지 (y좌표만 경계에 몰려있는지)
                y_coords = [keypoints[idx][1] for idx in active_idxs]
                y_std = float(np.std(y_coords)) if len(y_coords) > 1 else 0.0
                y_min, y_max = min(y_coords), max(y_coords)
                y_range = y_max - y_min
                y_near_boundary = sum(1 for y in y_coords if y >= h - 30) / max(1, len(y_coords))  # 경계 30px 이내 비율
                
                # 🔍 추가 지표: 신뢰도 분산 (실제 감지는 분산 큼, 추정은 분산 작음)
                confidences = [filtered_scores[idx] for idx in active_idxs]
                conf_std = float(np.std(confidences)) if len(confidences) > 1 else 0.0
                
                verdict = "KEEP"
                cause = ""
                
                # 🎯 DWPose 추정 감지 (더미 좌표 패턴)
                is_estimated = False
                estimation_score = 0  # 0-5점, 3점 이상이면 추정으로 판단
                
                # 지표 1: y좌표가 20px 범위에 몰려있음
                if y_range < 20:
                    estimation_score += 1
                    
                # 지표 2: 대부분이 경계 30px 이내
                if y_near_boundary > 0.7:  # 70% 이상이 경계 근처
                    estimation_score += 1
                    
                # 지표 3: 평균 신뢰도 낮음
                if avg_confidence < 3.0:
                    estimation_score += 1
                    
                # 지표 4: y좌표 분산 매우 작음 (수평 클러스터링)
                if y_std < 10:
                    estimation_score += 1
                    
                
                # 1.5) DWPose 추정 패턴 감지 (더 정확한 판단)
                if is_estimated and not skip_geometric_check:
                    verdict = "OCCLUDED"
                    if cause:
                        cause += f", DWPose추정(score={estimation_score}/5: y_range={y_range:.1f}, y_boundary={y_near_boundary:.1%}, y_std={y_std:.1f})"
                    else:
                        cause = f"DWPose추정(score={estimation_score}/5: y_range={y_range:.1f}, y_boundary={y_near_boundary:.1%}, y_std={y_std:.1f})"
                # 지표 5: 신뢰도 분산 작음 (모두 비슷한 낮은 값)
                if conf_std < 1.0:
                    estimation_score += 1
                
                if estimation_score >= 3:
                    is_estimated = True
                
                # 부위별 완화된 기준
                # ✅ 손: min_avg_conf=2.0, min_dist_std=30.0
                # ✅ 얼굴: min_avg_conf=4.0 (작은 영역, 보통 고신뢰도), dist_std=15.0
                # ✅ 팔/다리: 긴 영역이므로 기하 체크 무시, min_avg_conf=1.5
                # ✅ 발: 하단에 있어도 정상이므로 기하 체크 무시, min_avg_conf=1.5
                effective_min_dist_std = min_dist_std
                effective_min_avg_conf = min_avg_conf
                skip_dist_check = False
                skip_geometric_check = False  # near_ratio, far_count 체크 무시
                
                if name == "Face":
                    effective_min_dist_std = 15.0  # 얼굴은 작은 영역
                    effective_min_avg_conf = 4.0  # 얼굴은 보통 고신뢰도
                    if avg_confidence > 5.0:  # 신뢰도 높으면 모든 기하 체크 무시
                        skip_geometric_check = True
                        skip_dist_check = True
                elif name in ["LHand", "RHand"]:
                    effective_min_dist_std = 30.0  # ✅ 손: 작은 손도 정상
                    effective_min_avg_conf = 2.0  # ✅ 손: 2.0 미만은 추정
                    # 신뢰도 높으면 기하 체크 무시
                    if avg_confidence > 3.5:
                        skip_geometric_check = True
                        skip_dist_check = True
                elif name in ["LArm", "RArm", "LLeg", "RLeg"]:
                    skip_dist_check = True  # 팔/다리는 긴 영역이므로 dist_std 무시
                    skip_geometric_check = True  # near_ratio, far_count도 무의미
                    # ✅ 발 전용 기준 적용: 3개 키포인트만 있고 좁은 영역에 모이는 게 정상
                    effective_min_dist_std = foot_min_dist_std  # 10.0 (손: 30.0)
                    effective_min_avg_conf = foot_min_avg_conf  # 4.0 (손: 2.0)
                    skip_geometric_check = True  # 하단 위치로 인한 오판 방지
                    # dist_std는 체크하되 발 전용 낮은 기준(10.0) 적용
                    skip_dist_check = False
                    skip_geometric_check = True  # 하단 위치로 인한 오판 방지
                    effective_min_avg_conf = 1.5  # 발은 낮은 신뢰도도 정상
                
                # 1) 평균 신뢰도 체크 (추측 탐지) - 부위별 기준 적용
                if avg_confidence < effective_min_avg_conf:
                    verdict = "OCCLUDED"
                    cause = f"avg_conf {avg_confidence:.2f} < min {effective_min_avg_conf:.2f} (추측/환각)"
                
                # 2) 거리 분산 체크 (비정상적 균일성) - 부위별 조건부 적용
                if not skip_dist_check and dist_std < effective_min_dist_std:
                    verdict = "OCCLUDED"
                    cause = (cause + ", ") if cause else ""
                    cause += f"dist_std {dist_std:.1f} < min {effective_min_dist_std:.1f} (균일함)"
                
                # 3) 기존 near_ratio 체크 - 부위별 조건부 적용
                if not skip_geometric_check and near_ratio < min_near_ratio:
                    verdict = "OCCLUDED"
                    cause = (cause + ", ") if cause else ""
                    cause += f"near_ratio {near_ratio:.2f} < min {min_near_ratio:.2f}"
                
                # 4) 기존 far_count 체크 - 부위별 조건부 적용
                if not skip_geometric_check and far_count > max_far:
                    verdict = "OCCLUDED"
                    cause = (cause + ", ") if cause else ""
                    cause += f"far_count {far_count} > max {max_far}"

                if debug:
                    anchor_score = float(filtered_scores[anchor_idx])
                    parent_score = float(filtered_scores[parent_idx]) if parent_idx < len(filtered_scores) else 0.0
                    line = (
                        f"[BODY][Step3.5] {name} verdict={verdict} "
                        f"active={len(active_idxs)}/{total_pts} near={near_count} far={far_count} "
                        f"near_ratio={near_ratio:.3f} avg_conf={avg_confidence:.2f} dist_std={dist_std:.1f} "
                        f"thr={thr:.2f} min_pts={adjusted_min_pts} "
                        f"min_near={min_near_ratio:.2f} max_far={max_far} min_avg_conf={min_avg_conf:.2f} min_dist_std={min_dist_std:.1f} "
                        f"anchor_score={anchor_score:.3f} parent_score={parent_score:.3f} "
                        f"limb_len={limb_length:.1f} base_r={base_radius:.1f} far_r={far_radius:.1f} "
                        f"[추정감지: score={estimation_score}/5 y_range={y_range:.1f} y_boundary={y_near_boundary:.1%} y_std={y_std:.1f} conf_std={conf_std:.2f}]"
                    )
                    debug_lines.append(line)
                    if verdict == "OCCLUDED":  # REMOVE → OCCLUDED
                        note = f"[BODY][Step3.5] {name} 설명: {cause} -> 가림/환각 가능성으로 -1 레이어 마킹 (제거 안 됨!)"
                        debug_lines.append(note)
                    # Add a tiny hint about farthest points (top 5)
                    active_details.sort(key=lambda t: t[2], reverse=True)
                    top = active_details[:5]
                    if top:
                        top_s = ", ".join([f"idx{ii}:d={dd:.1f},c={cc:.2f}" for ii, cc, dd in top])
                        debug_lines.append(f"[BODY][Step3.5] {name} 가장먼(active): {top_s}")
                    if print_to_console:
                        print(f"   [GhostFilter]{line}")
                        if verdict == "REMOVE":
                            print(f"   [GhostFilter]{note}")

                # 🟡 가림/환각으로 판단되면 제거 대신 마킹 (verdict 기반)
                if verdict == "OCCLUDED":
                    for idx in active_idxs:
                        # 제거하지 않고 가려진 것으로 마킹!
                        occluded_indices.add(idx)  # -1 레이어: 가려진 키포인트
                        # filtered_scores[idx]는 그대로 유지 (score 0으로 만들지 않음!)
                        # 디버그 기록만 남김
                        if idx not in removal_reasons:  # 중복 방지
                            removal_reasons[idx] = f"occluded_{name}({cause})(마킹만)"
                    
                    # 🟡 가려진 키포인트 박스 표시
                    if debug and print_to_console:
                        print(f"")
                        print(f"   ┌────────────────────────────────────────────────────────────────────────────┐")
                        print(f"   │ 🟡 [-1 레이어] 가려진 키포인트 (occluded_indices)                          │")
                        print(f"   ├────────────────────────────────────────────────────────────────────────────┤")
                        # start, end는 sides 튜플의 값 사용 (하드코딩 제거)
                        print(f"   │ {name}: {len(active_idxs)}개 키포인트가 가려진 것으로 마킹됨 (idx {start}-{end})" + " "*(22-len(name)) + "│")
                        print(f"   │        → score는 유지됨 (제거 안 됨!)                                      │")
                        print(f"   │        → 렌더링 시 점선/투명도로 표현 가능                                 │")
                        print(f"   └────────────────────────────────────────────────────────────────────────────┘")

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
            feet_indices = [15, 16, 17, 18, 19, 20, 21, 22]
            for _ in range(3):
                for child, parent in self.hierarchy_rules.items():
                    if child < len(filtered_scores) and filtered_scores[child] > 0.01:
                        # 부모가 원래 없거나(score<threshold) OR 방금 삭제되었으면(filtered_scores==0)
                        parent_dead = False
                        if parent < len(filtered_scores):
                            if filtered_scores[parent] < self.config.confidence_threshold: parent_dead = True
                        
                        if parent_dead:
                            p_name = str(parent)
                            if child in feet_indices or parent in feet_indices:
                                print(f"   [GhostFilter] 🦶 발 체인킬: child={child} 제거됨 (parent={parent} 죽음)")
                            remove(child, f"orphan_node(parent_{p_name}_dead)")

        # 신체 부위 디버그 요약
        if debug:
            def _summarize(name: str, start: int, end: int, tag: str = "BODY") -> None:
                total_pts = end - start + 1
                alive = int(np.sum(filtered_scores[start:end + 1] > 0.0))
                occluded = len([i for i in range(start, end + 1) if i in occluded_indices])
                reasons: Dict[str, int] = {}
                for i in range(start, end + 1):
                    r = removal_reasons.get(i)
                    if not r:
                        continue
                    key = r.split("(", 1)[0]
                    reasons[key] = reasons.get(key, 0) + 1
                # order by count desc
                reasons_sorted = dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
                line = f"[{tag}] {name}: 남은={alive}/{total_pts} 가려진={occluded}/{total_pts} 제거사유={reasons_sorted if reasons_sorted else '{}'}"
                debug_lines.append(line)
                if print_to_console:
                    print(f"   [GhostFilter]{line}")

                if reasons_sorted:
                    top_reason = next(iter(reasons_sorted.keys()))
                    top_cnt = reasons_sorted[top_reason]
                    note = explain_reason(top_reason)
                    note_line = f"[{tag}] {name} 설명: 최다사유={top_reason} x{top_cnt} -> {note}"
                    debug_lines.append(note_line)
                    if print_to_console:
                        print(f"   [GhostFilter]{note_line}")

            _summarize("LHand", 91, 111, "HAND")
            _summarize("RHand", 112, 132, "HAND")
            _summarize("LFoot", 17, 19, "FOOT")
            _summarize("RFoot", 20, 22, "FOOT")
            _summarize("Face", 23, 90, "FACE")
            _summarize("LArm", 5, 9, "ARM")
            _summarize("RArm", 6, 10, "ARM")
            _summarize("LLeg", 11, 15, "LEG")
            _summarize("RLeg", 12, 16, "LEG")
                        
        return FilterResult(
            filtered_scores=filtered_scores,
            removed_indices=removed_indices,
            removal_reasons=removal_reasons,
            debug_lines=debug_lines,
            occluded_indices=occluded_indices,
            out_of_frame_indices=out_of_frame_indices
        )

    def compute_intersection(self, src_scores, ref_scores, threshold):
        """
        ❌ 기존: src AND ref 교집합 (src에 없으면 제외)
        ✅ 변경: ref 기준 (ref에 있으면 포함)
        """
        # ref에 있는 키포인트를 모두 포함 (src 여부 무관)
        return ref_scores > threshold
    
    def apply_intersection_mask(self, kpts, scores, mask):
        """
        ref 기준 마스크 적용
        """
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