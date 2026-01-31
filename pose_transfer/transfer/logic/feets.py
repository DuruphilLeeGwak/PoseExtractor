"""
Feet Transfer Logic Module (Refactored v6.0 - Ref Visibility Respect)

위치: pose_transfer/transfer/logic/feets.py

DWPose Foot Keypoint Indices (17-22):
  17: left_big_toe
  18: left_small_toe
  19: left_heel
  20: right_big_toe
  21: right_small_toe
  22: right_heel

원칙:
1. 발 크기(길이): Src의 비율 사용 (global_scale 적용)
2. 발 방향: Ref의 발 방향을 따름
3. **Ref에 없는 키포인트는 Trans에서도 생성하지 않음** (v6 핵심 수정)
"""
import numpy as np
from typing import Dict, Optional
from ..config import TransferConfig


class FeetTransfer:
    def __init__(self, config: TransferConfig):
        self.config = config
        
        # DWPose 공식 인덱스
        self.feet_indices = {
            'left': {
                'ankle': 15, 
                'knee': 13,
                'big_toe': 17,    # left_big_toe
                'small_toe': 18,  # left_small_toe
                'heel': 19        # left_heel
            },
            'right': {
                'ankle': 16, 
                'knee': 14,
                'big_toe': 20,    # right_big_toe
                'small_toe': 21,  # right_small_toe
                'heel': 22        # right_heel
            }
        }
        
        self.MIN_SCORE = 0.1

    def transfer_feet(
        self, 
        trans_kpts, trans_scores, 
        src_kpts, src_scores, 
        ref_kpts, ref_scores, 
        global_scale=1.0, 
        log=None
    ):
        """양쪽 발 전이"""
        for side in ['left', 'right']:
            self._process_foot(
                trans_kpts, trans_scores,
                src_kpts, src_scores,
                ref_kpts, ref_scores,
                self.feet_indices[side],
                side,
                global_scale
            )

    def _process_foot(
        self,
        trans_kpts, trans_scores,
        src_kpts, src_scores,
        ref_kpts, ref_scores,
        idx: Dict,
        side: str,
        global_scale: float
    ):
        """개별 발 처리
        
        핵심 원칙:
        - Ref에 있는 키포인트만 Trans에 생성
        - Ref에 없으면 (score < MIN_SCORE) Trans에서도 생성하지 않음
        """
        ankle = idx['ankle']
        knee = idx['knee']
        big_toe = idx['big_toe']
        small_toe = idx['small_toe']
        heel = idx['heel']
        
        # Trans ankle이 없으면 처리 불가
        if trans_scores[ankle] <= 0:
            return
        
        trans_ankle_pos = trans_kpts[ankle]
        
        # Src에서 발 길이 측정
        src_lengths = self._measure_foot_lengths(src_kpts, src_scores, idx)
        
        # 다리 회전 계산
        trans_leg_angle = self._get_leg_angle(trans_kpts, trans_scores, ankle, knee)
        ref_leg_angle = self._get_leg_angle(ref_kpts, ref_scores, ankle, knee)
        leg_rotation = trans_leg_angle - ref_leg_angle
        
        # =========================================================================
        # 각 발 키포인트: Ref에 있을 때만 생성
        # =========================================================================
        
        # Big Toe
        if ref_scores[big_toe] > self.MIN_SCORE:
            # Ref에 있음 → 생성
            ref_dir = self._get_direction(ref_kpts, ref_scores, ankle, big_toe)
            if ref_dir is not None and src_lengths['big_toe'] > 0:
                final_angle = ref_dir + leg_rotation
                length = src_lengths['big_toe'] * global_scale
                trans_kpts[big_toe] = trans_ankle_pos + self._polar_to_cartesian(length, final_angle)
                trans_scores[big_toe] = ref_scores[big_toe] * 0.8
        # else: Ref에 없음 → Trans에도 생성하지 않음 (score 유지)
        
        # Small Toe
        if ref_scores[small_toe] > self.MIN_SCORE:
            ref_dir = self._get_direction(ref_kpts, ref_scores, ankle, small_toe)
            if ref_dir is not None and src_lengths['small_toe'] > 0:
                final_angle = ref_dir + leg_rotation
                length = src_lengths['small_toe'] * global_scale
                trans_kpts[small_toe] = trans_ankle_pos + self._polar_to_cartesian(length, final_angle)
                trans_scores[small_toe] = ref_scores[small_toe] * 0.8
        
        # Heel
        if ref_scores[heel] > self.MIN_SCORE:
            ref_dir = self._get_direction(ref_kpts, ref_scores, ankle, heel)
            if ref_dir is not None and src_lengths['heel'] > 0:
                final_angle = ref_dir + leg_rotation
                length = src_lengths['heel'] * global_scale
                trans_kpts[heel] = trans_ankle_pos + self._polar_to_cartesian(length, final_angle)
                trans_scores[heel] = ref_scores[heel] * 0.8

    def _measure_foot_lengths(self, kpts, scores, idx: Dict) -> Dict[str, float]:
        """Src 발에서 각 부분의 길이 측정"""
        ankle = idx['ankle']
        
        result = {'big_toe': 0, 'small_toe': 0, 'heel': 0}
        
        if scores[ankle] <= self.MIN_SCORE:
            return result
        
        ankle_pos = kpts[ankle]
        
        for part in ['big_toe', 'small_toe', 'heel']:
            part_idx = idx[part]
            if scores[part_idx] > self.MIN_SCORE:
                result[part] = np.linalg.norm(kpts[part_idx] - ankle_pos)
        
        return result

    def _get_direction(self, kpts, scores, from_idx: int, to_idx: int) -> Optional[float]:
        """두 키포인트 사이의 방향(각도) 계산"""
        if scores[from_idx] <= self.MIN_SCORE or scores[to_idx] <= self.MIN_SCORE:
            return None
        
        vec = kpts[to_idx] - kpts[from_idx]
        return np.arctan2(vec[1], vec[0])

    def _get_leg_angle(self, kpts, scores, ankle_idx: int, knee_idx: int) -> float:
        """다리 방향 (Knee → Ankle)"""
        if scores[ankle_idx] <= 0 or scores[knee_idx] <= 0:
            return np.pi / 2  # 기본값: 아래
        
        vec = kpts[ankle_idx] - kpts[knee_idx]
        return np.arctan2(vec[1], vec[0])

    def _polar_to_cartesian(self, length: float, angle: float) -> np.ndarray:
        """극좌표 → 직교좌표"""
        return np.array([length * np.cos(angle), length * np.sin(angle)])
