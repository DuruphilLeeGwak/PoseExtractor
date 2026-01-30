"""
Hand Transfer Logic Module (Refactored v2.1)

역할:
- 손(Left: 91~111, Right: 112~132) 키포인트 전이 담당
- [원칙] Reference의 손 포즈를 100% 따릅니다.
- [예외] Reference에 손이 '아예 없을 때만' Source 손을 사용하여 손목 절단을 방지합니다.
"""
import numpy as np
from typing import Dict, Any, Optional

class HandTransfer:
    def __init__(self, config=None):
        self.config = config

    def transfer_hands(
        self,
        trans_kpts: np.ndarray,
        trans_scores: np.ndarray,
        src_kpts: np.ndarray,
        src_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        hand_scale_ratio: float,
        log: Dict[str, Any]
    ):
        """
        손 전이 실행
        """
        LW, RW = 9, 10
        
        # (손목 인덱스, 시작, 끝, 이름)
        hands_def = [
            (LW, 91, 112, "Left"), 
            (RW, 112, 133, "Right")
        ]
        
        for wrist_idx, start, end, side in hands_def:
            # 1. Trans에 손목조차 없으면 손을 붙일 곳이 없으므로 포기
            if trans_scores[wrist_idx] < 0.1:
                continue
                
            trans_wrist = trans_kpts[wrist_idx]
            
            # 유효성 검사 (Threshold 0.2)
            # Ref에 손이 있는가?
            ref_valid_cnt = sum(1 for i in range(start, end) if ref_scores[i] > 0.2)
            
            status = "skipped"
            
            # =========================================================
            # [Main Strategy] Reference 손 포즈 복사 (절대 원칙)
            # =========================================================
            if ref_valid_cnt > 3: # 손가락이 3개 이상만 보여도 Ref를 따름
                ref_wrist = ref_kpts[wrist_idx]
                scale = hand_scale_ratio
                
                for idx in range(start, end):
                    if ref_scores[idx] > 0.1: # 낮은 점수라도 최대한 가져옴
                        # Ref: 손목 -> 손가락 벡터
                        rel_vec = ref_kpts[idx] - ref_wrist
                        
                        # Trans: 손목 + (벡터 * 스케일)
                        trans_kpts[idx] = trans_wrist + (rel_vec * scale)
                        trans_scores[idx] = ref_scores[idx]
                
                status = f"ref_pose (cnt={ref_valid_cnt})"

            # =========================================================
            # [Emergency Fallback] Ref 손이 아예 없을 때 (결측치 보정)
            # =========================================================
            # Ref가 없고 Src에는 손이 있다면, 손목이 잘리는 것을 막기 위해 
            # '현재 손 모양'을 유지한 채 위치만 옮김.
            else:
                src_valid_cnt = sum(1 for i in range(start, end) if src_scores[i] > 0.2)
                
                if src_valid_cnt > 5:
                    src_wrist = src_kpts[wrist_idx]
                    scale = hand_scale_ratio # 몸집 비율에 맞춰 손 크기 조절
                    
                    for idx in range(start, end):
                        if src_scores[idx] > 0.2:
                            rel_vec = src_kpts[idx] - src_wrist
                            trans_kpts[idx] = trans_wrist + (rel_vec * scale)
                            trans_scores[idx] = src_scores[idx] * 0.5 # 신뢰도 낮춤 (원본이 아니므로)
                    
                    status = f"src_repair (Ref missing)"
                    # 로그에 경고 추가
                    if log is not None:
                        log.setdefault('warnings', []).append(f"{side} hand: Ref missing, used Src fallback")

            # 로그 기록
            if log is not None:
                log.setdefault('hand_debug', []).append({
                    'side': side,
                    'status': status
                })