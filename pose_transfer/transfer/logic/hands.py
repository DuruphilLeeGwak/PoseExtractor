"""
Hand Transfer Logic Module (Refactored v8.2 - Scale Fix)

위치: pose_transfer/transfer/logic/hands.py
변경사항:
- [Critical] 손 크기(Scale)를 Ref가 아닌 'Source * GlobalScale'로 강제 적용 (왕손 해결)
"""
import numpy as np
from typing import Dict, Any
from ..config import TransferConfig

class HandTransfer:
    def __init__(self, config: TransferConfig):
        self.config = config
        
        # 손가락 구조 (Root -> Tip)
        self.fingers = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]

    def transfer_hands(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, hand_scale_ratio=1.0, log=None):
        """
        손목(Wrist)을 기준으로 손가락 전이.
        hand_scale_ratio: Global Body Scale (이 값을 반드시 따라야 함)
        """
        # COCO Body Indices: L_Wrist=9, R_Wrist=10
        # Hands usually start from index 23 onwards in OpenPose/RTMPose extended format?
        # Or standard 133 format: Left Hand 91~111, Right Hand 112~132
        
        # 여기서는 133 Keypoint Format (WholeBody) 기준
        # Left Hand Root: 91 (Wrist) -> coincides with Body 9
        # Right Hand Root: 112 (Wrist) -> coincides with Body 10
        
        # Logic: 
        # 1. Wrist 위치는 BodyTransfer에서 이미 결정됨 (trans_kpts[9], [10])
        # 2. Source의 손 크기(Wrist -> Middle Finger Tip) 계산
        # 3. Ref의 손 자세(각도) 계산
        # 4. Trans = Wrist + (Ref_Dir * Src_Len * Scale)
        
        # Left Hand (91~111), Right Hand (112~132)
        self._process_hand(trans_kpts, trans_scores, src_kpts, ref_kpts, ref_scores, 
                           9, range(91, 112), hand_scale_ratio, "Left")
        
        self._process_hand(trans_kpts, trans_scores, src_kpts, ref_kpts, ref_scores, 
                           10, range(112, 133), hand_scale_ratio, "Right")

    def _process_hand(self, trans_kpts, trans_scores, src_kpts, ref_kpts, ref_scores, 
                      wrist_idx, hand_indices, scale, side):
        
        # 손목 위치 확인 (이미 전이되어 있어야 함)
        if trans_scores[wrist_idx] == 0: return # 손목이 없으면 손도 못 그림
        
        wrist_pos = trans_kpts[wrist_idx]
        
        # Source Hand Size Check (Wrist -> Middle MCP or Tip)
        # Middle MCP is usually index 9 in local hand (global: start + 9)
        mid_mcp_idx = hand_indices[9] 
        
        src_hand_len = 0
        # Source 손 크기 측정 (Wrist -> Middle Finger Base)
        # 유효하지 않으면 대략적인 값(Wrist-Elbow의 25% 등) 추정해야 하지만
        # 여기서는 단순히 전신 스케일만 믿고 기본값 사용
        src_vec = src_kpts[mid_mcp_idx] - src_kpts[wrist_idx]
        src_hand_len = np.linalg.norm(src_vec)
        
        if src_hand_len < 5: # Source 손이 안 보이거나 너무 작음
            src_hand_len = 50.0 # Default fallback
            
        # [Critical] 손 크기 결정: Source길이 * GlobalScale
        target_hand_len = src_hand_len * scale
        
        # Ref Hand Size (for normalizing Ref vectors)
        ref_vec = ref_kpts[mid_mcp_idx] - ref_kpts[wrist_idx]
        ref_len = np.linalg.norm(ref_vec)
        if ref_len < 1: ref_len = 1 # Avoid div by zero
        
        scale_factor = target_hand_len / ref_len
        
        # Transfer All Finger Points
        # Wrist(0) is fixed. Move others relative to Wrist.
        # Global Index Mapping: hand_indices[0] is Wrist duplicate usually
        
        for i in hand_indices:
            if ref_scores[i] > 0.1:
                # Relative vector from Ref Wrist
                vec = ref_kpts[i] - ref_kpts[wrist_idx]
                
                # Apply Scale
                new_pos = wrist_pos + (vec * scale_factor)
                
                trans_kpts[i] = new_pos
                trans_scores[i] = ref_scores[i]