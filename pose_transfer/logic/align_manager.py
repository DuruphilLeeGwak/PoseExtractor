"""
Align Manager Module (Refactored v6.0 - Feet Bottom Anchor Logic)

위치: pose_transfer/logic/align_manager.py
변경사항:
- [Critical] 소스 박스의 하단(y2)을 바닥(Ground) 기준으로 사용하여 발 정렬 완벽 복구
- [Fix] Engine (0,0)=Neck 특성에 맞춰 다리 길이를 역산(Back-calc)
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from .bbox_manager import BboxInfo

@dataclass
class TransferLayout:
    global_scale: float
    offset_vector: np.ndarray
    anchor_type: str
    anchor_point_src: Tuple[int, int]
    anchor_point_ref: Tuple[int, int]

class AlignManager:
    def __init__(self, config):
        self.config = config

    def analyze_layout(self, src: BboxInfo, ref: BboxInfo, src_kpts, src_scores, ref_kpts, ref_scores) -> TransferLayout:
        # 1. 정렬 모드 결정
        align_type = 'FACE'
        if src.has_lower_body and ref.has_lower_body:
            align_type = 'FEET'
        elif src.has_face and ref.has_face:
            align_type = 'HIP'

        # 2. 스케일 계산 (키 비율 기준)
        scale = 1.0
        if src.height > 50 and ref.height > 50:
            scale = src.height / ref.height
        elif src.width > 50 and ref.width > 50:
            scale = src.width / ref.width
        scale = float(np.clip(scale, 0.3, 3.0))

        # 3. [핵심] 오프셋 계산 (Feet Bottom 기준)
        # 소스 이미지에서 발이 위치해야 할 절대 좌표 (바닥)
        target_ground_y = src.y2 
        target_center_x = src.center[0]

        # Engine은 (0,0)을 Neck으로 잡고 그림.
        # 따라서 Ref 캐릭터의 "Neck에서 발바닥(Box Bottom)까지의 벡터"를 알아야 함.
        ref_neck = self._get_neck_point(ref_kpts, ref_scores, ref.center)
        
        # 참조 캐릭터의 다리 벡터 (Neck -> Bottom)
        ref_leg_vector = np.array([ref.center[0], ref.y2]) - np.array(ref_neck)
        
        if align_type == 'FEET':
            # 목표: 생성된 캐릭터의 발바닥이 src.y2에 닿아야 함
            # 식: Target_Bottom = Offset + (Ref_Leg_Vector * Scale)
            # -> Offset = Target_Bottom - (Ref_Leg_Vector * Scale)
            
            target_pos = np.array([target_center_x, target_ground_y])
            offset = target_pos - (ref_leg_vector * scale)
            
        elif align_type == 'HIP':
            # 상반신만 있으면 그냥 중심점 맞춤
            target_pos = np.array(src.center)
            ref_vec = np.array(ref.center) - np.array(ref_neck)
            offset = target_pos - (ref_vec * scale)
        else:
            # 얼굴 기준
            target_pos = np.array(src.face_center)
            ref_vec = np.array(ref.face_center) - np.array(ref_neck)
            offset = target_pos - (ref_vec * scale)

        print(f"   🧠 [AlignManager] Strategy: {align_type}")
        print(f"   📏 Scale: {scale:.3f}")
        print(f"   📍 Target Ground Y: {target_ground_y}")
        print(f"   🚚 Offset: {offset.astype(int)}")
        
        return TransferLayout(scale, offset, align_type, (int(target_center_x), int(target_ground_y)), tuple(ref.center))

    def _get_neck_point(self, kpts, scores, fallback) -> Tuple[int, int]:
        # Ref Neck (Shoulder Center)
        if kpts is not None and len(kpts) > 6 and scores[5] > 0.1 and scores[6] > 0.1:
            neck = (kpts[5] + kpts[6]) / 2.0
            return (int(neck[0]), int(neck[1]))
        return fallback