"""
Align Manager Module (Refactored v5.3 - Math Stabilized)

위치: pose_transfer/logic/align_manager.py
역할: Src와 Ref의 크기(Scale) 및 위치(Offset) 차이 계산
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

    def analyze_layout(self, src: BboxInfo, ref: BboxInfo, *args) -> TransferLayout:
        # 1. 전략 결정
        align_type = 'FACE'
        if src.has_lower_body and ref.has_lower_body: align_type = 'FEET'
        elif src.has_face and ref.has_face: align_type = 'HIP'
            
        # 2. 스케일 계산 (비율 제한 0.5 ~ 2.0)
        scale = 1.0
        if align_type == 'FEET' and src.height > 10:
            scale = ref.height / src.height
        elif align_type == 'HIP' and src.height > 10: # Torso 대신 Height 사용 (더 안정적)
            scale = ref.height / src.height
        elif src.width > 10:
            scale = ref.width / src.width
            
        scale = float(np.clip(scale, 0.5, 2.0))
        
        # 3. 오프셋 계산 (Target - Source*Scale)
        src_anchor = self._get_anchor(src, align_type)
        ref_anchor = self._get_anchor(ref, align_type)
        
        v_src = np.array(src_anchor, dtype=np.float32)
        v_ref = np.array(ref_anchor, dtype=np.float32)
        
        offset = v_ref - (v_src * scale)
        
        print(f"   🧠 [AlignManager] {align_type} | Scale: {scale:.3f} | Offset: {offset.astype(int)}")
        
        return TransferLayout(scale, offset, align_type, src_anchor, ref_anchor)

    def _get_anchor(self, bbox: BboxInfo, align_type: str) -> Tuple[int, int]:
        """안전한 앵커 포인트 반환"""
        pt = (0, 0)
        if align_type == 'FEET':
            pt = bbox.feet_center
        elif align_type == 'FACE':
            pt = bbox.face_center
        
        # 유효하지 않거나(0,0), HIP 모드면 중심점 사용
        if pt[0] <= 0 or pt[1] <= 0 or align_type == 'HIP':
            pt = bbox.center
            
        return pt