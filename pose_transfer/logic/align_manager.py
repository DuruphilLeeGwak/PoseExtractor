"""
Align Manager Module (Final Fix - Tuple Conversion & Center Align)

위치: pose_transfer/logic/align_manager.py
변경사항:
- [Critical] BboxInfo 객체(x1...)를 함수 진입 즉시 순수 튜플(list-like)로 변환
- [Align] X: BBox 정중앙, Y: 발바닥(Ground), Scale: 1.0 고정
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TransferLayout:
    anchor_type: str
    global_scale: float
    offset_vector: np.ndarray
    anchor_point_src: Tuple[int, int]
    anchor_point_ref: Tuple[int, int]

class AlignManager:
    def __init__(self, config):
        self.config = config

    def analyze_layout(self, src_bbox_input, ref_bbox_input, src_kpts, src_scores, ref_kpts, ref_scores) -> TransferLayout:
        # 1. [핵심] 입력받은 BBox를 무조건 튜플 (x1, y1, x2, y2)로 변환
        # 이제부터 아래 로직에서는 bbox[0], bbox[2] 접근이 100% 가능해집니다.
        src_bbox = self._to_tuple(src_bbox_input, "Src")
        ref_bbox = self._to_tuple(ref_bbox_input, "Ref")
        
        # 2. Scale 강제 고정 (사용자 요청: 1.0)
        global_scale = 1.0 
        
        # 3. Anchor Point 계산
        # X축: BBox의 정중앙 (Center)
        # Y축: 발의 가장 낮은 지점 (Ground)
        src_anchor = self._get_feet_anchor(src_kpts, src_scores, src_bbox)
        ref_anchor = self._get_feet_anchor(ref_kpts, ref_scores, ref_bbox)
        
        # 4. Offset 계산 (Src위치 - Ref위치)
        # Ref 캐릭터를 Src 캐릭터 위치로 옮기기 위한 이동량
        offset_x = src_anchor[0] - (ref_anchor[0] * global_scale)
        offset_y = src_anchor[1] - (ref_anchor[1] * global_scale)
        
        offset_vector = np.array([offset_x, offset_y])

        print(f"   🧠 [AlignManager] Strategy: FEET (Center X, Bottom Y)")
        print(f"   📏 Scale: {global_scale:.3f} (Forced 1.0)")
        print(f"   📦 Src BBox: {src_bbox}")
        print(f"   📍 Src Anchor: {src_anchor.astype(int)}")
        print(f"   📍 Ref Anchor: {ref_anchor.astype(int)}")
        print(f"   🚚 Offset: {offset_vector.astype(int)}")

        return TransferLayout(
            anchor_type="FEET",
            global_scale=global_scale,
            offset_vector=offset_vector,
            anchor_point_src=tuple(map(int, src_anchor)),
            anchor_point_ref=tuple(map(int, ref_anchor))
        )

    def _to_tuple(self, bbox, label) -> Tuple[float, float, float, float]:
        """
        BboxInfo 객체든, 리스트든 상관없이 (x1, y1, x2, y2) 튜플로 반환
        """
        # 1. BboxInfo 객체 (x1, x2 속성 사용) - 사용자 로그 기반
        if hasattr(bbox, 'x1') and hasattr(bbox, 'x2'):
            return (float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2))
        
        # 2. to_tuple() 메서드가 있는 경우
        if hasattr(bbox, 'to_tuple') and callable(bbox.to_tuple):
            val = bbox.to_tuple()
            return (float(val[0]), float(val[1]), float(val[2]), float(val[3]))
            
        # 3. 리스트나 튜플인 경우 (초기 방식 호환)
        if isinstance(bbox, (list, tuple, np.ndarray)):
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        # 변환 실패 시 로그 출력
        print(f"❌ [AlignManager] {label} BBox 변환 실패! Type: {type(bbox)}, Dir: {dir(bbox)}")
        raise TypeError(f"{label} BBox 형식을 변환할 수 없습니다.")

    def _get_feet_anchor(self, kpts, scores, bbox_tuple) -> np.ndarray:
        """
        X: BBox 정중앙 ((x1+x2)/2)
        Y: 발 관련 키포인트 중 최대값 (Ground)
        """
        x1, y1, x2, y2 = bbox_tuple

        # 1. X축: BBox Center (좌우 쏠림 방지)
        center_x = (x1 + x2) / 2.0

        # 2. Y축: 발바닥 (Ground) 찾기
        # DWPose Index: 15,16(Ankle), 17~22(Toes/Heels)
        foot_indices = [15, 16, 17, 18, 19, 20, 21, 22]
        valid_ys = []
        
        for idx in foot_indices:
            if idx < len(kpts) and scores[idx] > 0.1:
                valid_ys.append(kpts[idx][1])
        
        if valid_ys:
            ground_y = max(valid_ys) # 키포인트 중 가장 아래쪽
        else:
            ground_y = y2 # 키포인트 없으면 BBox 바닥 사용

        return np.array([center_x, ground_y])

    def load_image_safe(self, path):
        pass