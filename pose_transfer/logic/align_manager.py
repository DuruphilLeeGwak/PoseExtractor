"""
Align Manager Module (Refactored v2.0 - Smart Anchor & Offset)

역할:
- Source와 Reference의 포즈/BBox 상태를 분석하여 전이 전략(Layout) 수립
- Global Scale(크기 비율)과 Offset Vector(이동 좌표)를 계산하여 Engine에 전달
- '발 끝 맞추기(Grounding)' vs '중심 맞추기(Center)' 전략 자동 분기
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from .bbox_manager import BboxInfo

@dataclass
class TransferLayout:
    """전이 전략 및 배치 정보"""
    global_scale: float          # 적용할 확대/축소 배율
    offset_vector: np.ndarray    # 최종 이동 벡터 (dx, dy)
    anchor_type: str             # 'FEET', 'HIP', 'FACE'
    anchor_point_src: Tuple[int, int]
    anchor_point_ref: Tuple[int, int]

class AlignManager:
    def __init__(self, config):
        self.config = config

    def analyze_layout(
        self,
        src_bbox_info: BboxInfo,
        ref_bbox_info: BboxInfo,
        src_kpts: np.ndarray,
        src_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        src_depth_map=None,
        ref_depth_map=None,
        src_depth_vals=None
    ) -> TransferLayout:
        """
        [Main] 최적의 배치 전략(Layout) 계산
        """
        # 1. 정렬 기준(Anchor Type) 결정
        align_type = self._decide_alignment_type(src_bbox_info, ref_bbox_info)
        
        # 2. 스케일(Scale) 계산
        # 정렬 기준에 따라 스케일 계산 방식도 달라짐 (발 기준이면 키 비율, 힙 기준이면 몸통 비율 등)
        scale = self._calculate_scale(
            align_type, src_kpts, src_scores, ref_kpts, ref_scores, 
            src_bbox_info, ref_bbox_info
        )
        
        # 3. 앵커 포인트(Anchor Point) 및 오프셋(Offset) 계산
        offset, src_anchor, ref_anchor = self._calculate_offset(
            align_type, scale, src_bbox_info, ref_bbox_info
        )
        
        print(f"   🧠 [AlignManager] Strategy: {align_type}")
        print(f"      Scale: {scale:.3f}")
        print(f"      Offset: {offset.astype(int)}")
        
        return TransferLayout(
            global_scale=scale,
            offset_vector=offset,
            anchor_type=align_type,
            anchor_point_src=src_anchor,
            anchor_point_ref=ref_anchor
        )

    def _decide_alignment_type(self, src: BboxInfo, ref: BboxInfo) -> str:
        """
        Source와 Ref의 상태를 보고 정렬 기준 선택
        우선순위: FEET(전신) > HIP(상반신) > FACE(얼굴)
        """
        # 1. 둘 다 하체가 존재하면 -> 발 끝 정렬 (바닥 고정)
        if src.has_lower_body and ref.has_lower_body:
            return 'FEET'
        
        # 2. 둘 다 얼굴이 존재하면 (상반신 샷 등) -> 힙/몸통 중심 정렬
        # (얼굴 정렬보다 힙 정렬이 전체적인 포즈 안정성이 높음)
        if src.has_face and ref.has_face:
            return 'HIP'
            
        # 3. 그 외의 경우 (얼굴 클로즈업 등)
        return 'FACE'

    def _calculate_scale(
        self, 
        align_type: str,
        src_kpts, src_scores, 
        ref_kpts, ref_scores,
        src_bbox: BboxInfo, ref_bbox: BboxInfo
    ) -> float:
        """
        정렬 타입에 맞는 최적의 스케일 계산
        """
        scale = 1.0
        
        # Case A: FEET (전신) -> 키(Height) 비율 or 몸통 길이 비율
        # BBox 높이 비율을 사용하는 것이 가장 안정적 (노이즈에 강함)
        if align_type == 'FEET':
            if src_bbox.height > 0 and ref_bbox.height > 0:
                scale = ref_bbox.height / src_bbox.height
                
        # Case B: HIP (상반신) -> 몸통(Torso) 길이 비율
        elif align_type == 'HIP':
            # 키포인트 기반 척추 길이 계산 시도
            src_torso = self._calc_torso_len(src_kpts, src_scores)
            ref_torso = self._calc_torso_len(ref_kpts, ref_scores)
            
            if src_torso > 0 and ref_torso > 0:
                scale = ref_torso / src_torso
            else:
                # 척추 길이 모르면 BBox 높이 비율로 대체
                scale = ref_bbox.height / src_bbox.height if src_bbox.height > 0 else 1.0
                
        # Case C: FACE -> 얼굴 크기 비율 (BBox or 귀/눈 거리)
        else:
            # BBoxManager가 제공하는 Face BBox 사용
            # (Face BBox는 이미 BBoxManager에서 계산됨)
            # 여기서는 BBoxInfo 자체의 크기보다는, BBoxManager가 내부적으로 가지고 있는 Face Box가 필요함.
            # 하지만 BBoxInfo.has_face가 True라면 src_bbox 안에 얼굴 정보가 포함되어 있거나
            # 별도의 Face BBox가 넘어와야 함.
            # pipeline.py에서 src_face_bbox를 별도로 넘겨주지 않고 
            # src_bbox_info(Person)만 넘겨주고 있다면, 정밀도는 떨어질 수 있음.
            # *현재 구조상 Person BBox 비율 사용*
            scale = ref_bbox.width / src_bbox.width if src_bbox.width > 0 else 1.0

        # 안전장치 (너무 과도한 스케일링 방지)
        return float(np.clip(scale, 0.3, 3.0))

    def _calculate_offset(
        self, 
        align_type: str, 
        scale: float, 
        src: BboxInfo, 
        ref: BboxInfo
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Offset 계산: Ref_Anchor - (Src_Anchor * Scale)
        """
        src_anchor = (0, 0)
        ref_anchor = (0, 0)
        
        if align_type == 'FEET':
            # 발 중심점 (BBoxInfo에 미리 계산되어 있음)
            src_anchor = src.feet_center
            ref_anchor = ref.feet_center
            
        elif align_type == 'HIP':
            # 힙 중심점 (BBox의 중심 또는 하단 1/3 지점 등)
            # 여기서는 BBox의 중심(Center) 사용
            src_anchor = src.center
            ref_anchor = ref.center
            
        elif align_type == 'FACE':
            # 얼굴 중심점
            src_anchor = src.face_center
            ref_anchor = ref.face_center
            
        # 벡터 연산
        v_src = np.array(src_anchor, dtype=np.float32)
        v_ref = np.array(ref_anchor, dtype=np.float32)
        
        # 공식: Target = Source * Scale + Offset
        # 따라서 Offset = Target - (Source * Scale)
        offset = v_ref - (v_src * scale)
        
        return offset, src_anchor, ref_anchor

    def _calc_torso_len(self, kpts, scores):
        """[Helper] 척추 길이 계산 (Neck to Hip Center)"""
        # COCO: 5,6(Sh), 11,12(Hip)
        if (scores[5]>0.1 and scores[6]>0.1 and scores[11]>0.1 and scores[12]>0.1):
            neck = (kpts[5] + kpts[6]) / 2.0
            hip = (kpts[11] + kpts[12]) / 2.0
            return np.linalg.norm(hip - neck)
        return 0.0