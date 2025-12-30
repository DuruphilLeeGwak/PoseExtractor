"""
Simplified Align Manager (Refactored v2.0)

변경사항:
- Body Type 판별 로직 제거 (GhostFilter가 처리)
- Case Enum 제거 (Boolean 기반으로 단순화)
- 핵심 기능만 유지: 정렬 방식 결정 + 좌표 정렬
"""
import numpy as np
from typing import Tuple, Optional, Any

class AlignManager:
    """
    단순화된 정렬 관리자
    
    역할:
    1. 발 정렬 가능 여부 판단 (GhostFilter 처리 후 점수 기반)
    2. 좌표 정렬 수행 (발 또는 얼굴 기준)
    """
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config: Pipeline 설정 객체
        """
        self.config = config
        # 발목 인덱스 (left_ankle, right_ankle)
        self.ankle_indices = [15, 16]
    
    def should_align_by_feet(
        self, 
        src_scores: np.ndarray, 
        ref_scores: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[bool, bool]:
        """
        발 정렬 사용 가능 여부 판단
        
        Args:
            src_scores: Source 키포인트 신뢰도 점수
            ref_scores: Reference 키포인트 신뢰도 점수
            threshold: 유효 판단 임계값 (기본: 0.1)
        
        Returns:
            Tuple[bool, bool]:
                - should_transfer_lower: 하반신 전이 여부 (ref에 발 있으면 True)
                - align_by_feet: 발 정렬 사용 여부 (src, ref 둘 다 발 있으면 True)
        
        Note:
            GhostFilter가 이미 유효하지 않은 키포인트를 제거했으므로
            단순히 점수만 확인하면 됨
        """
        print("\n🔍 [AlignManager] Checking feet availability...")
        
        # Source 발목 체크
        src_left_ankle = src_scores[15] > threshold if 15 < len(src_scores) else False
        src_right_ankle = src_scores[16] > threshold if 16 < len(src_scores) else False
        src_has_feet = src_left_ankle and src_right_ankle
        
        # Reference 발목 체크
        ref_left_ankle = ref_scores[15] > threshold if 15 < len(ref_scores) else False
        ref_right_ankle = ref_scores[16] > threshold if 16 < len(ref_scores) else False
        ref_has_feet = ref_left_ankle and ref_right_ankle
        
        print(f"   Source feet: L={src_left_ankle}, R={src_right_ankle} → {src_has_feet}")
        print(f"   Reference feet: L={ref_left_ankle}, R={ref_right_ankle} → {ref_has_feet}")
        
        # 하반신 전이: Reference에 발이 있으면 가능
        should_transfer_lower = ref_has_feet
        
        # 발 정렬: 둘 다 발이 있어야 가능
        align_by_feet = src_has_feet and ref_has_feet
        
        print(f"   → should_transfer_lower: {should_transfer_lower}")
        print(f"   → align_by_feet: {align_by_feet}")
        
        return should_transfer_lower, align_by_feet
    
    def align_coordinates(
        self, 
        kpts: np.ndarray, 
        scores: np.ndarray, 
        align_by_feet: bool,
        src_person_bbox: Any, 
        src_face_bbox: Any, 
        face_bbox_func: callable
    ) -> np.ndarray:
        """
        좌표 정렬 수행
        
        Args:
            kpts: 전이된 키포인트 좌표 (133, 2)
            scores: 키포인트 신뢰도 점수 (133,)
            align_by_feet: True면 발 정렬, False면 얼굴 정렬
            src_person_bbox: Source의 Person bounding box
            src_face_bbox: Source의 Face bounding box
            face_bbox_func: 얼굴 bbox 계산 함수
        
        Returns:
            np.ndarray: 정렬된 키포인트 좌표 (133, 2)
        
        Note:
            - 발 정렬: 발 바닥을 Source 이미지 바닥에 맞춤
            - 얼굴 정렬: 얼굴 중심을 Source 얼굴 중심에 맞춤
        """
        print("\n" + "="*60)
        print(f"🔍 [AlignManager] align_coordinates(align_by_feet={align_by_feet})")
        print("="*60)
        
        aligned_kpts = kpts.copy()
        
        if align_by_feet:
            # ========================================
            # 발 정렬: 발 바닥을 Source 이미지 바닥에 맞춤
            # ========================================
            print("\n🦶 Feet-based alignment")
            
            # Source 이미지의 바닥 (person bbox의 y2)
            src_bottom = src_person_bbox.bbox[3]
            print(f"   src_person_bbox: {src_person_bbox.bbox}")
            print(f"   src_bottom (y2): {src_bottom}")
            
            # 전이된 키포인트의 발 관련 포인트 중 가장 아래 찾기
            feet_indices = [15, 16, 17, 18, 19, 20, 21, 22]  # ankles + toes + heels
            valid_y = []
            
            print(f"\n   Checking feet keypoints:")
            for i in feet_indices:
                if i < len(scores) and scores[i] > 0.1:
                    valid_y.append(kpts[i][1])
                    print(f"      idx={i}: score={scores[i]:.3f}, y={kpts[i][1]:.1f} ✅")
            
            if valid_y:
                trans_bottom = max(valid_y)
                print(f"   trans_bottom (max y): {trans_bottom:.1f}")
                
                # 수직 이동량 계산
                shift_y = src_bottom - trans_bottom
                aligned_kpts[:, 1] += shift_y
                
                print(f"   ✅ shift_y = {src_bottom:.1f} - {trans_bottom:.1f} = {shift_y:.1f}")
            else:
                print(f"   ❌ No valid feet keypoints found, NO SHIFT")
        
        else:
            # ========================================
            # 얼굴 정렬: 얼굴 중심을 Source 얼굴 중심에 맞춤
            # ========================================
            print(f"\n👤 Face-based alignment")
            
            # 1. Source 이미지의 얼굴 중심
            src_cx, src_cy = src_face_bbox.center
            
            # 2. 전이된 키포인트의 얼굴 중심 계산
            trans_face_info = face_bbox_func(kpts, scores)
            trans_cx, trans_cy = trans_face_info.center
            
            print(f"   Src Face Center: ({src_cx:.1f}, {src_cy:.1f})")
            print(f"   Trans Face Center: ({trans_cx:.1f}, {trans_cy:.1f})")
            
            # 3. 이동량 계산
            shift_x = src_cx - trans_cx
            shift_y = src_cy - trans_cy
            
            # 4. 전체 키포인트 이동
            aligned_kpts[:, 0] += shift_x
            aligned_kpts[:, 1] += shift_y
            
            print(f"   ✅ Shift Applied: x={shift_x:.1f}, y={shift_y:.1f}")
        
        print("="*60)
        return aligned_kpts