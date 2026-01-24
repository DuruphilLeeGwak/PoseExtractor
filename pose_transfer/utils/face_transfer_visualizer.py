"""
Face Transfer 디버그 시각화 생성 모듈
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def create_face_transfer_visualization(
    debug_info: Dict,
    src_kpts: np.ndarray,
    src_scores: np.ndarray,
    ref_kpts: np.ndarray,
    ref_scores: np.ndarray,
    trans_kpts: np.ndarray,
    trans_scores: np.ndarray,
    output_path: Path
) -> bool:
    """
    Face Transfer 디버그 시각화 이미지 생성
    
    Args:
        debug_info: Face transfer 디버그 정보
        src_kpts, src_scores: Source 키포인트
        ref_kpts, ref_scores: Reference 키포인트
        trans_kpts, trans_scores: Transfer 결과 키포인트
        output_path: 출력 경로
    
    Returns:
        성공 여부
    """
    if not debug_info:
        return False
    
    # 캔버스 크기 설정
    canvas_width = 1800
    canvas_height = 900
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 3열 레이아웃: Src | Ref | Trans
    col_width = canvas_width // 3
    
    # 색상 정의
    COLOR_NECK = (200, 100, 50)  # 갈색 (Neck)
    COLOR_NOSE = (0, 0, 255)     # 빨강 (Nose)
    COLOR_EYE = (0, 165, 255)    # 주황 (Eyes)
    COLOR_LINE = (100, 100, 100) # 회색 (연결선)
    COLOR_TEXT = (0, 0, 0)       # 검정 (텍스트)
    
    NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2
    LS, RS = 5, 6
    
    # ===== Src 그리기 =====
    if src_scores[LS] > 0.1 and src_scores[RS] > 0.1:
        src_neck = (src_kpts[LS] + src_kpts[RS]) / 2.0
        src_scale = 400 / max(abs(src_kpts[NOSE][1] - src_neck[1]), 1) * 0.8
        src_center = np.array([col_width // 2, canvas_height // 2])
        
        # Neck 위치 계산
        neck_vis = src_center
        nose_offset = (src_kpts[NOSE] - src_neck) * src_scale
        nose_vis = neck_vis + nose_offset
        
        # Neck -> Nose
        cv2.line(canvas, tuple(neck_vis.astype(int)), tuple(nose_vis.astype(int)), 
                 COLOR_LINE, 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(neck_vis.astype(int)), 8, COLOR_NECK, -1)
        cv2.circle(canvas, tuple(nose_vis.astype(int)), 8, COLOR_NOSE, -1)
        
        # Eyes
        if src_scores[LEFT_EYE] > 0.1:
            leye_offset = (src_kpts[LEFT_EYE] - src_neck) * src_scale
            leye_vis = neck_vis + leye_offset
            cv2.line(canvas, tuple(nose_vis.astype(int)), tuple(leye_vis.astype(int)), 
                     COLOR_LINE, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(leye_vis.astype(int)), 8, COLOR_EYE, -1)
        
        if src_scores[RIGHT_EYE] > 0.1:
            reye_offset = (src_kpts[RIGHT_EYE] - src_neck) * src_scale
            reye_vis = neck_vis + reye_offset
            cv2.line(canvas, tuple(nose_vis.astype(int)), tuple(reye_vis.astype(int)), 
                     COLOR_LINE, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(reye_vis.astype(int)), 8, COLOR_EYE, -1)
        
        # 텍스트 정보
        y_dist = debug_info.get('src_neck_nose_y', 0)
        cv2.putText(canvas, "Source", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)
        cv2.putText(canvas, f"Neck->Nose Y: {y_dist:.1f}px", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    
    # ===== Ref 그리기 =====
    if ref_scores[LS] > 0.1 and ref_scores[RS] > 0.1 and ref_scores[NOSE] > 0.1:
        ref_neck = (ref_kpts[LS] + ref_kpts[RS]) / 2.0
        ref_scale = 400 / max(abs(ref_kpts[NOSE][1] - ref_neck[1]), 1) * 0.8
        ref_center = np.array([col_width + col_width // 2, canvas_height // 2])
        
        neck_vis = ref_center
        nose_offset = (ref_kpts[NOSE] - ref_neck) * ref_scale
        nose_vis = neck_vis + nose_offset
        
        # Neck -> Nose
        cv2.line(canvas, tuple(neck_vis.astype(int)), tuple(nose_vis.astype(int)), 
                 COLOR_LINE, 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(neck_vis.astype(int)), 8, COLOR_NECK, -1)
        cv2.circle(canvas, tuple(nose_vis.astype(int)), 8, COLOR_NOSE, -1)
        
        # Eyes
        if ref_scores[LEFT_EYE] > 0.1:
            leye_offset = (ref_kpts[LEFT_EYE] - ref_neck) * ref_scale
            leye_vis = neck_vis + leye_offset
            cv2.line(canvas, tuple(nose_vis.astype(int)), tuple(leye_vis.astype(int)), 
                     COLOR_LINE, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(leye_vis.astype(int)), 8, COLOR_EYE, -1)
        
        if ref_scores[RIGHT_EYE] > 0.1:
            reye_offset = (ref_kpts[RIGHT_EYE] - ref_neck) * ref_scale
            reye_vis = neck_vis + reye_offset
            cv2.line(canvas, tuple(nose_vis.astype(int)), tuple(reye_vis.astype(int)), 
                     COLOR_LINE, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(reye_vis.astype(int)), 8, COLOR_EYE, -1)
        
        # 텍스트 정보
        y_dist = debug_info.get('ref_neck_nose_y', 0)
        x_ratio = debug_info.get('ref_neck_nose_x_ratio', 0)
        cv2.putText(canvas, "Reference", (col_width + 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)
        cv2.putText(canvas, f"Neck->Nose Y: {y_dist:.1f}px", (col_width + 20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        cv2.putText(canvas, f"X Ratio: {x_ratio:.3f}", (col_width + 20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    
    # ===== Trans 그리기 =====
    if trans_scores[LS] > 0.1 and trans_scores[RS] > 0.1 and trans_scores[NOSE] > 0.1:
        trans_neck = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
        trans_scale = 400 / max(abs(trans_kpts[NOSE][1] - trans_neck[1]), 1) * 0.8
        trans_center = np.array([2 * col_width + col_width // 2, canvas_height // 2])
        
        neck_vis = trans_center
        nose_offset = (trans_kpts[NOSE] - trans_neck) * trans_scale
        nose_vis = neck_vis + nose_offset
        
        # Neck -> Nose
        cv2.line(canvas, tuple(neck_vis.astype(int)), tuple(nose_vis.astype(int)), 
                 COLOR_LINE, 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(neck_vis.astype(int)), 8, COLOR_NECK, -1)
        cv2.circle(canvas, tuple(nose_vis.astype(int)), 8, COLOR_NOSE, -1)
        
        # Eyes
        if trans_scores[LEFT_EYE] > 0.1:
            leye_offset = (trans_kpts[LEFT_EYE] - trans_neck) * trans_scale
            leye_vis = neck_vis + leye_offset
            cv2.line(canvas, tuple(nose_vis.astype(int)), tuple(leye_vis.astype(int)), 
                     COLOR_LINE, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(leye_vis.astype(int)), 8, COLOR_EYE, -1)
        
        if trans_scores[RIGHT_EYE] > 0.1:
            reye_offset = (trans_kpts[RIGHT_EYE] - trans_neck) * trans_scale
            reye_vis = neck_vis + reye_offset
            cv2.line(canvas, tuple(nose_vis.astype(int)), tuple(reye_vis.astype(int)), 
                     COLOR_LINE, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(reye_vis.astype(int)), 8, COLOR_EYE, -1)
        
        # 텍스트 정보
        y_dist = debug_info.get('trans_neck_nose_y', 0)
        x_dist = debug_info.get('trans_neck_nose_x', 0)
        leye_y = debug_info.get('left_eye_y', 0)
        leye_x = debug_info.get('left_eye_x', 0)
        reye_y = debug_info.get('right_eye_y', 0)
        reye_x = debug_info.get('right_eye_x', 0)
        
        cv2.putText(canvas, "Transfer", (2 * col_width + 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)
        cv2.putText(canvas, f"Neck->Nose Y: {y_dist:.1f}px", (2 * col_width + 20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        cv2.putText(canvas, f"Neck->Nose X: {x_dist:.1f}px", (2 * col_width + 20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        cv2.putText(canvas, f"L Eye: Y={leye_y:.1f}, X={leye_x:.1f}", 
                    (2 * col_width + 20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        cv2.putText(canvas, f"R Eye: Y={reye_y:.1f}, X={reye_x:.1f}", 
                    (2 * col_width + 20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    
    # 구분선
    cv2.line(canvas, (col_width, 0), (col_width, canvas_height), (200, 200, 200), 2)
    cv2.line(canvas, (2 * col_width, 0), (2 * col_width, canvas_height), (200, 200, 200), 2)
    
    # 범례
    legend_y = canvas_height - 150
    cv2.circle(canvas, (30, legend_y), 8, COLOR_NECK, -1)
    cv2.putText(canvas, "Neck (Shoulder Center)", (50, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    cv2.circle(canvas, (30, legend_y + 35), 8, COLOR_NOSE, -1)
    cv2.putText(canvas, "Nose", (50, legend_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    cv2.circle(canvas, (30, legend_y + 70), 8, COLOR_EYE, -1)
    cv2.putText(canvas, "Eyes", (50, legend_y + 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    # 저장
    cv2.imwrite(str(output_path), canvas)
    return True
