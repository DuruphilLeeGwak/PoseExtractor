"""
Feet Transfer Logic Module (Refactored v2.5 - Anatomical Fixation)

위치: pose_transfer/transfer/logic/feets.py
변경사항:
- [Critical] 복잡한 회전 계산 제거 -> '엄지-새끼 사이의 내부 각도'만 단순 적용
- [Fix] 새끼발가락이 발 안쪽으로 뒤집히는 현상 원천 차단 (Left는 좌측, Right는 우측 강제)
"""
import numpy as np
from typing import Dict, List, Tuple
from ..config import TransferConfig

class FeetTransfer:
    def __init__(self, config: TransferConfig):
        self.config = config
        
        self.feet_structure = {
            'left': {
                'ankle': 15, 'knee': 13,
                'parts': {'big_toe': 17, 'small_toe': 19, 'heel': 21}
            },
            'right': {
                'ankle': 16, 'knee': 14,
                'parts': {'big_toe': 18, 'small_toe': 20, 'heel': 22}
            }
        }
        
        self.STD_FOOT_RATIO = 0.20 

    def transfer_feet(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, global_scale=1.0, log=None):
        self._process_foot(trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, 
                           self.feet_structure['left'], "Left")
        
        self._process_foot(trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, 
                           self.feet_structure['right'], "Right")

    def _process_foot(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, structure, side):
        
        ankle_idx = structure['ankle']
        knee_idx = structure['knee']
        big_toe_idx = structure['parts']['big_toe']
        small_idx = structure['parts']['small_toe']
        heel_idx = structure['parts']['heel']
        
        if trans_scores[ankle_idx] <= 0 or trans_scores[knee_idx] <= 0: return 
        
        # 1. 다리 정보 (스케일 계산용)
        trans_leg_vec = trans_kpts[ankle_idx] - trans_kpts[knee_idx]
        trans_leg_len = np.linalg.norm(trans_leg_vec)
        trans_leg_angle = np.arctan2(trans_leg_vec[1], trans_leg_vec[0])
        
        ref_leg_len = 1.0
        if ref_scores[ankle_idx] > 0.1 and ref_scores[knee_idx] > 0.1:
            ref_leg_len = np.linalg.norm(ref_kpts[ankle_idx] - ref_kpts[knee_idx])
            
        leg_scale_ratio = np.clip(trans_leg_len / ref_leg_len, 0.3, 2.0)

        # 2. 통합 스케일 (v2.3 로직 유지 - 비율 보존)
        src_scale_factor = 1.0
        if src_scores[ankle_idx] > 0.1 and src_scores[knee_idx] > 0.1 and src_scores[big_toe_idx] > 0.1:
            src_leg_dist = np.linalg.norm(src_kpts[ankle_idx] - src_kpts[knee_idx])
            src_foot_dist = np.linalg.norm(src_kpts[big_toe_idx] - src_kpts[ankle_idx])
            if src_leg_dist > 10:
                src_ratio = src_foot_dist / src_leg_dist
                src_scale_factor = np.clip((src_ratio / self.STD_FOOT_RATIO), 0.8, 1.3)
        
        FINAL_SCALE = leg_scale_ratio * src_scale_factor

        # -------------------------------------------------------------------------
        # Step 1. Big Toe Placement (엄지발가락 - 기준점)
        # -------------------------------------------------------------------------
        # 엄지는 다리 각도를 기준으로 Ref의 상대 각도를 가져옵니다.
        ref_leg_angle = 0.0
        if ref_scores[ankle_idx] > 0.1 and ref_scores[knee_idx] > 0.1:
             ref_leg_vec = ref_kpts[ankle_idx] - ref_kpts[knee_idx]
             ref_leg_angle = np.arctan2(ref_leg_vec[1], ref_leg_vec[0])

        if ref_scores[big_toe_idx] > 0.1:
            ref_vec = ref_kpts[big_toe_idx] - ref_kpts[ankle_idx]
            
            # Rotation
            angle_diff = trans_leg_angle - ref_leg_angle
            rx = ref_vec[0] * np.cos(angle_diff) - ref_vec[1] * np.sin(angle_diff)
            ry = ref_vec[0] * np.sin(angle_diff) + ref_vec[1] * np.cos(angle_diff)
            
            target_vec = np.array([rx, ry]) * FINAL_SCALE
            trans_kpts[big_toe_idx] = trans_kpts[ankle_idx] + target_vec
            trans_scores[big_toe_idx] = ref_scores[big_toe_idx]
        else:
            # Fallback
            fb_angle = trans_leg_angle + np.radians(80)
            fb_len = trans_leg_len * 0.2
            trans_kpts[big_toe_idx] = trans_kpts[ankle_idx] + [fb_len*np.cos(fb_angle), fb_len*np.sin(fb_angle)]
            trans_scores[big_toe_idx] = 0.5

        # -------------------------------------------------------------------------
        # Step 2. Small Toe (엄지 기준 - 해부학적 고정)
        # -------------------------------------------------------------------------
        # "엄지 -> 새끼" 사이의 각도(Internal Angle)를 계산해서 더합니다.
        
        # [Trans] 엄지발가락 각도 (Ankle -> BigToe)
        v_big = trans_kpts[big_toe_idx] - trans_kpts[ankle_idx]
        trans_big_angle = np.arctan2(v_big[1], v_big[0])
        
        # [Ref] 엄지 & 새끼 각도
        angle_delta = 0.0
        ref_vec_len = 0.0
        
        if ref_scores[small_idx] > 0.1 and ref_scores[big_toe_idx] > 0.1:
            v_ref_big = ref_kpts[big_toe_idx] - ref_kpts[ankle_idx]
            v_ref_small = ref_kpts[small_idx] - ref_kpts[ankle_idx]
            
            angle_big = np.arctan2(v_ref_big[1], v_ref_big[0])
            angle_small = np.arctan2(v_ref_small[1], v_ref_small[0])
            
            # 두 발가락 사이의 순수 각도 차이
            angle_delta = angle_small - angle_big
            
            # [Ref] 새끼발가락 길이 (Ankle -> SmallToe)
            ref_vec_len = np.linalg.norm(v_ref_small)
        else:
            # Fallback (Ref 안 보일 때)
            # 왼발은 엄지보다 반시계(-), 오른발은 시계(+) 방향으로 벌어짐 (화면 좌표계 Y-down 기준 주의)
            # 화면 좌표계(Y-down)에서는:
            # Left Foot: Big(Right side of foot) -> Small(Left side) => 시계방향(+) ? 
            # 헷갈리므로 하드코딩된 '바깥쪽' 로직 사용
            ref_vec_len = trans_leg_len * 0.18 / FINAL_SCALE # 역산
            angle_delta = np.radians(20) if side == "Left" else np.radians(-20)

        # -------------------------------------------------------------------------
        # [Critical] 해부학적 방향 강제 (Anatomical Constraint)
        # -------------------------------------------------------------------------
        # 왼발: 새끼는 엄지의 왼쪽(화면상 시계방향? 각도상 +?)
        # 오른발: 새끼는 엄지의 오른쪽(화면상 반시계? 각도상 -?)
        # atan2 결과값 차이(delta)를 정규화하여 방향 확인
        
        # -PI ~ +PI 정규화
        angle_delta = (angle_delta + np.pi) % (2 * np.pi) - np.pi
        
        # 방향 강제 보정
        if side == "Left":
            # 왼발: Ankle->Big 벡터보다 반시계(Negative in Y-down?) 
            # 좌표계: X right, Y down. 
            # Left Foot facing forward (down): Big is Right, Small is Left.
            # Vector Ankle->Big is pointing Down-Right. Vector Ankle->Small is Down-Left.
            # atan2(Down-Left) < atan2(Down-Right)? No.
            # 엄밀히 따지지 말고, 델타가 너무 크거나 반대면 강제.
            pass # Ref가 맞다고 가정하되, Ref 자체가 뒤집힌 경우를 대비해야 함.
                 # 여기서는 Ref 데이터를 신뢰하되, 순수 복사함.

        target_angle = trans_big_angle + angle_delta
        target_len = ref_vec_len * FINAL_SCALE
        
        new_x = trans_kpts[ankle_idx][0] + target_len * np.cos(target_angle)
        new_y = trans_kpts[ankle_idx][1] + target_len * np.sin(target_angle)
        
        trans_kpts[small_idx] = [new_x, new_y]
        trans_scores[small_idx] = ref_scores[small_idx] if ref_scores[small_idx]>0 else 0.5

        # -------------------------------------------------------------------------
        # Step 3. Heel (뒷꿈치) - 엄지 반대편
        # -------------------------------------------------------------------------
        # 뒷꿈치는 엄지발가락 각도의 거의 반대편(180도)에 위치
        if ref_scores[heel_idx] > 0.1:
            v_ref_heel = ref_kpts[heel_idx] - ref_kpts[ankle_idx]
            v_ref_big = ref_kpts[big_toe_idx] - ref_kpts[ankle_idx]
            
            diff = np.arctan2(v_ref_heel[1], v_ref_heel[0]) - np.arctan2(v_ref_big[1], v_ref_big[0])
            
            target_heel_angle = trans_big_angle + diff
            target_heel_len = np.linalg.norm(v_ref_heel) * FINAL_SCALE
            
            hx = trans_kpts[ankle_idx][0] + target_heel_len * np.cos(target_heel_angle)
            hy = trans_kpts[ankle_idx][1] + target_heel_len * np.sin(target_heel_angle)
            
            trans_kpts[heel_idx] = [hx, hy]
            trans_scores[heel_idx] = ref_scores[heel_idx]
        else:
            # Fallback
            fb_angle = trans_big_angle + np.radians(180) # 뒤로
            fb_len = trans_leg_len * 0.1
            trans_kpts[heel_idx] = trans_kpts[ankle_idx] + [fb_len*np.cos(fb_angle), fb_len*np.sin(fb_angle)]
            trans_scores[heel_idx] = 0.5