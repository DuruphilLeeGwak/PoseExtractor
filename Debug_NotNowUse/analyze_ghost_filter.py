"""Ghost Filter 결과 비교 분석"""
import json
import os
import numpy as np
from pathlib import Path

# 경로 설정
path_with = Path("test_io/outputs/ghost filter_use20260121")
path_without = Path("test_io/outputs/ghost filter_wihoutUse20260121")

# 샘플 파일 목록
samples = [
    "origin_full_0001",
    "origin_full_0006",
    "origin_full_0020",
    "origin_full_0030",
    "Bailarina",
    "martial art reference",
]

print("=" * 80)
print("Ghost Filter 비교 분석: WITH vs WITHOUT")
print("=" * 80)

for sample in samples:
    kp_file = f"{sample}_kp.json"
    file_with = path_with / kp_file
    file_without = path_without / kp_file
    
    if not file_with.exists() or not file_without.exists():
        print(f"\n⚠️ {sample}: 파일이 존재하지 않음")
        continue
    
    # JSON 로드 (OpenPose 형식)
    with open(file_with) as f:
        data_with = json.load(f)
    with open(file_without) as f:
        data_without = json.load(f)
    
    # OpenPose 형식: {"version": 1.3, "people": [{"pose_keypoints_2d": [x1,y1,c1,x2,y2,c2,...]}]}
    if data_with.get("people") and len(data_with["people"]) > 0:
        kp_flat_with = data_with["people"][0].get("pose_keypoints_2d", [])
        kpts_with = np.array(kp_flat_with).reshape(-1, 3)
        scores_with = kpts_with[:, 2]
        kpts_with = kpts_with[:, :2]
    else:
        scores_with = np.zeros(133)
        kpts_with = np.zeros((133, 2))
    
    if data_without.get("people") and len(data_without["people"]) > 0:
        kp_flat_without = data_without["people"][0].get("pose_keypoints_2d", [])
        kpts_without = np.array(kp_flat_without).reshape(-1, 3)
        scores_without = kpts_without[:, 2]
        kpts_without = kpts_without[:, :2]
    else:
        scores_without = np.zeros(133)
        kpts_without = np.zeros((133, 2))
    
    print(f"\n{'='*80}")
    print(f"[Sample] {sample}")
    print(f"{'='*80}")
    
    # 전체 통계
    print(f"\n[전체 통계]")
    print(f"  WITH    Ghost Filter: 활성 키포인트(score>0.3) = {np.sum(scores_with > 0.3)}/133")
    print(f"  WITHOUT Ghost Filter: 활성 키포인트(score>0.3) = {np.sum(scores_without > 0.3)}/133")
    print(f"  차이: {np.sum(scores_with > 0.3) - np.sum(scores_without > 0.3)}")
    
    # 발목/발 영역 (15-22)
    ankle_foot_indices = range(15, 23)
    print(f"\n[발목/발 영역 (idx 15-22)]")
    with_ankle_active = sum(1 for i in ankle_foot_indices if i < len(scores_with) and scores_with[i] > 0.3)
    without_ankle_active = sum(1 for i in ankle_foot_indices if i < len(scores_without) and scores_without[i] > 0.3)
    print(f"  WITH:    {with_ankle_active}/8 활성")
    print(f"  WITHOUT: {without_ankle_active}/8 활성")
    print(f"  차이: {with_ankle_active - without_ankle_active}")
    
    # 상세 발목/발 좌표 및 점수
    if with_ankle_active > 0 or without_ankle_active > 0:
        print(f"\n  [상세 비교]")
        labels = {15: "LAnkle", 16: "RAnkle", 17: "LBigToe", 18: "LSmallToe", 
                  19: "LHeel", 20: "RBigToe", 21: "RSmallToe", 22: "RHeel"}
        for idx in ankle_foot_indices:
            if idx >= len(scores_with):
                continue
            label = labels.get(idx, f"idx{idx}")
            sw = scores_with[idx]
            swo = scores_without[idx]
            if sw > 0.1 or swo > 0.1:
                print(f"    {label:12s} - WITH: {sw:.3f}  WITHOUT: {swo:.3f}", end="")
                if sw < 0.3 and swo > 0.3:
                    print(f"  ⚠️ Ghost Filter로 제거됨")
                elif sw > 0.3 and swo < 0.3:
                    print(f"  ✅ Ghost Filter로 복원됨")
                else:
                    print()
    
    # 팔 영역 (5-10)
    arm_indices = range(5, 11)
    print(f"\n[팔/손목 영역 (idx 5-10)]")
    with_arm_active = sum(1 for i in arm_indices if i < len(scores_with) and scores_with[i] > 0.3)
    without_arm_active = sum(1 for i in arm_indices if i < len(scores_without) and scores_without[i] > 0.3)
    print(f"  WITH:    {with_arm_active}/6 활성")
    print(f"  WITHOUT: {without_arm_active}/6 활성")
    print(f"  차이: {with_arm_active - without_arm_active}")
    
    # 손 영역 (91-132)
    hand_indices = range(91, 133)
    print(f"\n[손 영역 (idx 91-132)]")
    with_hand_active = sum(1 for i in hand_indices if i < len(scores_with) and scores_with[i] > 0.3)
    without_hand_active = sum(1 for i in hand_indices if i < len(scores_without) and scores_without[i] > 0.3)
    print(f"  WITH:    {with_hand_active}/42 활성")
    print(f"  WITHOUT: {without_hand_active}/42 활성")
    print(f"  차이: {with_hand_active - without_hand_active}")
    
    # 프레임 밖 키포인트 확인
    # 이미지 크기는 키포인트에서 추정 (최대 좌표값 기준)
    if len(kpts_with) > 0 and np.any(scores_with > 0.1):
        w = int(np.max(kpts_with[scores_with > 0.1, 0]) * 1.2) + 100
        h = int(np.max(kpts_with[scores_with > 0.1, 1]) * 1.2) + 100
    elif len(kpts_without) > 0 and np.any(scores_without > 0.1):
        w = int(np.max(kpts_without[scores_without > 0.1, 0]) * 1.2) + 100
        h = int(np.max(kpts_without[scores_without > 0.1, 1]) * 1.2) + 100
    else:
        w, h = 3000, 4000  # 기본값
    
    if w > 0 and h > 0:
        print(f"\n[이미지 크기] {w}x{h}")
        
        # 하단 경계 근처 키포인트 (y >= h - 50)
        boundary_zone = h - 50
        with_near_bottom = []
        without_near_bottom = []
        
        for idx in range(len(kpts_with)):
            if scores_with[idx] > 0.3:
                x, y = kpts_with[idx]
                if y >= boundary_zone or y <= 5 or x <= 5 or x >= w - 5:
                    with_near_bottom.append((idx, x, y, scores_with[idx]))
        
        for idx in range(len(kpts_without)):
            if scores_without[idx] > 0.3:
                x, y = kpts_without[idx]
                if y >= boundary_zone or y <= 5 or x <= 5 or x >= w - 5:
                    without_near_bottom.append((idx, x, y, scores_without[idx]))
        
        if with_near_bottom or without_near_bottom:
            print(f"\n[경계 근처 키포인트 (y>={boundary_zone} or x,y<=5 or x>={w-5})]")
            print(f"  WITH:    {len(with_near_bottom)}개")
            print(f"  WITHOUT: {len(without_near_bottom)}개")
            
            if len(with_near_bottom) < 10:
                for idx, x, y, s in with_near_bottom:
                    print(f"    WITH: idx={idx} xy=({x:.1f},{y:.1f}) score={s:.3f}")
            if len(without_near_bottom) < 10:
                for idx, x, y, s in without_near_bottom:
                    print(f"    WITHOUT: idx={idx} xy=({x:.1f},{y:.1f}) score={s:.3f}")

print("\n" + "=" * 80)
print("분석 완료")
print("=" * 80)
