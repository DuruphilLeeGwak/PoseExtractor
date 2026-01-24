import json
import numpy as np
from pathlib import Path

folder = Path('test_io/outputs/20260123_235523')
files = sorted(folder.glob('*_kp.json'))

print(f"📁 분석 폴더: {folder.name}")
print(f"📄 총 {len(files)}개 파일\n")

corner_count = 0
for f in files[:5]:
    d = json.load(open(f))
    kpts = np.array(d['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    
    # 좌상단 키포인트 (이미지 크기 따라 다르지만 일반적으로 작은 값)
    corner = kpts[(kpts[:, 0] < 500) & (kpts[:, 1] < 500) & (kpts[:, 2] > 0.1)]
    
    print(f"{f.stem}: 좌상단(<500, <500) 키포인트 {len(corner)}개")
    if len(corner) > 5:
        corner_count += 1
        print(f"  ⚠️ 이상 발견:")
        for i, kpt in enumerate(corner[:5]):
            print(f"    - x={kpt[0]:.1f}, y={kpt[1]:.1f}, score={kpt[2]:.2f}")

print(f"\n{'✅ 모든 파일 정상!' if corner_count == 0 else f'⚠️ {corner_count}개 파일에서 이상 발견'}")
