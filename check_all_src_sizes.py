import cv2
import os

# 모든 src 이미지 크기 확인
inputs_folder = 'test_io/inputs/'
files = sorted([f for f in os.listdir(inputs_folder) if f.endswith('.jpg') or f.endswith('.JPG')], 
               key=lambda x: int(x.split('.')[0]))

print(f"Src 이미지 크기 분석:")
print(f"{'파일':<15} {'너비':>8} {'높이':>8} {'비고'}")
print("=" * 50)

sizes = {}
for f in files:
    img = cv2.imread(os.path.join(inputs_folder, f))
    w, h = img.shape[1], img.shape[0]
    note = ""
    if w < 500 or h < 500:
        note = "⚠️ 매우 작음!"
    elif w < 1000 or h < 1000:
        note = "작음"
    
    print(f"{f:<15} {w:>8} {h:>8} {note}")
    sizes[f] = (w, h)

# 통계
widths = [s[0] for s in sizes.values()]
heights = [s[1] for s in sizes.values()]
print(f"\n통계:")
print(f"너비 범위: {min(widths)} ~ {max(widths)}")
print(f"높이 범위: {min(heights)} ~ {max(heights)}")
print(f"평균: {sum(widths)/len(widths):.0f} x {sum(heights)/len(heights):.0f}")

# 작은 이미지들 (너비 또는 높이 < 1000)
small_images = [(f, w, h) for f, (w, h) in sizes.items() if w < 1000 or h < 1000]
if small_images:
    print(f"\n작은 이미지 ({len(small_images)}개):")
    for f, w, h in small_images:
        print(f"  {f}: {w} x {h}")
