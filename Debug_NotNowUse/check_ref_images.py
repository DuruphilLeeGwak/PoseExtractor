import cv2
import os

# 입력 폴더 확인
inputs_folder = 'test_io/inputs/'
files = os.listdir(inputs_folder)
refs = [f for f in files if f.startswith('ref_')]

print(f"총 {len(refs)}개의 Ref 이미지 발견")
print(f"\n처음 5개:")
for f in refs[:5]:
    try:
        img = cv2.imread(os.path.join(inputs_folder, f))
        if img is not None:
            print(f"{f}: {img.shape[1]} x {img.shape[0]}")
    except:
        print(f"{f}: 읽기 실패")

# 14번이 어떤 ref인지 확인 (파일명 기준)
ref_14 = [f for f in refs if '14' in f or f.startswith('ref_014')]
if ref_14:
    print(f"\n14번과 관련된 ref 이미지:")
    for f in ref_14:
        try:
            img = cv2.imread(os.path.join(inputs_folder, f))
            if img is not None:
                print(f"{f}: {img.shape[1]} x {img.shape[0]}")
        except:
            print(f"{f}: 읽기 실패")
