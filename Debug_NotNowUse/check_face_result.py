"""
face_rendering 테스트 결과 확인 스크립트
- trans_kp.json을 읽어서 face keypoint들의 score를 확인
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 최신 결과 찾기
enabled_dir = PROJECT_ROOT / "test_io" / "face_rendering_test" / "enabled"
disabled_dir = PROJECT_ROOT / "test_io" / "face_rendering_test" / "disabled"

# 가장 최근 폴더 찾기
def find_latest_result(base_dir):
    if not base_dir.exists():
        return None
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda x: x.stat().st_mtime)
    return latest / "trans" / "trans_kp.json"

enabled_json = find_latest_result(enabled_dir)
disabled_json = find_latest_result(disabled_dir)

def check_face_keypoints(json_path, label):
    """Face keypoint 상태 확인"""
    print(f"\n{'='*70}")
    print(f"{label}: {json_path.name}")
    print('='*70)
    
    if not json_path.exists():
        print(f"❌ 파일이 없습니다: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # OpenPose 형식: people[0]['pose_keypoints_2d']는 평탄화된 배열 [x, y, score, x, y, score, ...]
    people = data.get('people', [])
    if not people:
        print("❌ 사람 데이터가 없습니다")
        return
    
    pose_kpts = people[0].get('pose_keypoints_2d', [])
    
    # 3개씩 묶어서 (x, y, score) 형태로 변환
    keypoints = []
    for i in range(0, len(pose_kpts), 3):
        if i+2 < len(pose_kpts):
            keypoints.append([pose_kpts[i], pose_kpts[i+1], pose_kpts[i+2]])
    
    print(f"\n총 {len(keypoints)}개 키포인트")
    
    # Body Face (0-4): nose, left_eye, right_eye, left_ear, right_ear
    print("\n[Body Face] 키포인트 (0-4):")
    body_face_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
    for i, name in enumerate(body_face_names):
        if i < len(keypoints):
            kpt = keypoints[i]
            score = kpt[2]
            pos = f"({kpt[0]:.1f}, {kpt[1]:.1f})"
            status = "✅" if score > 0.1 else "❌"
            print(f"  {status} [{i}] {name:12s}: score={score:.3f}, pos={pos}")
    
    # Face Landmarks (23-90)
    print("\n[Face Landmarks] 키포인트 (23-90):")
    face_count = 0
    face_active_count = 0
    sample_indices = []
    for i in range(23, min(91, len(keypoints))):
        kpt = keypoints[i]
        score = kpt[2]
        face_count += 1
        if score > 0.1:
            face_active_count += 1
            if len(sample_indices) < 5:
                sample_indices.append(i)
    
    print(f"  총 {face_count}개 중 {face_active_count}개 활성화 (score > 0.1)")
    
    if face_active_count > 0:
        print(f"  ✅ Face landmarks가 활성화되어 있습니다")
        # 샘플 출력
        print(f"\n  샘플 (처음 {len(sample_indices)}개):")
        for i in sample_indices:
            kpt = keypoints[i]
            score = kpt[2]
            pos = f"({kpt[0]:.1f}, {kpt[1]:.1f})"
            print(f"    [{i}] score={score:.3f}, pos={pos}")
    else:
        print(f"  ❌ Face landmarks가 비활성화되어 있습니다")

if __name__ == "__main__":
    check_face_keypoints(enabled_json, "enabled=true")
    check_face_keypoints(disabled_json, "enabled=false")
    
    print("\n" + "="*70)
    print("📊 결론")
    print("="*70)
    print("enabled와 disabled의 face keypoints를 비교하여")
    print("disabled에서 Body Face(0-4)와 Face Landmarks(23-90)가")
    print("모두 비활성화(score=0 또는 매우 낮음)되어 있어야 정상입니다.")
