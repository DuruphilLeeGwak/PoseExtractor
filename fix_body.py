"""body.py의 transfer_shoulders를 torso_ratio 기반으로 수정"""

with open(r'D:\2025\pose_extractor\pose_transfer\transfer\logic\body.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 기존 transfer_shoulders 메서드를 찾아서 교체
old_method_start = '    def transfer_shoulders(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, processed, log):'

new_method = '''    def transfer_shoulders(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, *args, **kwargs):
        """어깨 전이 (torso_ratio 기반)"""
        torso_ratio = kwargs.get('torso_ratio', None)
        processed = kwargs.get('processed', None)
        log = kwargs.get('log', None)
        r_scores = kwargs.get('r_scores', None)

        # Positional args 호환
        if len(args) == 1:
            processed = args[0]
        elif len(args) == 2:
            processed, log = args
        elif len(args) >= 3:
            torso_ratio, processed, log = args[0], args[1], args[2]

        if processed is None:
            processed = set()
        if log is None:
            log = {}

        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        # Src 어깨 중심
        src_center = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
        
        # Ref 어깨 방향과 넓이
        ref_vec = r_kpts[r_sh] - r_kpts[l_sh]
        ref_width = calculate_distance(r_kpts[l_sh], r_kpts[r_sh])

        # Ref 방향 유효성 체크
        ref_shoulders_ok = ref_width > 1e-6
        if r_scores is not None:
            ref_shoulders_ok = ref_shoulders_ok and (r_scores[l_sh] > 0.3 and r_scores[r_sh] > 0.3)

        if ref_shoulders_ok:
            ref_dir = normalize_vector(ref_vec)
            dir_source = 'ref_shoulders'
        else:
            # Fallback: hip 방향 또는 src 방향
            l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
            hip_vec = r_kpts[r_hip] - r_kpts[l_hip]
            hip_width = calculate_distance(r_kpts[l_hip], r_kpts[r_hip])
            hips_ok = hip_width > 1e-6
            if r_scores is not None:
                hips_ok = hips_ok and (r_scores[l_hip] > 0.3 and r_scores[r_hip] > 0.3)

            if hips_ok:
                ref_dir = normalize_vector(hip_vec)
                dir_source = 'ref_hips_fallback'
            else:
                src_vec = s_kpts[r_sh] - s_kpts[l_sh]
                ref_dir = normalize_vector(src_vec)
                dir_source = 'src_shoulders_fallback'

        # 어깨 넓이 결정
        if torso_ratio is not None:
            target_width = ref_width * float(torso_ratio)
            log['shoulder'] = f'torso_ratio={float(torso_ratio):.3f}'
            print(f"   📐 어깨 전이: ref_width={ref_width:.1f}, torso_ratio={float(torso_ratio):.3f} → target_width={target_width:.1f} (dir={dir_source})")
        else:
            src_width = calculate_distance(s_kpts[l_sh], s_kpts[r_sh])
            target_width = src_width
            log['shoulder'] = 'src_width_fallback'
            print(f"   📐 어깨 전이(호환): src_width={src_width:.1f} (dir={dir_source})")
        
        t_kpts[l_sh] = src_center - ref_dir * (target_width / 2)
        t_kpts[r_sh] = src_center + ref_dir * (target_width / 2)
        t_scores[l_sh] = t_scores[r_sh] = 0.9
        
        processed.add(l_sh)
        processed.add(r_sh)'''

if old_method_start in content:
    # 메서드 전체를 찾아서 교체
    import re
    pattern = r'    def transfer_shoulders\(self.*?\n(?=    def |\Z)'
    content = re.sub(pattern, new_method + '\n\n', content, count=1, flags=re.DOTALL)
    
    with open(r'D:\2025\pose_extractor\pose_transfer\transfer\logic\body.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ body.py transfer_shoulders 수정 완료")
else:
    print("⚠️ transfer_shoulders를 찾을 수 없습니다")

# 검증
import subprocess
result = subprocess.run(
    [r'C:\Users\durup\anaconda3\envs\jyk\python.exe', '-c',
     "from pose_transfer.transfer.logic.body import BodyTransfer; import inspect; print('Signature:', inspect.signature(BodyTransfer.transfer_shoulders))"],
    capture_output=True, text=True, cwd=r'D:\2025\pose_extractor'
)
print(result.stdout)
if result.stderr:
    print("ERROR:", result.stderr)
