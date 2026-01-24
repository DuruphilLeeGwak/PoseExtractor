import numpy as np
from ...extractors.keypoint_constants import BODY_KEYPOINTS, FEET_KEYPOINTS, get_keypoint_index
from ...utils.geometry import normalize_vector, calculate_distance

class BodyTransfer:
    def __init__(self):
        self.upper_body_order = [
            ('left_shoulder', 'left_shoulder', ['left_elbow']),
            ('left_elbow', 'left_elbow', ['left_wrist']),
            ('right_shoulder', 'right_shoulder', ['right_elbow']),
            ('right_elbow', 'right_elbow', ['right_wrist']),
        ]
        self.lower_body_order = [
            ('left_hip', 'left_hip', ['left_knee']),
            ('left_knee', 'left_knee', ['left_ankle']),
            ('right_hip', 'right_hip', ['right_knee']),
            ('right_knee', 'right_knee', ['right_ankle']),
        ]
        self.feet_order = [
            ('left_ankle', 'left_ankle', ['left_heel', 'left_big_toe', 'left_small_toe']),
            ('right_ankle', 'right_ankle', ['right_heel', 'right_big_toe', 'right_small_toe']),
        ]

    def transfer_shoulders(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, *args, **kwargs):
        """어깨 전이 (호환 모드).

        이 함수는 엔진/파이프라인 버전 차이로 인해 호출 시그니처가 섞여도
        크래시가 나지 않도록 "구버전/신버전" 호출을 모두 허용합니다.

        허용하는 호출 형태
          1) (t_kpts, t_scores, s_kpts, s_scores, r_kpts, processed)
          2) (t_kpts, t_scores, s_kpts, s_scores, r_kpts, processed, log)
          3) (t_kpts, t_scores, s_kpts, s_scores, r_kpts, torso_ratio, processed, log)
          4) 키워드 인자: torso_ratio=..., processed=..., log=...

        어깨 넓이 정책
          - torso_ratio가 주어지면: ref 어깨 방향을 따르되, 넓이는 ref_width * torso_ratio
            (측면 src에서도 어깨가 과하게 좁아지는 문제 완화)
          - torso_ratio가 없으면(구버전 호환): src 어깨 넓이를 그대로 사용(기존 동작)
        """
        torso_ratio = kwargs.get('torso_ratio', None)
        processed = kwargs.get('processed', None)
        log = kwargs.get('log', None)
        r_scores = kwargs.get('r_scores', None)

        # Positional args 해석
        if len(args) == 1:
            processed = args[0]
        elif len(args) == 2:
            processed, log = args
        elif len(args) == 3:
            torso_ratio, processed, log = args
        elif len(args) != 0:
            raise TypeError(f"transfer_shoulders() got unexpected args: {args}")

        if processed is None:
            processed = set()
        if log is None:
            log = {}
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        # Src 어깨 중심 (위치는 src 기준)
        src_center = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
        
        # Ref 어깨 방향과 넓이 (방향은 ref를 따라야 포즈가 자연스러움)
        ref_vec = r_kpts[r_sh] - r_kpts[l_sh]
        ref_width = calculate_distance(r_kpts[l_sh], r_kpts[r_sh])

        # Ref 방향 유효성 체크: 어깨가 누락/붕괴되면 방향이 0벡터가 되어 각도가 반영되지 않음
        ref_shoulders_ok = ref_width > 1e-6
        if r_scores is not None:
            ref_shoulders_ok = ref_shoulders_ok and (r_scores[l_sh] > 0.3 and r_scores[r_sh] > 0.3)

        if ref_shoulders_ok:
            ref_dir = normalize_vector(ref_vec)
            dir_source = 'ref_shoulders'
        else:
            # Fallback 1) Ref hip 방향 (상대적으로 안정)
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
                # Fallback 2) Src 어깨 방향
                src_vec = s_kpts[r_sh] - s_kpts[l_sh]
                ref_dir = normalize_vector(src_vec)
                dir_source = 'src_shoulders_fallback'

        # 어깨 넓이 결정
        if torso_ratio is not None:
            # 시점 독립적 보정
            target_width = ref_width * float(torso_ratio)
            log['shoulder'] = f'torso_ratio={float(torso_ratio):.3f}'
            print(
                f"   📐 어깨 전이: ref_width={ref_width:.1f}, torso_ratio={float(torso_ratio):.3f} → target_width={target_width:.1f} (dir={dir_source})"
            )
        else:
            # 구버전 호환(기존 동작): src 어깨 넓이 그대로
            src_width = calculate_distance(s_kpts[l_sh], s_kpts[r_sh])
            target_width = src_width
            log['shoulder'] = 'src_width_fallback'
            print(f"   📐 어깨 전이(호환): src_width={src_width:.1f} (torso_ratio 없음, dir={dir_source})")
        
        t_kpts[l_sh] = src_center - ref_dir * (target_width / 2)
        t_kpts[r_sh] = src_center + ref_dir * (target_width / 2)
        t_scores[l_sh] = t_scores[r_sh] = 0.9
        
        processed.add(l_sh); processed.add(r_sh)
        # log는 위에서 정책에 맞게 채움

    def transfer_torso(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, scale, processed, log):
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
        
        t_neck = (t_kpts[l_sh] + t_kpts[r_sh]) / 2
        r_neck = (r_kpts[l_sh] + r_kpts[r_sh]) / 2
        r_root = (r_kpts[l_hip] + r_kpts[r_hip]) / 2
        
        spine_vec = r_root - r_neck
        spine_dir = normalize_vector(spine_vec)
        
        if s_scores[l_hip] > 0.3:
            s_root = (s_kpts[l_hip] + s_kpts[r_hip]) / 2
            s_neck = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
            spine_len = calculate_distance(s_root, s_neck)
        else:
            spine_len = calculate_distance(r_root, r_neck) * scale
            
        new_root = t_neck + spine_dir * spine_len
        
        r_hip_vec = r_kpts[r_hip] - r_kpts[l_hip]
        r_hip_dir = normalize_vector(r_hip_vec)
        
        if s_scores[l_hip] > 0.3:
            hip_width = calculate_distance(s_kpts[l_hip], s_kpts[r_hip])
        else:
            hip_width = calculate_distance(r_kpts[l_hip], r_kpts[r_hip]) * scale
            
        t_kpts[l_hip] = new_root - r_hip_dir * (hip_width / 2)
        t_kpts[r_hip] = new_root + r_hip_dir * (hip_width / 2)
        t_scores[l_hip] = t_scores[r_hip] = 0.9
        
        processed.add(l_hip); processed.add(r_hip)
        log['torso'] = 'spine_calc'

    def transfer_chain(self, t_kpts, t_scores, lengths, r_kpts, r_scores, scale, processed, log, is_lower=False):
        order = self.lower_body_order if is_lower else self.upper_body_order
        chain_type = "LOWER" if is_lower else "UPPER"
        
        print(f"\n   🔍 [DEBUG] transfer_chain({chain_type}) START")
        print(f"      processed indices: {sorted(processed)}")
        
        for _, p_name, children in order:
            p_idx = get_keypoint_index(p_name)
            
            if p_idx not in processed:
                continue
            
            p_pos = t_kpts[p_idx]
            
            for c_name in children:
                c_idx = get_keypoint_index(c_name)
                if c_idx is None:
                    continue
                    
                r_score = r_scores[c_idx] if c_idx < len(r_scores) else 0
                
                if r_score < 0.1:
                    print(f"      ❌ {c_name} (idx={c_idx}): ref_score={r_score:.3f} < 0.1, SKIP")
                    continue
                
                bone = f"{p_name}_{c_name}"
                alt = f"{c_name}_{p_name}"
                
                # 뼈 길이 결정: SRC 우선 (신체 비율 유지)
                src_length = lengths.get(bone) or lengths.get(alt)
                ref_length = calculate_distance(r_kpts[p_idx], r_kpts[c_idx])
                
                if src_length:
                    length = src_length
                    source = "SRC"
                else:
                    length = ref_length * scale
                    source = f"REF*scale"
                
                # REF 방향 사용 (각도는 REF 따름)
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                direct = normalize_vector(vec)
                
                t_kpts[c_idx] = p_pos + direct * length
                t_scores[c_idx] = 0.8
                
                processed.add(c_idx)
                log[c_name] = 'chain'
                
                print(f"      ✅ {c_name} (idx={c_idx}): length={length:.1f} ({source})")
        
        print(f"   🔍 [DEBUG] transfer_chain({chain_type}) END")
        print(f"      final processed: {sorted(processed)}")

    def fine_tune_lower_ratio(
        self,
        t_kpts, t_scores,
        src_kpts, src_scores,
        ref_kpts, ref_scores,
        processed, log
    ):
        """
        하체(hip~feet) 비율 미세튜닝:
        - Src의 마디별 비율(좌우 평균)을 계산
        - Trans에 적용할 때 Ref의 방향을 따르되, 길이는 Src 비율 * Trans torso 길이로 설정
        """
        if log is None:
            log = {}

        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']

        # Src torso length
        if (src_scores[l_sh] > 0.2 and src_scores[r_sh] > 0.2 and src_scores[l_hip] > 0.2 and src_scores[r_hip] > 0.2):
            src_neck = (src_kpts[l_sh] + src_kpts[r_sh]) / 2.0
            src_root = (src_kpts[l_hip] + src_kpts[r_hip]) / 2.0
            src_torso = float(np.linalg.norm(src_neck - src_root))
        else:
            return

        if src_torso <= 1e-6:
            return

        # Trans torso length (apply scale to trans)
        if (t_scores[l_sh] > 0.1 and t_scores[r_sh] > 0.1 and t_scores[l_hip] > 0.1 and t_scores[r_hip] > 0.1):
            trans_neck = (t_kpts[l_sh] + t_kpts[r_sh]) / 2.0
            trans_root = (t_kpts[l_hip] + t_kpts[r_hip]) / 2.0
            trans_torso = float(np.linalg.norm(trans_neck - trans_root))
        else:
            return

        if trans_torso <= 1e-6:
            return

        # Src ratios (left/right avg)
        def _src_ratio(p_name, c_name):
            p_idx = get_keypoint_index(p_name)
            c_idx = get_keypoint_index(c_name)
            if p_idx is None or c_idx is None:
                return None
            if src_scores[p_idx] <= 0.2 or src_scores[c_idx] <= 0.2:
                return None
            length = float(np.linalg.norm(src_kpts[c_idx] - src_kpts[p_idx]))
            return length / src_torso if src_torso > 1e-6 else None

        ratio_map = {}
        ratio_pairs = {
            'hip_knee': ('left_hip', 'left_knee', 'right_hip', 'right_knee'),
            'knee_ankle': ('left_knee', 'left_ankle', 'right_knee', 'right_ankle'),
            'ankle_big_toe': ('left_ankle', 'left_big_toe', 'right_ankle', 'right_big_toe'),
            'ankle_small_toe': ('left_ankle', 'left_small_toe', 'right_ankle', 'right_small_toe'),
            'ankle_heel': ('left_ankle', 'left_heel', 'right_ankle', 'right_heel'),
        }

        for key, (lp, lc, rp, rc) in ratio_pairs.items():
            l_ratio = _src_ratio(lp, lc)
            r_ratio = _src_ratio(rp, rc)
            if l_ratio is not None and r_ratio is not None:
                ratio_map[key] = (l_ratio + r_ratio) / 2.0
            elif l_ratio is not None:
                ratio_map[key] = l_ratio
            elif r_ratio is not None:
                ratio_map[key] = r_ratio

        if not ratio_map:
            return

        # Apply ratios using ref direction
        def _apply_side(side_prefix):
            hip = f"{side_prefix}_hip"
            knee = f"{side_prefix}_knee"
            ankle = f"{side_prefix}_ankle"
            big_toe = f"{side_prefix}_big_toe"
            small_toe = f"{side_prefix}_small_toe"
            heel = f"{side_prefix}_heel"

            def _place(parent, child, ratio_key):
                if ratio_key not in ratio_map:
                    return
                p_idx = get_keypoint_index(parent)
                c_idx = get_keypoint_index(child)
                if p_idx is None or c_idx is None:
                    return
                if p_idx not in processed:
                    return
                if ref_scores[c_idx] <= 0.1 or ref_scores[p_idx] <= 0.1:
                    return
                ref_vec = ref_kpts[c_idx] - ref_kpts[p_idx]
                ref_dir = normalize_vector(ref_vec)
                length = ratio_map[ratio_key] * trans_torso
                t_kpts[c_idx] = t_kpts[p_idx] + ref_dir * length
                t_scores[c_idx] = 0.85
                processed.add(c_idx)

            _place(hip, knee, 'hip_knee')
            _place(knee, ankle, 'knee_ankle')
            _place(ankle, big_toe, 'ankle_big_toe')
            _place(ankle, small_toe, 'ankle_small_toe')
            _place(ankle, heel, 'ankle_heel')

        _apply_side('left')
        _apply_side('right')

        log['lower_ratio_tuning'] = {
            'src_torso': float(src_torso),
            'trans_torso': float(trans_torso),
            'ratios': {k: float(v) for k, v in ratio_map.items()}
        }

    def fine_tune_upper_ratio(
        self,
        t_kpts, t_scores,
        src_kpts, src_scores,
        ref_kpts, ref_scores,
        processed, log
    ):
        """
        상체(shoulder~wrist) 비율 미세튜닝:
        - Src의 마디별 비율(좌우 평균)을 계산
        - Trans에 적용할 때 Ref의 방향을 따르되, 길이는 Src 비율 * Trans torso 길이로 설정
        """
        if log is None:
            log = {}

        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']

        # Src torso length
        if (src_scores[l_sh] > 0.2 and src_scores[r_sh] > 0.2 and src_scores[l_hip] > 0.2 and src_scores[r_hip] > 0.2):
            src_neck = (src_kpts[l_sh] + src_kpts[r_sh]) / 2.0
            src_root = (src_kpts[l_hip] + src_kpts[r_hip]) / 2.0
            src_torso = float(np.linalg.norm(src_neck - src_root))
        else:
            return

        if src_torso <= 1e-6:
            return

        # Trans torso length
        if (t_scores[l_sh] > 0.1 and t_scores[r_sh] > 0.1 and t_scores[l_hip] > 0.1 and t_scores[r_hip] > 0.1):
            trans_neck = (t_kpts[l_sh] + t_kpts[r_sh]) / 2.0
            trans_root = (t_kpts[l_hip] + t_kpts[r_hip]) / 2.0
            trans_torso = float(np.linalg.norm(trans_neck - trans_root))
        else:
            return

        if trans_torso <= 1e-6:
            return

        def _src_ratio(p_name, c_name):
            p_idx = get_keypoint_index(p_name)
            c_idx = get_keypoint_index(c_name)
            if p_idx is None or c_idx is None:
                return None
            if src_scores[p_idx] <= 0.2 or src_scores[c_idx] <= 0.2:
                return None
            length = float(np.linalg.norm(src_kpts[c_idx] - src_kpts[p_idx]))
            return length / src_torso if src_torso > 1e-6 else None

        ratio_map = {}
        ratio_pairs = {
            'shoulder_elbow': ('left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'),
            'elbow_wrist': ('left_elbow', 'left_wrist', 'right_elbow', 'right_wrist'),
        }

        for key, (lp, lc, rp, rc) in ratio_pairs.items():
            l_ratio = _src_ratio(lp, lc)
            r_ratio = _src_ratio(rp, rc)
            if l_ratio is not None and r_ratio is not None:
                ratio_map[key] = (l_ratio + r_ratio) / 2.0
            elif l_ratio is not None:
                ratio_map[key] = l_ratio
            elif r_ratio is not None:
                ratio_map[key] = r_ratio

        if not ratio_map:
            return

        def _apply_side(side_prefix):
            shoulder = f"{side_prefix}_shoulder"
            elbow = f"{side_prefix}_elbow"
            wrist = f"{side_prefix}_wrist"

            def _place(parent, child, ratio_key):
                if ratio_key not in ratio_map:
                    return
                p_idx = get_keypoint_index(parent)
                c_idx = get_keypoint_index(child)
                if p_idx is None or c_idx is None:
                    return
                if p_idx not in processed:
                    return
                if ref_scores[c_idx] <= 0.1 or ref_scores[p_idx] <= 0.1:
                    return
                ref_vec = ref_kpts[c_idx] - ref_kpts[p_idx]
                ref_dir = normalize_vector(ref_vec)
                length = ratio_map[ratio_key] * trans_torso
                t_kpts[c_idx] = t_kpts[p_idx] + ref_dir * length
                t_scores[c_idx] = 0.85
                processed.add(c_idx)

            _place(shoulder, elbow, 'shoulder_elbow')
            _place(elbow, wrist, 'elbow_wrist')

        _apply_side('left')
        _apply_side('right')

        log['upper_ratio_tuning'] = {
            'src_torso': float(src_torso),
            'trans_torso': float(trans_torso),
            'ratios': {k: float(v) for k, v in ratio_map.items()}
        }

    def transfer_feet(self, t_kpts, t_scores, src_kpts, src_scores, lengths, r_kpts, r_scores, scale, processed, log):
        """
        Feet 키포인트 전이 (DEBUG VERSION)
        """
        print(f"\n" + "="*60)
        print(f"🦶 [DEBUG] transfer_feet()")
        print("="*60)
        print(f"   global_scale (어깨비율): {scale:.3f}")

        if log is not None:
            log.setdefault('foot_debug', [])
        
        # 발 관련 뼈 길이 확인
        feet_bones = [k for k in lengths.keys() if any(x in k for x in ['ankle', 'toe', 'heel'])]
        print(f"   src에서 계산된 발 뼈 길이: {feet_bones if feet_bones else 'NONE!'}")
        
        if not FEET_KEYPOINTS:
            print(f"   ⚠️ FEET_KEYPOINTS not defined, skipping")
            return

        def _foot_mean(kpts, scores, ankle_idx, foot_indices, thr=0.2):
            if ankle_idx is None or ankle_idx >= len(scores) or scores[ankle_idx] <= thr:
                return None, 0
            ankle = kpts[ankle_idx]
            dists = []
            for idx in foot_indices:
                if idx < len(scores) and scores[idx] > thr:
                    dists.append(np.linalg.norm(kpts[idx] - ankle))
            if len(dists) == 0:
                return None, 0
            return float(np.mean(dists)), len(dists)

        def _torso_length(kpts, scores, thr=0.2):
            l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
            l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
            if (scores[l_sh] > thr and scores[r_sh] > thr and scores[l_hip] > thr and scores[r_hip] > thr):
                neck = (kpts[l_sh] + kpts[r_sh]) / 2.0
                root = (kpts[l_hip] + kpts[r_hip]) / 2.0
                return float(np.linalg.norm(neck - root))
            return None

        left_ankle_idx = BODY_KEYPOINTS.get('left_ankle')
        right_ankle_idx = BODY_KEYPOINTS.get('right_ankle')
        left_foot_indices = [
            FEET_KEYPOINTS.get('left_big_toe'),
            FEET_KEYPOINTS.get('left_small_toe'),
            FEET_KEYPOINTS.get('left_heel')
        ]
        right_foot_indices = [
            FEET_KEYPOINTS.get('right_big_toe'),
            FEET_KEYPOINTS.get('right_small_toe'),
            FEET_KEYPOINTS.get('right_heel')
        ]

        src_left_mean, src_left_cnt = _foot_mean(src_kpts, src_scores, left_ankle_idx, left_foot_indices)
        src_right_mean, src_right_cnt = _foot_mean(src_kpts, src_scores, right_ankle_idx, right_foot_indices)
        src_base_candidates = [m for m, c in [(src_left_mean, src_left_cnt), (src_right_mean, src_right_cnt)] if m is not None and c > 0]
        src_base = float(max(src_base_candidates)) if len(src_base_candidates) > 0 else None

        ref_torso = _torso_length(r_kpts, r_scores)
        ref_left_mean, ref_left_cnt = _foot_mean(r_kpts, r_scores, left_ankle_idx, left_foot_indices)
        ref_right_mean, ref_right_cnt = _foot_mean(r_kpts, r_scores, right_ankle_idx, right_foot_indices)

        ref_ratios = {}
        if ref_torso is not None and ref_torso > 1e-6:
            if ref_left_mean is not None and ref_left_cnt > 0:
                ref_ratios['LEFT'] = float(ref_left_mean / ref_torso)
            if ref_right_mean is not None and ref_right_cnt > 0:
                ref_ratios['RIGHT'] = float(ref_right_mean / ref_torso)
        ref_ratio_max = max(ref_ratios.values()) if len(ref_ratios) > 0 else None
        
        for _, p_name, children in self.feet_order:
            p_idx = get_keypoint_index(p_name)
            
            if p_idx is None or p_idx not in processed:
                print(f"\n   ❌ {p_name} not in processed, SKIP")
                continue
            
            p_pos = t_kpts[p_idx]
            side = "LEFT" if "left" in p_name else "RIGHT"
            print(f"\n   [{side}] Foot from {p_name} ({p_idx})")
            print(f"      parent_pos: ({p_pos[0]:.1f}, {p_pos[1]:.1f})")

            foot_debug = {
                'side': side,
                'parent': p_name,
                'parent_idx': int(p_idx)
            }
            
            for c_name in children:
                c_idx = FEET_KEYPOINTS.get(c_name)
                if c_idx is None:
                    print(f"      ❌ {c_name}: not in FEET_KEYPOINTS")
                    continue
                
                r_score = r_scores[c_idx] if c_idx < len(r_scores) else 0
                
                if r_score < 0.1:
                    print(f"      ❌ {c_name} (idx={c_idx}): ref_score={r_score:.3f} < 0.1, SKIP")
                    continue
                
                # 뼈 길이 결정
                bone = f"{p_name}_{c_name}"
                alt = f"{c_name}_{p_name}"
                
                # 뼈 길이 결정: SRC 우선
                bone = f"{p_name}_{c_name}"
                alt = f"{c_name}_{p_name}"
                
                src_length = lengths.get(bone) or lengths.get(alt)
                ref_length = calculate_distance(r_kpts[p_idx], r_kpts[c_idx])

                length = None
                source = None

                # New strategy: ref foot/torso ratio normalized to src max foot size
                ref_ratio = ref_ratios.get(side, None)
                target_size = None
                scale_factor = None
                ref_mean_for_side = ref_left_mean if side == 'LEFT' else ref_right_mean
                if src_base is not None and ref_ratio_max is not None and ref_ratio is not None and ref_ratio_max > 1e-6:
                    target_size = src_base * (ref_ratio / ref_ratio_max)
                    if ref_mean_for_side is not None and ref_mean_for_side > 1e-6:
                        scale_factor = target_size / ref_mean_for_side

                if scale_factor is not None:
                    length = ref_length * scale_factor
                    source = f"REF*ratio ({scale_factor:.3f})"
                else:
                    if src_length:
                        length = src_length
                        source = "SRC"
                    else:
                        length = ref_length * scale
                        source = f"REF*scale ({ref_length:.1f}*{scale:.3f})"
                
                # REF 방향 사용
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                direct = normalize_vector(vec)
                
                t_kpts[c_idx] = p_pos + direct * length
                t_scores[c_idx] = 0.8
                
                processed.add(c_idx)
                log[c_name] = 'feet_chain'

                foot_debug.setdefault('children', []).append({
                    'name': c_name,
                    'idx': int(c_idx),
                    'ref_score': float(r_score),
                    'length': float(length),
                    'source': source,
                    'ref_vec': [float(vec[0]), float(vec[1])],
                    'ref_len': float(np.linalg.norm(vec))
                })

                print(f"      ✅ {c_name} (idx={c_idx}): length={length:.1f} ({source})")

            if log is not None:
                foot_debug.update({
                    'src_base': float(src_base) if src_base is not None else None,
                    'ref_torso_len': float(ref_torso) if ref_torso is not None else None,
                    'ref_ratio': float(ref_ratios.get(side)) if ref_ratios.get(side) is not None else None,
                    'ref_ratio_max': float(ref_ratio_max) if ref_ratio_max is not None else None,
                    'src_left_mean': float(src_left_mean) if src_left_mean is not None else None,
                    'src_right_mean': float(src_right_mean) if src_right_mean is not None else None,
                    'ref_left_mean': float(ref_left_mean) if ref_left_mean is not None else None,
                    'ref_right_mean': float(ref_right_mean) if ref_right_mean is not None else None,
                    'target_size': float(target_size) if target_size is not None else None,
                    'scale_factor': float(scale_factor) if scale_factor is not None else None,
                    'ref_mean_used': float(ref_mean_for_side) if ref_mean_for_side is not None else None
                })
                log['foot_debug'].append(foot_debug)
        
        print(f"\n   final feet processed: {sorted([i for i in processed if i >= 17 and i <= 22])}")