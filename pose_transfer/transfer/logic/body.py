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

    def transfer_chain(
        self,
        trans_kpts: np.ndarray,
        trans_scores: np.ndarray,
        corrected_lengths: dict,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        hand_scale_ratio: float,
        processed: set,
        log: dict = None,
        is_lower: bool = False,
        src_proportions=None, ref_proportions=None,
        src_depths=None, ref_depths=None, depth_z_scale=None
    ):
        """
        [수정됨] 사지(팔/다리) 전이 - Source 길이 보존 모드
        
        변경사항:
        - Ref 비율(ratio) 기반 길이 계산을 제거했습니다.
        - 무조건 corrected_lengths(Src 길이)를 사용하여, Ref가 롱다리여도 Src 비율을 유지합니다.
        """
        chain = self.lower_body_order if is_lower else self.upper_body_order
        chain_name = "LOWER" if is_lower else "UPPER"
        
        # 디버그 로그 시작
        if log is not None:
            if 'chain_debug' not in log: log['chain_debug'] = []
            print(f"   🔍 [DEBUG] transfer_chain({chain_name}) START [Source Length Priority]")
        
        processed_indices = []

        for parent_name, self_name, children_names in chain:
            parent_idx = get_keypoint_index(parent_name)
            self_idx = get_keypoint_index(self_name)
            
            # 부모가 처리되지 않았거나 신뢰도가 낮으면 스킵
            if trans_scores[parent_idx] == 0:
                continue
            
            # 자식 노드들 처리
            for child_name in children_names:
                child_idx = get_keypoint_index(child_name)
                if child_idx in processed:
                    continue
                
                # 1. 뼈 이름 및 Src 길이 가져오기
                bone_name = f"{self_name}_{child_name}"
                
                # [핵심] Ref 비율 계산 없이, Src의 길이를 직접 사용
                target_length = corrected_lengths.get(bone_name)
                
                # Src 길이가 없으면(감지 실패 등), Fallback으로 거리 직접 계산
                if target_length is None or target_length <= 0:
                    # Fallback: Ref 길이 * hand_scale_ratio (최후의 수단)
                    ref_dist = calculate_distance(ref_kpts[self_idx], ref_kpts[child_idx])
                    target_length = ref_dist * hand_scale_ratio
                    src_str = "Fallback(Ref*Scale)"
                else:
                    src_str = "SrcFixed"

                # 2. 방향 벡터 (Direction) - Ref 기준
                # Ref에 해당 뼈가 있으면 그 방향을 사용
                if ref_scores[self_idx] > 0.1 and ref_scores[child_idx] > 0.1:
                    direction_vec = ref_kpts[child_idx] - ref_kpts[self_idx]
                    direction = normalize_vector(direction_vec)
                else:
                    # Ref에도 없으면 Src 방향 유지 (거의 발생 안 함)
                    direction = np.array([0, 1]) # 기본 아래로

                # 3. 좌표 결정: 부모 위치 + (Ref 방향 * Src 길이)
                trans_kpts[child_idx] = trans_kpts[self_idx] + direction * target_length
                
                # 점수는 부모 점수와 Ref 점수 중 낮은 것 (보수적 접근)
                trans_scores[child_idx] = min(trans_scores[self_idx], ref_scores[child_idx]) \
                                          if ref_scores[child_idx] > 0 else trans_scores[self_idx] * 0.5
                
                processed.add(child_idx)
                processed_indices.append(child_idx)
                
                print(f"      ✅ {child_name} (idx={child_idx}): length={target_length:.1f} ({src_str})")

        if log is not None:
            print(f"   🔍 [DEBUG] transfer_chain({chain_name}) END")
            print(f"      final processed: {sorted(list(processed))}")

    def fine_tune_lower_ratio(
        self,
        t_kpts, t_scores,
        src_kpts, src_scores,
        ref_kpts, ref_scores,
        processed, log,
        source_depths=None,
        depth_z_scale: float = 1.0
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
        def _src_ratio_2d(p_name, c_name):
            p_idx = get_keypoint_index(p_name)
            c_idx = get_keypoint_index(c_name)
            if p_idx is None or c_idx is None:
                return None
            if src_scores[p_idx] <= 0.2 or src_scores[c_idx] <= 0.2:
                return None
            length = float(np.linalg.norm(src_kpts[c_idx] - src_kpts[p_idx]))
            return length / src_torso if src_torso > 1e-6 else None

        def _src_ratio_3d(p_name, c_name):
            if source_depths is None or depth_z_scale is None:
                return None
            p_idx = get_keypoint_index(p_name)
            c_idx = get_keypoint_index(c_name)
            if p_idx is None or c_idx is None:
                return None
            if src_scores[p_idx] <= 0.2 or src_scores[c_idx] <= 0.2:
                return None
            diff = src_kpts[c_idx] - src_kpts[p_idx]
            dz = (float(source_depths[c_idx]) - float(source_depths[p_idx])) * float(depth_z_scale)
            length = float(np.sqrt(diff[0] ** 2 + diff[1] ** 2 + dz ** 2))
            return length / src_torso if src_torso > 1e-6 else None

        ratio_map_2d = {}
        ratio_map_3d = {}
        ratio_map_used = {}
        ratio_pairs = {
            'hip_knee': ('left_hip', 'left_knee', 'right_hip', 'right_knee'),
            'knee_ankle': ('left_knee', 'left_ankle', 'right_knee', 'right_ankle'),
            'ankle_big_toe': ('left_ankle', 'left_big_toe', 'right_ankle', 'right_big_toe'),
            'ankle_small_toe': ('left_ankle', 'left_small_toe', 'right_ankle', 'right_small_toe'),
            'ankle_heel': ('left_ankle', 'left_heel', 'right_ankle', 'right_heel'),
        }

        for key, (lp, lc, rp, rc) in ratio_pairs.items():
            l_ratio_2d = _src_ratio_2d(lp, lc)
            r_ratio_2d = _src_ratio_2d(rp, rc)
            if l_ratio_2d is not None and r_ratio_2d is not None:
                ratio_map_2d[key] = (l_ratio_2d + r_ratio_2d) / 2.0
            elif l_ratio_2d is not None:
                ratio_map_2d[key] = l_ratio_2d
            elif r_ratio_2d is not None:
                ratio_map_2d[key] = r_ratio_2d

            l_ratio_3d = _src_ratio_3d(lp, lc)
            r_ratio_3d = _src_ratio_3d(rp, rc)
            if l_ratio_3d is not None and r_ratio_3d is not None:
                ratio_map_3d[key] = (l_ratio_3d + r_ratio_3d) / 2.0
            elif l_ratio_3d is not None:
                ratio_map_3d[key] = l_ratio_3d
            elif r_ratio_3d is not None:
                ratio_map_3d[key] = r_ratio_3d

        if source_depths is not None and len(ratio_map_3d) > 0:
            ratio_map_used = ratio_map_3d
            ratio_source = '3d'
        else:
            ratio_map_used = ratio_map_2d
            ratio_source = '2d'

        if not ratio_map_used:
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
                if ratio_key not in ratio_map_used:
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
                length = ratio_map_used[ratio_key] * trans_torso
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

        def _delta_map(base_map, used_map):
            delta = {}
            for k, v in used_map.items():
                if k in base_map:
                    delta[k] = float(v - base_map[k])
            return delta

        log['lower_ratio_tuning'] = {
            'src_torso': float(src_torso),
            'trans_torso': float(trans_torso),
            'ratio_source': ratio_source,
            'ratios_2d': {k: float(v) for k, v in ratio_map_2d.items()},
            'ratios_3d': {k: float(v) for k, v in ratio_map_3d.items()},
            'ratios_used': {k: float(v) for k, v in ratio_map_used.items()},
            'deltas_vs_2d': _delta_map(ratio_map_2d, ratio_map_used)
        }

    def fine_tune_upper_ratio(
        self,
        t_kpts, t_scores,
        src_kpts, src_scores,
        ref_kpts, ref_scores,
        processed, log,
        source_depths=None,
        depth_z_scale: float = 1.0
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

        def _src_ratio_2d(p_name, c_name):
            p_idx = get_keypoint_index(p_name)
            c_idx = get_keypoint_index(c_name)
            if p_idx is None or c_idx is None:
                return None
            if src_scores[p_idx] <= 0.2 or src_scores[c_idx] <= 0.2:
                return None
            length = float(np.linalg.norm(src_kpts[c_idx] - src_kpts[p_idx]))
            return length / src_torso if src_torso > 1e-6 else None

        def _src_ratio_3d(p_name, c_name):
            if source_depths is None or depth_z_scale is None:
                return None
            p_idx = get_keypoint_index(p_name)
            c_idx = get_keypoint_index(c_name)
            if p_idx is None or c_idx is None:
                return None
            if src_scores[p_idx] <= 0.2 or src_scores[c_idx] <= 0.2:
                return None
            diff = src_kpts[c_idx] - src_kpts[p_idx]
            dz = (float(source_depths[c_idx]) - float(source_depths[p_idx])) * float(depth_z_scale)
            length = float(np.sqrt(diff[0] ** 2 + diff[1] ** 2 + dz ** 2))
            return length / src_torso if src_torso > 1e-6 else None

        ratio_map_2d = {}
        ratio_map_3d = {}
        ratio_map_used = {}
        ratio_pairs = {
            'shoulder_elbow': ('left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'),
            'elbow_wrist': ('left_elbow', 'left_wrist', 'right_elbow', 'right_wrist'),
        }

        for key, (lp, lc, rp, rc) in ratio_pairs.items():
            l_ratio_2d = _src_ratio_2d(lp, lc)
            r_ratio_2d = _src_ratio_2d(rp, rc)
            if l_ratio_2d is not None and r_ratio_2d is not None:
                ratio_map_2d[key] = (l_ratio_2d + r_ratio_2d) / 2.0
            elif l_ratio_2d is not None:
                ratio_map_2d[key] = l_ratio_2d
            elif r_ratio_2d is not None:
                ratio_map_2d[key] = r_ratio_2d

            l_ratio_3d = _src_ratio_3d(lp, lc)
            r_ratio_3d = _src_ratio_3d(rp, rc)
            if l_ratio_3d is not None and r_ratio_3d is not None:
                ratio_map_3d[key] = (l_ratio_3d + r_ratio_3d) / 2.0
            elif l_ratio_3d is not None:
                ratio_map_3d[key] = l_ratio_3d
            elif r_ratio_3d is not None:
                ratio_map_3d[key] = r_ratio_3d

        if source_depths is not None and len(ratio_map_3d) > 0:
            ratio_map_used = ratio_map_3d
            ratio_source = '3d'
        else:
            ratio_map_used = ratio_map_2d
            ratio_source = '2d'

        if not ratio_map_used:
            return

        def _apply_side(side_prefix):
            shoulder = f"{side_prefix}_shoulder"
            elbow = f"{side_prefix}_elbow"
            wrist = f"{side_prefix}_wrist"

            def _place(parent, child, ratio_key):
                if ratio_key not in ratio_map_used:
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
                length = ratio_map_used[ratio_key] * trans_torso
                t_kpts[c_idx] = t_kpts[p_idx] + ref_dir * length
                t_scores[c_idx] = 0.85
                processed.add(c_idx)

            _place(shoulder, elbow, 'shoulder_elbow')
            _place(elbow, wrist, 'elbow_wrist')

        _apply_side('left')
        _apply_side('right')

        def _delta_map(base_map, used_map):
            delta = {}
            for k, v in used_map.items():
                if k in base_map:
                    delta[k] = float(v - base_map[k])
            return delta

        log['upper_ratio_tuning'] = {
            'src_torso': float(src_torso),
            'trans_torso': float(trans_torso),
            'ratio_source': ratio_source,
            'ratios_2d': {k: float(v) for k, v in ratio_map_2d.items()},
            'ratios_3d': {k: float(v) for k, v in ratio_map_3d.items()},
            'ratios_used': {k: float(v) for k, v in ratio_map_used.items()},
            'deltas_vs_2d': _delta_map(ratio_map_2d, ratio_map_used)
        }

    def transfer_feet(
        self,
        trans_kpts: np.ndarray,
        trans_scores: np.ndarray,
        s_kpts: np.ndarray,
        s_scores: np.ndarray,
        corrected_lengths: dict,
        r_kpts: np.ndarray,
        r_scores: np.ndarray,
        scale: float,
        processed: set,
        log: dict = None
    ):
        """
        발 전이 - Toe Chain Strategy (Heel&BigToe from Ankle, SmallToe from BigToe)
        """
        print(f"\n============================================================")
        print(f"🦶 [DEBUG] transfer_feet() - Toe Chain (Big->Small Vector)")
        print(f"============================================================")

        for side in ['left', 'right']:
            ankle_name = f'{side}_ankle'
            big_toe_name = f'{side}_big_toe'
            small_toe_name = f'{side}_small_toe'
            heel_name = f'{side}_heel'
            
            ankle_idx = get_keypoint_index(ankle_name)
            big_idx = get_keypoint_index(big_toe_name)
            
            if trans_scores[ankle_idx] == 0: continue
            
            print(f"   [{side.upper()}] Foot from {ankle_name}")
            
            # ----------------------------------------------------------
            # 1. Heel & Big Toe (From Ankle)
            # ----------------------------------------------------------
            primary_parts = [heel_name, big_toe_name]
            
            for part_name in primary_parts:
                part_idx = get_keypoint_index(part_name)
                
                # 길이: Src 대칭 평균
                bone_name = f"{ankle_name}_{part_name}"
                target_length = corrected_lengths.get(bone_name)
                
                if target_length is None or target_length <= 0:
                    if r_scores[ankle_idx] > 0.1 and r_scores[part_idx] > 0.1:
                        dist = np.linalg.norm(r_kpts[part_idx] - r_kpts[ankle_idx])
                        target_length = dist * scale
                    else: continue

                # 방향: Ref (Ankle -> Part)
                if r_scores[ankle_idx] > 0.1 and r_scores[part_idx] > 0.1:
                    ref_vec = r_kpts[part_idx] - r_kpts[ankle_idx]
                    norm = np.linalg.norm(ref_vec)
                    direction = ref_vec / norm if norm > 1e-6 else np.array([0, 1])
                    
                    trans_kpts[part_idx] = trans_kpts[ankle_idx] + direction * target_length
                    trans_scores[part_idx] = min(trans_scores[ankle_idx], r_scores[part_idx])
                    processed.add(part_idx)
                    print(f"      ✅ {part_name}: From Ankle, Dir=Ref")
                else:
                    print(f"      ❌ {part_name}: Ref missing")

            # ----------------------------------------------------------
            # 2. Small Toe (From Big Toe) - 핵심 수정!
            # ----------------------------------------------------------
            # 엄지발가락이 전이되었는지 확인
            if trans_scores[big_idx] > 0:
                small_idx = get_keypoint_index(small_toe_name)
                
                # 길이: Src 대칭 평균 (Big -> Small)
                # _correct_bone_lengths에 추가한 bone_name 사용
                bone_name = f"{big_toe_name}_{small_toe_name}"
                target_length = corrected_lengths.get(bone_name)
                
                # Fallback length
                if target_length is None or target_length <= 0:
                    if r_scores[big_idx] > 0.1 and r_scores[small_idx] > 0.1:
                        dist = np.linalg.norm(r_kpts[small_idx] - r_kpts[big_idx])
                        target_length = dist * scale
                
                if target_length is not None and target_length > 0:
                    # 방향: Ref (Big -> Small)
                    # 엄지에서 새끼로 가는 벡터를 가져옴으로써 Ref의 발끝 각도를 복사함
                    if r_scores[big_idx] > 0.1 and r_scores[small_idx] > 0.1:
                        ref_vec = r_kpts[small_idx] - r_kpts[big_idx]
                        norm = np.linalg.norm(ref_vec)
                        direction = ref_vec / norm if norm > 1e-6 else np.array([0, 1])
                        
                        # 배치: Trans_Big_Toe + (Ref_Dir * Src_Len)
                        trans_kpts[small_idx] = trans_kpts[big_idx] + direction * target_length
                        trans_scores[small_idx] = min(trans_scores[big_idx], r_scores[small_idx])
                        processed.add(small_idx)
                        print(f"      ✅ {small_toe_name}: From BigToe, Dir=Ref (Toe Chain)")
                    else:
                        # Ref 정보 없으면 Ankle 기준 Fallback (기존 방식)
                        bone_name_alt = f"{ankle_name}_{small_toe_name}"
                        len_alt = corrected_lengths.get(bone_name_alt)
                        if len_alt and r_scores[ankle_idx] > 0.1 and r_scores[small_idx] > 0.1:
                             ref_vec = r_kpts[small_idx] - r_kpts[ankle_idx]
                             norm = np.linalg.norm(ref_vec)
                             direction = ref_vec / norm if norm > 1e-6 else np.array([0, 1])
                             trans_kpts[small_idx] = trans_kpts[ankle_idx] + direction * len_alt
                             trans_scores[small_idx] = r_scores[small_idx]
                             processed.add(small_idx)
                             print(f"      ⚠️ {small_toe_name}: Fallback to Ankle-based")
                        else:
                             print(f"      ❌ {small_toe_name}: Ref missing")
                else:
                     print(f"      ❌ {small_toe_name}: Length missing")