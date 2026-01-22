"""
Keypoint Generator Module

src에 없는 키포인트를 해부학적 기준과 좌우대칭을 활용하여 생성하는 모듈
"""
import numpy as np
from typing import Tuple
from ..extractors.keypoint_constants import (
    BODY_KEYPOINTS, FEET_KEYPOINTS,
    LEFT_HAND_START_IDX, LEFT_HAND_END_IDX,
    RIGHT_HAND_START_IDX, RIGHT_HAND_END_IDX
)


class KeypointGenerator:
    """누락된 키포인트를 해부학적 기준과 좌우대칭으로 생성"""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.threshold = confidence_threshold
        
        # 좌우 대칭 쌍 정의
        self.symmetry_pairs_body = [
            ('left_eye', 'right_eye'),
            ('left_ear', 'right_ear'),
            ('left_shoulder', 'right_shoulder'),
            ('left_elbow', 'right_elbow'),
            ('left_wrist', 'right_wrist'),
            ('left_hip', 'right_hip'),
            ('left_knee', 'right_knee'),
            ('left_ankle', 'right_ankle'),
        ]
        
        self.symmetry_pairs_feet = [
            ('left_big_toe', 'right_big_toe'),
            ('left_small_toe', 'right_small_toe'),
            ('left_heel', 'right_heel'),
        ]
    
    def generate_missing_keypoints(
        self, 
        src_kpts: np.ndarray, 
        src_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ref에는 있지만 src에 없는 키포인트를 생성
        
        우선순위:
        1. 해부학적 관계 (부모-자식, 뼈 길이 비율 등)
        2. 좌우대칭 (한쪽이 있으면 반대편으로 미러링)
        
        Args:
            src_kpts: source keypoints (133, 2)
            src_scores: source scores (133,)
            ref_kpts: reference keypoints (133, 2)
            ref_scores: reference scores (133,)
        
        Returns:
            enhanced_kpts: 보강된 keypoints
            enhanced_scores: 보강된 scores
        """
        enhanced_kpts = src_kpts.copy()
        enhanced_scores = src_scores.copy()
        
        print("\n🔧 [Keypoint Generator] 누락된 키포인트 생성 중...")
        
        # ref에 있지만 src에 없는 키포인트 찾기
        missing_mask = (ref_scores > self.threshold) & (src_scores <= self.threshold)
        missing_indices = np.where(missing_mask)[0]
        
        if len(missing_indices) == 0:
            print("   ✅ 누락된 키포인트 없음")
            return enhanced_kpts, enhanced_scores
        
        print(f"   📊 누락된 키포인트 {len(missing_indices)}개 발견")
        
        # 스케일 비율 계산 (ref 대비 src의 크기)
        scale_ratio = self._calculate_scale_ratio(enhanced_kpts, enhanced_scores, ref_kpts, ref_scores)
        print(f"   📏 Ref → Src 스케일 비율: {scale_ratio:.3f}")
        
        generated_count = 0
        
        # 1단계: 해부학적 관계 기반 생성
        generated_count += self._generate_anatomical(
            enhanced_kpts, enhanced_scores, missing_indices, ref_scores
        )
        
        # 2단계: 좌우대칭 기반 생성
        generated_count += self._generate_symmetrical(
            enhanced_kpts, enhanced_scores, missing_indices, ref_scores, scale_ratio
        )
        
        # 3단계: Hand keypoints - 국소 스케일 적용
        print("\n   🖐️ 손 키포인트 생성 (국소 비율 기반)...")
        generated_count += self._generate_hand_keypoints_with_local_scale(
            enhanced_kpts, enhanced_scores, missing_indices, ref_kpts, ref_scores, scale_ratio
        )
        
        # 4단계: Feet keypoints - 발목 기준으로 생성 (스케일 적용)
        generated_count += self._generate_feet_keypoints(
            enhanced_kpts, enhanced_scores, missing_indices, ref_kpts, ref_scores, scale_ratio
        )
        
        print(f"   ✅ 총 {generated_count}개 키포인트 생성 완료")
        
        return enhanced_kpts, enhanced_scores
    
    def _calculate_scale_ratio(
        self,
        src_kpts: np.ndarray,
        src_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray
    ) -> float:
        """
        ref 대비 src의 크기 비율 계산 (전체 body 기준)
        우선순위: 어깨 너비 > 골반 너비 > 1.0(기본값)
        
        ⚠️ 주의: 이 스케일은 손/발 같은 국소 부위에는 부적합할 수 있음
        """
        left_shoulder = BODY_KEYPOINTS['left_shoulder']
        right_shoulder = BODY_KEYPOINTS['right_shoulder']
        left_hip = BODY_KEYPOINTS['left_hip']
        right_hip = BODY_KEYPOINTS['right_hip']
        
        # 1순위: 어깨 너비
        if (src_scores[left_shoulder] > self.threshold and src_scores[right_shoulder] > self.threshold and
            ref_scores[left_shoulder] > self.threshold and ref_scores[right_shoulder] > self.threshold):
            
            src_shoulder_width = np.linalg.norm(src_kpts[left_shoulder] - src_kpts[right_shoulder])
            ref_shoulder_width = np.linalg.norm(ref_kpts[left_shoulder] - ref_kpts[right_shoulder])
            
            if ref_shoulder_width > 1e-6:
                ratio = src_shoulder_width / ref_shoulder_width
                print(f"      📐 어깨 너비 기준: src={src_shoulder_width:.1f}px, ref={ref_shoulder_width:.1f}px")
                return ratio
        
        # 2순위: 골반 너비
        if (src_scores[left_hip] > self.threshold and src_scores[right_hip] > self.threshold and
            ref_scores[left_hip] > self.threshold and ref_scores[right_hip] > self.threshold):
            
            src_hip_width = np.linalg.norm(src_kpts[left_hip] - src_kpts[right_hip])
            ref_hip_width = np.linalg.norm(ref_kpts[left_hip] - ref_kpts[right_hip])
            
            if ref_hip_width > 1e-6:
                ratio = src_hip_width / ref_hip_width
                print(f"      📐 골반 너비 기준: src={src_hip_width:.1f}px, ref={ref_hip_width:.1f}px")
                return ratio
        
        # 기본값
        print(f"      ⚠️ 스케일 기준점 없음, 기본값 1.0 사용")
        return 1.0
    
    def _calculate_local_scale_for_hand(
        self,
        src_kpts: np.ndarray,
        src_scores: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        is_left: bool,
        body_scale: float
    ) -> float:
        """
        손에 특화된 국소 스케일 계산
        
        전략:
        1. src 대칭 손(있는 쪽)의 body 대비 비율 계산
        2. ref 손(생성할 쪽)의 body 대비 비율 계산
        3. 두 비율을 조합하여 최종 스케일 도출
        
        Args:
            is_left: True면 왼손, False면 오른손
            body_scale: 전체 body 스케일 (fallback)
        
        Returns:
            해당 손에 특화된 스케일
        """
        # 손목과 어깨 인덱스
        if is_left:
            wrist_idx = BODY_KEYPOINTS['left_wrist']
            shoulder_idx = BODY_KEYPOINTS['left_shoulder']
            opposite_wrist = BODY_KEYPOINTS['right_wrist']
            opposite_shoulder = BODY_KEYPOINTS['right_shoulder']
            hand_start = LEFT_HAND_START_IDX
            opposite_hand_start = RIGHT_HAND_START_IDX
        else:
            wrist_idx = BODY_KEYPOINTS['right_wrist']
            shoulder_idx = BODY_KEYPOINTS['right_shoulder']
            opposite_wrist = BODY_KEYPOINTS['left_wrist']
            opposite_shoulder = BODY_KEYPOINTS['left_shoulder']
            hand_start = RIGHT_HAND_START_IDX
            opposite_hand_start = LEFT_HAND_START_IDX
        
        # src 대칭 손(반대쪽)이 있는지 확인
        src_opposite_hand_count = sum(1 for i in range(opposite_hand_start, opposite_hand_start + 21) 
                                     if i < len(src_scores) and src_scores[i] > self.threshold)
        
        if src_opposite_hand_count > 5 and src_scores[opposite_wrist] > self.threshold and src_scores[opposite_shoulder] > self.threshold:
            # src 대칭 손이 있음 - 그 비율 사용
            # src 대칭 손의 평균 크기 (손목 → 손가락들)
            src_opposite_hand_vectors = []
            for i in range(opposite_hand_start, opposite_hand_start + 21):
                if i < len(src_scores) and src_scores[i] > self.threshold:
                    vec = src_kpts[i] - src_kpts[opposite_wrist]
                    src_opposite_hand_vectors.append(np.linalg.norm(vec))
            
            if len(src_opposite_hand_vectors) > 0:
                src_hand_avg_length = np.mean(src_opposite_hand_vectors)
                # src 팔 길이 (어깨 → 손목)
                src_arm_length = np.linalg.norm(src_kpts[opposite_wrist] - src_kpts[opposite_shoulder])
                
                if src_arm_length > 1e-6:
                    # src 손/팔 비율
                    src_hand_arm_ratio = src_hand_avg_length / src_arm_length
                    
                    # ref도 동일하게 계산
                    if ref_scores[wrist_idx] > self.threshold and ref_scores[shoulder_idx] > self.threshold:
                        ref_arm_length = np.linalg.norm(ref_kpts[wrist_idx] - ref_kpts[shoulder_idx])
                        
                        # ref 손 생성 목표 크기 = ref_arm × src_hand_arm_ratio × body_scale
                        target_hand_length = ref_arm_length * src_hand_arm_ratio * body_scale
                        
                        # ref 손의 원래 평균 크기
                        ref_hand_vectors = []
                        for i in range(hand_start, hand_start + 21):
                            if i < len(ref_scores) and ref_scores[i] > self.threshold:
                                vec = ref_kpts[i] - ref_kpts[wrist_idx]
                                ref_hand_vectors.append(np.linalg.norm(vec))
                        
                        if len(ref_hand_vectors) > 0:
                            ref_hand_avg_length = np.mean(ref_hand_vectors)
                            if ref_hand_avg_length > 1e-6:
                                local_scale = target_hand_length / ref_hand_avg_length
                                side = "왼손" if is_left else "오른손"
                                print(f"      🖐️ {side} 국소 스케일: {local_scale:.3f}")
                                print(f"         src 손/팔 비율: {src_hand_arm_ratio:.3f}")
                                print(f"         ref 팔 길이: {ref_arm_length:.1f}px")
                                print(f"         목표 손 크기: {target_hand_length:.1f}px")
                                print(f"         ref 손 원본: {ref_hand_avg_length:.1f}px")
                                return local_scale
        
        # fallback: body scale 사용
        print(f"      ⚠️ 대칭 손 없음, body_scale={body_scale:.3f} 사용")
        return body_scale
    
    def _generate_anatomical(
        self, 
        kpts: np.ndarray, 
        scores: np.ndarray, 
        missing_indices: np.ndarray,
        ref_scores: np.ndarray
    ) -> int:
        """해부학적 관계 기반으로 키포인트 생성"""
        generated = 0
        
        # 정의: (missing_kpt, parent, child, ratio)
        # missing = parent + (child - parent) * ratio
        anatomical_rules = [
            # 팔꿈치 = 어깨 + (손목 - 어깨) * 0.5
            ('left_elbow', 'left_shoulder', 'left_wrist', 0.5),
            ('right_elbow', 'right_shoulder', 'right_wrist', 0.5),
            
            # 무릎 = 골반 + (발목 - 골반) * 0.5
            ('left_knee', 'left_hip', 'left_ankle', 0.5),
            ('right_knee', 'right_hip', 'right_ankle', 0.5),
            
            # 손목 = 팔꿈치 + (팔꿈치 - 어깨) * 1.0 (팔 연장)
            ('left_wrist', 'left_shoulder', 'left_elbow', 2.0),
            ('right_wrist', 'right_shoulder', 'right_elbow', 2.0),
            
            # 발목 = 무릎 + (무릎 - 골반) * 1.0 (다리 연장)
            ('left_ankle', 'left_hip', 'left_knee', 2.0),
            ('right_ankle', 'right_hip', 'right_knee', 2.0),
        ]
        
        for missing_name, parent_name, child_name, ratio in anatomical_rules:
            missing_idx = BODY_KEYPOINTS.get(missing_name)
            parent_idx = BODY_KEYPOINTS.get(parent_name)
            child_idx = BODY_KEYPOINTS.get(child_name)
            
            if missing_idx is None or parent_idx is None or child_idx is None:
                continue
            
            # missing_idx가 누락되었고, parent와 child가 있는 경우
            if missing_idx in missing_indices and \
               scores[parent_idx] > self.threshold and \
               scores[child_idx] > self.threshold:
                
                # 해부학적 관계로 계산
                parent_pt = kpts[parent_idx]
                child_pt = kpts[child_idx]
                estimated_pt = parent_pt + (child_pt - parent_pt) * ratio
                
                kpts[missing_idx] = estimated_pt
                scores[missing_idx] = min(scores[parent_idx], scores[child_idx]) * 0.9
                generated += 1
                
                print(f"      ✨ {missing_name} 생성 (해부학적: {parent_name} → {child_name})")
        
        return generated
    
    def _generate_symmetrical(
        self, 
        kpts: np.ndarray, 
        scores: np.ndarray, 
        missing_indices: np.ndarray,
        ref_scores: np.ndarray,
        scale_ratio: float = 1.0
    ) -> int:
        """좌우대칭 기반으로 키포인트 생성"""
        generated = 0
        
        # Body 대칭 쌍
        for left_name, right_name in self.symmetry_pairs_body:
            left_idx = BODY_KEYPOINTS[left_name]
            right_idx = BODY_KEYPOINTS[right_name]
            
            generated += self._mirror_keypoint(
                kpts, scores, missing_indices, left_idx, right_idx, left_name, right_name
            )
        
        # Feet 대칭 쌍
        for left_name, right_name in self.symmetry_pairs_feet:
            left_idx = FEET_KEYPOINTS[left_name]
            right_idx = FEET_KEYPOINTS[right_name]
            
            generated += self._mirror_keypoint(
                kpts, scores, missing_indices, left_idx, right_idx, left_name, right_name
            )
        
        return generated
    
    def _mirror_keypoint(
        self,
        kpts: np.ndarray,
        scores: np.ndarray,
        missing_indices: np.ndarray,
        left_idx: int,
        right_idx: int,
        left_name: str,
        right_name: str
    ) -> int:
        """좌우 미러링으로 키포인트 생성"""
        generated = 0
        
        # 중심선 계산 (어깨 중점 또는 골반 중점)
        left_shoulder = BODY_KEYPOINTS['left_shoulder']
        right_shoulder = BODY_KEYPOINTS['right_shoulder']
        
        if scores[left_shoulder] > self.threshold and scores[right_shoulder] > self.threshold:
            center_x = (kpts[left_shoulder][0] + kpts[right_shoulder][0]) / 2
        else:
            # 어깨가 없으면 골반 사용
            left_hip = BODY_KEYPOINTS['left_hip']
            right_hip = BODY_KEYPOINTS['right_hip']
            if scores[left_hip] > self.threshold and scores[right_hip] > self.threshold:
                center_x = (kpts[left_hip][0] + kpts[right_hip][0]) / 2
            else:
                return 0  # 중심선을 찾을 수 없음
        
        # 왼쪽이 있고 오른쪽이 없으면
        if left_idx in missing_indices and scores[right_idx] > self.threshold:
            mirrored_x = 2 * center_x - kpts[right_idx][0]
            kpts[left_idx] = [mirrored_x, kpts[right_idx][1]]
            scores[left_idx] = scores[right_idx] * 0.8  # 신뢰도는 조금 낮춤
            generated += 1
            print(f"      ✨ {left_name} 생성 (대칭: {right_name} 미러링)")
        
        # 오른쪽이 있고 왼쪽이 없으면
        elif right_idx in missing_indices and scores[left_idx] > self.threshold:
            mirrored_x = 2 * center_x - kpts[left_idx][0]
            kpts[right_idx] = [mirrored_x, kpts[left_idx][1]]
            scores[right_idx] = scores[left_idx] * 0.8
            generated += 1
            print(f"      ✨ {right_name} 생성 (대칭: {left_name} 미러링)")
        
        return generated
    
    def _generate_hand_keypoints_with_local_scale(
        self,
        kpts: np.ndarray,
        scores: np.ndarray,
        missing_indices: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        body_scale: float
    ) -> int:
        """손 키포인트 생성 (국소 비율 기반)"""
        generated = 0
        
        # 왼손
        left_wrist_idx = BODY_KEYPOINTS['left_wrist']
        if scores[left_wrist_idx] > self.threshold:
            left_hand_missing = [idx for idx in range(LEFT_HAND_START_IDX, LEFT_HAND_END_IDX + 1) 
                                if idx in missing_indices]
            if len(left_hand_missing) > 0:
                local_scale = self._calculate_local_scale_for_hand(
                    kpts, scores, ref_kpts, ref_scores, is_left=True, body_scale=body_scale
                )
                generated += self._generate_hand_from_wrist(
                    kpts, scores, missing_indices, ref_kpts, ref_scores,
                    left_wrist_idx, LEFT_HAND_START_IDX, LEFT_HAND_END_IDX, "왼손", local_scale
                )
        
        # 오른손
        right_wrist_idx = BODY_KEYPOINTS['right_wrist']
        if scores[right_wrist_idx] > self.threshold:
            right_hand_missing = [idx for idx in range(RIGHT_HAND_START_IDX, RIGHT_HAND_END_IDX + 1) 
                                 if idx in missing_indices]
            if len(right_hand_missing) > 0:
                local_scale = self._calculate_local_scale_for_hand(
                    kpts, scores, ref_kpts, ref_scores, is_left=False, body_scale=body_scale
                )
                generated += self._generate_hand_from_wrist(
                    kpts, scores, missing_indices, ref_kpts, ref_scores,
                    right_wrist_idx, RIGHT_HAND_START_IDX, RIGHT_HAND_END_IDX, "오른손", local_scale
                )
        
        return generated
    
    def _generate_hand_keypoints(
        self,
        kpts: np.ndarray,
        scores: np.ndarray,
        missing_indices: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        scale_ratio: float = 1.0
    ) -> int:
        """손 키포인트 생성 (손목 위치 기준, 스케일 적용)"""
        generated = 0
        
        # 왼손
        left_wrist_idx = BODY_KEYPOINTS['left_wrist']
        if scores[left_wrist_idx] > self.threshold:
            generated += self._generate_hand_from_wrist(
                kpts, scores, missing_indices, ref_kpts, ref_scores,
                left_wrist_idx, LEFT_HAND_START_IDX, LEFT_HAND_END_IDX, "왼손", scale_ratio
            )
        
        # 오른손
        right_wrist_idx = BODY_KEYPOINTS['right_wrist']
        if scores[right_wrist_idx] > self.threshold:
            generated += self._generate_hand_from_wrist(
                kpts, scores, missing_indices, ref_kpts, ref_scores,
                right_wrist_idx, RIGHT_HAND_START_IDX, RIGHT_HAND_END_IDX, "오른손", scale_ratio
            )
        
        return generated
    
    def _generate_hand_from_wrist(
        self,
        kpts: np.ndarray,
        scores: np.ndarray,
        missing_indices: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        wrist_idx: int,
        hand_start: int,
        hand_end: int,
        hand_name: str,
        scale_ratio: float = 1.0
    ) -> int:
        """손목 기준으로 손 키포인트 생성 (스케일 비율 적용)"""
        generated = 0
        wrist_pos = kpts[wrist_idx]
        
        # ref의 손목 위치
        ref_wrist_pos = ref_kpts[wrist_idx]
        
        # ref에서 손목 대비 손가락의 상대 위치 계산
        for hand_idx in range(hand_start, hand_end + 1):
            if hand_idx in missing_indices and ref_scores[hand_idx] > self.threshold:
                # ref에서의 상대 오프셋
                ref_offset = ref_kpts[hand_idx] - ref_wrist_pos
                
                # 스케일 비율 적용 (ref 크기 → src 크기)
                scaled_offset = ref_offset * scale_ratio
                
                # src 손목에 스케일 조정된 오프셋 적용
                kpts[hand_idx] = wrist_pos + scaled_offset
                scores[hand_idx] = scores[wrist_idx] * 0.7  # 손목보다 낮은 신뢰도
                generated += 1
        
        if generated > 0:
            print(f"      ✨ {hand_name} 키포인트 {generated}개 생성 (손목 기준, scale={scale_ratio:.2f})")
        
        return generated
    
    def _generate_feet_keypoints(
        self,
        kpts: np.ndarray,
        scores: np.ndarray,
        missing_indices: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        scale_ratio: float = 1.0
    ) -> int:
        """발 키포인트 생성 (발목 위치 기준, 스케일 적용)"""
        generated = 0
        
        # 왼발
        left_ankle_idx = BODY_KEYPOINTS['left_ankle']
        if scores[left_ankle_idx] > self.threshold:
            generated += self._generate_foot_from_ankle(
                kpts, scores, missing_indices, ref_kpts, ref_scores,
                left_ankle_idx, "왼발", scale_ratio
            )
        
        # 오른발
        right_ankle_idx = BODY_KEYPOINTS['right_ankle']
        if scores[right_ankle_idx] > self.threshold:
            generated += self._generate_foot_from_ankle(
                kpts, scores, missing_indices, ref_kpts, ref_scores,
                right_ankle_idx, "오른발", scale_ratio
            )
        
        return generated
    
    def _generate_foot_from_ankle(
        self,
        kpts: np.ndarray,
        scores: np.ndarray,
        missing_indices: np.ndarray,
        ref_kpts: np.ndarray,
        ref_scores: np.ndarray,
        ankle_idx: int,
        foot_name: str,
        scale_ratio: float = 1.0
    ) -> int:
        """발목 기준으로 발 키포인트 생성 (스케일 비율 적용)"""
        generated = 0
        ankle_pos = kpts[ankle_idx]
        ref_ankle_pos = ref_kpts[ankle_idx]
        
        # 발목에 연결된 발 키포인트 찾기
        if "왼" in foot_name:
            foot_indices = [
                FEET_KEYPOINTS['left_big_toe'],
                FEET_KEYPOINTS['left_small_toe'],
                FEET_KEYPOINTS['left_heel']
            ]
        else:
            foot_indices = [
                FEET_KEYPOINTS['right_big_toe'],
                FEET_KEYPOINTS['right_small_toe'],
                FEET_KEYPOINTS['right_heel']
            ]
        
        for foot_idx in foot_indices:
            if foot_idx in missing_indices and ref_scores[foot_idx] > self.threshold:
                # ref에서의 상대 오프셋
                ref_offset = ref_kpts[foot_idx] - ref_ankle_pos
                
                # 스케일 비율 적용
                scaled_offset = ref_offset * scale_ratio
                
                # src 발목에 스케일 조정된 오프셋 적용
                kpts[foot_idx] = ankle_pos + scaled_offset
                scores[foot_idx] = scores[ankle_idx] * 0.7
                generated += 1
        
        if generated > 0:
            print(f"      ✨ {foot_name} 키포인트 {generated}개 생성 (발목 기준, scale={scale_ratio:.2f})")
        
        return generated
