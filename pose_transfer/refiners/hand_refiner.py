"""pose_transfer.refiners.hand_refiner

손 키포인트(21점) 정밀화 모듈.

목표
- 손이 화면에서 너무 작게 잡히는 경우(손가락 마디가 몇 픽셀로 뭉개지는 경우)
    전체 프레임에서 한 번 추론한 결과만으로는 손 21점의 위치/점수 품질이 낮아질 수 있습니다.
- 이 모듈은 "손 주변 ROI(Region Of Interest)만" 잘라서 더 크게(업스케일) 만든 뒤,
    그 업스케일된 crop에서 포즈 추론을 한 번 더 수행하여 손 21점의 품질을 개선하려고 합니다.

핵심 아이디어(요약)
1) 손목/팔꿈치로 손이 있을 만한 ROI를 추정한다.
2) ROI의 크기가 작으면(min_hand_size 미만) ROI만 업스케일한다.
3) 업스케일된 ROI에서 extractor를 다시 돌려 손 21점을 얻는다.
4) 업스케일 좌표를 원본 좌표계로 되돌린다.
5) "정말 좋아졌을 때만"(유효 점 개수 증가) refined 결과를 채택한다.

주의/한계
- 업스케일은 새로운 디테일을 '생성'하는 것이 아니라 보간(interpolation)으로 픽셀 격자를 촘촘히 만드는 것입니다.
    다만 모델 입력 해상도 민감도 때문에 실제 추론 안정성이 좋아질 수 있습니다.
- ROI 추정이 어긋나면(crop에 손이 충분히 안 들어오면) 재추론이 오히려 나빠질 수 있으므로,
    이 코드는 보수적으로 "좋아졌을 때만" 교체합니다.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict

from ..extractors.keypoint_constants import (
    BODY_KEYPOINTS,
    LEFT_HAND_START_IDX,
    RIGHT_HAND_START_IDX
)
from ..utils.geometry import expand_bbox


class HandRefiner:
    """
        손 키포인트 정밀화기.

        입력
        - image: 원본 이미지(또는 프레임)
        - keypoints/scores: 전체 신체 133점(wholebody) 키포인트 + 신뢰도
        - extractor: wholebody 포즈 추출기(DWPoseExtractor 등)

        출력
        - (변경될 수도 있는) 전체 keypoints/scores
        - refinement_info: 좌/우 손 각각 정밀화가 수행/채택되었는지

        정책(중요)
        - ROI가 충분히 크면(=손이 충분히 크게 잡히면) 재추론을 하지 않습니다.
        - 재추론을 하더라도, refined 결과가 "더 좋아졌다"고 판단될 때만 결과를 교체합니다.
            여기서 '더 좋아졌다'의 기준은 현재 구현에서는 "confidence_threshold 초과 점의 개수 증가"입니다.
    """
    
    def __init__(
        self,
        min_hand_size: int = 48,
        max_scale_factor: float = 4.0,
        roi_expand_ratio: float = 1.5,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            min_hand_size: 최소 손 크기 (픽셀)
            max_scale_factor: 최대 업스케일 배율
            roi_expand_ratio: ROI 확장 비율
            confidence_threshold: 유효 키포인트 판단 임계값
        """
        self.min_hand_size = min_hand_size
        self.max_scale_factor = max_scale_factor
        self.roi_expand_ratio = roi_expand_ratio
        self.confidence_threshold = confidence_threshold
    
    def estimate_hand_roi(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        is_left: bool,
        image_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """손 영역(ROI) 추정.

        ROI(Region Of Interest)란, 전체 이미지 중에서 "관심 있는 부분"만 잘라내기 위한 사각형 영역입니다.

        이 구현의 ROI 추정 로직
        - 손목이 확실히 잡혔을 때만(score >= confidence_threshold) ROI를 만듭니다.
        - 팔꿈치가 잡혔으면(=팔 방향을 신뢰할 수 있으면) 팔꿈치->손목 방향으로 ROI 중심을 약간 이동해
          손바닥/손가락이 ROI에 더 잘 포함되도록 합니다.
        - ROI 크기는 하완(팔꿈치~손목) 길이로 손 크기를 근사하고, roi_expand_ratio로 여유를 줍니다.

        Args:
            keypoints: (K, 2) 전체 키포인트
            scores: (K,) 키포인트 신뢰도
            is_left: 왼손 여부
            image_shape: (H, W) 이미지 크기

        Returns:
            (x1, y1, x2, y2) 바운딩 박스 또는 None
        """
        img_h, img_w = image_shape
        
        # 손목과 팔꿈치 인덱스
        wrist_name = 'left_wrist' if is_left else 'right_wrist'
        elbow_name = 'left_elbow' if is_left else 'right_elbow'
        
        wrist_idx = BODY_KEYPOINTS[wrist_name]
        elbow_idx = BODY_KEYPOINTS[elbow_name]
        
        # 손목 신뢰도 확인:
        # - 손 ROI의 기준점이 손목이므로, 손목이 불확실하면 ROI 추정 자체를 포기합니다.
        if scores[wrist_idx] < self.confidence_threshold:
            return None
        
        wrist = keypoints[wrist_idx]
        
        # 팔 방향 계산 (팔꿈치 -> 손목)
        # - 팔꿈치가 신뢰 가능하면 하완 길이를 이용해 손 크기를 추정합니다.
        if scores[elbow_idx] > self.confidence_threshold:
            elbow = keypoints[elbow_idx]
            forearm_vec = wrist - elbow
            forearm_length = np.linalg.norm(forearm_vec)
            
            # 손 크기 추정(heuristic): 하완 길이의 약 40%
            # - 손바닥~손가락 길이가 하완 길이의 일정 비율이라는 가정.
            # - 이 값은 데이터/모델에 따라 튜닝 포인트가 될 수 있습니다.
            hand_size = forearm_length * 0.4
        else:
            # 팔꿈치가 없거나 불확실하면 방향/길이 추정이 불가능하므로
            # 최소 크기(min_hand_size) 기반으로 보수적인 기본값을 사용합니다.
            hand_size = self.min_hand_size * 1.5
        
        # 손 방향으로 ROI 중심 이동
        # - ROI를 손목에 딱 맞추면, 손바닥/손가락 일부가 박스 밖으로 나갈 수 있어
        #   팔 방향(손목 바깥쪽)으로 절반 정도 이동시켜 손 전체를 포함시키려 합니다.
        if scores[elbow_idx] > self.confidence_threshold:
            direction = forearm_vec / (forearm_length + 1e-6)
            roi_center = wrist + direction * (hand_size * 0.5)
        else:
            roi_center = wrist
        
        # 바운딩 박스 생성
        # - roi_expand_ratio로 여유를 주어 손가락이 박스 경계에 걸리지 않도록 합니다.
        half_size = hand_size * self.roi_expand_ratio / 2
        
        x1 = max(0, int(roi_center[0] - half_size))
        y1 = max(0, int(roi_center[1] - half_size))
        x2 = min(img_w, int(roi_center[0] + half_size))
        y2 = min(img_h, int(roi_center[1] + half_size))
        
        # 유효한 ROI인지 확인
        # - 너무 작은 박스는 crop/upscale 가치가 없고 수치적으로도 불안정하므로 None 처리합니다.
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        
        return (x1, y1, x2, y2)
    
    def check_needs_upscale(
        self,
        roi: Tuple[int, int, int, int]
    ) -> Tuple[bool, float]:
        """업스케일 필요 여부 확인.

        해석
        - 여기서 업스케일은 "ROI crop 이미지의 해상도(픽셀 수)를 늘리는 것"을 의미합니다.
          (원본 전체 이미지를 키우는 것이 아니라, 손 주변 ROI만 국소적으로 리사이즈)

        Returns:
            (needs_upscale, scale_factor)
            - needs_upscale: ROI가 min_hand_size보다 작으면 True
            - scale_factor: (min_hand_size / roi_size) 이되, max_scale_factor로 상한을 둠
        """
        x1, y1, x2, y2 = roi
        roi_size = min(x2 - x1, y2 - y1)
        
        if roi_size >= self.min_hand_size:
            return False, 1.0
        
        scale_factor = self.min_hand_size / roi_size
        scale_factor = min(scale_factor, self.max_scale_factor)
        
        return True, scale_factor
    
    def crop_and_upscale(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int],
        scale_factor: float
    ) -> Tuple[np.ndarray, Dict]:
        """ROI 크롭 및 업스케일.

        처리
        - image에서 ROI 영역만 crop합니다.
        - crop을 scale_factor 배로 리사이즈(cv2.resize) 합니다.

        Returns:
            upscaled_crop: 업스케일된 crop 이미지
            transform_info: 업스케일 좌표계를 원본 좌표계로 되돌리기 위한 정보
                - roi: 원본 이미지에서 crop한 영역
                - scale_factor: 리사이즈 배율
                - offset: (x1, y1) 원본 이미지에서 crop의 좌상단 오프셋
        """
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        
        new_h = int((y2 - y1) * scale_factor)
        new_w = int((x2 - x1) * scale_factor)
        
        upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        transform_info = {
            'roi': roi,
            'scale_factor': scale_factor,
            'offset': (x1, y1)
        }
        
        return upscaled, transform_info
    
    def transform_keypoints_back(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        transform_info: Dict,
        is_left: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """업스케일 좌표를 원본 좌표로 역변환.

        업스케일 crop 좌표계에서 얻은 손 키포인트(21,2)를 원본 이미지 좌표계로 복원합니다.

        변환식
        - crop 좌표 -> 원본 좌표:
          1) 업스케일을 되돌림: (x, y) / scale_factor
          2) 원본에서 crop 위치를 더함: + (offset_x, offset_y)

        Args:
            keypoints: (21, 2) 손 키포인트 (업스케일 이미지 기준)
            scores: (21,) 신뢰도
            transform_info: 변환 정보
            is_left: 왼손 여부

        Returns:
            원본 좌표계의 키포인트, 신뢰도
        """
        scale = transform_info['scale_factor']
        offset_x, offset_y = transform_info['offset']
        
        # 스케일 역변환 + 오프셋 적용
        original_kpts = keypoints / scale
        original_kpts[:, 0] += offset_x
        original_kpts[:, 1] += offset_y
        
        return original_kpts, scores
    
    def refine_hand(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        extractor,  # DWPoseExtractor
        is_left: bool
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """단일 손(좌/우) 키포인트 정밀화.

        전체 흐름
        1) 현재 wholebody 결과에서 손 21점(original)을 가져옴
        2) 손 ROI를 추정
        3) ROI가 작으면 ROI만 crop & upscale
        4) 업스케일 crop에서 extractor를 다시 돌려 wholebody를 얻고, 그 중 손 21점을 사용
        5) 업스케일 좌표를 원본 좌표계로 복원
        6) "정말 좋아졌을 때만" refined로 교체

        왜 wholebody extractor를 다시 돌리나?
        - 현재 구현은 손 전용 모델이 아니라 기존 extractor.extract(...)를 그대로 사용합니다.
          즉, 업스케일 crop에서도 wholebody를 뽑되, 그 중 손 부분만 가져옵니다.

        채택 기준(현재 구현)
        - confidence_threshold 초과 점의 개수가 original보다 증가하면 refined를 채택합니다.
        - 개수가 같거나 줄면 original 유지(보수적).

        Args:
            image: 원본 이미지
            keypoints: (K, 2) 전체 키포인트
            scores: (K,) 신뢰도
            extractor: DWPose 추출기
            is_left: 왼손 여부

        Returns:
            refined_keypoints: 정밀화된 손 키포인트 (21, 2)
            refined_scores: 정밀화된 신뢰도 (21,)
            was_refined: 정밀화 수행 여부
        """
        hand_start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
        
        # 기존 손 키포인트(wholebody 결과에서 손 21점을 슬라이스)
        original_hand_kpts = keypoints[hand_start:hand_start + 21]
        original_hand_scores = scores[hand_start:hand_start + 21]
        
        # ROI 추정: 손이 작을 때만 의미가 있으므로, ROI가 없으면 즉시 종료
        img_h, img_w = image.shape[:2]
        roi = self.estimate_hand_roi(keypoints, scores, is_left, (img_h, img_w))
        
        if roi is None:
            return original_hand_kpts, original_hand_scores, False
        
        # 업스케일 필요 여부 확인
        # - ROI가 충분히 크면 재추론을 하지 않습니다(속도/안정성).
        needs_upscale, scale_factor = self.check_needs_upscale(roi)
        
        if not needs_upscale:
            return original_hand_kpts, original_hand_scores, False
        
        # 크롭 및 업스케일
        # - image에서 ROI 영역만 잘라내고, 그 crop을 scale_factor 배로 키웁니다.
        upscaled_crop, transform_info = self.crop_and_upscale(
            image, roi, scale_factor
        )
        
        # 업스케일 이미지에서 재추출
        # - crop을 키운 상태에서 extractor를 다시 실행하면, 작은 손 디테일이 더 잘 잡힐 수 있습니다.
        try:
            all_kpts, all_scores = extractor.extract(upscaled_crop)
            
            if len(all_kpts) == 0:
                return original_hand_kpts, original_hand_scores, False
            
            # 첫 번째 인물의 손 키포인트
            # - 주의: upscaled crop에서도 사람 검출이 여러 명일 수 있으나,
            #   현재 구현은 단순히 첫 번째 인물(all_kpts[0])을 사용합니다.
            new_hand_kpts = all_kpts[0][hand_start:hand_start + 21]
            new_hand_scores = all_scores[0][hand_start:hand_start + 21]
            
            # 원본 좌표로 역변환
            # - upscaled crop 좌표계를 원본 이미지 좌표계로 되돌립니다.
            refined_kpts, refined_scores = self.transform_keypoints_back(
                new_hand_kpts, new_hand_scores, transform_info, is_left
            )
            
            # 새 키포인트가 더 좋은지 확인
            # - "유효 점"의 개수가 늘어났을 때만 refined를 채택합니다.
            # - 유효 점의 개수는 confidence_threshold를 기준으로 셉니다.
            original_valid = np.sum(original_hand_scores > self.confidence_threshold)
            refined_valid = np.sum(refined_scores > self.confidence_threshold)
            
            if refined_valid > original_valid:
                return refined_kpts, refined_scores, True
            else:
                return original_hand_kpts, original_hand_scores, False
                
        except Exception as e:
            print(f"Hand refinement failed: {e}")
            return original_hand_kpts, original_hand_scores, False
    
    def refine_both_hands(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        extractor
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, bool]]:
        """양손(좌/우) 키포인트 정밀화.

        - 왼손/오른손 각각에 대해 refine_hand를 호출합니다.
        - 각 손별로 refined가 채택된 경우에만 전체 keypoints/scores 배열에 덮어씁니다.
        
        Returns:
            keypoints: 정밀화된 전체 키포인트
            scores: 정밀화된 신뢰도
            refinement_info: {'left': bool, 'right': bool}
        """
        result_kpts = keypoints.copy()
        result_scores = scores.copy()
        refinement_info = {'left': False, 'right': False}
        
        # 왼손 정밀화
        left_kpts, left_scores, left_refined = self.refine_hand(
            image, keypoints, scores, extractor, is_left=True
        )
        if left_refined:
            result_kpts[LEFT_HAND_START_IDX:LEFT_HAND_START_IDX + 21] = left_kpts
            result_scores[LEFT_HAND_START_IDX:LEFT_HAND_START_IDX + 21] = left_scores
            refinement_info['left'] = True
        
        # 오른손 정밀화
        right_kpts, right_scores, right_refined = self.refine_hand(
            image, keypoints, scores, extractor, is_left=False
        )
        if right_refined:
            result_kpts[RIGHT_HAND_START_IDX:RIGHT_HAND_START_IDX + 21] = right_kpts
            result_scores[RIGHT_HAND_START_IDX:RIGHT_HAND_START_IDX + 21] = right_scores
            refinement_info['right'] = True
        
        return result_kpts, result_scores, refinement_info
