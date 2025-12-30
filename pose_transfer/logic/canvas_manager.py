"""CanvasManager: 이미지 캔버스(바탕) 크기 보정 유틸.

이 모듈의 역할
    - 포즈 전이/정렬이 끝난 후, 스켈레톤 키포인트가 이미지 경계 밖으로 나가거나
        머리/상단 패딩(head padding) 때문에 위쪽 공간이 더 필요해지는 경우가 생깁니다.
    - 이런 상황에서 "이미지를 확장(padding)"하거나, 반대로 "불필요한 여백을 크롭"하여
        최종 출력 캔버스와 키포인트 좌표계를 일관되게 맞춥니다.

핵심 원칙
    - 이미지가 확장/크롭되면, 키포인트 좌표도 그만큼 평행이동(shift)되어야 합니다.
        그렇지 않으면 JSON/스켈레톤 렌더링이 이미지와 어긋납니다.

구성
    - `expand_canvas_to_fit(...)`:
            유효 키포인트 bbox가 이미지 밖으로 요구하는 영역이 있으면 흰색으로 캔버스를 확장하고,
            키포인트를 (+pad_left, +pad_top)만큼 이동시킵니다.
    - `crop_to_keypoints(...)`:
            유효 키포인트 bbox를 기준으로(설정/패딩 포함) 이미지를 크롭하고,
            키포인트를 (-x1, -y1)만큼 이동시킵니다.

참고
    - 현재 파일은 DEBUG 출력이 포함된 버전입니다.
        (로그를 끄거나 최소화하려면 print 블록을 정리해야 합니다.)
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class CanvasManager:
    def __init__(self, config):
        self.config = config

    def crop_to_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        head_pad_px: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """유효 키포인트 bbox 기준으로 이미지를 크롭하고, 키포인트 좌표를 함께 이동.

        언제 쓰나
            - 캔버스 확장까지 끝난 뒤, half 포즈 등으로 인해 남는 여백이 커서
              최종 결과를 더 타이트하게 만들고 싶을 때.

        크롭 규칙
            - scores > 0.01 인 키포인트만 "유효"로 보고 bbox(min/max)를 계산합니다.
            - 설정값 `crop_padding_px`(고정 픽셀 여백), `canvas_padding_ratio`(비율 여백),
              그리고 `head_pad_px`(상단 방향 추가 여백)를 반영합니다.

        좌표 보정
            - 이미지가 (x1,y1)에서 잘려나가면, 모든 키포인트는 (-x1, -y1)만큼 이동해야
              새 이미지 좌표계와 일치합니다.

        Returns:
            - cropped_image: 크롭된 이미지
            - shifted_keypoints: 크롭에 맞춰 이동된 키포인트
            - (new_h, new_w): 크롭 후 이미지 크기
        """
        h, w = image.shape[:2]

        # 유효 키포인트만 사용 (너무 낮은 score는 bbox를 흔들 수 있음)
        valid_mask = scores > 0.01
        if not np.any(valid_mask):
            return image, keypoints, (h, w)

        valid_kpts = keypoints[valid_mask]
        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)

        # 크롭 여백: 픽셀 기반 + 이미지 크기 비율 기반(가장자리 여유)
        fixed_pad = float(getattr(self.config, 'crop_padding_px', 0))
        ratio = float(getattr(self.config, 'canvas_padding_ratio', 0.0))
        ratio_pad_w = int(w * ratio)
        ratio_pad_h = int(h * ratio)

        # 요청 크롭 영역(이상적인 bbox + padding). head_pad_px는 "위쪽"에만 추가.
        req_x1 = int(np.floor(min_x - fixed_pad - ratio_pad_w))
        req_y1 = int(np.floor(min_y - fixed_pad - ratio_pad_h - head_pad_px))
        req_x2 = int(np.ceil(max_x + fixed_pad + ratio_pad_w))
        req_y2 = int(np.ceil(max_y + fixed_pad + ratio_pad_h))

        # 이미지 경계를 벗어나는 요청은 clamp (이미지 밖은 크롭 불가)
        x1 = max(0, req_x1)
        y1 = max(0, req_y1)
        x2 = min(w, req_x2)
        y2 = min(h, req_y2)

        # 유효하지 않은 크롭이면 스킵 (너무 작은 영역)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return image, keypoints, (h, w)

        # 크롭이 의미 없으면 스킵 (전체 이미지와 동일)
        if x1 == 0 and y1 == 0 and x2 == w and y2 == h:
            return image, keypoints, (h, w)

        # 이미지 크롭
        cropped = image[y1:y2, x1:x2]

        # 좌표계 보정: 새 이미지의 (0,0)은 원본의 (x1,y1)
        shifted = keypoints.copy()
        shifted[:, 0] -= x1
        shifted[:, 1] -= y1

        new_h, new_w = cropped.shape[:2]
        return cropped, shifted, (new_h, new_w)

    def expand_canvas_to_fit(
        self, 
        source_image: np.ndarray, 
        keypoints: np.ndarray, 
        scores: np.ndarray,
        head_pad_px: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        
        # 디버그 로그(현재는 항상 출력). 필요 시 config로 토글하는 형태로 바꿀 수 있습니다.
        print("\n" + "="*60)
        print("🔍 [DEBUG] CanvasManager.expand_canvas_to_fit()")
        print("="*60)
        
        # 원본 이미지 크기 (H,W)
        h, w = source_image.shape[:2]
        print(f"\n📐 Source Image: {w}x{h}")
        print(f"   head_pad_px: {head_pad_px}")
        print(f"   crop_padding_px: {self.config.crop_padding_px}")
        print(f"   canvas_padding_ratio: {self.config.canvas_padding_ratio}")
        
        # 1) 유효 키포인트 bbox 계산
        # - 여기서 bbox는 "현재 키포인트가 요구하는 최소 영역"입니다.
        # - 이 bbox가 이미지 경계를 넘어가면, 캔버스를 그 방향으로 padding해야 합니다.
        valid_mask = scores > 0.01
        valid_count = np.sum(valid_mask)
        print(f"\n📊 Valid Keypoints (score > 0.01): {valid_count}")
        
        if not np.any(valid_mask):
            print("   ❌ No valid keypoints!")
            return source_image, keypoints, (h, w)
        
        # (디버그) 하반신 키포인트 상태 확인
        # - hip/knee/ankle이 경계 밖으로 나가면 이후 GhostFilter에서 제거될 수 있으므로
        #   캔버스 확장이 제대로 이루어지는지 확인하기 위해 출력합니다.
        print("\n📊 Lower Body Keypoints Status:")
        lower_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        lower_indices = [11, 12, 13, 14, 15, 16]
        for name, idx in zip(lower_names, lower_indices):
            if idx < len(scores):
                score = scores[idx]
                pos = keypoints[idx]
                is_valid = score > 0.01
                in_bounds = 0 <= pos[0] <= w and 0 <= pos[1] <= h
                status = "✅" if is_valid else "❌"
                bounds = "📍" if in_bounds else "⚠️ OUT"
                print(f"   {status} {name:15}: score={score:.3f}, pos=({pos[0]:.1f}, {pos[1]:.1f}) {bounds}")
            
        # 유효 키포인트로 bbox(min/max) 계산
        valid_kpts = keypoints[valid_mask]
        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)
        
        print(f"\n📏 Valid Keypoints BBox:")
        print(f"   min: ({min_x:.1f}, {min_y:.1f})")
        print(f"   max: ({max_x:.1f}, {max_y:.1f})")
        print(f"   size: {max_x - min_x:.1f} x {max_y - min_y:.1f}")
        
        # 2) padding(여백) 계산
        # - fixed_pad: 절대 픽셀 여백
        # - ratio_pad: 이미지 크기에 비례한 여백
        fixed_pad = self.config.crop_padding_px
        ratio = self.config.canvas_padding_ratio
        ratio_pad_w = int(w * ratio)
        ratio_pad_h = int(h * ratio)
        
        print(f"\n📦 Padding Calculation:")
        print(f"   fixed_pad: {fixed_pad}")
        print(f"   ratio_pad: ({ratio_pad_w}, {ratio_pad_h})")
        
        # 3) "최종적으로 필요로 하는" 캔버스 경계(요청 bounds)
        # - head_pad_px는 위쪽에 추가로 공간이 필요할 때 사용(상단만 더 위로 확장 요구)
        req_x1 = int(min_x - fixed_pad - ratio_pad_w)
        req_y1 = int(min_y - fixed_pad - ratio_pad_h - head_pad_px)
        req_x2 = int(max_x + fixed_pad + ratio_pad_w)
        req_y2 = int(max_y + fixed_pad + ratio_pad_h)
        
        print(f"\n📐 Required Canvas Bounds:")
        print(f"   req_x1: {req_x1}, req_y1: {req_y1}")
        print(f"   req_x2: {req_x2}, req_y2: {req_y2}")
        
        # 4) 원본 이미지 경계(0..w, 0..h) 대비 부족한 부분을 padding 크기로 환산
        # - req_x1 < 0 이면 왼쪽으로 -req_x1 만큼 확장 필요
        # - req_x2 > w 이면 오른쪽으로 req_x2-w 만큼 확장 필요 (y도 동일)
        pad_l = max(0, -req_x1)
        pad_t = max(0, -req_y1)
        pad_r = max(0, req_x2 - w)
        pad_b = max(0, req_y2 - h)
        
        print(f"\n🔲 Padding Needed:")
        print(f"   Left:   {pad_l} (req_x1={req_x1} < 0? {req_x1 < 0})")
        print(f"   Top:    {pad_t} (req_y1={req_y1} < 0? {req_y1 < 0})")
        print(f"   Right:  {pad_r} (req_x2={req_x2} > w={w}? {req_x2 > w})")
        print(f"   Bottom: {pad_b} (req_y2={req_y2} > h={h}? {req_y2 > h})")
        
        # 5) padding이 필요 없다면 원본 그대로 반환
        if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
            print("\n   ✅ No padding needed, returning original")
            return source_image, keypoints, (h, w)
            
        # 6) 이미지 확장 (흰색 패딩)
        # - 배경을 흰색으로 두는 이유: 디버그/시각화 시 확장 영역이 명확하고,
        #   렌더링된 스켈레톤이 검정 배경에서 잘 보이도록 하기 위함입니다.
        print(f"\n   🖼️ Expanding Canvas: T={pad_t}, B={pad_b}, L={pad_l}, R={pad_r}")
        
        padded_image = cv2.copyMakeBorder(
            source_image, pad_t, pad_b, pad_l, pad_r, 
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        # 7) 키포인트 좌표 이동(Shift)
        # - 왼쪽/위로 padding이 추가되면, 원본 좌표계에서 (0,0)의 위치가
        #   새 이미지에서는 (pad_l, pad_t)로 밀립니다.
        # - 따라서 키포인트 전체를 (+pad_l, +pad_t)만큼 이동해야 동일한 위치를 가리킵니다.
        shifted_kpts = keypoints.copy()
        shifted_kpts[:, 0] += pad_l
        shifted_kpts[:, 1] += pad_t
        
        new_h, new_w = padded_image.shape[:2]
        
        print(f"\n📐 Final Canvas: {new_w}x{new_h}")
        print(f"   Keypoint Shift: (+{pad_l}, +{pad_t})")
        
        # (디버그) 시프트 후 하반신 위치 확인
        print("\n📊 After Shift - Lower Body Positions:")
        for name, idx in zip(lower_names, lower_indices):
            if idx < len(scores):
                score = scores[idx]
                pos = shifted_kpts[idx]
                is_valid = score > 0.01
                in_bounds = 0 <= pos[0] <= new_w and 0 <= pos[1] <= new_h
                status = "✅" if is_valid else "❌"
                bounds = "📍" if in_bounds else "⚠️ OUT"
                print(f"   {status} {name:15}: pos=({pos[0]:.1f}, {pos[1]:.1f}) {bounds}")
        
        print("\n" + "="*60)
        
        return padded_image, shifted_kpts, (new_h, new_w)