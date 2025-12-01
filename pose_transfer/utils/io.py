"""
파일 입출력 유틸리티
"""
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict


@dataclass
class PoseResult:
    """포즈 추출 결과 데이터 클래스"""
    keypoints: np.ndarray  # (N, 133, 2) 또는 (133, 2)
    scores: np.ndarray     # (N, 133) 또는 (133,)
    image_size: Tuple[int, int]  # (height, width)
    selected_person_idx: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'keypoints': self.keypoints.tolist(),
            'scores': self.scores.tolist(),
            'image_size': list(self.image_size),
            'selected_person_idx': self.selected_person_idx
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoseResult':
        """딕셔너리에서 생성"""
        return cls(
            keypoints=np.array(data['keypoints']),
            scores=np.array(data['scores']),
            image_size=tuple(data['image_size']),
            selected_person_idx=data.get('selected_person_idx')
        )


@dataclass
class TransferResult:
    """포즈 전이 결과 데이터 클래스"""
    source_keypoints: np.ndarray
    source_scores: np.ndarray
    reference_keypoints: np.ndarray
    reference_scores: np.ndarray
    transferred_keypoints: np.ndarray
    transferred_scores: np.ndarray
    bone_lengths: Dict[str, float]
    fallback_applied: Dict[str, bool]
    image_size: Tuple[int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'source_keypoints': self.source_keypoints.tolist(),
            'source_scores': self.source_scores.tolist(),
            'reference_keypoints': self.reference_keypoints.tolist(),
            'reference_scores': self.reference_scores.tolist(),
            'transferred_keypoints': self.transferred_keypoints.tolist(),
            'transferred_scores': self.transferred_scores.tolist(),
            'bone_lengths': self.bone_lengths,
            'fallback_applied': self.fallback_applied,
            'image_size': list(self.image_size)
        }


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """YAML 설정 파일 저장"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """이미지 로드 (BGR 포맷)"""
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """이미지 저장"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def save_json(data: Dict[str, Any], output_path: Union[str, Path], 
              indent: int = 2) -> None:
    """JSON 파일 저장"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """JSON 파일 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_openpose_format(keypoints: np.ndarray, scores: np.ndarray,
                                image_size: Tuple[int, int]) -> Dict[str, Any]:
    """
    OpenPose JSON 포맷으로 변환
    
    OpenPose format:
    - pose_keypoints_2d: [x1, y1, c1, x2, y2, c2, ...]
    - face_keypoints_2d: [x1, y1, c1, ...]
    - hand_left_keypoints_2d: [x1, y1, c1, ...]
    - hand_right_keypoints_2d: [x1, y1, c1, ...]
    """
    height, width = image_size
    
    # 키포인트 인덱스 매핑 (COCO-WholeBody 133 keypoints)
    # Body: 0-16 (17개) -> OpenPose BODY_25 매핑 필요
    # Feet: 17-22 (6개)
    # Face: 23-90 (68개)
    # Left Hand: 91-111 (21개)
    # Right Hand: 112-132 (21개)
    
    def flatten_keypoints(kpts: np.ndarray, scrs: np.ndarray) -> list:
        """키포인트를 [x, y, c, x, y, c, ...] 형태로 변환"""
        result = []
        for i in range(len(kpts)):
            result.extend([float(kpts[i, 0]), float(kpts[i, 1]), float(scrs[i])])
        return result
    
    # 단일 인물 처리
    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, ...]
        scores = scores[np.newaxis, ...]
    
    people = []
    for person_idx in range(len(keypoints)):
        kpts = keypoints[person_idx]
        scrs = scores[person_idx]
        
        # Body (0-22: body + feet)
        body_kpts = kpts[:23]
        body_scores = scrs[:23]
        
        # Face (23-90)
        face_kpts = kpts[23:91]
        face_scores = scrs[23:91]
        
        # Left Hand (91-111)
        left_hand_kpts = kpts[91:112]
        left_hand_scores = scrs[91:112]
        
        # Right Hand (112-132)
        right_hand_kpts = kpts[112:133]
        right_hand_scores = scrs[112:133]
        
        person_data = {
            'person_id': [-1],
            'pose_keypoints_2d': flatten_keypoints(body_kpts, body_scores),
            'face_keypoints_2d': flatten_keypoints(face_kpts, face_scores),
            'hand_left_keypoints_2d': flatten_keypoints(left_hand_kpts, left_hand_scores),
            'hand_right_keypoints_2d': flatten_keypoints(right_hand_kpts, right_hand_scores),
        }
        people.append(person_data)
    
    return {
        'version': 1.3,
        'people': people
    }


def get_image_files(folder_path: Union[str, Path], 
                    extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')) -> list:
    """폴더에서 이미지 파일 목록 가져오기"""
    folder_path = Path(folder_path)
    files = []
    for ext in extensions:
        files.extend(folder_path.glob(f'*{ext}'))
        files.extend(folder_path.glob(f'*{ext.upper()}'))
    return sorted(files)


def create_output_paths(input_path: Union[str, Path], 
                        output_dir: Union[str, Path],
                        suffixes: Dict[str, str]) -> Dict[str, Path]:
    """
    출력 파일 경로 생성
    
    Args:
        input_path: 입력 파일 경로
        output_dir: 출력 디렉토리
        suffixes: {'json': '_keypoints.json', 'skeleton': '_skeleton.png', ...}
    
    Returns:
        {'json': Path, 'skeleton': Path, ...}
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = input_path.stem
    
    result = {}
    for key, suffix in suffixes.items():
        result[key] = output_dir / f"{stem}{suffix}"
    
    return result
