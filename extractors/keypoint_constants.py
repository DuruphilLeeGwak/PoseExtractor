"""
COCO-WholeBody 133 키포인트 상수 정의

키포인트 구성:
- Body: 0-16 (17개)
- Feet: 17-22 (6개)  
- Face: 23-90 (68개)
- Left Hand: 91-111 (21개)
- Right Hand: 112-132 (21개)
"""

# =============================================================================
# 키포인트 인덱스
# =============================================================================

# Body 키포인트 (0-16)
BODY_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Feet 키포인트 (17-22)
FEET_KEYPOINTS = {
    'left_big_toe': 17,
    'left_small_toe': 18,
    'left_heel': 19,
    'right_big_toe': 20,
    'right_small_toe': 21,
    'right_heel': 22
}

# Face 키포인트 시작/끝 인덱스
FACE_START_IDX = 23
FACE_END_IDX = 90
FACE_COUNT = 68

# Hand 키포인트 인덱스
LEFT_HAND_START_IDX = 91
LEFT_HAND_END_IDX = 111
RIGHT_HAND_START_IDX = 112
RIGHT_HAND_END_IDX = 132
HAND_COUNT = 21

# 전체 키포인트 수
TOTAL_KEYPOINTS = 133

# =============================================================================
# 본(Bone) 연결 관계
# =============================================================================

# Body 본 연결
BODY_BONES = [
    # 몸통
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    
    # 왼쪽 팔
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    
    # 오른쪽 팔
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    
    # 왼쪽 다리
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    
    # 오른쪽 다리
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    
    # 머리
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    ('left_shoulder', 'left_ear'),
    ('right_shoulder', 'right_ear'),
]

# Feet 본 연결
FEET_BONES = [
    ('left_ankle', 'left_heel'),
    ('left_ankle', 'left_big_toe'),
    ('left_heel', 'left_big_toe'),
    ('left_big_toe', 'left_small_toe'),
    
    ('right_ankle', 'right_heel'),
    ('right_ankle', 'right_big_toe'),
    ('right_heel', 'right_big_toe'),
    ('right_big_toe', 'right_small_toe'),
]

# 손 본 연결 (각 손가락 기준)
HAND_BONES = [
    # 손목 -> 각 손가락 기저부
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    # 엄지
    (1, 2), (2, 3), (3, 4),
    # 검지
    (5, 6), (6, 7), (7, 8),
    # 중지
    (9, 10), (10, 11), (11, 12),
    # 약지
    (13, 14), (14, 15), (15, 16),
    # 소지
    (17, 18), (18, 19), (19, 20)
]

# Face 본 연결 (윤곽선)
FACE_CONTOUR_BONES = [(i, i+1) for i in range(16)]  # 0-16: 얼굴 윤곽

# Face 본 연결 (눈썹)
FACE_EYEBROW_LEFT = [(i, i+1) for i in range(17, 21)]   # 17-21: 왼쪽 눈썹
FACE_EYEBROW_RIGHT = [(i, i+1) for i in range(22, 26)]  # 22-26: 오른쪽 눈썹

# Face 본 연결 (코)
FACE_NOSE_BRIDGE = [(i, i+1) for i in range(27, 30)]    # 27-30: 코 브릿지
FACE_NOSE_TIP = [(i, i+1) for i in range(31, 35)]       # 31-35: 코끝

# Face 본 연결 (눈)
FACE_EYE_LEFT = [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)]
FACE_EYE_RIGHT = [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)]

# Face 본 연결 (입)
FACE_MOUTH_OUTER = [(i, i+1) for i in range(48, 59)] + [(59, 48)]
FACE_MOUTH_INNER = [(i, i+1) for i in range(60, 67)] + [(67, 60)]

# =============================================================================
# 대칭 쌍 (Symmetric Pairs)
# =============================================================================

SYMMETRIC_BODY_PAIRS = [
    ('left_eye', 'right_eye'),
    ('left_ear', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_elbow', 'right_elbow'),
    ('left_wrist', 'right_wrist'),
    ('left_hip', 'right_hip'),
    ('left_knee', 'right_knee'),
    ('left_ankle', 'right_ankle'),
]

SYMMETRIC_FEET_PAIRS = [
    ('left_big_toe', 'right_big_toe'),
    ('left_small_toe', 'right_small_toe'),
    ('left_heel', 'right_heel'),
]

# =============================================================================
# 계층 구조 (부모-자식 관계)
# =============================================================================

# 루트에서부터의 계층 구조
BODY_HIERARCHY = {
    # 루트 (엉덩이 중심)
    'root': ['left_hip', 'right_hip'],
    
    # 상체
    'left_hip': ['left_knee', 'left_shoulder'],
    'right_hip': ['right_knee', 'right_shoulder'],
    
    # 다리
    'left_knee': ['left_ankle'],
    'right_knee': ['right_ankle'],
    'left_ankle': ['left_heel', 'left_big_toe'],
    'right_ankle': ['right_heel', 'right_big_toe'],
    'left_big_toe': ['left_small_toe'],
    'right_big_toe': ['right_small_toe'],
    
    # 팔
    'left_shoulder': ['left_elbow', 'left_ear'],
    'right_shoulder': ['right_elbow', 'right_ear'],
    'left_elbow': ['left_wrist'],
    'right_elbow': ['right_wrist'],
    
    # 머리
    'left_ear': ['left_eye'],
    'right_ear': ['right_eye'],
    'left_eye': ['nose'],
    'right_eye': [],
}

# =============================================================================
# 렌더링 색상
# =============================================================================

# OpenPose 스타일 색상 (BGR)
BODY_COLORS = [
    (255, 0, 0),     # 0: 빨강
    (255, 85, 0),    # 1
    (255, 170, 0),   # 2
    (255, 255, 0),   # 3: 노랑
    (170, 255, 0),   # 4
    (85, 255, 0),    # 5
    (0, 255, 0),     # 6: 초록
    (0, 255, 85),    # 7
    (0, 255, 170),   # 8
    (0, 255, 255),   # 9: 시안
    (0, 170, 255),   # 10
    (0, 85, 255),    # 11
    (0, 0, 255),     # 12: 파랑
    (85, 0, 255),    # 13
    (170, 0, 255),   # 14
    (255, 0, 255),   # 15: 마젠타
    (255, 0, 170),   # 16
    (255, 0, 85),    # 17
]

FACE_COLOR = (255, 255, 255)  # 흰색
LEFT_HAND_COLOR = (0, 255, 255)  # 시안
RIGHT_HAND_COLOR = (255, 255, 0)  # 노랑

# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_keypoint_index(name: str) -> int:
    """키포인트 이름으로 인덱스 가져오기"""
    if name in BODY_KEYPOINTS:
        return BODY_KEYPOINTS[name]
    elif name in FEET_KEYPOINTS:
        return FEET_KEYPOINTS[name]
    else:
        raise ValueError(f"Unknown keypoint name: {name}")


def get_symmetric_pair(name: str) -> str:
    """대칭 키포인트 이름 가져오기"""
    for left, right in SYMMETRIC_BODY_PAIRS + SYMMETRIC_FEET_PAIRS:
        if name == left:
            return right
        elif name == right:
            return left
    return None


def get_body_bone_indices() -> list:
    """Body 본 연결을 인덱스로 변환"""
    indices = []
    for start_name, end_name in BODY_BONES:
        start_idx = get_keypoint_index(start_name)
        end_idx = get_keypoint_index(end_name)
        indices.append((start_idx, end_idx))
    return indices


def get_feet_bone_indices() -> list:
    """Feet 본 연결을 인덱스로 변환"""
    indices = []
    for start_name, end_name in FEET_BONES:
        start_idx = get_keypoint_index(start_name)
        end_idx = get_keypoint_index(end_name)
        indices.append((start_idx, end_idx))
    return indices


def get_hand_bone_indices(is_left: bool) -> list:
    """Hand 본 연결을 전체 키포인트 인덱스로 변환"""
    offset = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
    return [(start + offset, end + offset) for start, end in HAND_BONES]


def get_face_bone_indices() -> list:
    """Face 본 연결을 전체 키포인트 인덱스로 변환"""
    all_face_bones = (
        FACE_CONTOUR_BONES +
        FACE_EYEBROW_LEFT +
        FACE_EYEBROW_RIGHT +
        FACE_NOSE_BRIDGE +
        FACE_NOSE_TIP +
        FACE_EYE_LEFT +
        FACE_EYE_RIGHT +
        FACE_MOUTH_OUTER +
        FACE_MOUTH_INNER
    )
    return [(start + FACE_START_IDX, end + FACE_START_IDX) for start, end in all_face_bones]
