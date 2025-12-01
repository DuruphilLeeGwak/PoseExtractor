"""
COCO-WholeBody 133 키포인트 상수 정의
"""

# Body 키포인트 (0-16)
BODY_KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# Feet 키포인트 (17-22)
FEET_KEYPOINTS = {
    'left_big_toe': 17, 'left_small_toe': 18, 'left_heel': 19,
    'right_big_toe': 20, 'right_small_toe': 21, 'right_heel': 22
}

FACE_START_IDX = 23
FACE_END_IDX = 90
FACE_COUNT = 68

LEFT_HAND_START_IDX = 91
LEFT_HAND_END_IDX = 111
RIGHT_HAND_START_IDX = 112
RIGHT_HAND_END_IDX = 132
HAND_COUNT = 21
TOTAL_KEYPOINTS = 133

# Body 본 연결
BODY_BONES = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_shoulder', 'left_ear'), ('right_shoulder', 'right_ear'),
]

FEET_BONES = [
    ('left_ankle', 'left_heel'), ('left_ankle', 'left_big_toe'),
    ('left_heel', 'left_big_toe'), ('left_big_toe', 'left_small_toe'),
    ('right_ankle', 'right_heel'), ('right_ankle', 'right_big_toe'),
    ('right_heel', 'right_big_toe'), ('right_big_toe', 'right_small_toe'),
]

# 손 본 연결 (0-20 상대 인덱스)
HAND_BONES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20)
]

# Face 본 연결 (0-67 상대 인덱스)
FACE_BONES_RELATIVE = (
    [(i, i+1) for i in range(16)] +           # 윤곽 0-16
    [(i, i+1) for i in range(17, 21)] +       # 왼쪽 눈썹 17-21
    [(i, i+1) for i in range(22, 26)] +       # 오른쪽 눈썹 22-26
    [(i, i+1) for i in range(27, 30)] +       # 코 브릿지 27-30
    [(i, i+1) for i in range(31, 35)] +       # 코끝 31-35
    [(36,37),(37,38),(38,39),(39,40),(40,41),(41,36)] +  # 왼쪽 눈
    [(42,43),(43,44),(44,45),(45,46),(46,47),(47,42)] +  # 오른쪽 눈
    [(i, i+1) for i in range(48, 59)] + [(59, 48)] +     # 입 외곽
    [(i, i+1) for i in range(60, 67)] + [(67, 60)]       # 입 내곽
)

SYMMETRIC_BODY_PAIRS = [
    ('left_eye', 'right_eye'), ('left_ear', 'right_ear'),
    ('left_shoulder', 'right_shoulder'), ('left_elbow', 'right_elbow'),
    ('left_wrist', 'right_wrist'), ('left_hip', 'right_hip'),
    ('left_knee', 'right_knee'), ('left_ankle', 'right_ankle'),
]

SYMMETRIC_FEET_PAIRS = [
    ('left_big_toe', 'right_big_toe'),
    ('left_small_toe', 'right_small_toe'),
    ('left_heel', 'right_heel'),
]

BODY_HIERARCHY = {
    'root': ['left_hip', 'right_hip'],
    'left_hip': ['left_knee', 'left_shoulder'],
    'right_hip': ['right_knee', 'right_shoulder'],
    'left_knee': ['left_ankle'], 'right_knee': ['right_ankle'],
    'left_ankle': ['left_heel', 'left_big_toe'],
    'right_ankle': ['right_heel', 'right_big_toe'],
    'left_big_toe': ['left_small_toe'], 'right_big_toe': ['right_small_toe'],
    'left_shoulder': ['left_elbow', 'left_ear'],
    'right_shoulder': ['right_elbow', 'right_ear'],
    'left_elbow': ['left_wrist'], 'right_elbow': ['right_wrist'],
    'left_ear': ['left_eye'], 'right_ear': ['right_eye'],
    'left_eye': ['nose'], 'right_eye': [],
}

BODY_COLORS = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
    (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
    (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85),
]
FACE_COLOR = (255, 255, 255)
LEFT_HAND_COLOR = (0, 255, 255)
RIGHT_HAND_COLOR = (255, 255, 0)

def get_keypoint_index(name: str) -> int:
    if name in BODY_KEYPOINTS:
        return BODY_KEYPOINTS[name]
    elif name in FEET_KEYPOINTS:
        return FEET_KEYPOINTS[name]
    raise ValueError(f"Unknown keypoint: {name}")

def get_symmetric_pair(name: str) -> str:
    for left, right in SYMMETRIC_BODY_PAIRS + SYMMETRIC_FEET_PAIRS:
        if name == left: return right
        if name == right: return left
    return None

def get_body_bone_indices() -> list:
    return [(get_keypoint_index(s), get_keypoint_index(e)) for s, e in BODY_BONES]

def get_feet_bone_indices() -> list:
    return [(get_keypoint_index(s), get_keypoint_index(e)) for s, e in FEET_BONES]

def get_hand_bone_indices(is_left: bool) -> list:
    offset = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
    return [(s + offset, e + offset) for s, e in HAND_BONES]

def get_face_bone_indices() -> list:
    return [(s + FACE_START_IDX, e + FACE_START_IDX) for s, e in FACE_BONES_RELATIVE]
