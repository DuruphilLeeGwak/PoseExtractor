"""
COCO-WholeBody 133 키포인트 상수 정의

이 모듈은 COCO-WholeBody 형식의 전신 포즈 데이터를 다루기 위한
모든 상수, 인덱스, 연결 관계를 정의합니다.

COCO-WholeBody 구조:
- Body: 0-16 (17개) - 몸통과 팔다리 주요 관절
- Feet: 17-22 (6개) - 발가락과 발뒤꿈치
- Face: 23-90 (68개) - 얼굴 랜드마크
- Left Hand: 91-111 (21개) - 왼손 관절
- Right Hand: 112-132 (21개) - 오른손 관절

키포인트 그리기 순서:
- Body (몸통, 0~16번): 가장 먼저 중심이 되는 몸의 큰 줄기를 그립니다.

- Feet (발, 17~22번): 발가락과 뒤꿈치 등 발의 세부적인 위치를 그립니다.

- Face (얼굴, 23~90번): 눈, 코, 입과 얼굴 윤곽선 68개를 그립니다.

- Hands (손, 91~132번): 마지막으로 좌우 손가락 마디마디를 그립니다.
총 133개 키포인트
"""

# ============================================================================
# Body 키포인트 정의 (0-16, 총 17개)
# ============================================================================
# 몸통과 팔다리의 주요 관절 포인트
BODY_KEYPOINTS = {
    'nose': 0,              # 코
    'left_eye': 1,          # 왼쪽 눈
    'right_eye': 2,         # 오른쪽 눈
    'left_ear': 3,          # 왼쪽 귀
    'right_ear': 4,         # 오른쪽 귀
    'left_shoulder': 5,     # 왼쪽 어깨
    'right_shoulder': 6,    # 오른쪽 어깨
    'left_elbow': 7,        # 왼쪽 팔꿈치
    'right_elbow': 8,       # 오른쪽 팔꿈치
    'left_wrist': 9,        # 왼쪽 손목
    'right_wrist': 10,      # 오른쪽 손목
    'left_hip': 11,         # 왼쪽 엉덩이(골반)
    'right_hip': 12,        # 오른쪽 엉덩이(골반)
    'left_knee': 13,        # 왼쪽 무릎
    'right_knee': 14,       # 오른쪽 무릎
    'left_ankle': 15,       # 왼쪽 발목
    'right_ankle': 16       # 오른쪽 발목
}

# ============================================================================
# Feet 키포인트 정의 (17-22, 총 6개)
# ============================================================================
# 발의 세부 포인트 (발가락, 발뒤꿈치)
FEET_KEYPOINTS = {
    'left_big_toe': 17,     # 왼쪽 엄지발가락
    'left_small_toe': 18,   # 왼쪽 새끼발가락
    'left_heel': 19,        # 왼쪽 발뒤꿈치
    'right_big_toe': 20,    # 오른쪽 엄지발가락
    'right_small_toe': 21,  # 오른쪽 새끼발가락
    'right_heel': 22        # 오른쪽 발뒤꿈치
}

# ============================================================================
# Face, Hand 인덱스 범위 정의
# ============================================================================
# Face: 23-90 (68개의 얼굴 랜드마크)
FACE_START_IDX = 23
FACE_END_IDX = 90
FACE_COUNT = 68

# Left Hand: 91-111 (21개의 왼손 관절)
LEFT_HAND_START_IDX = 91
LEFT_HAND_END_IDX = 111

# Right Hand: 112-132 (21개의 오른손 관절)
RIGHT_HAND_START_IDX = 112
RIGHT_HAND_END_IDX = 132

# Hand 관절 개수 (각 손마다 21개)
HAND_COUNT = 21

# 전체 키포인트 개수
TOTAL_KEYPOINTS = 133

# ============================================================================
# Body 본(뼈) 연결 관계
# ============================================================================
# 키포인트 간의 연결선을 정의하여 스켈레톤을 구성
# 각 튜플은 (시작 키포인트, 끝 키포인트) 쌍을 나타냄
BODY_BONES = [
    # 어깨선 (좌우 어깨 연결)
    ('left_shoulder', 'right_shoulder'),
    
    # 몸통 (어깨-골반 연결)
    ('left_shoulder', 'left_hip'), 
    ('right_shoulder', 'right_hip'),
    
    # 골반선 (좌우 골반 연결)
    ('left_hip', 'right_hip'),
    
    # 왼팔 (어깨-팔꿈치-손목)
    ('left_shoulder', 'left_elbow'), 
    ('left_elbow', 'left_wrist'),
    
    # 오른팔 (어깨-팔꿈치-손목)
    ('right_shoulder', 'right_elbow'), 
    ('right_elbow', 'right_wrist'),
    
    # 왼다리 (골반-무릎-발목)
    ('left_hip', 'left_knee'), 
    ('left_knee', 'left_ankle'),
    
    # 오른다리 (골반-무릎-발목)
    ('right_hip', 'right_knee'), 
    ('right_knee', 'right_ankle'),
    
    # 얼굴 (코-눈 연결)
    ('nose', 'left_eye'), 
    ('nose', 'right_eye'),
    
    # 얼굴 (눈-귀 연결)
    ('left_eye', 'left_ear'), 
    ('right_eye', 'right_ear'),
    
    # 목 (어깨-귀 연결)
    ('left_shoulder', 'left_ear'), 
    ('right_shoulder', 'right_ear'),
]

# ============================================================================
# Feet 본(뼈) 연결 관계
# ============================================================================
# 발의 세부 구조를 나타내는 연결선
FEET_BONES = [
    # 왼발 (발목-발뒤꿈치-발가락 연결)
    ('left_ankle', 'left_heel'), 
    ('left_ankle', 'left_big_toe'),
    ('left_heel', 'left_big_toe'), 
    ('left_big_toe', 'left_small_toe'),
    
    # 오른발 (발목-발뒤꿈치-발가락 연결)
    ('right_ankle', 'right_heel'), 
    ('right_ankle', 'right_big_toe'),
    ('right_heel', 'right_big_toe'), 
    ('right_big_toe', 'right_small_toe'),
]

# ============================================================================
# Hand 본(뼈) 연결 관계 (0-20 상대 인덱스)
# ============================================================================
# 손 관절 구조: 손목(0)에서 각 손가락으로 뻗어나가는 형태
# 실제 사용 시 LEFT_HAND_START_IDX 또는 RIGHT_HAND_START_IDX를 더해야 함
HAND_BONES = [
    # 손목에서 각 손가락 뿌리로 연결
    (0, 1),    # 손목 -> 엄지 뿌리
    (0, 5),    # 손목 -> 검지 뿌리
    (0, 9),    # 손목 -> 중지 뿌리
    (0, 13),   # 손목 -> 약지 뿌리
    (0, 17),   # 손목 -> 새끼 뿌리
    
    # 엄지손가락 (3개 관절)
    (1, 2), (2, 3), (3, 4),
    
    # 검지손가락 (3개 관절)
    (5, 6), (6, 7), (7, 8),
    
    # 중지손가락 (3개 관절)
    (9, 10), (10, 11), (11, 12),
    
    # 약지손가락 (3개 관절)
    (13, 14), (14, 15), (15, 16),
    
    # 새끼손가락 (3개 관절)
    (17, 18), (18, 19), (19, 20)
]

# ============================================================================
# Face 본(뼈) 연결 관계 (0-67 상대 인덱스)
# ============================================================================
# 얼굴 랜드마크의 연결 관계 (윤곽, 눈썹, 눈, 코, 입)
# 실제 사용 시 FACE_START_IDX를 더해야 함
FACE_BONES_RELATIVE = (
    # 얼굴 윤곽선 (0-16)
    [(i, i+1) for i in range(16)] +
    
    # 왼쪽 눈썹 (17-21)
    [(i, i+1) for i in range(17, 21)] +
    
    # 오른쪽 눈썹 (22-26)
    [(i, i+1) for i in range(22, 26)] +
    
    # 코 브릿지 (27-30)
    [(i, i+1) for i in range(27, 30)] +
    
    # 코끝 (31-35)
    [(i, i+1) for i in range(31, 35)] +
    
    # 왼쪽 눈 (36-41, 닫힌 루프)
    [(36,37),(37,38),(38,39),(39,40),(40,41),(41,36)] +
    
    # 오른쪽 눈 (42-47, 닫힌 루프)
    [(42,43),(43,44),(44,45),(45,46),(46,47),(47,42)] +
    
    # 입 외곽선 (48-59, 닫힌 루프)
    [(i, i+1) for i in range(48, 59)] + [(59, 48)] +
    
    # 입 내곽선 (60-67, 닫힌 루프)
    [(i, i+1) for i in range(60, 67)] + [(67, 60)]
)

# ============================================================================
# 대칭 키포인트 쌍 정의
# ============================================================================
# 좌우 대칭인 키포인트들의 매핑 (포즈 미러링, 대칭성 검사 등에 사용)

# Body 대칭 쌍
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

# Feet 대칭 쌍
SYMMETRIC_FEET_PAIRS = [
    ('left_big_toe', 'right_big_toe'),
    ('left_small_toe', 'right_small_toe'),
    ('left_heel', 'right_heel'),
]

# ============================================================================
# Body 계층 구조 (Hierarchy)
# ============================================================================
# 부모-자식 관계를 정의하여 포즈 전송 시 계층적 처리를 가능하게 함
# 부모 키포인트가 이동/변형되면 자식들도 따라서 변형되어야 함
BODY_HIERARCHY = {
    'root': ['left_hip', 'right_hip'],              # 루트: 골반(중심)
    'left_hip': ['left_knee', 'left_shoulder'],     # 왼쪽 골반 -> 무릎, 어깨
    'right_hip': ['right_knee', 'right_shoulder'],  # 오른쪽 골반 -> 무릎, 어깨
    'left_knee': ['left_ankle'],                    # 왼쪽 무릎 -> 발목
    'right_knee': ['right_ankle'],                  # 오른쪽 무릎 -> 발목
    'left_ankle': ['left_heel', 'left_big_toe'],    # 왼쪽 발목 -> 발 세부
    'right_ankle': ['right_heel', 'right_big_toe'], # 오른쪽 발목 -> 발 세부
    'left_big_toe': ['left_small_toe'],             # 왼쪽 엄지 -> 새끼발가락
    'right_big_toe': ['right_small_toe'],           # 오른쪽 엄지 -> 새끼발가락
    'left_shoulder': ['left_elbow', 'left_ear'],    # 왼쪽 어깨 -> 팔꿈치, 귀
    'right_shoulder': ['right_elbow', 'right_ear'], # 오른쪽 어깨 -> 팔꿈치, 귀
    'left_elbow': ['left_wrist'],                   # 왼쪽 팔꿈치 -> 손목
    'right_elbow': ['right_wrist'],                 # 오른쪽 팔꿈치 -> 손목
    'left_ear': ['left_eye'],                       # 왼쪽 귀 -> 눈
    'right_ear': ['right_eye'],                     # 오른쪽 귀 -> 눈
    'left_eye': ['nose'],                           # 왼쪽 눈 -> 코
    'right_eye': [],                                # 오른쪽 눈은 자식 없음
}

# ============================================================================
# 시각화 색상 정의
# ============================================================================
# 스켈레톤 렌더링 시 사용할 색상 (BGR 형식)

# Body 각 본마다 다른 색상 (무지개 스펙트럼)
BODY_COLORS = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
    (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
    (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85),
]

# 각 부위별 고정 색상
FACE_COLOR = (255, 255, 255)        # 얼굴: 흰색
LEFT_HAND_COLOR = (0, 255, 255)     # 왼손: 노란색(시안)
RIGHT_HAND_COLOR = (255, 255, 0)    # 오른손: 청록색

# ============================================================================
# 유틸리티 함수들
# ============================================================================

def get_keypoint_index(name: str) -> int:
    """
    키포인트 이름으로 인덱스 찾기
    
    Args:
        name: 키포인트 이름 (예: 'left_shoulder', 'right_knee')
    
    Returns:
        int: 해당 키포인트의 인덱스 (0-22 범위)
    
    Raises:
        ValueError: 알 수 없는 키포인트 이름인 경우
    
    Example:
        >>> get_keypoint_index('left_hip')
        11
        >>> get_keypoint_index('right_ankle')
        16
    """
    if name in BODY_KEYPOINTS:
        return BODY_KEYPOINTS[name]
    elif name in FEET_KEYPOINTS:
        return FEET_KEYPOINTS[name]
    raise ValueError(f"Unknown keypoint: {name}")

def get_symmetric_pair(name: str) -> str:
    """
    대칭 키포인트 찾기 (좌 <-> 우 매핑)
    
    Args:
        name: 키포인트 이름
    
    Returns:
        str: 대칭 쌍의 이름, 없으면 None
    
    Example:
        >>> get_symmetric_pair('left_shoulder')
        'right_shoulder'
        >>> get_symmetric_pair('right_knee')
        'left_knee'
        >>> get_symmetric_pair('nose')
        None
    """
    for left, right in SYMMETRIC_BODY_PAIRS + SYMMETRIC_FEET_PAIRS:
        if name == left: return right
        if name == right: return left
    return None

def get_body_bone_indices() -> list:
    """
    Body 본의 인덱스 쌍 리스트 반환
    
    Returns:
        list: [(시작_인덱스, 끝_인덱스), ...] 형태의 리스트
    
    Example:
        >>> bones = get_body_bone_indices()
        >>> bones[0]  # 왼쪽-오른쪽 어깨 연결
        (5, 6)
    """
    return [(get_keypoint_index(s), get_keypoint_index(e)) for s, e in BODY_BONES]

def get_feet_bone_indices() -> list:
    """
    Feet 본의 인덱스 쌍 리스트 반환
    
    Returns:
        list: [(시작_인덱스, 끝_인덱스), ...] 형태의 리스트
    
    Example:
        >>> bones = get_feet_bone_indices()
        >>> bones[0]  # 왼쪽 발목-발뒤꿈치 연결
        (15, 19)
    """
    return [(get_keypoint_index(s), get_keypoint_index(e)) for s, e in FEET_BONES]

def get_hand_bone_indices(is_left: bool) -> list:
    """
    Hand 본의 절대 인덱스 쌍 리스트 반환
    
    Args:
        is_left: True면 왼손, False면 오른손
    
    Returns:
        list: [(시작_인덱스, 끝_인덱스), ...] 형태의 리스트
              왼손: 91-111, 오른손: 112-132 범위
    
    Example:
        >>> left_bones = get_hand_bone_indices(True)
        >>> left_bones[0]  # 왼손 손목에서 엄지 뿌리
        (91, 92)
        >>> right_bones = get_hand_bone_indices(False)
        >>> right_bones[0]  # 오른손 손목에서 엄지 뿌리
        (112, 113)
    """
    # 상대 인덱스에 시작 오프셋을 더해 절대 인덱스로 변환
    offset = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
    return [(s + offset, e + offset) for s, e in HAND_BONES]

def get_face_bone_indices() -> list:
    """
    Face 본의 절대 인덱스 쌍 리스트 반환
    
    Returns:
        list: [(시작_인덱스, 끝_인덱스), ...] 형태의 리스트
              얼굴: 23-90 범위
    
    Example:
        >>> face_bones = get_face_bone_indices()
        >>> face_bones[0]  # 얼굴 윤곽선 첫 번째 연결
        (23, 24)
    """
    # 상대 인덱스에 얼굴 시작 오프셋을 더해 절대 인덱스로 변환
    return [(s + FACE_START_IDX, e + FACE_START_IDX) for s, e in FACE_BONES_RELATIVE]
