import numpy as np
from ...extractors.keypoint_constants import BODY_KEYPOINTS, FACE_START_IDX, FACE_END_IDX

FACE_PARTS_IDX = {
    'jawline': range(0, 17), 'left_eyebrow': range(17, 22), 'right_eyebrow': range(22, 27),
    'nose': range(27, 36), 'left_eye': range(36, 42), 'right_eye': range(42, 48),
    'mouth_outer': range(48, 60), 'mouth_inner': range(60, 68),
}

class FaceTransfer:
    def __init__(self, config):
        self.config = config

    def transfer(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, log):
        if not self.config.face_rendering.enabled: return
        
        nose = BODY_KEYPOINTS['nose']
        l_eye = BODY_KEYPOINTS['left_eye']
        r_eye = BODY_KEYPOINTS['right_eye']
        
        # 회전 매트릭스 (Ref 각도 적용)
        s_vec = s_kpts[r_eye] - s_kpts[l_eye]
        r_vec = r_kpts[r_eye] - r_kpts[l_eye]
        
        angle = np.arctan2(r_vec[1], r_vec[0]) - np.arctan2(s_vec[1], s_vec[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        anchor = s_kpts[nose]
        src_face_nose = s_kpts[FACE_START_IDX + 30]
        
        # 1. 68 랜드마크 전이
        for i in range(FACE_START_IDX, FACE_END_IDX + 1):
            local_idx = i - FACE_START_IDX
            part_name = self._get_part_name(local_idx)
            
            # YAML 설정 체크
            part_config = self.config.face_rendering.parts.get(part_name)
            if part_config and not part_config.enabled:
                t_scores[i] = 0.0
                continue
                
            if s_scores[i] > 0.3:
                rel = s_kpts[i] - src_face_nose
                t_kpts[i] = anchor + rot_mat @ rel
                t_scores[i] = s_scores[i]
                log[f'face_{i}'] = 'rot'

        # 2. COCO 기본 얼굴
        head_parts = [nose, l_eye, r_eye, BODY_KEYPOINTS['left_ear'], BODY_KEYPOINTS['right_ear']]
        for idx in head_parts:
             if s_scores[idx] > 0.3:
                 rel = s_kpts[idx] - s_kpts[nose]
                 t_kpts[idx] = anchor + rot_mat @ rel
                 t_scores[idx] = s_scores[idx]

    def _get_part_name(self, idx):
        for name, r in FACE_PARTS_IDX.items():
            if idx in r: return name
        return None