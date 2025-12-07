#!/usr/bin/env python3
"""
engine.py 패치 스크립트

사용법:
1. 이 파일의 내용을 참고하여 engine.py 수정
2. 또는 sed 명령어로 자동 패치

"""

import re
import sys

def patch_engine_py(file_path):
    """engine.py를 패치하여 Case 정보 기반 하반신 전이 제어 추가"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # [패치 1] transfer 메서드 시그니처에 alignment_case 추가
    # 기존: def transfer(self, source_keypoints, source_scores, reference_keypoints, reference_scores, src_img_size, ref_img_size, transfer_log):
    # 수정: def transfer(self, source_keypoints, source_scores, reference_keypoints, reference_scores, src_img_size, ref_img_size, transfer_log, alignment_case=None):
    
    pattern1 = r'(def transfer\(self,.*?transfer_log)(\):\s*)'
    replacement1 = r'\1, alignment_case=None\2'
    content = re.sub(pattern1, replacement1, content, count=1)
    
    # [패치 2] _check_lower_body_valid 호출 전에 Case 체크 추가
    # 찾기: ref_lower_valid = self._check_lower_body_valid(
    # 추가: Case B, C 체크
    
    check_pattern = r'(\s+)(ref_lower_valid\s*=\s*self\._check_lower_body_valid\()'
    
    case_check_code = '''\\1# Case B, C는 REF가 상반신 → 하반신 전이 스킵
\\1if alignment_case in ['B', 'C']:
\\1    print(f"   ⏭️ [Skip] Lower Body (REF is UPPER, Case {alignment_case})")
\\1    ref_lower_valid = False
\\1else:
\\1    \\2'''
    
    content = re.sub(check_pattern, case_check_code, content, count=1)
    
    # 결과 저장
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Patched: {file_path}")


# =============================================================================
# 수동 패치 가이드 (sed 명령어)
# =============================================================================
"""
# 1. transfer 시그니처 수정
sed -i 's/def transfer(self, source_keypoints, source_scores, reference_keypoints, reference_scores, src_img_size, ref_img_size, transfer_log):/def transfer(self, source_keypoints, source_scores, reference_keypoints, reference_scores, src_img_size, ref_img_size, transfer_log, alignment_case=None):/' engine.py

# 2. _check_lower_body_valid 호출 전에 Case 체크 추가 (수동으로 해야 함)
# 아래 코드를 ref_lower_valid = self._check_lower_body_valid(...) 위에 추가:

        # Case B, C는 REF가 상반신 → 하반신 전이 스킵
        if alignment_case in ['B', 'C']:
            print(f"   ⏭️ [Skip] Lower Body (REF is UPPER, Case {alignment_case})")
            ref_lower_valid = False
        else:
            ref_lower_valid = self._check_lower_body_valid(...)
"""


# =============================================================================
# pipeline.py 수정 가이드
# =============================================================================
"""
# transfer() 호출 부분 수정:

# 기존:
trans_keypoints, trans_scores, log = self.engine.transfer(
    source_keypoints, source_scores,
    reference_keypoints, reference_scores,
    src_img_size, ref_img_size, {}
)

# 수정:
trans_keypoints, trans_scores, log = self.engine.transfer(
    source_keypoints, source_scores,
    reference_keypoints, reference_scores,
    src_img_size, ref_img_size, {},
    alignment_case=case.value  # 'A', 'B', 'C', 'D'
)
"""


if __name__ == '__main__':
    if len(sys.argv) > 1:
        patch_engine_py(sys.argv[1])
    else:
        print("Usage: python patch_engine.py /path/to/engine.py")
        print("\n또는 수동으로 위의 가이드를 따라 수정하세요.")