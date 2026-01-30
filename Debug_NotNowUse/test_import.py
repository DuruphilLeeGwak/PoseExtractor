"""Import 테스트"""
print("=" * 80)
print("Cross-Filter 구현 Import 테스트")
print("=" * 80)

# 1. 기본 패키지
import cv2
import numpy as np
import rtmlib
print(f"✅ OpenCV: {cv2.__version__}")
print(f"✅ NumPy: {np.__version__}")
try:
    print(f"✅ rtmlib: {rtmlib.__version__}")
except:
    print(f"✅ rtmlib: (버전 정보 없음)")
print()

# 2. BodyExtractor
try:
    from pose_transfer.extractors import BodyExtractor
    print("✅ BodyExtractor import 성공")
except Exception as e:
    print(f"❌ BodyExtractor import 실패: {e}")
    import traceback
    traceback.print_exc()

# 3. CrossFilter
try:
    from pose_transfer.logic import CrossFilter, CrossFilterConfig
    print("✅ CrossFilter import 성공")
except Exception as e:
    print(f"❌ CrossFilter import 실패: {e}")
    import traceback
    traceback.print_exc()

# 4. BodyExtractor 인스턴스 생성
try:
    body_extractor = BodyExtractor(mode='balanced', backend='onnxruntime', device='cpu')
    print("✅ BodyExtractor 인스턴스 생성 성공")
except Exception as e:
    print(f"❌ BodyExtractor 인스턴스 생성 실패: {e}")
    import traceback
    traceback.print_exc()

# 5. CrossFilter 인스턴스 생성
try:
    cross_filter = CrossFilter(
        config=CrossFilterConfig(
            body_confidence_threshold=0.3,
            enable_hand_dependency=True,
            enable_foot_dependency=True,
            enable_face_dependency=True,
            dw_min_confidence=0.05
        )
    )
    print("✅ CrossFilter 인스턴스 생성 성공")
except Exception as e:
    print(f"❌ CrossFilter 인스턴스 생성 실패: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("테스트 완료")
print("=" * 80)
