import sys

print("-" * 50)
print(f"🐍 Python Version: {sys.version}")
print("-" * 50)

# 1. Numpy 확인
try:
    import numpy
    print(f"✅ Numpy Version: {numpy.__version__}")
except ImportError as e:
    print(f"❌ Numpy Import Failed: {e}")

# 2. Onnxruntime 확인 (rtmlib 의존성)
try:
    import onnxruntime
    print(f"✅ Onnxruntime Version: {onnxruntime.__version__}")
    print(f"   Device: {onnxruntime.get_device()}")
except ImportError as e:
    print(f"❌ Onnxruntime Import Failed: {e}")
    print("   👉 팁: Windows에서 'DLL load failed'가 뜨면 'VC++ 재배포 패키지'가 없거나, CUDA 버전이 안 맞는 경우입니다.")

# 3. rtmlib 확인
try:
    import rtmlib
    print(f"✅ rtmlib Version: {rtmlib.__version__}")
    from rtmlib import Wholebody
    print("🎉 rtmlib Class Import Success!")
except ImportError as e:
    print(f"❌ rtmlib Import Failed: {e}")
except Exception as e:
    print(f"❌ rtmlib Error: {e}")

print("-" * 50)