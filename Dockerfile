FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 패키지 설치
COPY requirements.txt .
RUN pip3 install --upgrade pip

# PyTorch (CUDA 12.1)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ONNX Runtime GPU (CUDA 12.x 호환 버전)
RUN pip3 install onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 나머지 의존성
RUN pip3 install -r requirements.txt

# 소스코드 복사
COPY . .

CMD ["/bin/bash"]