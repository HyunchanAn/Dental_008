# 베이스 이미지 (RTX 5080 Blackwell 지원을 위해 CUDA 12.6 공식 런타임 사용)
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트, Python 3.10 및 OpenCV 의존성 설치
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# python 명령어 링크 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# pip 업그레이드
RUN python -m pip install --upgrade pip

# PyTorch (CUDA 12.6) 설치
# RTX 5080 (sm_120)을 정상 지원하기 위해 cu126 휠 설치
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 요구사항 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 전체 복사
COPY . .

# 패키지 설치
RUN pip install -e .

# 기본 실행 명령어 (학습 코드 실행)
CMD ["python", "src/dentex_seg/train.py"]
