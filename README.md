![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-YOLOv8-red "Backend") ![UI](https://img.shields.io/badge/UI-Streamlit-orange "UI") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")
# Dental_008 (Tooth Masking & FDI Assignment)

파노라마 X-ray에서 개별 치아를 Instance Segmentation하고, FDI 시스템(치식)을 매칭하는 모듈입니다.

## 💻 Hardware & Infrastructure
- **Development / Inference Env**: Intel Core i5-14450HX, NVIDIA RTX 4060 Laptop (8GB VRAM), 16GB RAM
- **Training Env (Main Workstation)**: AMD Ryzen 9 9900X, NVIDIA RTX 5080 (16GB VRAM), 64GB RAM
- **데이터 교정**: `Dental_000`의 HiTL UI 도구를 통해 치식 밀림 현상(drift)이 발생한 Edge Case의 마스크를 교정 후 YOLOv8m-seg 재학습 수행.

## 📌 개요

이 프로젝트는 파노라마 X-ray 영상에서 유치(Deciduous)와 영구치(Permanent, 치배 포함)를 개별적으로 분할(Instance Segmentation)하고 FDI 치식 번호를 부여하는 PyTorch 파이프라인입니다. 혼합치열기 환자의 구강 구조에서 유치와 영구치가 겹치거나 치배가 매복된 형태를 픽셀 단위로 정확하게 식별합니다.

추가적으로, 유치가 포함된 소아/혼합치열기 환자 이미지와 성인 영구치열기 이미지를 사전에 구분하는 **유치 존재 여부 이진 분류 모델(Binary Classifier)**이 포함되어, 영구치 전용 모델의 에러를 방지하는 라우터 역할을 수행합니다.

### 🎯 모델 파이프라인 구성 (How it works)

현재 본 프로젝트는 다음과 같이 유기적으로 연결된 다중 모델(Multi-Model) 아키텍처로 구성되어 있습니다:

1. **사전 라우터 모델 (Binary Classifier)**
   * 유치가 포함된 소아/혼합치열기 환자 이미지와 성인 영구치열기 이미지를 사전에 구분하여 영구치 전용 모델의 오류를 방지하는 역할을 수행합니다.
2. **치아 탐지 및 분할 (YOLOv8m-seg Object Detection)**
   * 최신 1-Stage 모델(YOLOv8m-seg)을 사용하여 빠르고 정확하게 치아의 위치(Bounding Box)와 픽셀 윤곽(Mask)을 추출해 냅니다. 특히 75장의 무치악(Edentulous) Hard Negative 데이터를 학습하여 턱뼈나 임플란트 픽스처를 치아로 착각하는 오탐지(FP)를 100% 억제하도록 튜닝되었습니다.
3. **치식 번호 매칭 알고리즘 (Heuristic Sequence Matcher)**
   * 딥러닝이 치아를 찾아내면, 이 알고리즘(Rule-based)이 구강 내 해부학적 기하 구조(Geometry)를 분석하여 각 치아에 올바른 **FDI 치식 번호(예: 11, 48 등)**를 2-Stage로 매핑합니다.


## 📦 Model Weights (Hugging Face)
이 모듈의 학습된 가중치 모델은 Hugging Face 저장소에 연동되어 있습니다. 
아래 링크에서 다운로드할 수 있습니다:
- [Hugging Face Repository (chemahc94/Dental-AI-Models)](https://huggingface.co/chemahc94/Dental-AI-Models/tree/main/Dental_008)

다운로드한 가중치 파일은 이 레포지토리의 해당 모델 폴더에 배치하여 사용하세요.

## 설치 및 실행 방법

본 프로젝트는 표준 Python 라이브러리(`dentex_seg`) 형태로 패키징되어 있습니다.

### 1. 패키지 설치

로컬 가상환경에서 디렉토리 최상단으로 이동한 후 다음 명령어를 실행합니다.

```bash
pip install -e .
```

이는 `requirements.txt`에 명시된 의존성 패키지들과 함께 본 파이프라인을 시스템에 등록합니다.

### 2. 데이터 학습 (Training)

**\[영구치 분할 모델 학습]**

```bash
python src/dentex_seg/train.py
```

* 스크립트를 실행하면 Hugging Face의 `datasets` 라이브러리를 통해 DENTEX 데이터셋이 자동으로 다운로드 및 캐싱됩니다.

**\[유치 존재 여부 이진 분류기 학습]**

```bash
python src/dentex_seg/train_classifier.py
```

* 스크립트 실행 시 로컬의 Kaggle 소아/성인 폴더 구조를 기반으로 분류 모델을 학습합니다.
* 학습된 가중치는 `weights/` 폴더에 저장됩니다.

### 3. 추론 및 시각화 (Inference)

```bash
python src/dentex_seg/inference.py
```

* 추론된 결과 이미지는 `output/` 폴더에 생성됩니다.

***

## 🐳 도커(Docker) 환경 실행 가이드

다른 컴퓨터에서 환경 설정의 번거로움 없이 완벽히 동일한 환경에서 데이터 검증 및 학습을 수행하려면 Docker를 활용하는 것이 가장 좋습니다.

### 1. 도커 컨테이너 빌드 및 백그라운드 학습 시작

최상단 디렉토리에서 다음 명령어를 실행합니다. (NVIDIA GPU 드라이버 및 nvidia-container-toolkit 설치 필수)

```bash
docker-compose up --build -d
```

이 명령어는 다음 작업을 자동으로 수행합니다:

1. PyTorch & CUDA 런타임 환경 이미지 빌드.
2. 파이프라인 의존성 패키지 자동 설치.
3. 데이터셋 다운로드 및 모델 학습(`train.py`) 자동 백그라운드 실행.

### 2. 도커 학습 로그 확인

백그라운드에서 진행 중인 학습 로그를 한글 텍스트로 확인하려면 아래 명령어를 입력합니다.

```bash
docker logs -f dentex_seg_runner
```

### 3. 볼륨(Volume) 마운트 안내

도커 컨테이너 내부에서 생성된 파일은 호스트(현재 컴퓨터)와 동기화됩니다.

* 학습된 가중치는 호스트의 `./weights/` 폴더에 저장됩니다.
* 추론 결과물은 호스트의 `./output/` 폴더에 생성됩니다.
* 데이터셋 다운로드 캐시는 `~/.cache/huggingface`를 공유하여 매번 재다운로드하는 것을 방지합니다.

***

## 📦 데이터셋 및 사전 학습 가중치 다운로드

### 1. 데이터셋 자동 다운로드 및 구성

* **영구치 데이터셋 (DENTEX)**: Hugging Face Open Dataset [LUNA0206/DENTEX](https://huggingface.co/datasets/LUNA0206/DENTEX)
  * `datasets.load_dataset("LUNA0206/DENTEX")` 모듈을 통해 파이프라인 구동 시 자동 다운로드 및 캐싱됩니다.
* **유치 이진 분류 데이터셋 (Kaggle Children's)**: [Childrens dental panoramic radiographs dataset](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset)
  * 이 데이터셋은 `Adult tooth segmentation dataset`과 `Childrens dental caries segmentation dataset` 두 폴더로 나뉘며, 유치/혼합치열기 이진 분류기의 훈련에 사용됩니다.

### 2. 사전 학습 가중치 다운로드 (Git Submodule 연동)

모델 학습에 소요되는 긴 시간을 절약하기 위해, 훈련된 DENTEX 가중치 및 유치 이진 분류기 가중치는 허깅페이스 저장소에 서브모듈로 연동되어 있습니다. 다른 컴퓨터(오케스트레이션 PC 등)에서 이 작업을 그대로 이어가려면 클론 시 아래 명령어를 통해 코드와 가중치를 한 번에 받아올 수 있습니다.

```bash
git clone --recurse-submodules https://github.com/HyunchanAn/Dental_008.git
```

* **가중치 저장소**: [chemahc94/dentex-tooth-segmentation](https://huggingface.co/chemahc94/dentex-tooth-segmentation)
* **참고 사항**: 모델 접근 권한이나 사용에 대해 필요하면 레포지토리 주인에게 연락하세요. 가져온 가중치는 `weights/pretrained/` 경로에 위치하며 바로 검증, 추론 및 라우터 기능에 사용할 수 있습니다.

***

## 📊 E2E 벤치마크 및 성능 검증

본 모델은 영구치열기, 유치열기, 혼합치열기 전반에 걸쳐 강력한 인스턴스 세그멘테이션 성능을 입증했습니다.

* **Average Inference Time**: 0.0596 s/image (16.77 FPS)
* **Average Bounding Box IoU**: 0.8245
* **Average Mask IoU**: 0.7965

상세한 시각화 자료와 성능 평가 지표는 루트 경로에 생성된 **[최종 E2E 검증 리포트](reports_archive/260710_1300_Dentex_E2E_Validation_Report.md)**에서 확인하실 수 있습니다.

***

## 🚀 향후 과제 (TODO)

- **ONNX 포팅 (ONNX Conversion)**: 현재 `.pth` 형태의 모델(객체 분할)은 파이토치 의존성으로 인해 추론 시 오버헤드가 발생할 수 있습니다. 향후 전체 서빙 파이프라인의 추론 속도를 극대화하고 타 모듈들과 추론 엔진(ONNX Runtime 등)을 표준화하기 위해, 복잡한 후처리 로직을 최적화하여 ONNX 포맷으로 변환하는 작업을 진행해야 합니다.

## 📚 References

* [DENTEX Dataset (Hugging Face Open Dataset) - Permanent Teeth Segmentation](https://huggingface.co/datasets/LUNA0206/DENTEX)
* [Childrens Dental Panoramic Radiographs Dataset (Kaggle) - Deciduous Binary Classification](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset)

## 학습 환경 (Training Environment)
> **[학습 환경 사양]** 실질적 모델 학습은 **RTX 5080 + 라이젠9-6 9900x** 환경에서 진행되었습니다.
