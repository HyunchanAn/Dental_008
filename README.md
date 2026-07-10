# Dental\_008: DENTEX 혼합치열기 인스턴스 세그멘테이션

![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Backend](https://img.shields.io/badge/Backend-YOLOv8-red) ![UI](https://img.shields.io/badge/UI-Streamlit-orange) ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github)

## 📌 개요

이 프로젝트는 파노라마 X-ray 영상에서 유치(Deciduous)와 영구치(Permanent, 치배 포함)를 개별적으로 분할(Instance Segmentation)하고 FDI 치식 번호를 부여하는 PyTorch 파이프라인입니다. 혼합치열기 환자의 구강 구조에서 유치와 영구치가 겹치거나 치배가 매복된 형태를 픽셀 단위로 정확하게 식별합니다.

추가적으로, 유치가 포함된 소아/혼합치열기 환자 이미지와 성인 영구치열기 이미지를 사전에 구분하는 **유치 존재 여부 이진 분류 모델(Binary Classifier)**이 포함되어, 영구치 전용 모델의 에러를 방지하는 라우터 역할을 수행합니다.

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

## 📚 References

* [DENTEX Dataset (Hugging Face Open Dataset) - Permanent Teeth Segmentation](https://huggingface.co/datasets/LUNA0206/DENTEX)
* [Childrens Dental Panoramic Radiographs Dataset (Kaggle) - Deciduous Binary Classification](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset)
