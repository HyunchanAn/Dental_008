![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-YOLOv8-red "Backend") ![UI](https://img.shields.io/badge/UI-Streamlit-orange "UI") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")
# Dental_008 (Tooth Masking & FDI Assignment)

파노라마 X-ray에서 개별 치아를 Instance Segmentation하고, FDI 시스템(치식)을 매칭하는 모듈입니다.

## 💻 Hardware & Infrastructure
- [Development / Inference Env]: Intel Core i5-14450HX, NVIDIA RTX 4060 Laptop (8GB VRAM), 16GB RAM
- [Training Env (Main Workstation)]: AMD Ryzen 9 9900X, NVIDIA RTX 5080 (16GB VRAM), 64GB RAM
- [데이터 교정 및 무누수 분할]: `Dental_000/reports_archive/dental_008_dentex_manifest.csv` (Train 507장 / Val 127장, 파일명 및 SHA-256 해시 교집합 0건) 및 무치악 Hard Negative 분리(Train 60장 / 독립 Holdout 15장, 교집합 0건) 기반 `YOLOv8m-seg` 100 에포크 재학습 수행.

## 📌 개요

이 프로젝트는 파노라마 X-ray 영상에서 유치(Deciduous)와 영구치(Permanent, 치배 포함)를 개별적으로 분할(Instance Segmentation)하고 FDI 치식 번호를 부여하는 PyTorch 파이프라인입니다. 혼합치열기 환자의 구강 구조에서 유치와 영구치가 겹치거나 치배가 매복된 형태를 픽셀 단위로 정확하게 식별합니다.

추가적으로, 유치가 포함된 소아/혼합치열기 환자 이미지와 성인 영구치열기 이미지를 사전에 구분하는 [유치 존재 여부 이진 분류 모델(Binary Classifier)]이 포함되어, 영구치 전용 모델의 에러를 방지하는 라우터 역할을 수행합니다.

### 🎯 모델 파이프라인 구성 (How it works)

현재 본 프로젝트는 다음과 같이 유기적으로 연결된 다중 모델(Multi-Model) 아키텍처로 구성되어 있습니다:

1. [사전 라우터 모델 (Binary Classifier)]
   - 유치가 포함된 소아/혼합치열기 환자 이미지와 성인 영구치열기 이미지를 사전에 구분하여 영구치 전용 모델의 오류를 방지하는 역할을 수행합니다.
2. [치아 탐지 및 분할 (YOLOv8m-seg Instance Segmentation)]
   - 1-Stage 모델(`YOLOv8m-seg`)을 사용하여 치아의 위치(Bounding Box)와 픽셀 윤곽(Mask)을 추출합니다. 전체 75장의 무치악(Edentulous) 영상을 고정 시드(`seed=42`) 기준 `Train 60장(Hard Negative)`과 학습에 전혀 사용되지 않은 `독립 Holdout Val 15장`으로 분리(SHA-256 해시 교집합 0건)하여 학습 및 검증한 결과, 독립 무치악 홀드아웃 15장에서 영상당 평균 오탐 `0.20 FP/image` (총 3건 오탐, 무오탐 영상 비율 `86.7%`, 13/15장)를 기록하였습니다.
3. [치식 번호 매칭 알고리즘 (Heuristic Sequence Matcher)]
   - 딥러닝이 치아를 찾아내면, 구강 내 해부학적 기하 구조(Geometry) 및 DP 시퀀스 정렬을 통해 각 치아에 올바른 `FDI 치식 번호(예: 11, 48 등)`를 2-Stage로 매핑합니다.

## 📦 Model Weights (Hugging Face)
이 모듈의 학습된 가중치 모델은 Hugging Face 저장소에 연동되어 있습니다.
아래 링크에서 다운로드할 수 있습니다:
- [Hugging Face Repository (chemahc94/Dental-AI-Models)](https://huggingface.co/chemahc94/Dental-AI-Models) (`Dental_008_weights_best.pt`, `Dental_008/yolov8m_best.pt`)

다운로드한 가중치 파일은 이 레포지토리의 `weights/` 및 `models/` 폴더에 배치하여 사용하세요.

## 설치 및 실행 방법

본 프로젝트는 표준 Python 라이브러리(`dentex_seg`) 형태로 패키징되어 있습니다.

### 1. 패키지 설치

```bash
pip install -e .
```

### 2. 무누수 데이터셋 생성 및 학습 (Training)

[무누수 YOLO 데이터셋 및 무치악 홀드아웃 생성]
```bash
python prepare_yolo_data.py
python scripts/prepare_v3_dataset.py
```

[영구치 분할 모델 100 에포크 재학습 (RTX 5080)]
```bash
python train.py
```

[독립 무치악 홀드아웃(15장) 오탐(FP) 평가]
```bash
python scripts/test_on_edentulous.py
```

[유치 존재 여부 이진 분류기 학습]
```bash
python src/dentex_seg/train_classifier.py
```

### 3. 추론 및 시각화 (Inference)

```bash
python src/dentex_seg/inference.py
```

***

## 📊 E2E 벤치마크 및 무누수 검증셋(507/127) 실측 성적표

본 모델은 결정론적 무누수 분할 매니페스트(`Dental_000/reports_archive/dental_008_dentex_manifest.csv`, Train 507장 / Val 127장, 파일명 및 SHA-256 해시 교집합 `0건`)와 무치악 Hard Negative 분리(`Train 60장` / 독립 `Holdout 15장`, 교집합 `0건`) 기준으로 메인 워크스테이션(`RTX 5080`, 100 Epochs, `batch=16`, `imgsz=640`)에서 재학습 및 검증되었습니다.

- [YOLOv8m-seg 무누수 독립 검증셋 실측 지표 (Val 127장, 3,598개 치아 인스턴스)]:
  - 가중치 파일: `weights/best.pt` (`weights/yolov8m_best.pt`, SHA-256: `88db875aa9ec209731dfce6fa630b4ee4169e3a57b084d585d5aa150263fe9b7`)
  - Bounding Box: Precision `99.06%` (`0.9906`), Recall `99.06%` (`0.9906`), `mAP@0.5` `99.15%` (`0.9915`), `mAP@0.5:0.95` `60.37%` (`0.6037`)
  - Segmentation Mask: Precision `97.75%` (`0.9775`), Recall `97.75%` (`0.9775`), `mAP@0.5` `98.55%` (`0.9855`), `mAP@0.5:0.95` `55.64%` (`0.5564`)
  - 추론 속도(RTX 5080): `7.1 ms/image` (`0.8ms` preprocess + `3.3ms` inference + `3.0ms` postprocess, `140.8 FPS`)
- [독립 무치악 홀드아웃 검증셋 실측 지표 (`images/edentulous_val` 15장, 학습 미사용)]:
  - 평가 조건: `conf=0.25` 및 `conf=0.20` (학습에 투입된 60장과 파일명·SHA-256 해시 교집합 `0건`)
  - 총 오탐(FP) 수: 15장 중 총 `3건` (`edentulous_0072.jpg` 1건, `edentulous_0078.jpg` 2건)
  - 영상당 평균 오탐: `0.20 FP/image`
  - 무오탐 영상 비율(Zero-FP Image Rate): `86.67%` (`13 / 15장` 완전 무오탐)
- [10례 임상 정량 벤치마크 (`Dental_000/reports_archive/10_cases_quantitative_benchmark.json`)]:
  - 검출률(Detection Rate): `100.0%` (10/10 증례, 증례당 평균 28.9개 치아 식별)
- [데이터 누수(Data Leakage) 감사 및 조치 기록]:
  - 과거 `data/yolo_fdi/images/val` (38장)에서 산출되었던 구 지표는 학습셋 중복으로 확인되어 공식 폐기되었으며, `prepare_yolo_data.py` 및 `src/dentex_seg/dataset.py`에 파일명·SHA-256 해시 교집합 `0건` 런타임 검증(`AssertionError`)이 적용되었습니다.

***

## 🚀 완료 현황 (Status)

- [완료] Train/Val 중복 누수 차단 가드 및 무누수 분할 매니페스트(`dental_008_dentex_manifest.csv`, 507/127, SHA-256 교집합 0) 구축 완료
- [완료] 무치악 Hard Negative 75장의 학습(60장) / 독립 홀드아웃 검증(15장) 분리 및 오탐률(`0.20 FP/img`, 무오탐 `86.7%`) 실측 완료
- [완료] 메인 워크스테이션(RTX 5080) 무누수 분할(507/127) 기반 `YOLOv8m-seg` 100 에포크 재학습 및 `Box/Mask mAP` 실측 완료
- [완료] 재학습 가중치(`best.pt`) Hugging Face (`chemahc94/Dental-AI-Models`) 배포 및 `weights_manifest.yaml` SHA-256 동기화 완료

## 📚 References

- [DENTEX Dataset (Hugging Face Open Dataset) - Permanent Teeth Segmentation](https://huggingface.co/datasets/LUNA0206/DENTEX)
- [Childrens Dental Panoramic Radiographs Dataset (Kaggle) - Deciduous Binary Classification](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset)

## 학습 환경 (Training Environment)
> [학습 환경 사양] 실질적 모델 학습은 RTX 5080 + 라이젠9-6 9900X 환경에서 진행되었습니다.

## 연구용 실행 주의사항
- 기본 실행 주소: 127.0.0.1 (로컬 전용)
- 외부 공개/상용 배포 금지
- API 키·토큰은 환경변수로만 관리
- 오픈 데이터셋 외 민감 의료데이터 사용 금지
- 외부 접근이 필요하면 ALLOW_EXTERNAL_ACCESS=true 로만 명시적 허용
