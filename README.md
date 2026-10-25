![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-YOLOv8m--seg-red "Backend") ![UI](https://img.shields.io/badge/UI-Streamlit-orange "UI") ![License](https://img.shields.io/badge/License-MIT-green "License") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")

# Dental_008: Tooth Masking & FDI Assignment (파노라마 치아 인스턴스 분할 및 FDI 치식 부여)

## 개요

`Dental_008`은 파노라마 방사선 영상에서 유치(Deciduous)와 영구치(Permanent, 치배 포함)를 개별적으로 분할(Instance Segmentation)하고 국제 표준 FDI 치식 번호(`11`~`48`)를 부여하는 코어 파이프라인입니다. 추가적으로 유치가 포함된 소아/혼합치열기 환자 이미지와 성인 영구치열기 이미지를 사전에 구분하는 유치 존재 여부 이진 분류기(ResNet-18 Binary Classifier)가 포함되어 하류 치조골 소실 계측(`Dental_003`) 등의 분기를 제어하는 라우터 역할을 수행합니다.

- [학습 및 추론 하드웨어 사양]:
  - Training Env (메인 워크스테이션): AMD Ryzen 9 9900X, NVIDIA GeForce RTX 5080 (16GB VRAM), 64GB RAM
  - Development / Inference Env (랩탑 워크스테이션): Intel Core i5-14450HX, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), 16GB RAM
- [데이터 교정 및 무누수 분할]: `Dental_000/reports_archive/dental_008_dentex_manifest.csv` (Train 507장 / Val 127장, 파일명 및 SHA-256 해시 교집합 0건) 및 무치악 Hard Negative 분리(Train 60장 / 독립 Holdout 15장, 교집합 0건) 기반 `YOLOv8m-seg` 100 에포크 재학습 수행.

---

## 주요 기능 및 아키텍처

### 1. 다중 모델 파이프라인 구성 (Multi-Model Architecture)
1. [사전 라우터 모델 (Deciduous Binary Classifier)]:
   - 유치가 포함된 소아/혼합치열기 환자 이미지(BGR->RGB 정규화 및 불균형 클래스 가중치 `pos_weight=10.4` 로짓 보정 적용)와 성인 영구치열기 이미지를 구분합니다.
2. [치아 탐지 및 분할 (`YOLOv8m-seg` Instance Segmentation)]:
   - 1-Stage 모델(`YOLOv8m-seg`)을 사용하여 치아의 바운딩 박스와 픽셀 윤곽(Mask)을 추출하며, 무치악(Edentulous) Hard Negative 학습을 통해 보철물·치조능 오탐을 억제합니다.
3. [치식 번호 매칭 알고리즘 (2-Stage Heuristic & DP Sequence Matcher)]:
   - 구강 내 해부학적 악궁 기하 구조(Arch Geometry)와 YOLO 예측 라벨 Prior 보너스(`-4.0`)가 결합된 동적 계획법(DP) 시퀀스 정렬을 통해 도미노 치식 밀림을 방어하고 각 치아에 `FDI 치식 번호(11~48)` 및 불확실(`uncertain`) 플래그를 부여합니다.

### 2. 모델 가중치 (Model Weights on Hugging Face)
- [Hugging Face Repository (`chemahc94/Dental-AI-Models`)](https://huggingface.co/chemahc94/Dental-AI-Models) (`Dental_008_weights_best.pt`, `Dental_008/yolov8m_best.pt`)
- 다운로드한 가중치 파일은 본 레포지토리의 `weights/` 및 `models/` 폴더에 배치하여 사용합니다.

---

## 벤치마크 및 검증 성능

본 모델은 결정론적 무누수 분할 매니페스트(`Dental_000/reports_archive/dental_008_dentex_manifest.csv`, Train 507장 / Val 127장, 파일명 및 SHA-256 해시 교집합 `0건`)와 무치악 Hard Negative 분리(`Train 60장` / 독립 `Holdout 15장`, 교집합 `0건`) 기준으로 메인 워크스테이션(`RTX 5080`, 100 Epochs, `batch=16`, `imgsz=640`)에서 재학습 및 검증되었습니다.

- [YOLOv8m-seg 무누수 독립 검증셋 실측 지표 (`Val 127장`, `3,598개` 치아 인스턴스)]:
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

---

## 데이터셋 출처 및 라이선스

- [DENTEX Dataset (Hugging Face Open Dataset) - Permanent Teeth Segmentation](https://huggingface.co/datasets/LUNA0206/DENTEX) (`CC BY-NC-SA 4.0`, Hamamci et al., 2023)
- [Childrens Dental Panoramic Radiographs Dataset (Kaggle) - Deciduous Binary Classification](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset) (`CC0 1.0`)
- [무치악 Hard Negative 참조 샘플 (`edentulous_images/`)]:
  - 웹 수집 무치악 및 임플란트 파노라마 참조 영상(`*.jpg`, `*.gif`)은 공개 Git 버전 관리에서 제외(`.gitignore` 등록 및 `git rm --cached`)되었으며, 로컬 연구 환경에서만 비영리 오탐(FP) 억제 평가 용도로 보존됩니다([edentulous_images/README.md](edentulous_images/README.md) 참조).
- [소스 코드 라이선스]: `MIT License` ([LICENSE](LICENSE)), 단 `CC BY-NC-SA 4.0` 데이터 기반 학습 가중치는 비영리 연구 조건을 승계합니다.

---

## 설치 및 실행 방법

본 프로젝트는 표준 Python 라이브러리(`dentex_seg`) 형태로 패키징되어 있습니다.

1. 패키지 설치:
   ```bash
   pip install -e .
   ```
2. 무누수 YOLO 데이터셋 생성 및 재학습:
   ```bash
   python prepare_yolo_data.py
   python scripts/prepare_v3_dataset.py
   python train.py
   ```
3. 독립 무치악 홀드아웃(15장) 오탐(FP) 평가 및 유치 분류기 학습:
   ```bash
   python scripts/test_on_edentulous.py
   python src/dentex_seg/train_classifier.py
   ```
4. 추론 및 단위 테스트 실행:
   ```bash
   python src/dentex_seg/inference.py
   pytest ../Dental_000/tests/tests_008 -v
   ```

---

## 참고 문헌 및 연구용 실행 주의사항

- Hamamci, I. E. et al. (2023). DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays. MICCAI 2023.
- [연구용 실행 주의사항]:
  - 기본 실행 주소: `127.0.0.1` (로컬 전용)
  - 외부 공개/상용 배포 금지 및 오픈 데이터셋 외 민감 의료데이터 사용 금지
  - API 키·토큰은 환경변수로만 관리하며, 외부 접근이 필요하면 `ALLOW_EXTERNAL_ACCESS=true`로만 명시적 허용
